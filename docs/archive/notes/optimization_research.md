# Optimization Research: Paged Attention, Prefix Caching, CUDA Graph

> Based on current codebase analysis (post R1-R6 + cuDNN cleanup).
> Target: single-GPU multi-user serving on B200/4090.

---

## 1. Paged Attention Decode Kernel Optimization

### 1.1 Current Performance Problem

nsys profiling on B200 (DeepSeek-R1-Distill-Qwen-1.5B, BF16):

| Kernel | Avg per call | Total (10K calls) | % of GPU time |
|--------|-------------|-------------------|---------------|
| `paged_attention_decode_kernel` | **129.5us** | 1,381ms | **64.4%** |
| All cuBLAS GEMV combined | ~5-8us each | ~490ms | 23% |
| All other kernels | 2-3us each | ~250ms | 12% |

The paged decode kernel is the **dominant bottleneck**, taking 64% of total GPU time. For comparison, the deleted contiguous decode kernel averaged 25.5us — **5x faster**.

### 1.2 Root Cause Analysis

Current kernel: `paged_attention_nvidia.cu:72-184`

```
Config: Grid=(nhead), Block=256, smem=(d + PA_TILE_KV + NUM_WARPS) floats
```

**Problem 1: Block-table indirect memory access**

Every K/V access requires:
```cpp
int block_idx = j_global / block_size;        // integer division
int block_offset = j_global % block_size;      // modulo
int k_physical = k_block_table[block_idx] * block_size + block_offset;
const T *k_ptr = pool_base + k_physical * nkvhead * d + kvh * d;
```

This is 3 extra operations + 1 global memory read (block table) per K/V token access. The indirect address prevents GPU hardware prefetching — the memory controller can't predict the next address because it depends on the block table content.

With blocks scattered across a 15GB pool, the L2 cache hit rate is low. Each block table lookup fetches a cache line (128 bytes) for a single 4-byte int, wasting bandwidth.

**Problem 2: V aggregation thread utilization**

```cpp
const int dv_idx = tid % dv;              // dv = head_dim = 128
const int dv_group = blockDim.x / dv;     // 256 / 128 = 2
const int dv_rank = tid / dv;             // 0 or 1
```

With Block=256 and head_dim=128: `dv_group=2`. Each V element is touched by 2 threads with strided access. This creates bank conflicts and suboptimal memory coalescing.

For the contiguous kernel (deleted), V access was sequential — thread `tid` accessed `V[j * nkvhead * dv + kvh * dv + tid]` which was perfectly coalesced. With paged access, `V[v_physical * nkvhead * dv + ...]` where `v_physical` jumps across blocks, destroying coalescing.

**Problem 3: High variance (max 7.96ms)**

For late decode steps (seq_len ~255), the kernel iterates over 16 blocks (256/16). If blocks are scattered across the pool, each block access may miss L2 cache, causing ~200ns per miss. With 255 tokens × 2 (K+V) × 200ns = ~100us just for cache misses. The 7.96ms max suggests occasional severe TLB misses or page table walks.

### 1.3 Optimization Approaches

#### Approach A: Integrate FlashDecoding / FlashInfer

**FlashDecoding** (Dao-AILab) splits the KV sequence across multiple thread blocks, each computing a partial attention, then reduces. Key benefit: parallelism across KV length.

**FlashInfer** (flashinfer.ai) provides highly optimized paged attention kernels with:
- Ragged/paged KV cache native support
- Split-K parallelism for long sequences
- Cooperative fetch of K/V blocks into shared memory
- Supports `block_table` parameter natively
- C++ header-only API, BSD-3 license

```cpp
// FlashInfer API sketch:
flashinfer::BatchDecodeWithPagedKVCache(
    q, paged_kv_cache, output,
    /*rotary_mode=*/flashinfer::RotaryMode::kNone,
    /*q_rope_offset=*/nullptr, /*kv_rope_offset=*/nullptr);
```

**Expected speedup:** 3-5x for decode (bringing it close to or better than contiguous)
**Integration effort:** Medium — need to adapt block pool layout to FlashInfer's expected format

#### Approach B: Optimize current kernel

1. **Shared memory K/V tile loading:** Load an entire block (16 tokens × head_dim) into shared memory in one coalesced transaction, then compute attention scores from shared memory. Eliminates per-token global memory access.

```
Current:  for each KV token → read K from global (scattered) → compute score
Proposed: for each block → load block to smem (coalesced) → compute scores from smem
```

2. **Block-aligned tiling:** Set `PA_TILE_KV = block_size` (16). Each tile corresponds to exactly one physical block, so the K/V data is contiguous within the tile. This restores coalescing within tiles.

3. **V aggregation restructure:** Use all 256 threads for both K scoring and V aggregation phases. For K: warp-level dot products (current). For V: each thread accumulates a partial sum across tiles, then reduce.

4. **Split-K for long sequences:** When seq_len > threshold (e.g., 512), launch multiple blocks per head, each handling a subset of KV tiles. Final reduction produces the output.

**Expected speedup:** 2-3x for decode
**Integration effort:** Low — kernel rewrite within existing framework

#### Approach C: Paged prefill with FlashAttention C++ API

The current paged prefill kernel (`paged_attention_nvidia.cu:360-454`) is naive:
- Grid=(seqlen_q, nhead): one block per (query, head)
- No IO-aware tiling, no Q-Q reuse
- Each block re-reads K independently

For long prompts (2K+), FlashAttention-2 with paged support would be 2-4x faster.

**flash-attn C++ API** (Dao-AILab/flash-attention, BSD-3):
- `flash_attn_with_kvcache()` supports `block_table` parameter
- Supports `cu_seqlens` for variable-length batched prefill
- Highly optimized for Hopper/Blackwell

**Expected speedup:** 2-4x for prefill
**Integration effort:** Medium-High — compile flash-attn from source, adapt tensor layout

### 1.4 Recommended Approach

**Phase 1 (High priority):** Optimize decode kernel (Approach B) — immediate 2-3x win with no external dependency.

**Phase 2 (Medium priority):** Integrate FlashInfer or flash-attn for both decode and prefill — additional 1.5-2x on top of Phase 1.

---

## 2. CUDA Graph Optimization

### 2.1 Current Decode Path Overhead

Each decode step (single token) launches approximately:

```
Per layer (28 layers for Qwen-1.5B):
  1× rms_norm kernel
  3× cuBLAS GEMV (Q, K, V projections)
  [2× rms_norm if Qwen3 Q/K norm]
  2× rope kernel
  write_kv: 2× cudaMemcpyAsync (K scatter + V scatter)
  1× paged_attention_decode kernel
  1× cuBLAS GEMV (O projection)
  1× add kernel
  1× rms_norm kernel
  3× cuBLAS GEMV (gate, up, down)
  1× swiglu kernel
  1× add kernel
= ~15 kernel launches + 2 memcpy per layer

Plus outside layers:
  1× embedding kernel
  1× rms_norm kernel
  1× cuBLAS GEMV (lm_head)
  1× argmax kernel
  1× cudaMemcpy D2H (argmax result)

Total: ~15 × 28 + 5 = ~425 kernel launches per decode step
       + ~56 cudaMemcpyAsync calls
```

From nsys data, kernel launch overhead:
- `cudaLaunchKernel`: avg 4.9us × 87K calls = 427ms total
- `cuLaunchKernelEx`: avg 5.1us × 75K calls = 384ms total
- `cudaMemcpyAsync`: avg 4.3us × 28K calls = 123ms total

That's ~934ms of CPU-side launch overhead for ~10K decode steps. Per decode step: ~93us of pure launch overhead. At 2.7ms per token (370 tok/s contiguous), that's 3.4% overhead. At 9.1ms per token (110 tok/s paged), it's 1% — less significant because the kernel itself is slow.

**After paged attention optimization (target ~3ms/token): launch overhead becomes ~3% again.** CUDA graph would eliminate this 3%.

### 2.2 Requirements for CUDA Graph Capture

CUDA graph capture requires:

| Requirement | Current Status | Blocker? |
|-------------|---------------|----------|
| Fixed GPU memory addresses | `Tensor::create()` per step via BestFitMemoryPool → may return different addresses | **Yes** |
| No dynamic cudaMalloc inside capture | Pool may grow (new `cudaMalloc`) if needed | **Yes** |
| Fixed kernel configurations | All kernel configs fixed for decode (N=1, same shapes) | OK |
| No CPU-side data-dependent branches | `build_decode_cache()` has `if (cached_num_decode_)` | Minor (host-side only) |
| Fixed memcpy addresses | KV scatter addresses depend on block table → change per step | **Yes** |
| cuBLAS handle reused | Singleton `handle()` in `linear_cublas.cu` | OK |

### 2.3 What Must Change

**2.3.1 Pre-allocated decode scratch buffers**

Replace `Tensor::create()` in `transformer_forward` decode path with pre-allocated buffers:

```cpp
struct DecodeScratch {
    tensor_t hidden;    // [1, hidden_size]
    tensor_t normed;    // [1, hidden_size]
    tensor_t q, k, v;   // projections
    tensor_t q_rope, k_rope;
    tensor_t attn_out;  // [1, nhead, head_dim]
    tensor_t gate, up, act, down;
    tensor_t logits;    // [1, vocab_size]
    tensor_t ids;       // [1] I32
    tensor_t pos_ids;   // [1] I64
};
```

Allocated once at engine init. All `make({1, dim})` calls in `transformer_forward` replaced with views into these buffers. For decode (N=1), every buffer shape is fixed → same GPU addresses every step.

**Total scratch memory for Qwen-1.5B (BF16):** ~600KB. Negligible.

**2.3.2 KV scatter must use fixed addresses or be part of the graph**

Current KV scatter does `cudaMemcpyAsync` with addresses computed from `block_table`. These addresses change per step (new token → potentially new block offset).

Options:
a) **Exclude KV scatter from the graph.** Capture only the compute kernels (embedding → attention → logits). KV scatter runs before/after the graph. Simple but doesn't eliminate KV scatter overhead.

b) **Replace KV scatter with a custom CUDA kernel.** The scatter kernel takes `(src, block_table, past_len, pool_base)` as arguments. The kernel computes the destination address internally. Since kernel arguments can be updated between graph replays via `cudaGraphExecKernelNodeSetParams`, this works with CUDA graph.

c) **Use CUDA graph with graph update.** Capture the graph including memcpy nodes, then update the memcpy addresses before each replay via `cudaGraphExecMemcpyNodeSetParams`. More complex but captures everything.

**Recommended: Option (b)** — a small scatter kernel is simple and eliminates the memcpy API calls entirely.

**2.3.3 Attention kernel block table argument**

The paged attention decode kernel takes `k_block_table` and `v_block_table` as kernel arguments (pointers). These point to the same `SequenceBlockTable` data every step — the block table grows but existing entries don't change. The kernel reads `seq_len` as an argument which does change.

For CUDA graph: capture the kernel once with initial `seq_len`, then update `seq_len` via `cudaGraphExecKernelNodeSetParams` before each replay. Or pass `seq_len` via a device-side memory location.

**2.3.4 Argmax sync point**

`ArgmaxSampler::sample()` does a D2H memcpy to get the sampled token. This is a sync point between graph replays. The graph captures up to logits, then CPU reads the argmax result, updates the input token, and replays.

### 2.4 Implementation Plan

```
Phase 1: DecodeScratch pre-allocation
  - Allocate fixed buffers at engine init
  - Modify transformer_forward to use scratch for N=1
  - Verify bit-identical output
  - Benchmark: should see ~5-10% improvement from reduced malloc/free

Phase 2: CUDA graph capture for compute kernels
  - Capture: embedding → layers → logits (excluding KV scatter and argmax)
  - KV scatter: custom CUDA kernel with graph-compatible arguments
  - Argmax: remains outside the graph (sync point)
  - Graph update: set input_token, position_id, seq_len per replay
  - Benchmark: expect 10-20% decode latency reduction

Phase 3: Full graph with argmax
  - Move argmax into the graph
  - Read result from pinned memory (no device sync needed)
  - Benchmark: expect additional 2-5% improvement
```

### 2.5 Expected Impact

| Scenario | Current | With CUDA Graph | Speedup |
|----------|---------|-----------------|---------|
| 425 kernel launches × ~5us overhead | ~2.1ms/step | ~0.01ms/step (single launch) | Saves ~2ms |
| 56 memcpy calls × ~4us overhead | ~0.2ms/step | 0 (scatter kernel in graph) | Saves ~0.2ms |
| Tensor::create/destroy overhead | ~0.5ms/step | 0 (pre-allocated) | Saves ~0.5ms |
| **Total per step** | **~2.8ms overhead** | **~0.01ms** | **Saves ~2.8ms** |

At current 9.1ms/step (110 tok/s): 9.1 - 2.8 = 6.3ms → 203 tok/s (+85%)
After paged attention optimization (~4ms/step): 4.0 - 2.8 = 1.2ms → 1067 tok/s (theoretical max, likely limited by kernel time)

**CUDA graph is most impactful after paged attention is optimized** — when kernel time is reduced, launch overhead becomes a larger fraction.

### 2.6 Dependency

CUDA graph requires **DecodeScratch** (fixed memory addresses). This is independent of paged attention optimization. Can be developed in parallel:

```
Paged attention optimization ─┐
                               ├──→ Combined: ~300-400 tok/s
DecodeScratch + CUDA graph ───┘
```

---

## 3. Prefix Caching

### 3.1 What Is Prefix Caching

When multiple requests share the same prompt prefix (e.g., system prompt "You are a helpful assistant..."), each request currently computes the KV cache for that prefix independently. Prefix caching stores the KV blocks for common prefixes and **shares** them across requests.

Example:
```
Request A: [system_prompt] + "What is 2+2?"
Request B: [system_prompt] + "Tell me a joke"

Without prefix caching:
  A prefills: system_prompt (200 tokens) + question → 200 blocks computed
  B prefills: system_prompt (200 tokens) + question → 200 blocks computed (redundant!)

With prefix caching:
  A prefills: system_prompt → blocks stored in hash table
  B matches: system_prompt → reuse A's blocks → only prefill question
  Savings: 200 tokens of prefill skipped for B (and C, D, E...)
```

### 3.2 Current Block Pool Architecture

```
BlockPool: single contiguous allocation [num_blocks × block_bytes]
  ├── allocated_: vector<bool> bitmap (no reference counting)
  ├── allocate(): scan bitmap, return free block_id
  └── free(): mark block as available

SequenceBlockTable (per request):
  ├── k_blocks[layer][block_idx] = physical block_id
  ├── v_blocks[layer][block_idx] = physical block_id
  └── seq_len = tokens cached

BlockAllocator:
  ├── allocate_sequence(estimated_tokens): pre-allocate blocks
  ├── extend_sequence(table, layer, is_k): add one block
  └── free_sequence(table): free all blocks
```

**No reference counting.** Each block is exclusively owned by one sequence. When the sequence completes, all its blocks are freed. No sharing mechanism exists.

### 3.3 What Needs to Change

#### 3.3.1 Block Reference Counting

```cpp
// Replace vector<bool> allocated_ with:
struct BlockMeta {
    int ref_count = 0;       // 0 = free, 1 = exclusive, >1 = shared
    uint64_t content_hash = 0; // hash of token IDs in this block
    bool complete = false;    // true if block is full (block_size tokens)
};
std::vector<BlockMeta> block_meta_; // [num_blocks]
```

- `allocate()`: find block with `ref_count == 0`, set to 1
- `share(block_id)`: increment `ref_count`
- `release(block_id)`: decrement `ref_count`, free if reaches 0

#### 3.3.2 Content Hash Table

Map token content to physical blocks:

```cpp
// Hash key: (layer, token_hash) → physical block_id
// token_hash = hash of the block_size token IDs that fill this block
struct PrefixKey {
    int layer;
    uint64_t token_hash;  // hash of tokens [block_start .. block_start + block_size)
};
std::unordered_map<PrefixKey, int> prefix_cache_;
```

When a new request's prompt is tokenized:
1. Divide into blocks of `block_size` tokens
2. For each block, compute hash of the token IDs
3. Look up in `prefix_cache_`
4. If found: share the existing block (increment ref_count)
5. If not found: allocate new block, compute KV, insert into cache

#### 3.3.3 Copy-on-Write (COW)

When a shared block needs modification (shouldn't happen for prefix — prefix blocks are read-only), allocate a new block and copy data. For prefix caching this is rarely needed since prefix blocks are immutable.

However, the **last block** of a prefix may be partially filled. If two requests share a prefix of 200 tokens with block_size=16, blocks 0-11 are full (192 tokens) and block 12 has 8 tokens. If request A and B both continue from token 200:
- Block 12 is shared but needs different new tokens for A vs B
- COW: allocate a new block for B, copy the 8 shared tokens, then append B's new tokens

```cpp
int cow_block(int block_id, int layer, bool is_k) {
    if (block_meta_[block_id].ref_count <= 1) return block_id; // exclusive, no copy needed

    int new_block = allocate();
    memcpy(block_data(new_block), block_data(block_id), block_bytes_);
    release(block_id);
    return new_block;
}
```

#### 3.3.4 Scheduler Integration

The scheduler's admission control must account for shared blocks:

```
can_admit(request):
  tokens_needed = request.prompt_tokens
  blocks_needed = ceil(tokens_needed / block_size) * num_layers * 2

  // Check how many blocks can be reused from prefix cache
  shared_blocks = count_prefix_matches(request.prompt_tokens)
  new_blocks_needed = blocks_needed - shared_blocks

  return allocator.available_blocks() >= new_blocks_needed
```

#### 3.3.5 Eviction Policy

When the block pool is full and new blocks are needed, evict cached (non-active) prefix blocks:

```
evict_prefix_blocks(count):
  candidates = blocks where ref_count == 1 && in prefix_cache
  sort by LRU (least recently accessed)
  for block in candidates[:count]:
    prefix_cache_.erase(block.hash_key)
    release(block)
```

### 3.4 How vLLM Does It

vLLM's prefix caching (Automatic Prefix Caching, APC):

1. **Hash-based block matching:** Each block's content is hashed by its token IDs. The hash includes the parent block's hash (chain hash) to ensure prefix ordering.

2. **Evictor:** LRU evictor for cached blocks. Blocks are evicted when memory pressure is high.

3. **Tree structure:** Prefixes form a tree — "You are a helpful" is a parent of "You are a helpful assistant" and "You are a helpful bot".

4. **Integration with scheduler:** The scheduler checks prefix matches before allocating blocks, reducing the number of new blocks needed.

### 3.5 Impact Estimate

For a typical chat serving scenario:
- System prompt: 100-500 tokens (shared across all requests)
- User message: 10-200 tokens (unique per request)

With system prompt of 200 tokens:
- Without caching: each request prefills 200 + message tokens
- With caching: first request prefills 200 tokens, subsequent requests prefill only the message

For 100 concurrent requests with the same system prompt:
- Without: 100 × 200 = 20,000 tokens of redundant prefill
- With: 200 tokens (once) + 100 × message tokens
- Savings: ~99% of system prompt prefill compute

### 3.6 Implementation Complexity

| Component | Complexity | Risk |
|-----------|-----------|------|
| Block ref counting | Low | Low — simple counter |
| Content hash table | Medium | Low — standard hash map |
| COW for partial blocks | Medium | Medium — edge cases |
| Scheduler integration | Medium | Low — additive change |
| Eviction policy | Low | Low — LRU is straightforward |
| Testing (correctness) | High | Medium — must verify shared blocks produce identical KV |

### 3.7 Dependency

Prefix caching is **independent** of paged attention optimization and CUDA graph. Can be developed in parallel. But it's most valuable after the HTTP server is in production (many users sharing system prompts).

---

## 4. Priority and Ordering

| Optimization | Impact | Effort | Dependency |
|-------------|--------|--------|-----------|
| **Paged attention decode optimization** | Critical (5x kernel speedup) | Medium | None |
| **DecodeScratch pre-allocation** | Medium (saves ~0.5ms/step) | Low | None |
| **CUDA graph** | High (saves ~2ms/step) | Medium | DecodeScratch |
| **Paged attention prefill (FlashAttn)** | High for long prompts | Medium-High | None |
| **Prefix caching** | High for multi-user serving | Medium | None |

Recommended order:

```
Step 1: Paged attention decode kernel optimization  ← most urgent
Step 2: DecodeScratch pre-allocation                ← foundation for CUDA graph
Step 3: CUDA graph capture for decode               ← reduces launch overhead
Step 4: Prefix caching                              ← multi-user efficiency
Step 5: FlashAttention for paged prefill            ← long-prompt optimization
```

Steps 1-3 are the critical path for decode throughput. Step 4 is independently valuable for serving. Step 5 is valuable when long-context workloads are common.

---

## 5. Combined Impact Estimate (Qwen-1.5B, B200, Decode)

| State | tok/s | Per token | Notes |
|-------|-------|-----------|-------|
| Current (paged, no opt) | ~110 | 9.1ms | Attention kernel dominates |
| + Paged attention opt (2-3x) | ~250-300 | 3.3-4.0ms | Kernel time reduced |
| + DecodeScratch + CUDA graph | ~400-500 | 2.0-2.5ms | Launch overhead eliminated |
| + FlashInfer decode kernel | ~500-600 | 1.7-2.0ms | Near-optimal kernel |
| Theoretical limit (kernel only) | ~700+ | <1.5ms | Compute-bound |
