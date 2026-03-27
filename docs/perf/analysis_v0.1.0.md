# Performance Analysis: ZedInfer v0.1.0

> Analysis of the performance regression/improvement after paged KV cache migration,
> and gap analysis against mainstream inference engines.
>
> Test hardware: NVIDIA RTX 4090, BF16
> Test date: 2026-03-26

---

## 1. What Changed: Linear → Paged

| Component | Before (2026/01/19) | After (2026/03/26) |
|-----------|--------------------|--------------------|
| KV Cache | Contiguous (DynamicKVCache) | Paged (BlockPool + BlockAllocator) |
| Attention (decode) | Contiguous kernel, Grid=(nhead,) | Paged kernel, Grid=(nhead,), block table indirection |
| Attention (prefill) | Contiguous kernel, Grid=(seqlen, nhead) | Paged kernel, Grid=(seqlen, nhead), block table indirection |
| Attention (batched) | N/A | Paged kernel, Grid=(num_reqs, nhead) |
| Memory management | Per-request malloc/free | Pre-allocated block pool, ref counting |
| Prefix caching | None | Chain-hashed block sharing |
| Decode allocation | Tensor::create per step | DecodeScratch pre-allocated buffers |

---

## 2. Old vs New: What Improved, What Regressed

### 2.1 DeepSeek-R1-Distill-Qwen-1.5B (12 heads, head_dim=128)

**Prefill (tok/s) — REGRESSED significantly:**

| Prompt Length | Old (linear) | New (paged) | Delta |
|--------------|-------------|-------------|-------|
| 128 | 8,908 | 4,661 | **-48%** |
| 256 | 9,714 | 5,329 | **-45%** |
| 512 | 12,148 | 5,566 | **-54%** |
| 1024 | 13,579 | 5,612 | **-59%** |

Prefill throughput dropped by ~50% across all sequence lengths, with the gap widening as sequences get longer.

**Decode (tok/s) — improved slightly:**

| Prompt Length | Old (linear) | New (paged) | Delta |
|--------------|-------------|-------------|-------|
| 128 | 172.5 | 189.3 | **+10%** |
| 256 | 153.8 | 170.2 | **+11%** |
| 512 | 126.6 | 141.2 | **+12%** |
| 1024 | 93.0 | 105.3 | **+13%** |

Decode improved ~10-13%, likely from DecodeScratch pre-allocation (eliminating ~500 Tensor::create per step) and shared memory address precomputation in the paged decode kernel.

### 2.2 DeepSeek-R1-0528-Qwen3-8B (32 heads, head_dim=128)

**Prefill (tok/s) — regressed, worse at long sequences:**

| Prompt Length | Old (linear) | New (paged) | Delta |
|--------------|-------------|-------------|-------|
| 128 | 2,529 | 2,980 | **+18%** |
| 256 | 3,567 | 3,374 | **-5%** |
| 512 | 4,061 | 2,914 | **-28%** |
| 1024 | 3,316 | 2,261 | **-32%** |

Short prefill (128) actually improved (more heads = more blocks = better SM utilization), but 512+ degraded significantly.

**Decode (tok/s) — mixed:**

| Prompt Length | Old (linear) | New (paged) | Delta |
|--------------|-------------|-------------|-------|
| 128 | 52.6 | 54.0 | **+3%** |
| 256 | 48.3 | 50.4 | **+4%** |
| 512 | 44.7 | 43.3 | **-3%** |
| 1024 | 39.2 | 35.3 | **-10%** |

For 8B model, decode at 512+ tokens is worse. The paged kernel's block table indirection overhead becomes more visible with 32 KV heads and longer sequences.

---

## 3. New ZedInfer vs Other Engines

### 3.1 Prefill Comparison (tok/s)

**1.5B:**

| Prompt | ZedInfer | llama.cpp | vLLM | SGLang | Gap vs best |
|--------|----------|-----------|------|--------|-------------|
| 128 | 4,661 | 15,076 | 13,332 | 5,967 | **3.2x** behind |
| 256 | 5,329 | 21,949 | 21,961 | 12,178 | **4.1x** behind |
| 512 | 5,566 | 24,647 | 29,476 | 23,853 | **5.3x** behind |
| 1024 | 5,612 | 24,259 | 36,611 | 34,162 | **6.5x** behind |

**8B:**

| Prompt | ZedInfer | llama.cpp | vLLM | SGLang | Gap vs best |
|--------|----------|-----------|------|--------|-------------|
| 128 | 2,980 | 4,442 | 4,666 | 3,001 | **1.6x** behind |
| 256 | 3,374 | 6,601 | 7,467 | 5,246 | **2.2x** behind |
| 512 | 2,914 | 7,430 | 8,563 | 6,860 | **2.9x** behind |
| 1024 | 2,261 | 7,040 | 9,284 | 8,213 | **4.1x** behind |

Key observation: the gap **widens with sequence length**. At prompt=128 the gap is 1.6-3.2x, at 1024 it's 4.1-6.5x. This is the hallmark of a naive attention kernel that doesn't scale.

### 3.2 Decode Comparison (tok/s)

**1.5B:**

| Prompt | ZedInfer | llama.cpp | vLLM | SGLang | Gap vs best |
|--------|----------|-----------|------|--------|-------------|
| 128 | 189.3 | 245.4 | 209.8 | 247.4 | **1.3x** behind |
| 256 | 170.2 | 245.6 | 209.8 | 246.4 | **1.4x** behind |
| 512 | 141.2 | 241.5 | 209.2 | 245.8 | **1.7x** behind |
| 1024 | 105.3 | 241.0 | 208.3 | 244.4 | **2.3x** behind |

**8B:**

| Prompt | ZedInfer | llama.cpp | vLLM | SGLang | Gap vs best |
|--------|----------|-----------|------|--------|-------------|
| 128 | 54.0 | 58.3 | 57.0 | 59.5 | **1.1x** behind |
| 256 | 50.4 | 58.3 | 56.7 | 59.2 | **1.2x** behind |
| 512 | 43.3 | 57.7 | 56.5 | 58.9 | **1.4x** behind |
| 1024 | 35.3 | 57.8 | 56.1 | 58.3 | **1.7x** behind |

Key observation: other engines maintain **near-constant decode throughput** across all sequence lengths. ZedInfer's decode throughput **drops linearly** with sequence length. At 1024, the gap is 1.7-2.3x.

---

## 4. Root Cause Analysis

### 4.1 Prefill Kernel: No Q-Tiling (Primary Bottleneck)

**Current implementation** (`paged_attention_nvidia.cu:350-450`):

```
Grid:  (seqlen_q, nhead)
Block: (256,)   // 8 warps
```

Each CUDA block handles **one query position** against all KV tokens. This means:

- **No Q-tiling**: K/V data is loaded from global memory once per query position. With `seqlen_q=1024` and `nhead=12`, there are 12,288 blocks, each independently loading the full KV sequence. No shared memory reuse of K/V across neighboring Q positions.
- **No IO-aware tiling**: Flash Attention's core insight is to tile both Q and K/V so that K/V blocks stay in shared memory and are reused across multiple Q rows. Our kernel misses this entirely.
- **Block table overhead scales linearly**: For each KV position, the kernel computes `block_table[j/block_size] * block_size + j%block_size`. This indirection is in the inner loop (line 394-397) without precomputation (unlike the decode kernel which uses `s_phys[]`).

**Why it scales poorly**: At seqlen=128, each block reads a small KV range. At seqlen=1024, each block reads 1024 KV positions — the memory bandwidth demand grows linearly while compute stays the same (memory-bound regime). Flash Attention amortizes this by keeping KV tiles in SRAM and processing multiple Q rows per tile load.

**Impact**: Explains the 3-6x gap vs vLLM/SGLang (which use FlashAttention/FlashInfer) and the widening gap at longer sequences.

### 4.2 Decode Kernel: No Split-K / Flash-Decoding (Secondary Bottleneck)

**Current implementation** (`paged_attention_nvidia.cu:71-189`):

```
Grid:  (nhead,)     // 12 blocks for 1.5B, 32 blocks for 8B
Block: (256,)       // 8 warps
Tile:  PA_TILE_KV = 256 tokens per iteration
```

Each CUDA block processes **one attention head** and iterates over all KV tokens sequentially (in tiles of 256). Problems:

- **Low SM occupancy**: RTX 4090 has 128 SMs. With 12 heads (1.5B model), only 12 SMs are active — **9% occupancy**. With 32 heads (8B), 32 SMs — 25% occupancy. The rest of the GPU sits idle.
- **No split-K**: FlashDecoding (used by FlashInfer, vLLM) splits the KV sequence across multiple thread blocks per head, then does a lightweight reduction. This enables `Grid=(nhead, num_splits)` — at seq_len=1024 with split_size=256, that's `(12, 4) = 48 blocks` instead of 12.
- **Sequential KV processing**: Within each block, KV tiles are processed sequentially in a loop (line 113-174). For seq_len=1024, that's 4 iterations of 256 tokens each — all serialized within one block.

**Why decode degrades with sequence length**: Other engines maintain constant throughput because split-K parallelizes the longer KV sequence across more blocks — more work gets mapped to more SMs. Our kernel has a fixed number of blocks (nhead) regardless of sequence length, so longer sequences just mean more serial work per block.

**Quantifying the occupancy problem (1.5B, seq_len=1024)**:

| Engine | Estimated active blocks | SM utilization |
|--------|------------------------|----------------|
| ZedInfer | 12 (nhead) | 9% |
| FlashDecoding | 12 * 4 = 48 (nhead * splits) | 38% |
| Ideal | 128 (fill all SMs) | 100% |

### 4.3 Block Table Indirection Overhead

Paged attention adds an extra memory indirection compared to contiguous attention:

```
Contiguous:  K[token_idx * stride + dim]              // 1 global load
Paged:       block_table[token_idx / block_size]       // 1 global load (indirection)
             → physical_addr                           // compute
             → K[physical_addr * stride + dim]         // 1 global load (data)
```

The decode kernel mitigates this with shared memory precomputation (`s_k_phys[]`, `s_v_phys[]`), but the prefill kernel does not — it computes `block_table[blk] * block_size + off` inside the inner loop for every KV position.

### 4.4 No CUDA Graph

Each decode step launches ~30 CUDA kernels (embedding, linear, norm, rope, attention, swiglu, argmax). At ~5-10us launch overhead per kernel, that's ~150-300us of CPU-side overhead per step. For a decode step that takes ~5ms of GPU time, this is ~3-6% overhead. Minor compared to the attention kernel issues but still measurable.

### 4.5 Why Decode Improved Despite Paged Overhead

The ~10-13% decode improvement (1.5B) despite paged overhead is explained by:

1. **DecodeScratch pre-allocation**: Eliminated ~500 `Tensor::create` / `BestFitMemoryPool` round-trips per decode step. Previously each decode step allocated and freed temporary tensors through the memory pool, adding latency.
2. **Shared memory address precomputation** (`s_k_phys[]`, `s_v_phys[]`): Breaks the dependent-load chain for block table lookups. The decode kernel precomputes all physical addresses for a tile cooperatively, then accesses K/V with a single-hop global load instead of a 2-hop dependent chain.

These optimizations offset the inherent paged overhead for single-request decode. For long sequences (1024 on 8B), the offset is not enough and decode still regresses.

---

## 5. Improvement Roadmap

### Priority 1: FlashInfer Integration (Expected: 3-5x decode, 2-4x prefill)

Replace all three paged attention kernels with FlashInfer's optimized implementations:

| Issue | FlashInfer Solution |
|-------|--------------------|
| Prefill no Q-tiling | IO-aware tiling of both Q and K/V in SRAM |
| Decode low SM occupancy | Split-K (flash-decoding): splits KV across multiple blocks per head |
| Block table inner-loop overhead | Optimized page table handling with batch-level indirection |
| No vectorized load | 128-bit vectorized K/V loads |

**Prerequisite**: K/V block unification — current separate `k_blocks` / `v_blocks` per layer must be unified into shared page indices for FlashInfer's API. See `docs/plan/flashinfer_integration.md`.

### Priority 2: CUDA Graph (Expected: ~3-6% overall improvement)

Capture the decode forward pass as a CUDA graph to eliminate kernel launch overhead. Most impactful for small models (1.5B) where per-step GPU time is shorter and launch overhead is a larger fraction.

### Priority 3: Quantization (INT8/INT4)

Reduces memory bandwidth pressure on attention (smaller KV cache reads) and linear (smaller weight reads). Indirectly improves both prefill and decode throughput. Also enables running larger models on the same hardware.

---

## 6. Summary

| Metric | Status | Root Cause | Fix |
|--------|--------|-----------|-----|
| Prefill regression (old → new) | -45% to -59% (1.5B) | Paged prefill kernel naive: no Q-tiling, no smem reuse, inner-loop block table lookup | FlashInfer |
| Decode improvement (old → new) | +10% to +13% (1.5B) | DecodeScratch + smem precompute offset paged overhead | — |
| Decode degrades with seq_len | 189→105 tok/s at 128→1024 (1.5B) | No split-K, fixed Grid=(nhead,), 9% SM occupancy | FlashInfer (split-K) |
| Prefill gap vs competitors | 3-6x behind vLLM at 1024 | No IO-aware tiling, no Q-tiling | FlashInfer |
| Decode gap vs competitors | 1.3-2.3x behind at 128-1024 (1.5B) | Low occupancy + no split-K | FlashInfer |
| Near-constant decode in others | They don't degrade with seq_len | Split-K parallelizes longer sequences | FlashInfer |

**Bottom line**: The attention kernel is the single biggest performance bottleneck. FlashInfer integration is the critical path to closing the gap with vLLM/SGLang/llama.cpp.
