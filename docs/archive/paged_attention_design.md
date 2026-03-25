# Paged Attention Design

## Motivation

The current `DynamicKVCache` (`include/backend/kvcache/dynamic.hpp`) stores KV data in contiguous tensors of shape `[capacity, num_kv_heads, head_dim]`. Growth requires full reallocation + memcpy (`dynamic.cpp:61-97`). This wastes memory (over-allocation) and causes latency spikes (copy on grow).

Paged KV cache eliminates both problems by storing KV data in fixed-size blocks, managed by a block table. No reallocation. No copying. O(1) append.

## Page / Block Structure

```cpp
// include/backend/kvcache/paged.hpp

// One block stores a fixed number of tokens for one layer, one cache type (K or V)
// Shape: [block_size, num_kv_heads, head_dim]
constexpr int DEFAULT_BLOCK_SIZE = 16;  // tokens per block

struct BlockConfig {
    int block_size = DEFAULT_BLOCK_SIZE;
    int num_kv_heads;
    int head_dim;
    zedinferDataType_t dtype;
    zedinferDeviceType_t device_type;
    int device_id;

    size_t block_bytes() const {
        return block_size * num_kv_heads * head_dim * utils::dsize(dtype);
    }
};
```

### Block Size Choice

- **16 tokens**: Good default. Balances internal fragmentation (~8 tokens wasted on average per sequence) against block table overhead.
- GPU: Must be a multiple of warp size divisor (16 or 32) for coalesced memory access in paged attention kernels.
- Can be tuned per deployment. Larger blocks = less metadata overhead but more internal fragmentation.

## KV Cache Layout

### Physical Layer: Block Pool

```cpp
// A pool of pre-allocated blocks on a single device
class BlockPool {
public:
    BlockPool(const BlockConfig &config, int num_blocks);

    // Allocate a free block, returns physical block ID
    int allocate();

    // Free a block back to the pool
    void free(int block_id);

    // Get raw pointer to block's memory
    std::byte* block_ptr(int block_id) const;

    // Stats
    int total_blocks() const;
    int free_blocks() const;
    float utilization() const;

private:
    BlockConfig config_;
    tensor_t storage_;           // [num_blocks, block_size, num_kv_heads, head_dim]
    std::vector<bool> in_use_;   // per-block allocation status
    std::vector<int> free_list_; // stack of free block IDs
    int num_blocks_;
};
```

The pool allocates one large contiguous tensor `[num_blocks * block_size, num_kv_heads, head_dim]` and carves it into fixed-size blocks. Block `i` starts at offset `i * block_size * num_kv_heads * head_dim * dsize`.

### Logical Layer: Block Table

Each sequence (request) maintains a block table per layer:

```cpp
// Per-sequence KV cache metadata
struct SequenceKVMeta {
    // block_table[layer_idx] = list of physical block IDs in sequence order
    std::vector<std::vector<int>> k_block_table;  // [num_layers][num_blocks_used]
    std::vector<std::vector<int>> v_block_table;

    int seq_len = 0;           // total tokens cached
    int last_block_offset = 0; // fill level of last block (0..block_size-1)
};
```

### Block Allocator

```cpp
class BlockAllocator {
public:
    BlockAllocator(const BlockConfig &config, int total_gpu_blocks, int total_cpu_blocks = 0);

    // Allocate blocks for a new sequence
    // Returns block IDs for all layers (K and V)
    SequenceKVMeta allocate_initial(int num_layers, int estimated_tokens);

    // Append: allocate a new block when current block is full
    int allocate_block();

    // Free all blocks for a completed sequence
    void free_sequence(SequenceKVMeta &meta);

    // Query
    int available_gpu_blocks() const;
    int available_cpu_blocks() const;
    bool can_allocate(int num_blocks) const;

private:
    BlockPool gpu_pool_;
    BlockPool cpu_pool_;  // for swap/offload (optional)
};
```

## PagedKVCache Interface

```cpp
class PagedKVCache {
public:
    PagedKVCache(BlockAllocator &allocator, int num_layers, const BlockConfig &config);

    // Write K/V for new tokens into the cache
    // Automatically allocates new blocks as needed
    void append(int layer_idx,
                tensor_t k_new,  // [num_new_tokens, num_kv_heads, head_dim]
                tensor_t v_new,
                SequenceKVMeta &meta);

    // Get block table for attention kernel
    // Returns flattened block pointer table for GPU kernel consumption
    tensor_t get_block_table_tensor(const SequenceKVMeta &meta, int layer_idx);

    // Get K/V for a specific block (for CPU attention that iterates blocks)
    std::pair<tensor_t, tensor_t> get_block(int layer_idx, int physical_block_id);

    // Get contiguous view (for non-paged attention fallback, copies data)
    tensor_t get_contiguous_k(int layer_idx, const SequenceKVMeta &meta);
    tensor_t get_contiguous_v(int layer_idx, const SequenceKVMeta &meta);

private:
    BlockAllocator &allocator_;
    int num_layers_;
    BlockConfig config_;
    BlockPool &gpu_pool_;
};
```

## Attention Interface Changes

### Current Interface

```cpp
// include/backend/ops/ops.hpp
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale);

// CPU implementation expects:
// q: [seq_len, nhead, head_dim]
// k: [total_len, nkvhead, head_dim]   <- contiguous
// v: [total_len, nkvhead, head_dim]   <- contiguous
```

### New Paged Attention Interface

```cpp
// include/backend/ops/ops.hpp (additions)

// Non-paged attention (kept for compatibility and testing)
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale);

// Paged attention for decode (single query token per sequence, batched)
void paged_attention_decode(
    tensor_t out,                // [num_seqs, nhead, head_dim]
    tensor_t query,              // [num_seqs, nhead, head_dim]
    tensor_t k_cache,            // [num_blocks * block_size, nkvhead, head_dim] (pool)
    tensor_t v_cache,            // same
    tensor_t block_tables,       // [num_seqs, max_blocks_per_seq] int32
    tensor_t seq_lens,           // [num_seqs] int32 - actual sequence length per request
    float scale,
    int block_size);

// Paged attention for prefill (multiple query tokens, single sequence)
void paged_attention_prefill(
    tensor_t out,                // [seq_len, nhead, head_dim]
    tensor_t query,              // [seq_len, nhead, head_dim]
    tensor_t k_cache,            // pool
    tensor_t v_cache,            // pool
    tensor_t block_table,        // [max_blocks] int32 (single sequence)
    int seq_len,
    int past_len,
    float scale,
    int block_size);
```

### CPU Paged Attention Implementation Strategy

For CPU, paged attention can be implemented by iterating over blocks:

```
for each query position i:
    for each head h:
        for each block in block_table:
            for each token in block (up to block_size or remaining seq_len):
                compute Q*K score
                accumulate softmax denominator
            accumulate V weighted sum
        finalize softmax normalization
```

This is a straightforward modification of the current `self_attention_cpu.cpp` logic, replacing the contiguous K/V access with block-table-indexed access.

### NVIDIA Paged Attention Implementation Strategy

For GPU, two options:

1. **Adapt vLLM paged attention kernels** (Apache 2.0): Battle-tested, supports variable sequence lengths, optimized for decode. Would need adaptation to match our tensor layout and dtype support.

2. **Custom implementation**: Based on FlashAttention-2 with block-table indirection. More control but higher risk.

Recommendation: Start with option 1 for decode (most critical), custom for prefill if needed.

## Compatibility with Continuous Batching

### Batch Assembly

The scheduler constructs `BatchContext` with per-request block tables:

```cpp
// Scheduler builds:
struct BatchContext {
    // ... (from continuous_batching_design.md)

    // Paged attention metadata
    tensor_t block_tables;  // [num_requests, max_blocks_per_seq] padded
    tensor_t seq_lens;      // [num_requests] actual seq lengths
    int max_blocks_per_seq; // for padding block_tables
};
```

### Write Path (KV Population)

During forward pass:
1. Q/K/V projections produce `[total_tokens, dim]`
2. K and V are split per-request based on `BatchContext.slots`
3. Each request's K/V chunk is appended to its paged cache via `PagedKVCache::append()`
4. New blocks allocated on the fly if current block is full

### Read Path (Attention)

1. `paged_attention_decode()` receives the block table tensor and processes all decode requests in one kernel launch
2. Prefill requests use `paged_attention_prefill()` separately (or are chunked into the decode batch)

## Migration from DynamicKVCache

### Phase 1: Non-Paged Batched Attention
- Keep `DynamicKVCache` for single-sequence use
- Add batch dimension to attention kernels (multiple sequences, each with contiguous KV)
- This is the intermediate step

### Phase 2: Paged KV Cache
- Implement `BlockPool`, `BlockAllocator`, `PagedKVCache`
- Implement `paged_attention_decode` (GPU first, CPU second)
- Scheduler uses paged allocation
- `DynamicKVCache` remains for testing/comparison

### Phase 3: Paged Prefill
- Implement `paged_attention_prefill`
- Handle mixed prefill+decode batches
- Full paged pipeline

## Block Size Tuning

| Block Size | Fragmentation | Metadata Overhead | GPU Kernel Efficiency |
|-----------|--------------|-------------------|----------------------|
| 8 | Low (~4 tokens avg waste) | High (2x block IDs) | Lower (small reads) |
| 16 | Medium (~8) | Medium | Good |
| 32 | Higher (~16) | Low | Best (aligned reads) |
| 64 | High (~32) | Very low | Best |

Recommendation: **16** as default, configurable. For GPU-only deployment, 32 may be better.

## Memory Budget Example

For Qwen3-8B (BF16, 36 layers, 8 KV heads, head_dim=128):
- Per-token KV: `2 * 36 * 8 * 128 * 2 bytes = 147,456 bytes = 144 KB`
- Block (16 tokens): `16 * 144 KB = 2.25 MB`
- 24 GB GPU: `~10,600 blocks` = `~170K tokens` of KV capacity
- With INT8 KV: `~21,200 blocks` = `~340K tokens`
