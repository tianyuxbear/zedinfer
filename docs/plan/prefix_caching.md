# Prefix Caching Implementation Plan

> Date: 2026-03-24
> Status: Draft
> Depends on: Paged KV cache (already implemented), continuous batching scheduler (already implemented)

---

## 1. What Is Prefix Caching

Prefix caching is an optimization that avoids redundant KV cache computation when multiple inference requests share the same token prefix. The most common case is a **system prompt** shared across all users of a service.

### Concrete Example

Consider a chat service where every request includes the same system prompt:

```
System prompt: "You are a helpful AI assistant. Answer concisely and accurately."
  -> tokenized to ~50 tokens

Request A: [system_prompt] + "What is 2+2?"        (50 + 12 = 62 tokens)
Request B: [system_prompt] + "Tell me a joke"       (50 + 10 = 60 tokens)
Request C: [system_prompt] + "Explain gravity"      (50 + 8  = 58 tokens)
```

**Without prefix caching:** Each request independently runs prefill on the full prompt. The system prompt's 50 tokens are computed through all transformer layers three separate times. With 28 layers and block_size=16, that is `ceil(50/16) * 28 * 2 = 4 * 28 * 2 = 224` blocks allocated and computed per request, totaling 672 blocks for the shared prefix alone across three requests.

**With prefix caching:** Request A computes and stores the system prompt's KV blocks. Requests B and C look up the cached blocks by content hash, find a match, and **reuse** them. Only the unique user message tokens require new computation. The system prompt blocks are computed once (224 blocks) and shared with ref_count=3, saving 448 block allocations and 50 tokens of prefill compute per subsequent request.

### How It Saves Compute

Prefill is the most compute-intensive phase -- it processes all prompt tokens through every transformer layer. For a 200-token system prompt on a 28-layer model, that is 200 * 28 = 5,600 token-layer computations. Prefix caching converts these to simple block table pointer copies for all requests after the first, skipping the prefill entirely for the shared portion.

The savings scale with:
- Number of concurrent users sharing the same prefix
- Length of the shared prefix
- Model depth (number of layers)

---

## 2. Current Block Lifecycle Analysis

This section documents how blocks flow through the system today, based on the actual code.

### 2.1 Block Allocation

Blocks are allocated at two points:

**Point 1: Session creation** (`session.cpp:42-55`, `engine.hpp` via `create_session`)

When `InferenceEngine::create_session()` is called (triggered by `HttpServer::get_or_create_session` at `http_server.cpp:155`), the engine calls `block_allocator_->allocate_sequence(estimated_tokens)` which pre-allocates blocks for the session's `SequenceBlockTable`. The session owns these blocks for its entire lifetime.

**Point 2: Scheduler admission** (`scheduler.cpp:138-161`)

When the scheduler admits a new prefill request:
- If the request has no existing block table (`bt.num_layers == 0`), it calls `block_allocator_->allocate_sequence(est_tokens)` at line 148. For session-mode requests, this writes into `req->block_table_ref`; for stateless requests, into `req->block_table`.
- If the request already has blocks (session multi-turn), it extends the existing table with `block_allocator_->extend_sequence()` at lines 156-159.

During decode, the scheduler also extends blocks as needed at lines 100-110.

### 2.2 Block Freeing

Blocks are freed at two points:

**Point 1: Request completion** (`scheduler.cpp:262-268`)

`complete_request()` frees blocks only for **owned** tables (not borrowed session tables):
```cpp
if (block_allocator_ && !req.block_table_ref && req.block_table.num_layers > 0) {
    block_allocator_->free_sequence(req.block_table);
}
```
Session-mode requests (where `block_table_ref` points to the session's table) do NOT free blocks on completion -- the session retains them for future turns.

**Point 2: Session destruction** (`session.cpp:57-61`)

When `InferenceSession::~InferenceSession()` runs (triggered by `HttpServer::cleanup_idle_sessions` at `http_server.cpp:192-216` or `delete_session` at `http_server.cpp:183-190`):
```cpp
if (allocator_) {
    allocator_->free_sequence(block_table_);
}
```
All blocks held by the session are unconditionally freed.

**Point 3: Session reset** (`session.cpp:117-125`)

`InferenceSession::reset()` frees all current blocks and allocates fresh ones.

### 2.3 Current SequenceBlockTable Structure

Defined at `block_pool.hpp:26-32`:

```cpp
struct SequenceBlockTable {
    std::vector<std::vector<int>> k_blocks; // [num_layers][blocks_per_layer]
    std::vector<std::vector<int>> v_blocks; // [num_layers][blocks_per_layer]
    int seq_len = 0;
    int num_layers = 0;
};
```

Key observations:
- K and V blocks are **separate** vectors per layer. Each layer has independent physical block IDs for K and V.
- `block_allocator.cpp:129-143` allocates K and V blocks independently: `pool_.allocate()` is called separately for `kid` and `vid`.
- Total blocks for a sequence = `blocks_per_layer * num_layers * 2` (the `* 2` is K + V).

### 2.4 What Is Missing for Prefix Caching

| Feature | Current State | Required |
|---------|--------------|----------|
| Reference counting | None. `allocated_` is `vector<bool>` (`block_pool.hpp:72`). A block is either free or owned by exactly one sequence. | `ref_count` per block to support sharing. |
| Content hash | None. No record of what tokens a block contains. | Hash of token IDs per block to identify reusable content. |
| Sharing mechanism | None. `free_sequence()` unconditionally frees all blocks (`block_pool.cpp:161-175`). | `release()` that decrements ref_count; only frees when count reaches 0. |
| Prefix lookup | None. Every request allocates fresh blocks. | Hash table mapping token content to cached physical blocks. |
| COW | None. No concept of shared-but-mutable blocks. | Copy-on-write for the last (partially filled) block of a shared prefix. |
| Eviction | None. Freed blocks are immediately available for reallocation. | LRU eviction of cached prefix blocks under memory pressure. |

---

## 3. Design: Block Reference Counting

### 3.1 Block Metadata

Replace the `vector<bool> allocated_` bitmap in `BlockPool` (`block_pool.hpp:72`) with a richer metadata structure:

```cpp
// In block_pool.hpp, within namespace zedinfer::kvcache

struct BlockMeta {
    int ref_count = 0;          // 0 = free, 1 = exclusive, >1 = shared
    uint64_t content_hash = 0;  // hash of token IDs stored in this block (0 = unset)
    uint64_t last_access = 0;   // monotonic counter for LRU eviction
    bool immutable = false;     // true when block is full and content is finalized
};
```

### 3.2 New BlockPool Operations

Add to `BlockPool` (alongside existing `allocate()` and `free()`):

```cpp
class BlockPool {
public:
    // Existing
    int allocate();           // returns block with ref_count=0, sets ref_count=1
    void free(int block_id);  // DEPRECATED for shared blocks; use release()

    // New operations for prefix caching
    void share(int block_id);               // increment ref_count (assert > 0)
    void release(int block_id);             // decrement ref_count; if 0, mark as eviction candidate
    int ref_count(int block_id) const;      // query current ref_count

    // Metadata access
    void set_content_hash(int block_id, uint64_t hash);
    uint64_t content_hash(int block_id) const;
    void set_immutable(int block_id, bool immutable);
    bool is_immutable(int block_id) const;
    void touch(int block_id);               // update last_access for LRU

    // Eviction support
    int evict_one();                         // find and free the LRU block with ref_count==0
    int evictable_count() const;             // blocks with ref_count==0 and content_hash!=0

    // Stats (updated)
    int free_blocks() const;                 // ref_count == 0 AND content_hash == 0
    int evictable_blocks() const;            // ref_count == 0 AND content_hash != 0
    int available_blocks() const;            // free + evictable (total reclaimable)

private:
    std::vector<BlockMeta> block_meta_;      // replaces vector<bool> allocated_
    uint64_t access_counter_ = 0;            // monotonic counter for LRU ordering
};
```

### 3.3 Semantics

- `allocate()`: Finds a block with `ref_count == 0` and `content_hash == 0` (truly free). If none, calls `evict_one()` to reclaim an evictable block. Sets `ref_count = 1`. Returns block_id or -1.
- `share(block_id)`: Asserts `ref_count > 0`, increments it. Used when a new request reuses a cached prefix block.
- `release(block_id)`: Decrements `ref_count`. If it reaches 0, the block is NOT immediately freed -- it remains in the pool with its `content_hash` intact, eligible for future reuse or eviction. This is the key difference from `free()`.
- `free(block_id)`: Hard free. Sets `ref_count = 0`, `content_hash = 0`, `immutable = false`. Used when a block's content is no longer useful (e.g., decode-phase blocks that are sequence-specific).
- `touch(block_id)`: Updates `last_access = ++access_counter_`. Called whenever a block is accessed (shared or looked up in cache).

### 3.4 Block State Diagram

```
                   allocate()
  [FREE] ─────────────────────────> [EXCLUSIVE]
  ref=0, hash=0                      ref=1, hash=0
                                         │
                                         │ set_content_hash() + set_immutable()
                                         v
                                    [CACHED_EXCLUSIVE]
                                     ref=1, hash=H
                                         │
                              share()    │    release()
                           ┌─────────────┤────────────────┐
                           v             │                v
                      [SHARED]           │         [EVICTABLE]
                      ref>1, hash=H      │         ref=0, hash=H
                           │             │                │
                           │ release()   │    allocate()  │ (reuse by hash match)
                           │ (ref→1)     │    (eviction)  │ share()
                           v             │                v
                      [CACHED_EXCLUSIVE] │         [SHARED] or [EXCLUSIVE]
                                         │
                                    evict_one()
                                         │
                                         v
                                      [FREE]
                                    ref=0, hash=0
```

### 3.5 Impact on BlockAllocator

`BlockAllocator::free_sequence()` (`block_pool.cpp:161-175`) currently calls `pool_.free(bid)` for every block. This must be changed to call `pool_.release(bid)` for immutable (prefix) blocks and `pool_.free(bid)` for mutable (decode-phase) blocks:

```cpp
void BlockAllocator::release_sequence(SequenceBlockTable &table) {
    for (auto &layer_blocks : table.k_blocks) {
        for (int bid : layer_blocks) {
            pool_.release(bid);  // decrements ref_count; keeps hash if evictable
        }
        layer_blocks.clear();
    }
    for (auto &layer_blocks : table.v_blocks) {
        for (int bid : layer_blocks) {
            pool_.release(bid);
        }
        layer_blocks.clear();
    }
    table.seq_len = 0;
}
```

The old `free_sequence()` remains for cases where blocks should be hard-freed (e.g., session reset where cached content is invalidated).

---

## 4. Design: Content Hash Table

### 4.1 Hash Key Design

The hash must uniquely identify the **content** of a block, taking into account its position in the prefix chain. Two blocks with the same tokens at different positions in a sequence must NOT match, because their KV cache values differ (different positional encodings, different causal attention context).

**Chain hashing** solves this: each block's hash incorporates the hash of the previous block, creating a hash chain that encodes the full prefix up to that block.

```cpp
// Hash computation for block at logical index `block_idx` containing `tokens`
uint64_t compute_block_hash(
    const std::vector<int> &tokens,    // token IDs in this block (up to block_size)
    int block_start,                    // start index within tokens array
    int block_end,                      // end index (exclusive)
    uint64_t parent_hash                // hash of the preceding block (0 for first block)
) {
    uint64_t hash = parent_hash;
    for (int i = block_start; i < block_end; ++i) {
        // FNV-1a style mixing
        hash ^= static_cast<uint64_t>(tokens[i]);
        hash *= 0x100000001b3ULL;
    }
    return hash;
}
```

The chain property ensures:
- `hash(block_0)` = hash of tokens [0..15]
- `hash(block_1)` = hash of tokens [0..31] (because it incorporates `hash(block_0)`)
- Two requests with the same first 32 tokens will have identical `hash(block_0)` and `hash(block_1)`
- A request with different tokens at position 5 will have a different `hash(block_0)` and therefore different `hash(block_1)` even if tokens [16..31] happen to match

### 4.2 Hash Table Structure

```cpp
// In a new header: include/backend/kvcache/prefix_cache.hpp

namespace zedinfer::kvcache {

class PrefixCache {
public:
    explicit PrefixCache(BlockPool &pool, int num_layers);

    // Lookup: given a chain hash, return the cached physical block IDs for all layers.
    // Returns true if found, filling k_block_ids and v_block_ids.
    bool lookup(uint64_t content_hash,
                std::vector<int> &k_block_ids,   // [num_layers] output
                std::vector<int> &v_block_ids     // [num_layers] output
    ) const;

    // Insert: cache a set of block IDs (one per layer for K and V) under a content hash.
    // All referenced blocks must already be allocated and have ref_count >= 1.
    void insert(uint64_t content_hash,
                const std::vector<int> &k_block_ids,  // [num_layers]
                const std::vector<int> &v_block_ids    // [num_layers]
    );

    // Remove a single entry (used during eviction).
    void remove(uint64_t content_hash);

    // Match a full prompt: given token IDs, compute chain hashes for each
    // block-aligned segment and return the number of blocks that hit the cache.
    // Fills matched_table with the cached block IDs for the matched prefix.
    int match_prefix(
        const std::vector<int> &token_ids,
        int block_size,
        SequenceBlockTable &matched_table   // output: partially filled
    );

    // Stats
    size_t cache_size() const { return cache_.size(); }
    size_t hit_count() const { return hits_; }
    size_t miss_count() const { return misses_; }

private:
    struct CacheEntry {
        std::vector<int> k_block_ids;  // [num_layers]
        std::vector<int> v_block_ids;  // [num_layers]
    };

    BlockPool &pool_;
    int num_layers_;
    std::unordered_map<uint64_t, CacheEntry> cache_;  // content_hash -> block IDs

    // Stats
    mutable size_t hits_ = 0;
    mutable size_t misses_ = 0;
};

} // namespace zedinfer::kvcache
```

### 4.3 Lookup Flow

When a new request is submitted with `input_ids`, the prefix matching proceeds:

```
1. Divide input_ids into block-aligned chunks of block_size tokens.
   - Block 0: tokens [0 .. block_size-1]
   - Block 1: tokens [block_size .. 2*block_size-1]
   - ...
   - Only FULL blocks are candidates for prefix matching.
     The last partial block is never cached (it may grow).

2. Compute chain hashes:
   hash_0 = compute_block_hash(tokens, 0, block_size, 0)
   hash_1 = compute_block_hash(tokens, block_size, 2*block_size, hash_0)
   ...

3. For each hash, look up in cache_:
   - hash_0 found? -> reuse those blocks. Continue to hash_1.
   - hash_0 NOT found? -> stop. No prefix match beyond this point.
     (Chain hashing means a miss at block N invalidates all blocks N+1, N+2, ...)

4. For each matched block (all layers, K and V):
   - Call pool_.share(block_id) to increment ref_count.
   - Call pool_.touch(block_id) to update LRU.
   - Copy block_id into the request's SequenceBlockTable.

5. Return the number of matched blocks.
   The scheduler only needs to allocate blocks for the unmatched remainder.
```

### 4.4 Cache Insertion Flow

After a request completes prefill, its fully-computed KV blocks become cache candidates:

```
1. For each FULL block in the request's block table:
   - Compute the chain hash from the request's input_ids.
   - If not already in cache_: insert the entry.
   - Mark the block as immutable via pool_.set_immutable(block_id, true).
   - Set the content hash via pool_.set_content_hash(block_id, hash).

2. The last partial block is NOT inserted (it is mutable and sequence-specific).
```

### 4.5 Why One Hash Entry Covers All Layers

A single `content_hash` maps to `num_layers` K block IDs and `num_layers` V block IDs. This is because the token content determines the KV values at ALL layers (given the same model weights). If two requests have the same token prefix, their KV caches at every layer are identical. Therefore, one hash lookup retrieves all `num_layers * 2` block IDs at once. This is more efficient than a per-layer hash table.

---

## 5. Design: Copy-on-Write (COW)

### 5.1 When COW Is Needed

COW is needed when a **shared block needs to be modified**. In prefix caching, fully filled (immutable) blocks are read-only and never need COW. The only case where COW arises is the **last block of a shared prefix** that is partially filled:

```
Prefix: 200 tokens, block_size = 16
Full blocks: 0..11 (192 tokens) -- immutable, safe to share
Partial block: 12 (8 tokens filled, 8 slots empty) -- mutable

Request A continues from token 200: writes token 200 into block 12, slot 8
Request B continues from token 200: also needs to write into block 12, slot 8

If block 12 is shared (ref_count > 1), writing different tokens would corrupt data.
```

### 5.2 COW Operation

```cpp
// In BlockAllocator or a helper

int copy_on_write(SequenceBlockTable &table, int layer, int block_idx, bool is_k) {
    auto &blocks = is_k ? table.k_blocks[layer] : table.v_blocks[layer];
    int old_block = blocks[block_idx];

    if (pool_.ref_count(old_block) <= 1) {
        // Exclusive ownership -- no copy needed.
        return old_block;
    }

    // Allocate a new block and copy data.
    int new_block = pool_.allocate();
    if (new_block < 0) return -1;  // allocation failed

    // Copy block content (GPU-to-GPU memcpy for CUDA, plain memcpy for CPU).
    device_memcpy(pool_.block_data(new_block), pool_.block_data(old_block),
                  pool_.config().block_bytes());

    // Release old shared block (decrements ref_count).
    pool_.release(old_block);

    // Update the block table to point to the new exclusive block.
    blocks[block_idx] = new_block;

    return new_block;
}
```

### 5.3 When to Trigger COW

COW must be triggered **before writing new KV data into a shared block**. This happens in the model forward path when `write_kv` (or equivalent KV scatter operation) targets a block with `ref_count > 1`.

The check point is in the scheduler's block extension logic (`scheduler.cpp:100-110` for decode, `scheduler.cpp:151-160` for prefill multi-turn):

```
Before writing to the last block of a sequence:
  1. Determine which block will receive the next token's KV data.
  2. Check if that block has ref_count > 1.
  3. If yes, perform COW: allocate new block, copy, update table.
  4. Proceed with the write to the now-exclusive new block.
```

### 5.4 Optimization: Avoid COW for Full Blocks

Only the **last** block in a prefix can be partial. All preceding blocks are full and immutable. Since prefix caching only shares full blocks, COW is only ever needed for at most one block per layer per sequence. In practice, if the prefix length is a multiple of `block_size`, no COW is needed at all.

---

## 6. Design: Scheduler Integration

### 6.1 Modified Admission Control

Currently, `Scheduler::can_admit()` (`scheduler.cpp:63-85`) estimates blocks needed based on prompt length:

```cpp
int blocks_per_layer = (est_tokens + bs - 1) / bs;
int blocks_needed = blocks_per_layer * num_layers * 2;
return block_allocator_->available_blocks() >= blocks_needed;
```

With prefix caching, shared blocks reduce the allocation requirement:

```cpp
bool Scheduler::can_admit(const InferenceRequest &req) const {
    if (!block_allocator_) return true;

    int bs = block_allocator_->block_size();
    int num_layers = block_allocator_->num_layers();
    int prompt_len = static_cast<int>(req.input_ids.size());

    // Count how many blocks can be reused from prefix cache
    int cached_blocks_per_layer = 0;
    if (prefix_cache_) {
        cached_blocks_per_layer = prefix_cache_->count_prefix_matches(
            req.input_ids, bs);
    }

    int total_blocks_per_layer = (prompt_len + std::min(req.config.max_new_tokens, 256)
                                  + bs - 1) / bs;
    int new_blocks_per_layer = std::max(0, total_blocks_per_layer - cached_blocks_per_layer);
    int new_blocks_needed = new_blocks_per_layer * num_layers * 2;

    return block_allocator_->available_blocks() >= new_blocks_needed;
}
```

This allows admitting more requests when prefix cache hits are high, directly improving throughput.

### 6.2 Modified Prefill Admission

When the scheduler admits a request (`scheduler.cpp:138-161`), the allocation path changes:

```
Current flow:
  1. allocate_sequence(est_tokens) -- allocates ALL blocks fresh

New flow:
  1. Call prefix_cache_->match_prefix(input_ids, block_size, matched_table)
  2. matched_blocks = number of blocks already cached
  3. For matched blocks: share them (ref_count++) and copy IDs into request's block table
  4. For remaining blocks: allocate fresh from pool
  5. Set request's prefill_progress = matched_blocks * block_size
     (skip prefill for the cached portion)
```

The `prefill_progress` field already exists on `InferenceRequest` (`request.hpp:60`) and is used for chunked prefill. Setting it to `matched_blocks * block_size` means the prefill phase starts from where the cache ends, naturally skipping the shared prefix computation.

### 6.3 Modified Request Completion

Currently, `complete_request()` (`scheduler.cpp:262-268`) calls `block_allocator_->free_sequence()` for owned tables. With prefix caching:

```
New completion flow:
  1. For each block in the request's table:
     - If block is immutable (prefix block): call release() (decrement ref_count)
     - If block is mutable (decode block): call free() (hard free, no caching)
  2. After releasing, immutable blocks with ref_count > 0 remain alive (shared by
     other requests or held in cache). Blocks with ref_count == 0 become evictable.
```

### 6.4 Prefill Result Caching

After prefill completes for a request, newly computed KV blocks should be inserted into the prefix cache:

```
After prefill completion (in process_results, scheduler.cpp:227-253):
  1. For each full block that was freshly computed (not from cache):
     - Compute the chain hash
     - Insert into prefix_cache_
     - Mark block as immutable
  2. This makes the blocks available for subsequent requests with the same prefix.
```

---

## 7. Design: Session Integration

### 7.1 Multi-Turn Conversations with Cached Prefixes

Sessions (`session.hpp`) maintain a persistent `block_table_` across conversation turns. With prefix caching, the first turn's system prompt blocks can be shared with other sessions.

**First turn of a session:**
1. `prepare_prompt()` builds the full prompt: `bos_token + system_prompt + user_message + generation_prompt`.
2. The scheduler matches the prefix (system prompt portion) against the cache.
3. Matched prefix blocks are shared (ref_count incremented) into the session's block table.
4. Only the user message portion is freshly computed.
5. Decode-phase blocks are exclusive to the session.

**Subsequent turns:**
1. `prepare_prompt()` builds only the new turn: `user_prefix + user_message + user_suffix + generation_prompt`.
2. The session already holds blocks from previous turns (including shared prefix blocks).
3. Only the new turn's tokens need fresh blocks.
4. No prefix cache lookup needed for subsequent turns -- the session's existing blocks already contain the full conversation history.

### 7.2 Session Destruction with Shared Blocks

Currently, `~InferenceSession()` (`session.cpp:57-61`) calls `allocator_->free_sequence(block_table_)`, which hard-frees all blocks. With prefix caching, this must change to `release_sequence()`:

```cpp
InferenceSession::~InferenceSession() {
    if (allocator_) {
        allocator_->release_sequence(block_table_);
        // Shared prefix blocks survive with ref_count decremented.
        // Exclusive decode blocks are freed (ref_count reaches 0, no hash -> truly freed).
    }
}
```

Similarly, `InferenceSession::reset()` (`session.cpp:117-125`) must use `release_sequence()` instead of `free_sequence()`.

### 7.3 Session Expiry and Cache Interaction

When `HttpServer::cleanup_idle_sessions()` (`http_server.cpp:192-216`) evicts an idle session, the session destructor runs, releasing its blocks. If the system prompt blocks are still in the prefix cache (ref_count was > 1 due to other active sessions or the cache entry), they survive. If the evicted session was the last holder, the blocks become evictable (ref_count == 0, content_hash != 0) and can be reclaimed under memory pressure.

---

## 8. Design: Eviction Policy

### 8.1 When Eviction Occurs

Eviction is triggered when `BlockPool::allocate()` cannot find a truly free block (ref_count == 0, content_hash == 0) but evictable blocks exist (ref_count == 0, content_hash != 0).

### 8.2 LRU Eviction

Each block tracks `last_access` (a monotonic counter updated by `touch()`). When eviction is needed:

```cpp
int BlockPool::evict_one() {
    int victim = -1;
    uint64_t oldest = UINT64_MAX;

    for (int i = 0; i < num_blocks_; ++i) {
        if (block_meta_[i].ref_count == 0 &&
            block_meta_[i].content_hash != 0 &&
            block_meta_[i].last_access < oldest) {
            oldest = block_meta_[i].last_access;
            victim = i;
        }
    }

    if (victim >= 0) {
        // Remove from prefix cache
        // (requires reverse lookup or callback to PrefixCache)
        block_meta_[victim].ref_count = 0;
        block_meta_[victim].content_hash = 0;
        block_meta_[victim].immutable = false;
        free_count_++;
    }
    return victim;
}
```

### 8.3 Eviction Coordination with PrefixCache

When a block is evicted, the corresponding entry in `PrefixCache::cache_` must be removed. This requires either:

**Option A: Reverse index.** `BlockMeta` stores the `content_hash` it was cached under, so `evict_one()` can call `prefix_cache_->remove(content_hash)`.

**Option B: Callback.** `BlockPool` holds a callback to `PrefixCache::remove()`, invoked during eviction.

**Recommended: Option A.** The `content_hash` is already stored in `BlockMeta`, so the reverse lookup is free. However, since one cache entry covers all layers, evicting one block requires evicting all blocks under the same hash. The eviction must therefore evict an entire "block set" (all layers' K and V blocks for a given hash) atomically:

```
evict_prefix_entry(content_hash):
  entry = prefix_cache_.lookup(content_hash)
  for each block_id in entry.k_block_ids and entry.v_block_ids:
    assert ref_count == 0
    hard_free(block_id)
  prefix_cache_.remove(content_hash)
```

This ensures consistency: either all layers of a cached prefix block are available, or none are.

### 8.4 Proactive Eviction

Beyond on-demand eviction in `allocate()`, the scheduler can proactively evict when free block count drops below a threshold (e.g., 10% of total blocks). This prevents allocation failures during time-sensitive scheduling.

---

## 9. Implementation Phases

### Phase 1: Block Reference Counting (Foundation)

**Scope:** Replace `vector<bool> allocated_` with `vector<BlockMeta>`. Add `share()`, `release()`, `ref_count()`, `touch()` to `BlockPool`. Update `BlockAllocator` with `release_sequence()`.

**Rationale:** This is the minimal, non-breaking foundation. All existing code that calls `free()` continues to work unchanged. The new operations are additive.

**Files affected:**
- `include/backend/kvcache/block_pool.hpp` -- Add `BlockMeta` struct, new method declarations, replace `allocated_` with `block_meta_`
- `src/backend/kvcache/block_pool.cpp` -- Implement new methods, update `allocate()` and `free()` to use `block_meta_`

**Public interface changes:**
- `BlockPool` gains new methods: `share()`, `release()`, `ref_count()`, `touch()`, `set_content_hash()`, `content_hash()`, `set_immutable()`, `is_immutable()`
- `BlockAllocator` gains `release_sequence()` (existing `free_sequence()` unchanged)
- No existing API removed or changed in behavior

**Test plan:**
- Unit test: allocate block, verify ref_count==1, share -> ref_count==2, release -> ref_count==1, release -> ref_count==0 (evictable)
- Unit test: free() hard-frees regardless of hash
- Unit test: allocate exhaustion -> evict_one recovers a block
- Regression: existing bench/chat tests must pass unmodified

**Risks:**
- Low. Purely additive change. Existing `free()` path untouched.

### Phase 2: Content Hash Table and Prefix Matching

**Scope:** Implement `PrefixCache` class with `lookup()`, `insert()`, `remove()`, `match_prefix()`. Integrate with scheduler's prefill admission path.

**Rationale:** This enables the core prefix caching functionality. After this phase, requests with shared prefixes will reuse KV blocks and skip redundant prefill.

**Files affected:**
- `include/backend/kvcache/prefix_cache.hpp` -- New file: `PrefixCache` class declaration
- `src/backend/kvcache/prefix_cache.cpp` -- New file: implementation
- `include/zedinfer/scheduler.hpp` -- Add `PrefixCache *prefix_cache_` member, `set_prefix_cache()` method
- `src/zedinfer/scheduler.cpp` -- Modify `can_admit()`, prefill admission in `schedule()`, and `process_results()` to use prefix cache
- `include/zedinfer/request.hpp` -- Add `int prefix_cached_tokens = 0` field to track how many tokens were served from cache
- `src/zedinfer/engine.cpp` (or equivalent init path) -- Create `PrefixCache` instance and pass to scheduler

**Public interface changes:**
- New class `PrefixCache` in `zedinfer::kvcache` namespace
- `Scheduler` gains `set_prefix_cache()` method
- `InferenceRequest` gains `prefix_cached_tokens` field
- `prefill_progress` on requests with cache hits starts at `cached_blocks * block_size` instead of 0

**Test plan:**
- Unit test: insert a block hash, look it up, verify hit
- Unit test: chain hash correctness -- different token at position 5 invalidates all subsequent block hashes
- Unit test: match_prefix with partial match (first 3 blocks match, 4th does not)
- Integration test: submit two requests with identical 64-token prefix, verify second request's prefill_progress skips 64 tokens
- Benchmark: measure prefill time for second request with/without prefix cache (expect near-zero for cached portion)

**Risks:**
- Medium. Scheduler integration touches the scheduling hot path. Must verify no correctness regression with chunked prefill.
- Hash collisions: FNV-1a on block_size tokens has extremely low collision probability (2^-64). Add an assertion in debug builds that verifies token equality on hash match.

### Phase 3: Eviction Policy Under Memory Pressure

**Scope:** Implement LRU eviction in `BlockPool::evict_one()`. Coordinate eviction with `PrefixCache::remove()`. Add proactive eviction threshold to scheduler.

**Rationale:** Without eviction, the prefix cache grows until the pool is full and no new blocks can be allocated. Eviction enables the cache to work within bounded memory.

**Files affected:**
- `src/backend/kvcache/block_pool.cpp` -- Implement `evict_one()`, modify `allocate()` to call eviction on exhaustion
- `src/backend/kvcache/prefix_cache.cpp` -- Add `remove()` invocation path from eviction
- `include/backend/kvcache/block_pool.hpp` -- Add eviction-related methods and stats
- `src/zedinfer/scheduler.cpp` -- Optional: proactive eviction before scheduling

**Public interface changes:**
- `BlockPool::evict_one()` public method
- `BlockPool::evictable_count()` and `available_blocks()` stats

**Test plan:**
- Unit test: fill pool to capacity, insert cached blocks, allocate new block triggers eviction of LRU cached block
- Unit test: eviction removes corresponding PrefixCache entry
- Stress test: rapid allocation/eviction cycles, verify no memory leaks or dangling references
- Benchmark: measure overhead of eviction scan (linear scan of block_meta_; should be <1us for typical pool sizes)

**Risks:**
- Low-Medium. Eviction adds a linear scan in the allocation path, but only triggers when the pool is exhausted. For typical pool sizes (thousands of blocks), this is negligible.
- Must ensure atomicity: evicting a prefix entry must free all `num_layers * 2` blocks for that hash, not a subset.

### Phase 4: Copy-on-Write for Partial Blocks

**Scope:** Implement COW logic for the last block of a shared prefix. Integrate with block extension in the scheduler.

**Rationale:** COW handles the edge case where the prefix length is not a multiple of `block_size`. Without COW, the last partial block cannot be shared, limiting cache effectiveness for non-aligned prefixes.

**Files affected:**
- `include/backend/kvcache/block_pool.hpp` -- Add `copy_on_write()` to `BlockAllocator`
- `src/backend/kvcache/block_pool.cpp` -- Implement `copy_on_write()`
- `src/zedinfer/scheduler.cpp` -- Call COW before writing to shared blocks during decode extension and prefill continuation

**Public interface changes:**
- `BlockAllocator::copy_on_write(table, layer, block_idx, is_k)` method

**Test plan:**
- Unit test: share a block (ref_count=2), COW produces a new block with ref_count=1, old block ref_count decremented to 1
- Unit test: COW on exclusive block (ref_count=1) is a no-op
- Integration test: two requests sharing a prefix of 25 tokens (block_size=16: 1 full block + 1 partial with 9 tokens), verify each request's continuation writes to different physical blocks for the partial block
- Data correctness test: verify KV data in COW'd block matches the original

**Risks:**
- Medium. COW requires a device memcpy (GPU-to-GPU for CUDA). For a single block of `block_size * num_kv_heads * head_dim * dtype_size` bytes (e.g., 16 * 8 * 128 * 2 = 32KB for BF16), this is fast (~1us on modern GPUs) but must be synchronized correctly with the compute stream.
- Edge case: COW during decode when the block being extended is shared. The scheduler must check ref_count before every block write.

---

## 10. Files Affected Per Phase

### Phase 1: Block Reference Counting

| File | Change |
|------|--------|
| `include/backend/kvcache/block_pool.hpp` | Add `BlockMeta`, new methods, replace `allocated_` |
| `src/backend/kvcache/block_pool.cpp` | Implement ref counting logic |

### Phase 2: Content Hash Table + Prefix Matching

| File | Change |
|------|--------|
| `include/backend/kvcache/prefix_cache.hpp` | **New file**: PrefixCache class |
| `src/backend/kvcache/prefix_cache.cpp` | **New file**: implementation |
| `include/zedinfer/scheduler.hpp` | Add `prefix_cache_` member |
| `src/zedinfer/scheduler.cpp` | Modify `can_admit()`, `schedule()`, `process_results()` |
| `include/zedinfer/request.hpp` | Add `prefix_cached_tokens` field |
| `src/zedinfer/engine.cpp` | Create and wire PrefixCache |
| `CMakeLists.txt` (or build config) | Add new source file |

### Phase 3: Eviction Policy

| File | Change |
|------|--------|
| `include/backend/kvcache/block_pool.hpp` | Add eviction methods and stats |
| `src/backend/kvcache/block_pool.cpp` | Implement `evict_one()`, LRU logic |
| `src/backend/kvcache/prefix_cache.cpp` | Add eviction callback / coordination |
| `src/zedinfer/scheduler.cpp` | Optional proactive eviction |

### Phase 4: Copy-on-Write

| File | Change |
|------|--------|
| `include/backend/kvcache/block_pool.hpp` | Add `copy_on_write()` to BlockAllocator |
| `src/backend/kvcache/block_pool.cpp` | Implement COW with device memcpy |
| `src/zedinfer/scheduler.cpp` | Insert COW checks before block writes |

---

## 11. Impact Estimate

### Scenario: N Concurrent Users with Shared System Prompt

**Parameters:**
- System prompt: 200 tokens
- User message: 50 tokens average
- `block_size` = 16 (from `block_pool.hpp:14`)
- `num_layers` = 28 (Qwen-1.5B)
- System prompt blocks per layer: `ceil(200 / 16) = 13` blocks
- System prompt blocks total: `13 * 28 * 2 = 728` blocks (K + V)

| N users | Without cache (total blocks for prefix) | With cache (total blocks for prefix) | Blocks saved | Prefill tokens saved |
|---------|----------------------------------------|--------------------------------------|-------------|---------------------|
| 1 | 728 | 728 | 0 | 0 |
| 2 | 1,456 | 728 | 728 (50%) | 200 |
| 5 | 3,640 | 728 | 2,912 (80%) | 800 |
| 10 | 7,280 | 728 | 6,552 (90%) | 1,800 |
| 50 | 36,400 | 728 | 35,672 (98%) | 9,800 |
| 100 | 72,800 | 728 | 72,072 (99%) | 19,800 |

### Memory Savings

For Qwen-1.5B with BF16:
- `block_bytes = block_size * num_kv_heads * head_dim * 2 = 16 * 8 * 128 * 2 = 32,768 bytes = 32 KB`
- 728 shared blocks = 728 * 32 KB = **23.3 MB** saved per additional user

For 100 users: `72,072 * 32 KB = 2.25 GB` of GPU memory saved.

### Prefill Compute Savings

Each token of prefill requires a forward pass through all transformer layers. For a 200-token system prompt on 28 layers:
- Without cache: 200 tokens * 28 layers = 5,600 token-layer computations per request
- With cache: 0 token-layer computations for the cached prefix per subsequent request

At typical prefill throughput of ~10,000 tokens/sec, saving 200 tokens saves ~20ms per request. For 100 concurrent requests, that is ~2 seconds of total GPU prefill time saved.

### Throughput Impact

With prefix caching, the scheduler can admit more requests (fewer new blocks needed per request), increasing effective batch size and GPU utilization. The combination of reduced prefill compute and increased batch density can improve overall serving throughput by 20-50% in system-prompt-heavy workloads.

---

## 12. Risks and Edge Cases

### 12.1 Hash Collisions

**Risk:** Two different token sequences produce the same hash, causing incorrect KV cache reuse.

**Mitigation:**
- FNV-1a on 16 int32 tokens (64 bytes of input) with chain hashing has a collision probability of approximately 2^-64 per block pair. For a cache with 10,000 entries, the probability of any collision is ~10^-15 -- astronomically unlikely.
- **Debug-mode verification:** On cache hit, optionally compare the actual token IDs (stored alongside the hash) to confirm the match. This adds negligible overhead in debug builds and catches any collision.
- **Production safeguard:** Store a truncated token fingerprint (e.g., first and last 4 token IDs) in the cache entry for a lightweight sanity check.

### 12.2 Race Conditions with Concurrent Block Access

**Risk:** The HTTP server submits requests from multiple threads (`scheduler.cpp:27-34` uses `submit_mutex_`). If two requests concurrently look up the same prefix hash, both might try to share the same blocks simultaneously.

**Mitigation:**
- Prefix cache lookup and sharing happen inside the scheduler's `schedule()` method, which runs on a single scheduling thread (the engine loop). The `submit_mutex_` only protects the waiting queue. Since `schedule()` is not called concurrently, there is no race on prefix cache operations.
- If future changes introduce concurrent scheduling, `PrefixCache` must be protected with a mutex or made lock-free.

### 12.3 Memory Accounting with Shared Blocks

**Risk:** Shared blocks are counted once in the pool but logically "used" by multiple sequences. The scheduler's admission control (`can_admit`) must not double-count shared blocks.

**Mitigation:**
- `BlockPool::free_blocks()` returns blocks with `ref_count == 0 AND content_hash == 0`.
- `BlockPool::available_blocks()` returns `free_blocks() + evictable_blocks()` (blocks that CAN be reclaimed).
- The scheduler uses `available_blocks()` for admission, which correctly accounts for reclaimable cache entries.
- When counting blocks needed for a new request, subtract the number of blocks that will be shared from the prefix cache (already proposed in Section 6.1).

### 12.4 Cache Invalidation When Model Changes

**Risk:** If the model weights change (e.g., after fine-tuning or loading a different checkpoint), cached KV blocks contain stale data.

**Mitigation:**
- The prefix cache is created at engine initialization and tied to the loaded model. If the model is reloaded, the entire `PrefixCache` and `BlockPool` are destroyed and recreated.
- A model fingerprint (hash of model config or checkpoint path) can be stored in `PrefixCache` and verified at lookup time as an additional safeguard.

### 12.5 Prefix Cache Pollution

**Risk:** Infrequent or unique prefixes consume cache space without providing reuse benefits, evicting more valuable entries.

**Mitigation:**
- Only insert into the cache after prefill completes (not speculatively). This ensures the KV data is actually computed.
- Consider a minimum reuse threshold: only cache blocks that have been computed at least once. Since the first request always computes fresh blocks and inserts them, and eviction uses LRU, one-shot prefixes will naturally be evicted first.
- Optional: require a minimum prefix length (e.g., 2+ full blocks = 32+ tokens) to avoid caching trivially short prefixes.

### 12.6 Chat Template Interaction

**Risk:** Different chat templates (`chat_template.cpp`) produce different token sequences for the same logical system prompt, preventing cache hits across template formats.

**Mitigation:**
- This is expected and correct behavior. The KV cache depends on the exact token sequence, not the semantic content. Two different tokenizations of the same text produce different KV values and must NOT share blocks.
- The hash is computed on `input_ids` (token IDs), not on the raw text, so template differences are automatically handled.

### 12.7 Session Multi-Turn with Shared Prefix Blocks

**Risk:** A session holds shared prefix blocks. If the session is idle and its shared blocks are evicted, the session's block table contains dangling block IDs.

**Mitigation:**
- Active sessions (with ongoing requests or non-idle) have `ref_count > 0` on their blocks, preventing eviction.
- Idle sessions should be cleaned up by `cleanup_idle_sessions()` BEFORE their blocks are evicted. The idle timeout (`config_.session_idle_timeout` at `http_server.cpp:194`) should be configured to be shorter than the expected cache lifetime.
- As a safety measure, `InferenceSession::is_valid()` (`session.hpp:45`) can be extended to verify that the session's blocks are still allocated (ref_count > 0). If not, the session is invalidated and rebuilt from scratch on the next request.

### 12.8 Non-Aligned Prefix Lengths

**Risk:** If the shared prefix length is not a multiple of `block_size`, the last partial block cannot be cached, reducing cache effectiveness.

**Mitigation:**
- Phase 4 (COW) handles this by allowing the partial block to be shared read-only and copied on write.
- For maximum cache effectiveness, system prompts can be padded to a multiple of `block_size` (16 tokens). This is an operational optimization, not a code requirement.

---

## 13. Observability and Metrics

The following metrics should be exposed via the `/health` endpoint (`http_server.cpp:302-322`) and logging:

```json
{
    "prefix_cache": {
        "entries": 42,
        "hits": 1523,
        "misses": 89,
        "hit_rate": 0.945,
        "evictions": 12,
        "shared_blocks": 728,
        "evictable_blocks": 156
    }
}
```

Per-request logging should include:
```
[Request] chatcmpl-42 | prefix_cached=192 tokens (12 blocks) | new_prefill=58 tokens
```

This enables monitoring cache effectiveness and tuning pool size.

---

## 14. Summary and Dependencies

```
Phase 1: Block Ref Counting          ← Foundation, no external dependency
    │
    v
Phase 2: Content Hash Table          ← Core feature, depends on Phase 1
    │
    ├──> Phase 3: Eviction Policy     ← Memory management, depends on Phase 2
    │
    └──> Phase 4: Copy-on-Write       ← Edge case handling, depends on Phase 1
```

Phase 1 can be implemented and merged independently. Phase 2 delivers the core value. Phases 3 and 4 can be developed in parallel after Phase 2.

Prefix caching is independent of paged attention kernel optimization and CUDA graph (documented in `docs/notes/optimization_research.md`). All three can be developed in parallel.
