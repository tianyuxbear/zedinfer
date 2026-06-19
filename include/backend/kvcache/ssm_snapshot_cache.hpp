#pragma once

#include "frontend/models/ssm_state_pool.hpp"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace zedinfer::kvcache {

// Companion to PrefixCache for hybrid models (Qwen3.5 family). PrefixCache
// shares KV pages across requests with identical token prefixes; for hybrid
// models that also have linear-attention layers, reusing the KV without
// restoring the linear-attention state produces output that diverges from a
// cold-cache run (the linear-attn layers would see a post-reset zero state
// while the full-attn layers see prefilled K/V — "prefix-blind" output).
//
// This cache fixes that by storing, for each fully-prefilled prompt, a host
// snapshot of the entire SSM + conv state slot. The scheduler consults it on
// admission: only when the SSM snapshot for the full prompt is present does
// it honor a PrefixCache hit and skip prefill. Partial KV matches without a
// matching SSM snapshot are rejected so the request goes through a cold
// prefill instead.
//
// Sizing notes: one snapshot is ssm_bytes_per_slot + conv_bytes_per_slot bytes
// (~52 MB for Qwen3.5-27B). Held in host memory; D2H on record, H2D on
// restore. LRU-evicted at `max_entries`.
class SSMSnapshotCache {
public:
    explicit SSMSnapshotCache(model::SSMStatePool& pool, size_t max_entries);

    // Look up the snapshot for `token_ids` without mutating LRU. Returns true
    // when an entry exists. Used by the scheduler to decide whether a
    // PrefixCache hit can be honored on hybrid models before paying the
    // pool.share() cost.
    bool has(const std::vector<int>& token_ids) const;

    // If a snapshot exists for `token_ids`, copy it into the given slot
    // (H2D on NVIDIA) and bump LRU. Returns true on hit, false on miss. The
    // pool's slot must already be acquired by the caller; this method does
    // not touch the slot bitmap.
    bool try_restore(const std::vector<int>& token_ids, int slot_idx);

    // Snapshot the slot's current state and store it under `token_ids` hash.
    // Replaces any prior entry for the same hash. Evicts the LRU entry when
    // the cache exceeds `max_entries`.
    void record(const std::vector<int>& token_ids, int slot_idx);

    // Drop a specific entry (used by future GC paths; currently only LRU
    // eviction calls in). Safe to call when the entry does not exist.
    void remove(const std::vector<int>& token_ids);

    // Diagnostics.
    size_t cache_size() const { return entries_.size(); }
    size_t hit_count() const { return hits_; }
    size_t miss_count() const { return misses_; }

    // Public for tests; computes a stable hash for the full token sequence.
    // FNV-1a 64-bit over little-endian token id bytes.
    static uint64_t compute_hash(const std::vector<int>& token_ids);

private:
    struct Entry {
        model::SSMStateSnapshot snapshot;
        std::uint64_t last_access = 0;
    };

    model::SSMStatePool& pool_;
    size_t max_entries_;

    std::unordered_map<std::uint64_t, Entry> entries_;
    std::uint64_t access_counter_ = 0;

    mutable size_t hits_ = 0;
    mutable size_t misses_ = 0;

    // Evict the entry with the smallest `last_access`. Called from record()
    // when entries_.size() > max_entries_.
    void evict_lru();
};

} // namespace zedinfer::kvcache
