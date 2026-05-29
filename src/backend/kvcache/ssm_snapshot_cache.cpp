#include "backend/kvcache/ssm_snapshot_cache.hpp"

#include <plog/Log.h>

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace zedinfer::kvcache {

SSMSnapshotCache::SSMSnapshotCache(model::SSMStatePool& pool, size_t max_entries)
    : pool_(pool), max_entries_(max_entries == 0 ? 1 : max_entries) {
    LOGI.printf("[SSMSnapshotCache] capacity=%zu entries, ~%zu MB/snapshot", max_entries_,
                pool.snapshot_bytes() / (1024 * 1024));
}

uint64_t SSMSnapshotCache::compute_hash(const std::vector<int>& token_ids) {
    // FNV-1a 64. Treat each int as 4 little-endian bytes so the hash is stable
    // across runs / endianness assumptions consistent with PrefixCache's chain
    // hash style.
    constexpr uint64_t FNV_OFFSET = 1469598103934665603ULL;
    constexpr uint64_t FNV_PRIME = 1099511628211ULL;
    uint64_t h = FNV_OFFSET;
    for (int tok : token_ids) {
        uint32_t u = static_cast<uint32_t>(tok);
        for (int b = 0; b < 4; ++b) {
            uint8_t byte = static_cast<uint8_t>((u >> (b * 8)) & 0xFF);
            h ^= static_cast<uint64_t>(byte);
            h *= FNV_PRIME;
        }
    }
    return h;
}

bool SSMSnapshotCache::has(const std::vector<int>& token_ids) const {
    if (token_ids.empty()) {
        return false;
    }
    return entries_.find(compute_hash(token_ids)) != entries_.end();
}

bool SSMSnapshotCache::try_restore(const std::vector<int>& token_ids, int slot_idx) {
    if (token_ids.empty()) {
        ++misses_;
        return false;
    }
    auto it = entries_.find(compute_hash(token_ids));
    if (it == entries_.end()) {
        ++misses_;
        return false;
    }
    // Restore snapshot bytes back into the pool slot. May synchronously copy
    // through PCIe on NVIDIA, but that is ~5 ms for ~52 MB and only happens
    // on a cache hit, which by construction avoids a much more expensive
    // cold prefill.
    pool_.restore_slot(slot_idx, it->second.snapshot);
    it->second.last_access = ++access_counter_;
    ++hits_;
    return true;
}

void SSMSnapshotCache::record(const std::vector<int>& token_ids, int slot_idx) {
    if (token_ids.empty()) {
        return;
    }
    const uint64_t key = compute_hash(token_ids);
    auto snap = pool_.snapshot_slot(slot_idx);

    auto it = entries_.find(key);
    if (it != entries_.end()) {
        it->second.snapshot = std::move(snap);
        it->second.last_access = ++access_counter_;
        return;
    }

    if (entries_.size() >= max_entries_) {
        evict_lru();
    }
    Entry entry;
    entry.snapshot = std::move(snap);
    entry.last_access = ++access_counter_;
    entries_.emplace(key, std::move(entry));
}

void SSMSnapshotCache::remove(const std::vector<int>& token_ids) {
    if (token_ids.empty()) {
        return;
    }
    entries_.erase(compute_hash(token_ids));
}

void SSMSnapshotCache::evict_lru() {
    if (entries_.empty()) {
        return;
    }
    auto victim_it = entries_.begin();
    std::uint64_t oldest = victim_it->second.last_access;
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
        if (it->second.last_access < oldest) {
            oldest = it->second.last_access;
            victim_it = it;
        }
    }
    entries_.erase(victim_it);
}

} // namespace zedinfer::kvcache
