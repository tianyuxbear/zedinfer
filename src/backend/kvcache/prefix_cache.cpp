#include "backend/kvcache/prefix_cache.hpp"

namespace zedinfer::kvcache {

PrefixCache::PrefixCache(BlockPool& pool, int num_layers) : pool_(pool), num_layers_(num_layers) {}

uint64_t PrefixCache::compute_block_hash(const std::vector<int>& tokens, int start, int end, uint64_t parent_hash) {
    uint64_t hash = parent_hash;
    for (int i = start; i < end; ++i) {
        // FNV-1a mixing
        hash ^= static_cast<uint64_t>(static_cast<unsigned int>(tokens[i]));
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

int PrefixCache::match_prefix(const std::vector<int>& token_ids, int block_size, SequenceBlockTable& matched_table) {
    int num_tokens = static_cast<int>(token_ids.size());
    int num_full_blocks = num_tokens / block_size; // only full blocks are cacheable

    matched_table.num_layers = num_layers_;
    matched_table.k_blocks.resize(num_layers_);
    matched_table.v_blocks.resize(num_layers_);
    for (int l = 0; l < num_layers_; ++l) {
        matched_table.k_blocks[l].clear();
        matched_table.v_blocks[l].clear();
    }

    uint64_t parent_hash = 0;
    int matched_tokens = 0;

    for (int b = 0; b < num_full_blocks; ++b) {
        int start = b * block_size;
        int end = start + block_size;
        uint64_t block_hash = compute_block_hash(token_ids, start, end, parent_hash);

        auto it = cache_.find(block_hash);
        if (it == cache_.end()) {
            misses_++;
            break; // chain broken — no further matches possible
        }

        hits_++;
        const auto& entry = it->second;

        // Share all blocks across all layers
        for (int l = 0; l < num_layers_; ++l) {
            int kid = entry.k_block_ids[l];
            int vid = entry.v_block_ids[l];
            pool_.share(kid);
            pool_.share(vid);
            pool_.touch(kid);
            pool_.touch(vid);
            matched_table.k_blocks[l].push_back(kid);
            matched_table.v_blocks[l].push_back(vid);
        }

        matched_tokens += block_size;
        parent_hash = block_hash;
    }

    matched_table.seq_len = matched_tokens;
    return matched_tokens;
}

void PrefixCache::insert_blocks(const std::vector<int>& token_ids, int block_size, const SequenceBlockTable& table) {
    int num_tokens = static_cast<int>(token_ids.size());
    int num_full_blocks = num_tokens / block_size;
    int blocks_in_table = table.k_blocks.empty() ? 0 : static_cast<int>(table.k_blocks[0].size());
    int blocks_to_cache = std::min(num_full_blocks, blocks_in_table);

    uint64_t parent_hash = 0;

    for (int b = 0; b < blocks_to_cache; ++b) {
        int start = b * block_size;
        int end = start + block_size;
        uint64_t block_hash = compute_block_hash(token_ids, start, end, parent_hash);

        // Skip if already cached
        if (cache_.find(block_hash) != cache_.end()) {
            parent_hash = block_hash;
            continue;
        }

        CacheEntry entry;
        entry.k_block_ids.resize(num_layers_);
        entry.v_block_ids.resize(num_layers_);

        for (int l = 0; l < num_layers_; ++l) {
            int kid = table.k_blocks[l][b];
            int vid = table.v_blocks[l][b];
            entry.k_block_ids[l] = kid;
            entry.v_block_ids[l] = vid;

            pool_.set_content_hash(kid, block_hash);
            pool_.set_content_hash(vid, block_hash);
            pool_.set_immutable(kid, true);
            pool_.set_immutable(vid, true);
        }

        cache_[block_hash] = std::move(entry);
        parent_hash = block_hash;
    }
}

void PrefixCache::remove(uint64_t content_hash) {
    cache_.erase(content_hash);
}

} // namespace zedinfer::kvcache
