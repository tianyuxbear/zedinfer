#include "backend/kvcache/base.hpp"
#include "utils/types.hpp"
#include <iomanip>
#include <iostream>

namespace neollm::kvcache {

void KVCacheConfig::validate() const {
    if (num_layers <= 0) {
        throw std::invalid_argument("num_layers must be positive");
    }
    if (num_kv_heads <= 0) {
        throw std::invalid_argument("num_kv_heads must be positive");
    }
    if (head_dim <= 0 || head_dim % 8 != 0) {
        throw std::invalid_argument("head_dim must be positive and divisible by 8");
    }
}

size_t KVCacheConfig::bytes_per_token() const {
    size_t element_size = utils::dsize(dtype);
    return 2 * num_kv_heads * head_dim * element_size;
}

KVCacheManager::KVCacheManager(const KVCacheConfig &config)
    : config_(config), current_length_(0) {
    config_.validate();
    k_caches_.reserve(config_.num_layers);
    v_caches_.reserve(config_.num_layers);
}

void KVCacheManager::update_seq_len(int new_tokens) {
    if (new_tokens < 0) {
        throw std::invalid_argument("new_tokens must be non-negative");
    }
    if (new_tokens == 0) {
        return;
    }

    current_length_ += new_tokens;

    if (current_length_ > allocated_capacity()) {
        throw std::runtime_error(
            "Sequence length exceeds allocated capacity");
    }
}

void KVCacheManager::reset() {
    current_length_ = 0;
}

void KVCacheManager::print_stats() const {
    std::cout << "\n=== KV Cache Statistics ===" << std::endl;
    std::cout << "  Num layers: " << config_.num_layers << std::endl;
    std::cout << "  Num KV heads: " << config_.num_kv_heads << std::endl;
    std::cout << "  Head dim: " << config_.head_dim << std::endl;
    std::cout << "  Current length: " << current_length_ << " tokens" << std::endl;
    std::cout << "  Allocated capacity: " << allocated_capacity() << " tokens" << std::endl;
    std::cout << "  Utilization: " << std::fixed << std::setprecision(1)
              << (utilization() * 100.0f) << "%" << std::endl;
    std::cout << "  Memory usage: " << std::fixed << std::setprecision(2)
              << (memory_usage() / 1024.0 / 1024.0) << " MB" << std::endl;
}

void KVCacheManager::validate_layer_idx(int layer_idx) const {
    if (layer_idx < 0 || layer_idx >= config_.num_layers) {
        throw std::out_of_range("Invalid layer_idx");
    }
}

void KVCacheManager::validate_length_params(int past_len, int seq_len) const {
    if (past_len < 0 || seq_len < 0) {
        throw std::invalid_argument("Lengths must be non-negative");
    }
    if (past_len > current_length_) {
        std::cout << "past_len: " << past_len << ", current_length_: " << current_length_ << std::endl;
        throw std::invalid_argument("past_len exceeds current_length");
    }
}
} // namespace neollm::kvcache