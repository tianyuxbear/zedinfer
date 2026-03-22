#include "backend/kvcache/base.hpp"
#include "utils/types.hpp"
#include <stdexcept>

namespace zedinfer::kvcache {

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
    return 2 * num_layers * num_kv_heads * head_dim * element_size;
}

} // namespace zedinfer::kvcache
