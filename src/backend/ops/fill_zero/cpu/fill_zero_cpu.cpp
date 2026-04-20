#include "backend/ops/fill_zero/cpu/fill_zero_cpu.hpp"

#include <cstring>

namespace zedinfer::ops::cpu {

void fill_zero(std::byte* data, size_t size_bytes) {
    std::memset(data, 0, size_bytes);
}

} // namespace zedinfer::ops::cpu
