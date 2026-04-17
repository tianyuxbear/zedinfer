#include "backend/core/context/context.hpp"
#include "backend/ops/fill_zero/nvidia/fill_zero_nvidia.cuh"

#include <cuda_runtime.h>

namespace zedinfer::ops::nvidia {

void fill_zero(std::byte* data, size_t size_bytes) {
    // Use async memset on the runtime's stream to avoid race with other ops.
    auto stream = reinterpret_cast<cudaStream_t>(zedinfer::core::context().runtime().stream());
    cudaMemsetAsync(data, 0, size_bytes, stream);
}

} // namespace zedinfer::ops::nvidia
