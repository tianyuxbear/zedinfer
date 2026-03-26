#pragma once

#include "utils/nvidia/common.cuh"
#include "utils/nvidia/types.cuh"

#include <cstdint>
#include <cuda_runtime.h>

namespace zedinfer::ops::nvidia {

template <typename T>
void launch_quantize_q8_row(const T* input, int8_t* q_out, T* scale_out, size_t M, size_t K,
                            cudaStream_t stream = nullptr);

template <typename T>
void launch_quantize_q8_row_grouped(const T* input, int8_t* q_out, T* scale_out, size_t M, size_t K, int group_size,
                                    cudaStream_t stream);

} // namespace zedinfer::ops::nvidia
