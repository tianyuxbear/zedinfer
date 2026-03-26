#pragma once

#include "backend/tensor/tensor.hpp"

namespace zedinfer::ops::cpu {

// Reorder each input row by column indices:
// output[m, k] = input[m, indices[k]]
tensor_t permute_cols(tensor_t input, tensor_t indices);

} // namespace zedinfer::ops::cpu
