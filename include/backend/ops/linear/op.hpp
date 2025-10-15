#pragma once

#include "backend/tensor/tensor.hpp"

namespace neollm::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias = nullptr);
}
