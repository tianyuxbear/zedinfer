#pragma once

#include "backend/tensor/tensor.hpp"

namespace neollm::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps);
}
