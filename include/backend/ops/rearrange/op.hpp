#pragma once

#include "backend/tensor/tensor.hpp"

namespace neollm::ops {
void rearrange(tensor_t out, tensor_t in);
}
