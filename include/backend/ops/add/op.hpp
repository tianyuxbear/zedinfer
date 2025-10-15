#pragma once

#include "backend/tensor/tensor.hpp"

namespace neollm::ops {
void add(tensor_t c, tensor_t a, tensor_t b);
}
