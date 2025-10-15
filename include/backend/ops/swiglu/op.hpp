#pragma once

#include "backend/tensor/tensor.hpp"

namespace neollm::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up);
}
