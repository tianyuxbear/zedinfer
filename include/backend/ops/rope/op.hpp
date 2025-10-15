#pragma once

#include "backend/tensor/tensor.hpp"

namespace neollm::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta);
}
