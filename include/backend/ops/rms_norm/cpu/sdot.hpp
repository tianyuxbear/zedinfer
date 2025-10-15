#pragma once

#include <cstddef>

namespace neollm::ops::cpu {
float sdot(const float *x, const float *y, const size_t N);
}