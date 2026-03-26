#pragma once

#include <vector>

namespace zedinfer {

namespace ops {
inline size_t indexToOffset(size_t flat_index, size_t ndim, const std::vector<size_t>& shape,
                            const std::vector<ptrdiff_t>& strides) {
    size_t res = 0;
    for (size_t i = ndim; i-- > 0;) {
        res += (flat_index % shape[i]) * strides[i];
        flat_index /= shape[i];
    }
    return res;
}
} // namespace ops

} // namespace zedinfer