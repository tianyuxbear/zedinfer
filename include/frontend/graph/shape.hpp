#pragma once

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <vector>

namespace neollm::graph {

// Type of shape dimension
enum class ShapeDimType {
    FIXED,      // Compile-time constant
    BATCH_SIZE, // Runtime batch size
    SEQ_LEN,    // Runtime sequence length
    DERIVED     // Computed from context
};

// Runtime context for shape resolution
struct ExecutionContext {
    int batch_size;
    int seq_len;
    int past_len;
};

// Single dimension in a shape template
struct ShapeDim {
    ShapeDimType type;
    size_t fixed_value;
    std::function<size_t(const ExecutionContext &)> compute_fn;

    // Factory methods
    static ShapeDim Fixed(size_t value) {
        return {ShapeDimType::FIXED, value, nullptr};
    }

    static ShapeDim BatchSize() {
        return {ShapeDimType::BATCH_SIZE, 0, nullptr};
    }

    static ShapeDim SeqLen() {
        return {ShapeDimType::SEQ_LEN, 0, nullptr};
    }

    static ShapeDim Derived(std::function<size_t(const ExecutionContext &)> fn) {
        return {ShapeDimType::DERIVED, 0, std::move(fn)};
    }

    // Resolve to concrete value at runtime
    size_t resolve(const ExecutionContext &ctx) const {
        switch (type) {
        case ShapeDimType::FIXED:
            return fixed_value;
        case ShapeDimType::BATCH_SIZE:
            return static_cast<size_t>(ctx.batch_size);
        case ShapeDimType::SEQ_LEN:
            return static_cast<size_t>(ctx.seq_len);
        case ShapeDimType::DERIVED:
            return compute_fn(ctx);
        }
        throw std::runtime_error("Unknown ShapeDimType");
    }
};

// Template for tensor shape with symbolic dimensions
struct ShapeTemplate {
    std::vector<ShapeDim> dims;

    // Resolve all dimensions to concrete shape
    std::vector<size_t> resolve(const ExecutionContext &ctx) const {
        std::vector<size_t> shape;
        shape.reserve(dims.size());
        for (const auto &dim : dims) {
            shape.push_back(dim.resolve(ctx));
        }
        return shape;
    }

    // Validate shape template
    bool validate() const {
        return !dims.empty();
    }

    size_t ndim() const {
        return dims.size();
    }
};

} // namespace neollm::graph