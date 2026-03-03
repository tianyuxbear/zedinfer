/**
 * @file zedinfer_ops.cpp
 * @brief pybind11 module exposing zedinfer operators for Python testing.
 *
 * Each operator is wrapped as a Python function that accepts numpy arrays
 * and a device string ("cpu" or "cuda"). The module handles tensor creation,
 * host-device data transfers, operator dispatch, and result extraction.
 */

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "backend/core/context/context.hpp"
#include "backend/ops/ops.hpp"
#include "backend/tensor/tensor.hpp"
#include "zedinfer.h"

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using zedinfer::tensor_t;
using zedinfer::Tensor;

// ============================================================================
// Helpers
// ============================================================================

static zedinferDataType_t numpy_to_dtype(const py::dtype &dt) {
    if (dt.is(py::dtype::of<float>())) return ZEDINFER_DTYPE_F32;
    if (dt.is(py::dtype::of<double>())) return ZEDINFER_DTYPE_F64;
    if (dt.is(py::dtype::of<int32_t>())) return ZEDINFER_DTYPE_I32;
    if (dt.is(py::dtype::of<int64_t>())) return ZEDINFER_DTYPE_I64;
    // numpy float16
    if (dt.kind() == 'f' && dt.itemsize() == 2) return ZEDINFER_DTYPE_F16;
    throw std::invalid_argument("Unsupported numpy dtype: " + std::string(py::str(dt)));
}

static py::dtype dtype_to_numpy(zedinferDataType_t dt) {
    switch (dt) {
    case ZEDINFER_DTYPE_F32: return py::dtype::of<float>();
    case ZEDINFER_DTYPE_F64: return py::dtype::of<double>();
    case ZEDINFER_DTYPE_I32: return py::dtype::of<int32_t>();
    case ZEDINFER_DTYPE_I64: return py::dtype::of<int64_t>();
    case ZEDINFER_DTYPE_F16: return py::dtype("float16");
    default: throw std::invalid_argument("Cannot convert zedinfer dtype to numpy");
    }
}

static zedinferDeviceType_t parse_device(const std::string &device) {
    if (device == "cpu") return ZEDINFER_DEVICE_CPU;
    if (device == "cuda" || device == "nvidia") return ZEDINFER_DEVICE_NVIDIA;
    throw std::invalid_argument("Unknown device: " + device + ". Use 'cpu' or 'cuda'.");
}

static std::vector<size_t> get_shape(const py::array &arr) {
    std::vector<size_t> shape;
    for (py::ssize_t i = 0; i < arr.ndim(); i++) {
        shape.push_back(static_cast<size_t>(arr.shape(i)));
    }
    return shape;
}

/**
 * @brief Create a zedinfer tensor from a numpy array, optionally on a specified device.
 */
static tensor_t from_numpy(const py::array &arr, zedinferDeviceType_t device_type) {
    auto dtype = numpy_to_dtype(arr.dtype());
    auto shape = get_shape(arr);

    // Always create on CPU first and load data
    auto tensor = Tensor::create(shape, dtype, ZEDINFER_DEVICE_CPU);
    // Ensure contiguous C-order array before loading
    auto contiguous = py::array::ensure(arr, py::array::c_style);
    tensor->load(contiguous.data());

    if (device_type == ZEDINFER_DEVICE_NVIDIA) {
        tensor = tensor->to(ZEDINFER_DEVICE_NVIDIA);
    }

    return tensor;
}

/**
 * @brief Convert a zedinfer tensor back to a numpy array (always returns CPU data).
 */
static py::array to_numpy(tensor_t tensor) {
    auto cpu_tensor = tensor;
    if (tensor->deviceType() != ZEDINFER_DEVICE_CPU) {
        cpu_tensor = tensor->to(ZEDINFER_DEVICE_CPU);
    }

    auto np_dtype = dtype_to_numpy(cpu_tensor->dtype());
    std::vector<py::ssize_t> shape;
    for (auto s : cpu_tensor->shape()) {
        shape.push_back(static_cast<py::ssize_t>(s));
    }

    auto result = py::array(np_dtype, shape);
    std::memcpy(result.mutable_data(), cpu_tensor->data(),
                cpu_tensor->numel() * cpu_tensor->elementSize());
    return result;
}

/**
 * @brief Create a zedinfer tensor filled with zeros.
 */
static tensor_t create_zeros(const std::vector<size_t> &shape,
                             zedinferDataType_t dtype,
                             zedinferDeviceType_t device_type) {
    auto tensor = Tensor::create(shape, dtype, device_type);
    // GPU tensors are zero-initialized by the allocator;
    // CPU tensors need explicit zeroing
    if (device_type == ZEDINFER_DEVICE_CPU) {
        std::memset(tensor->data(), 0, tensor->numel() * tensor->elementSize());
    }
    return tensor;
}

// ============================================================================
// Operator wrappers
// ============================================================================

static py::array op_add(py::array a, py::array b, const std::string &device) {
    auto dev = parse_device(device);
    auto ta = from_numpy(a, dev);
    auto tb = from_numpy(b, dev);
    auto tc = create_zeros(ta->shape(), ta->dtype(), dev);

    zedinfer::ops::add(tc, ta, tb);
    return to_numpy(tc);
}

static py::tuple op_argmax(py::array vals, const std::string &device) {
    auto dev = parse_device(device);
    auto t_vals = from_numpy(vals, dev);

    auto t_max_idx = Tensor::create({1}, ZEDINFER_DTYPE_I64, dev);
    auto t_max_val = Tensor::create({1}, t_vals->dtype(), dev);

    zedinfer::ops::argmax(t_max_idx, t_max_val, t_vals);

    auto np_idx = to_numpy(t_max_idx);
    auto np_val = to_numpy(t_max_val);
    return py::make_tuple(np_idx, np_val);
}

static py::array op_embedding(py::array index, py::array weight,
                              const std::string &device) {
    auto dev = parse_device(device);

    // index must be int32
    auto index_i32 = py::array_t<int32_t>::ensure(index);
    if (!index_i32) {
        throw std::invalid_argument("embedding: index must be convertible to int32");
    }

    auto t_weight = from_numpy(weight, dev);
    auto t_index = from_numpy(index_i32, dev);

    size_t seq_len = static_cast<size_t>(index_i32.size());
    size_t hidden_size = t_weight->dim(1);
    auto t_out = create_zeros({seq_len, hidden_size}, t_weight->dtype(), dev);

    zedinfer::ops::embedding(t_out, t_index, t_weight);
    return to_numpy(t_out);
}

static py::array op_linear(py::array input, py::array weight,
                           py::object bias, const std::string &device) {
    auto dev = parse_device(device);
    auto t_in = from_numpy(input, dev);
    auto t_weight = from_numpy(weight, dev);

    // out[M, N] = in[M, K] @ weight[N, K]^T + bias[N]
    size_t M = t_in->dim(0);
    size_t N = t_weight->dim(0);
    auto t_out = create_zeros({M, N}, t_in->dtype(), dev);

    tensor_t t_bias = nullptr;
    if (!bias.is_none()) {
        t_bias = from_numpy(py::array(bias), dev);
    }

    zedinfer::ops::linear(t_out, t_in, t_weight, t_bias);
    return to_numpy(t_out);
}

static py::array op_rms_norm(py::array input, py::array weight,
                             float eps, const std::string &device) {
    auto dev = parse_device(device);
    auto t_in = from_numpy(input, dev);
    auto t_weight = from_numpy(weight, dev);
    auto t_out = create_zeros(t_in->shape(), t_in->dtype(), dev);

    zedinfer::ops::rms_norm(t_out, t_in, t_weight, eps);
    return to_numpy(t_out);
}

static py::array op_rope(py::array input, py::array pos_ids,
                         float theta, const std::string &device) {
    auto dev = parse_device(device);
    auto t_in = from_numpy(input, dev);

    // pos_ids must be int64
    auto pos_i64 = py::array_t<int64_t>::ensure(pos_ids);
    if (!pos_i64) {
        throw std::invalid_argument("rope: pos_ids must be convertible to int64");
    }
    auto t_pos = from_numpy(pos_i64, dev);
    auto t_out = create_zeros(t_in->shape(), t_in->dtype(), dev);

    zedinfer::ops::rope(t_out, t_in, t_pos, theta);
    return to_numpy(t_out);
}

static py::array op_self_attention(py::array q, py::array k, py::array v,
                                   float scale, const std::string &device) {
    auto dev = parse_device(device);
    auto t_q = from_numpy(q, dev);
    auto t_k = from_numpy(k, dev);
    auto t_v = from_numpy(v, dev);

    // q: [seq_len, num_heads, head_dim]
    // k: [total_len, num_kv_heads, head_dim]
    // v: [total_len, num_kv_heads, dv]
    // out: [seq_len, num_heads, dv]
    size_t seq_len = t_q->dim(0);
    size_t num_heads = t_q->dim(1);
    size_t dv = t_v->dim(2);

    auto t_out = create_zeros({seq_len, num_heads, dv}, t_q->dtype(), dev);

    zedinfer::ops::self_attention(t_out, t_q, t_k, t_v, scale);
    return to_numpy(t_out);
}

static py::array op_swiglu(py::array gate, py::array up, const std::string &device) {
    auto dev = parse_device(device);
    auto t_gate = from_numpy(gate, dev);
    auto t_up = from_numpy(up, dev);
    auto t_out = create_zeros(t_gate->shape(), t_gate->dtype(), dev);

    zedinfer::ops::swiglu(t_out, t_gate, t_up);
    return to_numpy(t_out);
}

// ============================================================================
// Device utilities
// ============================================================================

static bool has_cuda() {
#ifdef ENABLE_NVIDIA_API
    return true;
#else
    return false;
#endif
}

static void device_synchronize() {
#ifdef ENABLE_NVIDIA_API
    zedinfer::core::context().setDevice(ZEDINFER_DEVICE_NVIDIA, 0);
    zedinfer::core::context().runtime().api()->device_synchronize();
#endif
}

// ============================================================================
// Module definition
// ============================================================================

PYBIND11_MODULE(zedinfer_ops, m) {
    m.doc() = "zedinfer operator bindings for correctness and performance testing";

    // Device utilities
    m.def("has_cuda", &has_cuda, "Check if CUDA support was compiled in");
    m.def("device_synchronize", &device_synchronize,
          "Synchronize CUDA device (no-op if CUDA not available)");

    // Operators
    m.def("add", &op_add,
          "Element-wise addition: c = a + b",
          py::arg("a"), py::arg("b"), py::arg("device") = "cpu");

    m.def("argmax", &op_argmax,
          "Global argmax: returns (max_index, max_value)",
          py::arg("vals"), py::arg("device") = "cpu");

    m.def("embedding", &op_embedding,
          "Embedding lookup: out = weight[index]",
          py::arg("index"), py::arg("weight"), py::arg("device") = "cpu");

    m.def("linear", &op_linear,
          "Linear transform: out = input @ weight^T + bias",
          py::arg("input"), py::arg("weight"),
          py::arg("bias") = py::none(), py::arg("device") = "cpu");

    m.def("rms_norm", &op_rms_norm,
          "RMS normalization: out = weight * input / rms(input)",
          py::arg("input"), py::arg("weight"),
          py::arg("eps") = 1e-6f, py::arg("device") = "cpu");

    m.def("rope", &op_rope,
          "Rotary Position Embedding",
          py::arg("input"), py::arg("pos_ids"),
          py::arg("theta") = 10000.0f, py::arg("device") = "cpu");

    m.def("self_attention", &op_self_attention,
          "Grouped-Query Self-Attention with causal mask",
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("scale"), py::arg("device") = "cpu");

    m.def("swiglu", &op_swiglu,
          "SwiGLU activation: out = up * silu(gate)",
          py::arg("gate"), py::arg("up"), py::arg("device") = "cpu");
}
