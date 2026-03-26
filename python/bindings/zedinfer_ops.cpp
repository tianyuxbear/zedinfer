/**
 * @file zedinfer_ops.cpp
 * @brief pybind11 module exposing zedinfer Tensor and Ops for Python testing.
 *
 * Exposes:
 * - Tensor class: create from torch, convert back, supports bf16
 * - Ops class: static methods dispatching to zedinfer operators
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "backend/core/context/context.hpp"
#include "backend/ops/ops.hpp"
#include "backend/tensor/tensor.hpp"
#include "zedinfer.h"

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using zedinfer::Tensor;
using zedinfer::tensor_t;

// ============================================================================
// Internal helpers
// ============================================================================

static zedinferDataType_t parse_dtype(const std::string &dtype) {
    if (dtype == "f32") {
        return ZEDINFER_DTYPE_F32;
    }
    if (dtype == "f16") {
        return ZEDINFER_DTYPE_F16;
    }
    if (dtype == "bf16") {
        return ZEDINFER_DTYPE_BF16;
    }
    if (dtype == "i8") {
        return ZEDINFER_DTYPE_I8;
    }
    if (dtype == "i16") {
        return ZEDINFER_DTYPE_I16;
    }
    if (dtype == "i32") {
        return ZEDINFER_DTYPE_I32;
    }
    if (dtype == "i64") {
        return ZEDINFER_DTYPE_I64;
    }
    throw std::invalid_argument("Unknown dtype: " + dtype);
}

static std::string dtype_to_string(zedinferDataType_t dt) {
    switch (dt) {
    case ZEDINFER_DTYPE_F32:
        return "f32";
    case ZEDINFER_DTYPE_F16:
        return "f16";
    case ZEDINFER_DTYPE_BF16:
        return "bf16";
    case ZEDINFER_DTYPE_I8:
        return "i8";
    case ZEDINFER_DTYPE_I16:
        return "i16";
    case ZEDINFER_DTYPE_I32:
        return "i32";
    case ZEDINFER_DTYPE_I64:
        return "i64";
    default:
        return "unknown";
    }
}

static zedinferDeviceType_t parse_device(const std::string &device) {
    if (device == "cpu") {
        return ZEDINFER_DEVICE_CPU;
    }
    if (device == "cuda" || device == "nvidia") {
        return ZEDINFER_DEVICE_NVIDIA;
    }
    throw std::invalid_argument("Unknown device: " + device);
}

static std::string device_to_string(zedinferDeviceType_t dt) {
    switch (dt) {
    case ZEDINFER_DEVICE_CPU:
        return "cpu";
    case ZEDINFER_DEVICE_NVIDIA:
        return "nvidia";
    default:
        return "unknown";
    }
}

// --- torch dtype <-> zedinfer dtype (string-based, no libtorch dependency) ---

static zedinferDataType_t torch_dtype_to_zedinfer(py::object dtype) {
    auto name = py::str(dtype).cast<std::string>();
    if (name == "torch.float32") {
        return ZEDINFER_DTYPE_F32;
    }
    if (name == "torch.float16") {
        return ZEDINFER_DTYPE_F16;
    }
    if (name == "torch.bfloat16") {
        return ZEDINFER_DTYPE_BF16;
    }
    if (name == "torch.int8") {
        return ZEDINFER_DTYPE_I8;
    }
    if (name == "torch.int16") {
        return ZEDINFER_DTYPE_I16;
    }
    if (name == "torch.int32") {
        return ZEDINFER_DTYPE_I32;
    }
    if (name == "torch.int64") {
        return ZEDINFER_DTYPE_I64;
    }
    throw std::invalid_argument("Unsupported torch dtype: " + name);
}

static py::object zedinfer_to_torch_dtype(zedinferDataType_t dt) {
    auto torch = py::module_::import("torch");
    switch (dt) {
    case ZEDINFER_DTYPE_F32:
        return torch.attr("float32");
    case ZEDINFER_DTYPE_F16:
        return torch.attr("float16");
    case ZEDINFER_DTYPE_BF16:
        return torch.attr("bfloat16");
    case ZEDINFER_DTYPE_I8:
        return torch.attr("int8");
    case ZEDINFER_DTYPE_I16:
        return torch.attr("int16");
    case ZEDINFER_DTYPE_I32:
        return torch.attr("int32");
    case ZEDINFER_DTYPE_I64:
        return torch.attr("int64");
    default:
        throw std::invalid_argument("Cannot convert zedinfer dtype to torch");
    }
}

// ============================================================================
// PyTensor: Python-visible tensor wrapper
// ============================================================================

class PyTensor {
public:
    tensor_t tensor;

    explicit PyTensor(tensor_t t) : tensor(std::move(t)) {}

    /// Create a zero-filled tensor.
    static PyTensor zeros(const std::vector<size_t> &shape,
                          const std::string &dtype,
                          const std::string &device) {
        auto dt = parse_dtype(dtype);
        auto dev = parse_device(device);
        auto t = Tensor::create(shape, dt, dev);
        if (dev == ZEDINFER_DEVICE_CPU) {
            std::memset(t->data(), 0, t->numel() * t->elementSize());
        }
        return PyTensor(t);
    }

    /// Create from a torch.Tensor (supports all dtypes including bf16).
    /// Copies data through the raw data_ptr — no numpy involved.
    static PyTensor from_torch(py::object torch_tensor) {
        // Ensure contiguous memory layout
        torch_tensor = torch_tensor.attr("contiguous")();

        // Extract dtype
        auto dtype = torch_dtype_to_zedinfer(torch_tensor.attr("dtype"));

        // Extract shape
        auto py_shape = torch_tensor.attr("shape");
        std::vector<size_t> shape;
        for (auto s : py_shape) {
            shape.push_back(s.cast<size_t>());
        }

        // Determine source device
        auto dev_type = torch_tensor.attr("device")
                            .attr("type")
                            .cast<std::string>();

        // Always build on CPU first: move torch tensor to CPU if needed,
        // then copy bytes via data_ptr.
        auto cpu_torch = (dev_type == "cpu")
                           ? torch_tensor
                           : torch_tensor.attr("cpu")();
        auto data_ptr = reinterpret_cast<const void *>(
            cpu_torch.attr("data_ptr")().cast<intptr_t>());

        auto t = Tensor::create(shape, dtype, ZEDINFER_DEVICE_CPU);
        t->load(data_ptr);

        // Move to GPU if the original torch tensor was on CUDA
        if (dev_type == "cuda") {
            t = t->to(ZEDINFER_DEVICE_NVIDIA, 0);
        }
        return PyTensor(t);
    }

    /// Convert back to a torch.Tensor (supports all dtypes including bf16).
    py::object to_torch() const {
        auto torch = py::module_::import("torch");

        // Move to CPU if needed
        auto cpu_t = tensor;
        if (tensor->deviceType() != ZEDINFER_DEVICE_CPU) {
            cpu_t = tensor->to(ZEDINFER_DEVICE_CPU, 0);
        }

        // Build shape tuple
        py::list py_shape;
        for (auto s : cpu_t->shape()) {
            py_shape.append(static_cast<int64_t>(s));
        }

        // Create empty torch tensor with matching dtype, then memcpy
        auto torch_dtype = zedinfer_to_torch_dtype(cpu_t->dtype());
        auto result = torch.attr("empty")(
            py::tuple(py_shape), py::arg("dtype") = torch_dtype);
        auto dst = reinterpret_cast<void *>(
            result.attr("data_ptr")().cast<intptr_t>());
        std::memcpy(dst, cpu_t->data(),
                    cpu_t->numel() * cpu_t->elementSize());

        // If the original was on GPU, move the torch tensor to CUDA too
        if (tensor->deviceType() == ZEDINFER_DEVICE_NVIDIA) {
            result = result.attr("cuda")();
        }
        return result;
    }

    py::tuple get_shape() const {
        py::list l;
        for (auto s : tensor->shape()) {
            l.append(s);
        }
        return py::tuple(l);
    }
    std::string get_dtype() const { return dtype_to_string(tensor->dtype()); }
    std::string get_device() const { return device_to_string(tensor->deviceType()); }
};

// ============================================================================
// Ops: static operator dispatch
// ============================================================================

struct Ops {
    static void add(PyTensor &out, PyTensor &a, PyTensor &b) {
        zedinfer::ops::add(out.tensor, a.tensor, b.tensor);
    }
    static void argmax(PyTensor &max_idx, PyTensor &max_val, PyTensor &vals) {
        zedinfer::ops::argmax(max_idx.tensor, max_val.tensor, vals.tensor);
    }
    static void embedding(PyTensor &out, PyTensor &index, PyTensor &weight) {
        zedinfer::ops::embedding(out.tensor, index.tensor, weight.tensor);
    }
    static void linear(PyTensor &out, PyTensor &in, PyTensor &weight,
                       py::object bias) {
        tensor_t t_bias = nullptr;
        if (!bias.is_none()) {
            t_bias = py::cast<PyTensor &>(bias).tensor;
        }
        zedinfer::ops::linear(out.tensor, in.tensor, weight.tensor, t_bias);
    }
    static void rms_norm(PyTensor &out, PyTensor &in, PyTensor &weight,
                         float eps) {
        zedinfer::ops::rms_norm(out.tensor, in.tensor, weight.tensor, eps);
    }
    static void rope(PyTensor &out, PyTensor &in, PyTensor &pos_ids,
                     float theta) {
        zedinfer::ops::rope(out.tensor, in.tensor, pos_ids.tensor, theta);
    }
    static void self_attention(PyTensor &out, PyTensor &q, PyTensor &k,
                               PyTensor &v, float scale) {
        zedinfer::ops::self_attention(out.tensor, q.tensor, k.tensor,
                                      v.tensor, scale);
    }
    static void swiglu(PyTensor &out, PyTensor &gate, PyTensor &up) {
        zedinfer::ops::swiglu(out.tensor, gate.tensor, up.tensor);
    }
};

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
    m.doc() = "zedinfer operator bindings for Python testing";

    m.def("has_cuda", &has_cuda, "Check if CUDA support was compiled in");
    m.def("device_synchronize", &device_synchronize,
          "Synchronize CUDA device (no-op if not available)");

    // --- Tensor class ---
    py::class_<PyTensor>(m, "Tensor")
        .def_static("zeros", &PyTensor::zeros,
                    py::arg("shape"), py::arg("dtype") = "f32",
                    py::arg("device") = "cpu")
        .def_static("from_torch", &PyTensor::from_torch,
                    py::arg("torch_tensor"),
                    "Create from torch.Tensor (supports bf16)")
        .def("to_torch", &PyTensor::to_torch,
             "Convert to torch.Tensor (supports bf16)")
        .def_property_readonly("shape", &PyTensor::get_shape)
        .def_property_readonly("dtype", &PyTensor::get_dtype)
        .def_property_readonly("device", &PyTensor::get_device);

    // --- Ops class ---
    py::class_<Ops>(m, "Ops")
        .def_static("add", &Ops::add)
        .def_static("argmax", &Ops::argmax)
        .def_static("embedding", &Ops::embedding)
        .def_static("linear", &Ops::linear,
                    py::arg("out"), py::arg("input"), py::arg("weight"),
                    py::arg("bias") = py::none())
        .def_static("rms_norm", &Ops::rms_norm,
                    py::arg("out"), py::arg("input"), py::arg("weight"),
                    py::arg("eps") = 1e-6f)
        .def_static("rope", &Ops::rope,
                    py::arg("out"), py::arg("input"), py::arg("pos_ids"),
                    py::arg("theta") = 10000.0f)
        .def_static("self_attention", &Ops::self_attention,
                    py::arg("out"), py::arg("q"), py::arg("k"), py::arg("v"),
                    py::arg("scale"))
        .def_static("swiglu", &Ops::swiglu);
}