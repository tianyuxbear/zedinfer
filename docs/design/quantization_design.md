# Quantization Design

## Goals

1. **INT8 weight-only quantization**: Reduce model memory by ~2x (BF16 -> INT8). Enable larger batch sizes and longer sequences.
2. **INT4 weight-only quantization**: Reduce model memory by ~4x. Enable running 30B+ models on 24GB GPUs.
3. **Compatibility with existing formats**: Support loading GPTQ and AWQ quantized models.

## Weight Representation

### INT8 (Per-Channel Symmetric)

Each linear layer's weight matrix `W[N, K]` is quantized:

```
W_int8[N, K] = round(W_fp / scale)
scale[N]     = max(abs(W_fp[n, :])) / 127.0   for each output channel n
```

Storage:
- `W_int8`: `int8_t[N][K]` — the quantized weight matrix
- `scale`: `float[N]` or `bfloat16[N]` — per-output-channel scale factor

Dequantization (on-the-fly during GEMM):
```
W_fp[n, k] = W_int8[n, k] * scale[n]
```

### INT4 (Per-Group Asymmetric)

Each linear layer's weight matrix `W[N, K]` is quantized in groups of `group_size` (typically 128):

```
num_groups = K / group_size
W_int4[N, K]      = round((W_fp - zero_point) / scale)  // 4-bit, packed 2 per byte
scale[N, num_groups]      = (max - min) / 15.0
zero_point[N, num_groups] = round(-min / scale)          // 4-bit
```

Storage (per layer):
- `W_packed`: `uint8_t[N][K/2]` — two INT4 values packed per byte
- `scales`: `float16[N][K/group_size]`
- `zeros`: `uint8_t[N][K/group_size/2]` — packed 4-bit zero-points

### Data Type Extensions

Add to `zedinfer.h`:
```c
ZEDINFER_DTYPE_I4 = 14,   // 4-bit integer (packed)
ZEDINFER_DTYPE_COUNT = 15
```

### Quantization Config in ModelConfig

```cpp
// Additions to include/frontend/models/base.hpp

struct QuantizationConfig {
    enum class Method { NONE, INT8_PER_CHANNEL, INT4_GPTQ, INT4_AWQ };
    Method method = Method::NONE;
    int group_size = 128;       // for INT4
    int bits = 16;              // 4, 8, or 16
    bool sym = true;            // symmetric quantization
    std::string quant_method;   // "gptq", "awq", or empty

    bool is_quantized() const { return method != Method::NONE; }
};

struct ModelConfig {
    // ... existing fields ...
    QuantizationConfig quantization;
};
```

## Quantized Weight Storage

```cpp
// include/backend/tensor/quantized_tensor.hpp

struct QuantizedLinearWeight {
    tensor_t weight;       // int8[N, K] or packed int4[N, K/2]
    tensor_t scales;       // float16[N] (INT8) or float16[N, K/group_size] (INT4)
    tensor_t zeros;        // nullptr (INT8 sym) or uint8[N, K/group_size/2] (INT4)
    QuantizationConfig config;

    zedinferDataType_t compute_dtype; // BF16 or FP16 for accumulation
};
```

In `ModelWeights`, quantized weights are stored alongside the `QuantizationConfig`:

```cpp
class ModelWeights {
    // Existing
    std::unordered_map<std::string, tensor_t> weights_;

    // New: quantized weight metadata
    std::unordered_map<std::string, QuantizedLinearWeight> quantized_weights_;

    bool has_quantized_weight(const std::string &name) const;
    const QuantizedLinearWeight& get_quantized_weight(const std::string &name) const;
};
```

## Operator Dispatch

### Linear Operator

The `ops::linear()` function checks weight dtype and dispatches accordingly:

```cpp
// Modified include/backend/ops/ops.hpp
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias);

// New quantized variant
void linear_quantized(
    tensor_t out,                   // [M, N] in compute_dtype
    tensor_t in,                    // [M, K] in compute_dtype
    const QuantizedLinearWeight &qw);
```

### Dispatch Logic

```cpp
// In Model::forward(), for each linear layer:
if (weights_.has_quantized_weight(weight_name)) {
    ops::linear_quantized(out, in, weights_.get_quantized_weight(weight_name));
} else {
    ops::linear(out, in, weights_.get_tensor(weight_name), bias);
}
```

### CPU INT8 Linear

```
// Dequantize-multiply approach (simpler, lower perf)
for n in 0..N:
    for k in 0..K:
        w_fp32 = W_int8[n][k] * scale[n]
        out[m][n] += in[m][k] * w_fp32

// Better: oneDNN INT8 GEMM (if available)
// Uses VNNI instructions (AVX-512 VNNI, AMX-INT8)
onednn_gemm(in_fp32, W_int8, scale, out_fp32)
```

### NVIDIA INT8 Linear

```
// cuBLAS INT8 GEMM (cublasLtMatmul with CUDA_R_8I compute)
// Input: FP16/BF16, Weight: INT8, Output: FP16/BF16
// Scale applied post-GEMM: out = matmul(in, W_int8) * scale
cublasLtMatmul(handle, ..., CUDA_R_8I, ...);
```

### CPU INT4 Linear

```
// Dequant-on-the-fly per group
for n in 0..N:
    for g in 0..num_groups:
        s = scales[n][g]
        z = zeros[n][g]
        for k_offset in 0..group_size:
            k = g * group_size + k_offset
            w4 = unpack_int4(W_packed[n][k/2], k % 2)
            w_fp = (w4 - z) * s
            out[m][n] += in[m][k] * w_fp
```

### NVIDIA INT4 Linear

Options:
1. **CUTLASS INT4 GEMM**: Header-only, supports mixed-precision (INT4 weight x FP16 activation)
2. **Custom dequant kernel + cuBLAS FP16 GEMM**: Dequantize INT4 -> FP16 in a separate kernel, then use cuBLAS for GEMM. Simpler but uses more memory.
3. **Marlin kernels**: Highly optimized INT4 GEMM kernels from the vLLM project (Apache 2.0).

Recommendation: Start with option 2 (simplest), optimize to option 1 or 3 later.

## Weight Loading

### INT8 Loading

For INT8 models saved in SafeTensors format:
- Weight tensor: `layers.N.self_attn.q_proj.weight` with dtype `int8`
- Scale tensor: `layers.N.self_attn.q_proj.weight_scale` with dtype `float16`

```cpp
// In Model::load_weights()
if (tensor_info.dtype == ZEDINFER_DTYPE_I8 && has_tensor(name + "_scale")) {
    QuantizedLinearWeight qw;
    qw.weight = load_tensor(name);
    qw.scales = load_tensor(name + "_scale");
    qw.config.method = QuantizationConfig::Method::INT8_PER_CHANNEL;
    weights->add_quantized_weight(mapped_name, qw);
} else {
    weights->add_tensor(mapped_name, tensor);
}
```

### GPTQ INT4 Loading

GPTQ models store:
- `qweight`: packed INT4 weights (uint32 with 8 INT4 values each)
- `qzeros`: packed INT4 zero-points
- `scales`: FP16 per-group scales
- `g_idx`: group assignment index (optional, for non-sequential grouping)

```cpp
// GPTQ weight format conversion
QuantizedLinearWeight load_gptq_weight(
    tensor_t qweight,   // [K/8, N] uint32 (8 INT4 values packed)
    tensor_t qzeros,    // [K/group_size, N/8] uint32
    tensor_t scales,    // [K/group_size, N] float16
    int group_size);
```

### AWQ INT4 Loading

AWQ format is similar to GPTQ but uses different packing and may include activation-aware scales.

## Calibration

This engine does **not** perform quantization itself. It only loads pre-quantized models. Quantization is done offline using:
- GPTQ: `auto_gptq` Python library
- AWQ: `autoawq` Python library
- INT8: Simple per-channel min/max calibration (can provide a script)

## Rollout Plan

### Phase 1: INT8 Per-Channel (Symmetric)

**Scope**: Linear layers only (most memory and compute)

1. Add `QuantizationConfig` to `ModelConfig` — detect from model files
2. Add `QuantizedLinearWeight` storage in `ModelWeights`
3. Implement INT8 weight loading in SafeTensors loader
4. Implement `ops::linear_quantized()` dispatch
5. CPU: Dequant-multiply first, then oneDNN INT8 GEMM
6. GPU: cuBLAS INT8 GEMM
7. Validate: compare output logits against FP16 reference

**Files**:
- `include/zedinfer.h` — no change needed (I8 already exists)
- `include/frontend/models/base.hpp` — add QuantizationConfig
- `include/backend/tensor/quantized_tensor.hpp` — new
- `src/backend/ops/linear/cpu/linear_int8.cpp` — new
- `src/backend/ops/linear/nvidia/linear_int8.cu` — new
- `src/frontend/models/base.cpp` — quantized weight loading

### Phase 2: INT4 GPTQ

**Scope**: Linear layers, per-group (group_size=128)

1. Add `ZEDINFER_DTYPE_I4` to dtype enum
2. Implement GPTQ weight unpacking/loading
3. Implement INT4 dequant-multiply kernels
4. CPU: Dequant-on-the-fly with group-wise scales
5. GPU: Dequant kernel + cuBLAS FP16, or CUTLASS INT4
6. Validate against GPTQ reference outputs

### Phase 3: INT4 AWQ

Same infrastructure as GPTQ with different packing format. Minimal additional work.

## Memory Savings Estimate

| Model | FP16 | INT8 | INT4 | Savings |
|-------|------|------|------|---------|
| Qwen3-8B | ~16 GB | ~8 GB | ~4 GB | 4x |
| Qwen-30B-A3B | ~60 GB | ~30 GB | ~15 GB | 4x (fits 24GB with INT4) |

(Active parameters for MoE models: only top-K experts active per token)
