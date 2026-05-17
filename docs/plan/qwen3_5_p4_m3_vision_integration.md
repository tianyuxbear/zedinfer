# Qwen3.5 P4 (M3) — Vision Tower Integration

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `xmake run ping ~/data/models/Qwen3.5-27B-GPTQ-Int4 --nvidia --image <path>.jpg --prompt "describe this"` emit a sensible image description. DoD: vision tower forward outputs image embeddings, scatter_image_embeds injects them into hidden, hybrid forward runs end-to-end with both text + image.

**Architecture:** New ops: `ops::vision_attention` (FlashInfer ragged prefill wrapper), `ops::layer_norm_bias`, `ops::gelu_tanh`, `ops::scatter_image_embeds`. `VisionTower::forward` implemented (27 ViT blocks + merger). `MultiModalProcessor` decodes image, resizes, patchifies. Chat template expands `<|image_pad|>` placeholder. pos_ids_thw populates real T/H/W for image tokens.

**Tech Stack:** C++17, CUDA, FlashInfer (ragged prefill), stb_image (vendored).

**Reference:** Design doc `docs/plan/qwen3_5_support.md` §5.2 / §5.6 / §6.6 / §7.3.

**Pre-condition:** P3 (M2) complete. 27B text byte-exact.

---

### Task 1: `ops::layer_norm_bias` (ViT path)

**Files:**
- Create: `include/backend/ops/layer_norm/layer_norm_bias.hpp`
- Create: `src/backend/ops/layer_norm/{cpu,nvidia}/layer_norm_bias.{cpp,cu}`
- Create: `tests/unit/test_ops_layer_norm_bias.cpp`

- [ ] **Step 1: Header**

```cpp
// include/backend/ops/layer_norm/layer_norm_bias.hpp
#pragma once
#include "backend/tensor/tensor.hpp"

namespace zedinfer::ops {

// y = ((x - mean) / sqrt(var + eps)) * weight + bias
// x, y: [N, D] (last dim normalized);
// weight, bias: [D]
void layer_norm_bias(tensor_t y, tensor_t x, tensor_t weight, tensor_t bias, float eps);

} // namespace zedinfer::ops
```

- [ ] **Step 2: NVIDIA impl (one CTA per row)**

```cuda
// src/backend/ops/layer_norm/nvidia/layer_norm_bias.cu
#include "backend/ops/layer_norm/layer_norm_bias.hpp"
#include "backend/core/context/context.hpp"
#include <cuda_bf16.h>

namespace zedinfer::ops {

__global__ void layer_norm_bias_kernel(__nv_bfloat16* y, const __nv_bfloat16* x,
                                         const __nv_bfloat16* w, const __nv_bfloat16* b,
                                         int N, int D, float eps) {
    int n = blockIdx.x;
    int tid = threadIdx.x;
    if (n >= N) return;
    const __nv_bfloat16* xr = x + n * D;
    __nv_bfloat16* yr = y + n * D;

    // Compute mean (block reduction)
    __shared__ float s_mean, s_inv_std;
    float local_sum = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) local_sum += __bfloat162float(xr[i]);
    __shared__ float reduce[32];
    // Warp reduce + block reduce — use cooperative groups
    typedef cub::BlockReduce<float, 256> BR;
    __shared__ typename BR::TempStorage tmp;
    float total = BR(tmp).Sum(local_sum);
    if (tid == 0) s_mean = total / D;
    __syncthreads();

    float local_sq = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        float d = __bfloat162float(xr[i]) - s_mean;
        local_sq += d * d;
    }
    float var_total = BR(tmp).Sum(local_sq);
    if (tid == 0) s_inv_std = rsqrtf(var_total / D + eps);
    __syncthreads();

    for (int i = tid; i < D; i += blockDim.x) {
        float normed = (__bfloat162float(xr[i]) - s_mean) * s_inv_std;
        float wv = __bfloat162float(w[i]);
        float bv = __bfloat162float(b[i]);
        yr[i] = __float2bfloat16(normed * wv + bv);
    }
}

void layer_norm_bias(tensor_t y, tensor_t x, tensor_t w, tensor_t b, float eps) {
    int N = x->shape()[0];
    int D = x->shape()[1];
    auto stream = core::context().runtime().compute_stream();
    dim3 block(256);
    dim3 grid(N);
    layer_norm_bias_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(y->data()),
        reinterpret_cast<const __nv_bfloat16*>(x->data()),
        reinterpret_cast<const __nv_bfloat16*>(w->data()),
        reinterpret_cast<const __nv_bfloat16*>(b->data()),
        N, D, eps);
}

} // namespace zedinfer::ops
```

(Add `#include <cub/cub.cuh>` at top.)

- [ ] **Step 3: CPU impl (reference)**

```cpp
// src/backend/ops/layer_norm/cpu/layer_norm_bias.cpp
#include "backend/ops/layer_norm/layer_norm_bias.hpp"
#include <cmath>

namespace zedinfer::ops {
void layer_norm_bias(tensor_t y, tensor_t x, tensor_t w, tensor_t b, float eps) {
    int N = x->shape()[0], D = x->shape()[1];
    float* xp = reinterpret_cast<float*>(x->data());
    float* yp = reinterpret_cast<float*>(y->data());
    float* wp = reinterpret_cast<float*>(w->data());
    float* bp = reinterpret_cast<float*>(b->data());
    for (int n = 0; n < N; ++n) {
        float mean = 0.0f;
        for (int i = 0; i < D; ++i) mean += xp[n*D + i];
        mean /= D;
        float var = 0.0f;
        for (int i = 0; i < D; ++i) { float d = xp[n*D + i] - mean; var += d*d; }
        var /= D;
        float inv_std = 1.0f / std::sqrt(var + eps);
        for (int i = 0; i < D; ++i) {
            yp[n*D + i] = ((xp[n*D + i] - mean) * inv_std) * wp[i] + bp[i];
        }
    }
}
}
```

- [ ] **Step 4: Unit test**

Test with `N=2, D=4` known input/weight/bias and compare to manual computation.

```cpp
// tests/unit/test_ops_layer_norm_bias.cpp — abbreviated
// Input x = [[1,2,3,4],[2,4,6,8]], w=[1,1,1,1], b=[0,0,0,0], eps=1e-5
// Expected row 0: mean=2.5, var=1.25, inv_std=0.894 → [-1.342, -0.447, 0.447, 1.342]
```

- [ ] **Step 5: Commit**

```bash
git add include/backend/ops/layer_norm src/backend/ops/layer_norm \
        tests/unit/test_ops_layer_norm_bias.cpp xmake.lua
git commit -m "feat(ops): layer_norm_bias (ViT path; LayerNorm with bias)"
```

---

### Task 2: `ops::gelu_tanh` (ViT activation)

**Files:**
- Create: `include/backend/ops/gelu_tanh/gelu_tanh.hpp`
- Create: `src/backend/ops/gelu_tanh/{cpu,nvidia}/gelu_tanh.{cpp,cu}`
- Create: `tests/unit/test_ops_gelu_tanh.cpp`

- [ ] **Step 1: Header**

```cpp
// include/backend/ops/gelu_tanh/gelu_tanh.hpp
#pragma once
#include "backend/tensor/tensor.hpp"

namespace zedinfer::ops {

// PyTorch gelu_pytorch_tanh:
// y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
void gelu_tanh(tensor_t y, tensor_t x);

} // namespace zedinfer::ops
```

- [ ] **Step 2: NVIDIA impl**

```cuda
// src/backend/ops/gelu_tanh/nvidia/gelu_tanh.cu
#include "backend/ops/gelu_tanh/gelu_tanh.hpp"
#include "backend/core/context/context.hpp"
#include <cuda_bf16.h>

namespace zedinfer::ops {

__global__ void gelu_tanh_kernel(__nv_bfloat16* y, const __nv_bfloat16* x, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float v = __bfloat162float(x[i]);
    float c = 0.7978845608f;  // sqrt(2/pi)
    float inner = c * (v + 0.044715f * v * v * v);
    float t = tanhf(inner);
    float r = 0.5f * v * (1.0f + t);
    y[i] = __float2bfloat16(r);
}

void gelu_tanh(tensor_t y, tensor_t x) {
    int total = x->numel();
    auto stream = core::context().runtime().compute_stream();
    dim3 block(256); dim3 grid((total + 255) / 256);
    gelu_tanh_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(y->data()),
        reinterpret_cast<const __nv_bfloat16*>(x->data()), total);
}

} // namespace zedinfer::ops
```

- [ ] **Step 3: CPU impl + test**

Skipped here (analogous to attn_output_gate); use `0.5 * x * (1 + tanhf(0.7978845608f * (x + 0.044715f * x*x*x)))`.

Test: `gelu_tanh(0) == 0`, `gelu_tanh(1) ≈ 0.8412`, `gelu_tanh(-1) ≈ -0.1588`.

- [ ] **Step 4: Commit**

```bash
git add include/backend/ops/gelu_tanh src/backend/ops/gelu_tanh \
        tests/unit/test_ops_gelu_tanh.cpp xmake.lua
git commit -m "feat(ops): gelu_tanh (PyTorch tanh-approx GELU; ViT path)"
```

---

### Task 3: `ops::vision_attention` (FlashInfer ragged prefill wrapper)

**Files:**
- Create: `include/backend/ops/vision_attention/vision_attention.hpp`
- Create: `src/backend/ops/vision_attention/nvidia/vision_attention.cu`
- Create: `tests/unit/test_ops_vision_attention.cpp`

- [ ] **Step 1: Header**

```cpp
// include/backend/ops/vision_attention/vision_attention.hpp
#pragma once
#include "backend/tensor/tensor.hpp"

namespace zedinfer::ops {

struct VisionAttentionParams {
    tensor_t q;       // [N, H, D]
    tensor_t k;       // [N, H, D]
    tensor_t v;       // [N, H, D]
    tensor_t out;     // [N, H, D]
    float    scale;
};

void vision_attention(const VisionAttentionParams& p);

} // namespace zedinfer::ops
```

- [ ] **Step 2: Impl using FlashInfer ragged prefill (single-batch)**

```cuda
// src/backend/ops/vision_attention/nvidia/vision_attention.cu
#include "backend/ops/vision_attention/vision_attention.hpp"
#include "backend/core/context/context.hpp"

#ifdef USE_FLASHINFER
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/variants.cuh>
#include <cuda_bf16.h>

namespace zedinfer::ops {

void vision_attention(const VisionAttentionParams& p) {
    int N = p.q->shape()[0];
    int H = p.q->shape()[1];
    int D = p.q->shape()[2];

    auto stream = core::context().runtime().compute_stream();

    // FlashInfer ragged batch prefill: single sequence, no causal mask (ViT is bidirectional)
    using namespace flashinfer;
    DefaultAttention<false, false, false, false> variant;

    // Build cu_seqlens = [0, N] on device (one batch element of length N)
    static thread_local int32_t cu_host[2] = {0, 0};
    cu_host[1] = N;
    auto cu_dev = Tensor::create({2}, ZEDINFER_DTYPE_I32, ZEDINFER_DEVICE_NVIDIA, 0);
    cudaMemcpyAsync(cu_dev->data(), cu_host, 8, cudaMemcpyHostToDevice, stream);

    BatchPrefillRaggedParams<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, int32_t> params;
    params.q = reinterpret_cast<__nv_bfloat16*>(p.q->data());
    params.k = reinterpret_cast<__nv_bfloat16*>(p.k->data());
    params.v = reinterpret_cast<__nv_bfloat16*>(p.v->data());
    params.o = reinterpret_cast<__nv_bfloat16*>(p.out->data());
    params.qo_indptr = reinterpret_cast<int32_t*>(cu_dev->data());
    params.kv_indptr = reinterpret_cast<int32_t*>(cu_dev->data());
    params.num_qo_heads = H;
    params.num_kv_heads = H;
    params.head_dim_qk = D;
    params.head_dim_vo = D;
    params.sm_scale = p.scale;
    params.causal = false;

    // Use BatchPrefillWithRaggedKVCacheDispatched (the simplest dispatcher).
    auto status = BatchPrefillWithRaggedKVCacheDispatched(params, variant, stream);
    if (status != cudaSuccess) {
        throw std::runtime_error("FlashInfer ragged prefill failed");
    }
}

} // namespace zedinfer::ops
#else
namespace zedinfer::ops {
void vision_attention(const VisionAttentionParams&) {
    throw std::runtime_error("vision_attention requires --flashinfer=y");
}
}
#endif
```

(Exact FlashInfer parameter struct names may differ. Cross-check against `third_party/flashinfer/include/flashinfer/attention/default_prefill_params.cuh`. If templates need explicit dispatch, follow the pattern in `flashinfer_wrapper.cu` for paged prefill.)

- [ ] **Step 3: Unit test (correctness vs naive softmax)**

```cpp
// tests/unit/test_ops_vision_attention.cpp
// Generate random Q/K/V, compute reference attention with cuBLAS GEMM + naive softmax,
// run ops::vision_attention, compare. Tolerance: max_abs_diff < 0.02 BF16.
```

- [ ] **Step 4: Commit**

```bash
git add include/backend/ops/vision_attention src/backend/ops/vision_attention \
        tests/unit/test_ops_vision_attention.cpp xmake.lua
git commit -m "feat(ops): vision_attention (FlashInfer ragged prefill wrapper for ViT)"
```

---

### Task 4: `ops::scatter_image_embeds`

**Files:**
- Create: `include/backend/ops/scatter_image_embeds/scatter_image_embeds.hpp`
- Create: `src/backend/ops/scatter_image_embeds/{cpu,nvidia}/scatter_image_embeds.{cpp,cu}`
- Create: `tests/unit/test_ops_scatter_image_embeds.cpp`

- [ ] **Step 1: Header**

```cpp
// include/backend/ops/scatter_image_embeds/scatter_image_embeds.hpp
#pragma once
#include "backend/tensor/tensor.hpp"

namespace zedinfer::ops {

// For each i where input_ids[i] == image_token_id, replace hidden[i, :] with
// image_embeds[k, :] where k is the running count of matched positions.
//
//   hidden:        [N_total, H]   (in/out)
//   input_ids:     [N_total]      int32
//   image_embeds:  [N_img_total, H]
//   image_token_id: scalar
void scatter_image_embeds(tensor_t hidden, tensor_t input_ids,
                            tensor_t image_embeds, int image_token_id);

} // namespace zedinfer::ops
```

- [ ] **Step 2: NVIDIA impl (two-pass: prefix-sum index, then copy)**

```cuda
// src/backend/ops/scatter_image_embeds/nvidia/scatter_image_embeds.cu
#include "backend/ops/scatter_image_embeds/scatter_image_embeds.hpp"
#include "backend/core/context/context.hpp"
#include <cub/cub.cuh>
#include <cuda_bf16.h>

namespace zedinfer::ops {

__global__ void mark_and_copy(__nv_bfloat16* hidden, const int* input_ids,
                               const __nv_bfloat16* image_embeds, int N, int H,
                               int image_token_id, const int* prefix_sum) {
    int n = blockIdx.x;
    int tid = threadIdx.x;
    if (n >= N) return;
    if (input_ids[n] != image_token_id) return;
    int img_idx = prefix_sum[n];  // # of image tokens before this one
    for (int i = tid; i < H; i += blockDim.x) {
        hidden[n * H + i] = image_embeds[img_idx * H + i];
    }
}

void scatter_image_embeds(tensor_t hidden, tensor_t input_ids,
                            tensor_t image_embeds, int image_token_id) {
    int N = hidden->shape()[0];
    int H = hidden->shape()[1];
    auto stream = core::context().runtime().compute_stream();

    // Step 1: build mask (1 if input_ids[i] == image_token_id else 0)
    auto mask = Tensor::create({(size_t)N}, ZEDINFER_DTYPE_I32, ZEDINFER_DEVICE_NVIDIA, 0);
    // Mask kernel
    auto build_mask = [] __device__ (int x, int tid_target) -> int {
        return (x == tid_target) ? 1 : 0;
    };
    // Simpler: write a small kernel
    int* mask_ptr = reinterpret_cast<int*>(mask->data());
    const int* ids_ptr = reinterpret_cast<const int*>(input_ids->data());

    // mask = (ids == image_token_id)
    auto build_kernel = [] __global__ (int* m, const int* ids, int N, int tgt) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N) m[i] = (ids[i] == tgt) ? 1 : 0;
    };
    dim3 block(256); dim3 grid((N + 255) / 256);
    build_kernel<<<grid, block, 0, stream>>>(mask_ptr, ids_ptr, N, image_token_id);

    // Step 2: exclusive scan to get prefix_sum (# of image tokens before i)
    auto prefix = Tensor::create({(size_t)N}, ZEDINFER_DTYPE_I32, ZEDINFER_DEVICE_NVIDIA, 0);
    size_t temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, mask_ptr,
                                    reinterpret_cast<int*>(prefix->data()), N, stream);
    auto temp = Tensor::create({(size_t)temp_bytes}, ZEDINFER_DTYPE_BYTE, ZEDINFER_DEVICE_NVIDIA, 0);
    cub::DeviceScan::ExclusiveSum(temp->data(), temp_bytes, mask_ptr,
                                    reinterpret_cast<int*>(prefix->data()), N, stream);

    // Step 3: copy image embeds into hidden at matched positions
    mark_and_copy<<<dim3(N), dim3(256), 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(hidden->data()),
        ids_ptr,
        reinterpret_cast<const __nv_bfloat16*>(image_embeds->data()),
        N, H, image_token_id,
        reinterpret_cast<const int*>(prefix->data()));
}

} // namespace zedinfer::ops
```

(Adapt mask-building kernel to a named device function; inline lambdas inside the host function are not valid CUDA.)

- [ ] **Step 3: CPU impl (reference, simpler)**

```cpp
// src/backend/ops/scatter_image_embeds/cpu/scatter_image_embeds.cpp
#include "backend/ops/scatter_image_embeds/scatter_image_embeds.hpp"
#include <cstring>

namespace zedinfer::ops {
void scatter_image_embeds(tensor_t hidden, tensor_t input_ids, tensor_t image_embeds, int tok) {
    int N = hidden->shape()[0];
    int H = hidden->shape()[1];
    int* ids = reinterpret_cast<int*>(input_ids->data());
    float* hp = reinterpret_cast<float*>(hidden->data());
    float* ep = reinterpret_cast<float*>(image_embeds->data());
    int k = 0;
    for (int n = 0; n < N; ++n) {
        if (ids[n] == tok) {
            std::memcpy(hp + n * H, ep + k * H, H * sizeof(float));
            k++;
        }
    }
}
}
```

- [ ] **Step 4: Unit test**

```cpp
// tests/unit/test_ops_scatter_image_embeds.cpp
// input_ids = [10, 248056, 248056, 11, 248056]
// image_embeds = [[1,1,1],[2,2,2],[3,3,3]]
// Expected: hidden[1] = [1,1,1], hidden[2] = [2,2,2], hidden[4] = [3,3,3]
```

- [ ] **Step 5: Commit**

```bash
git add include/backend/ops/scatter_image_embeds src/backend/ops/scatter_image_embeds \
        tests/unit/test_ops_scatter_image_embeds.cpp xmake.lua
git commit -m "feat(ops): scatter_image_embeds (inject ViT embeddings at image_pad positions)"
```

---

### Task 5: `MultiModalProcessor` impl

**Files:**
- Modify: `include/zedinfer/multimodal_processor.hpp`
- Create: `src/zedinfer/multimodal_processor.cpp`
- Create: `tests/unit/test_multimodal_processor.cpp`

- [ ] **Step 1: Header (final)**

```cpp
// include/zedinfer/multimodal_processor.hpp
#pragma once
#include "backend/tensor/tensor.hpp"
#include "frontend/models/base.hpp"
#include <string>
#include <vector>

namespace zedinfer {

struct ImagePayload {
    int height = 0, width = 0;
    std::vector<uint8_t> rgb_pixels;  // [H, W, 3]
};

struct ProcessedImage {
    tensor_t patches;        // [N_patches, in_channels * T_patch * H_patch * W_patch]
    tensor_t pos_ids_thw;    // [N_patches, 3]
    int      num_image_tokens = 0;  // = N_patches / spatial_merge_size^2
    int      grid_t = 0, grid_h = 0, grid_w = 0;
};

class MultiModalProcessor {
public:
    MultiModalProcessor(const model::VisionConfig& cfg);
    ProcessedImage process(const ImagePayload& img, const ExecutorConfig& exec);

    // Decode "data:image/...;base64,XXX" → raw RGB bytes
    static ImagePayload decode_base64(std::string_view data_uri);
    // Compute number of image tokens BEFORE we run vision tower
    int num_image_tokens_for(int h, int w) const;

private:
    model::VisionConfig cfg_;
};

} // namespace zedinfer
```

- [ ] **Step 2: Impl using stb_image**

```cpp
// src/zedinfer/multimodal_processor.cpp
#include "zedinfer/multimodal_processor.hpp"
#include "backend/core/context/context.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize2.h>

#include <algorithm>
#include <cuda_runtime.h>
#include <plog/Log.h>
#include <stdexcept>

namespace zedinfer {

namespace {

// Standard base64 decoder
std::vector<uint8_t> base64_decode(std::string_view s) {
    static const int8_t T[256] = { /* fill 0..255 with -1 except A-Z, a-z, 0-9, +, /, = */ };
    // (Implement properly; many open-source one-screen impls exist.)
    // Or vendor an existing one.
    std::vector<uint8_t> out;
    int val = 0, bits = -8;
    for (char c : s) {
        int v = T[(uint8_t)c];
        if (v < 0) continue;
        val = (val << 6) + v;
        bits += 6;
        if (bits >= 0) {
            out.push_back((val >> bits) & 0xFF);
            bits -= 8;
        }
    }
    return out;
}

} // namespace

ImagePayload MultiModalProcessor::decode_base64(std::string_view data_uri) {
    // Strip "data:image/...;base64,"
    auto comma = data_uri.find(',');
    auto b64 = (comma != std::string_view::npos) ? data_uri.substr(comma + 1) : data_uri;
    auto compressed = base64_decode(b64);

    int w, h, ch;
    auto* pixels = stbi_load_from_memory(compressed.data(), compressed.size(),
                                          &w, &h, &ch, 3);
    if (!pixels) throw std::runtime_error(std::string("stb_image: ") + stbi_failure_reason());

    ImagePayload pl{h, w, std::vector<uint8_t>(pixels, pixels + w * h * 3)};
    stbi_image_free(pixels);
    return pl;
}

MultiModalProcessor::MultiModalProcessor(const model::VisionConfig& cfg) : cfg_(cfg) {}

int MultiModalProcessor::num_image_tokens_for(int h, int w) const {
    // After resize: aligned to patch_size * spatial_merge_size = 32
    int align = cfg_.patch_size * cfg_.spatial_merge_size;
    int ah = ((h + align - 1) / align) * align;
    int aw = ((w + align - 1) / align) * align;
    int patches_h = ah / cfg_.patch_size;
    int patches_w = aw / cfg_.patch_size;
    // Static image: T=1 chunk after temporal_patch_size=2 replication → 1 token in T axis
    int patches_t = 1;
    int merged_h = patches_h / cfg_.spatial_merge_size;
    int merged_w = patches_w / cfg_.spatial_merge_size;
    return patches_t * merged_h * merged_w;
}

ProcessedImage MultiModalProcessor::process(const ImagePayload& img, const ExecutorConfig& exec) {
    // 1. Resize so dimensions align to patch_size * spatial_merge_size
    int align = cfg_.patch_size * cfg_.spatial_merge_size;
    int new_h = ((img.height + align - 1) / align) * align;
    int new_w = ((img.width  + align - 1) / align) * align;

    std::vector<uint8_t> resized(new_h * new_w * 3);
    stbir_resize_uint8_linear(img.rgb_pixels.data(), img.width, img.height, 0,
                                resized.data(), new_w, new_h, 0, STBIR_RGB);

    // 2. Normalize: (x/255 - 0.5)/0.5 = x/127.5 - 1.0
    std::vector<float> normed(new_h * new_w * 3);
    for (size_t i = 0; i < resized.size(); ++i) normed[i] = resized[i] / 127.5f - 1.0f;

    // 3. Patchify: split into (T=2, 16, 16) blocks for static images replicate to T=2.
    //    Output shape: [N_patches, 3*2*16*16] where N_patches = T * (H/16) * (W/16) / 1 (T_chunks=1)
    int ps = cfg_.patch_size;
    int tps = cfg_.temporal_patch_size;
    int patches_h = new_h / ps;
    int patches_w = new_w / ps;
    int n_patches = patches_h * patches_w;  // T=1 for static image (one temporal chunk)
    int patch_dim = 3 * tps * ps * ps;

    std::vector<float> patches(n_patches * patch_dim);
    for (int ph = 0; ph < patches_h; ++ph) {
        for (int pw = 0; pw < patches_w; ++pw) {
            int patch_idx = ph * patches_w + pw;
            // Replicate static image across tps temporal slots
            for (int t = 0; t < tps; ++t) {
                for (int c = 0; c < 3; ++c) {
                    for (int y = 0; y < ps; ++y) {
                        for (int x = 0; x < ps; ++x) {
                            int src = ((ph*ps + y) * new_w + (pw*ps + x)) * 3 + c;
                            int dst = patch_idx * patch_dim
                                    + ((c * tps + t) * ps + y) * ps + x;
                            patches[dst] = normed[src];
                        }
                    }
                }
            }
        }
    }

    // 4. H2D
    auto pt = Tensor::create({(size_t)n_patches, (size_t)patch_dim},
                              ZEDINFER_DTYPE_BF16, exec.device_type, exec.device_id);
    // Convert fp32 → bf16 during copy
    std::vector<uint16_t> bf16_buf(patches.size());
    for (size_t i = 0; i < patches.size(); ++i) {
        uint32_t u = *reinterpret_cast<uint32_t*>(&patches[i]);
        bf16_buf[i] = (uint16_t)(u >> 16);  // truncate to bf16
    }
    cudaMemcpy(pt->data(), bf16_buf.data(), bf16_buf.size() * 2, cudaMemcpyHostToDevice);

    // 5. pos_ids_thw: row-major (t=0, h, w) for each patch
    int sms = cfg_.spatial_merge_size;
    auto pos = Tensor::create({(size_t)n_patches, 3}, ZEDINFER_DTYPE_I32,
                                exec.device_type, exec.device_id);
    std::vector<int> pos_host(n_patches * 3);
    for (int ph = 0; ph < patches_h; ++ph) {
        for (int pw = 0; pw < patches_w; ++pw) {
            int i = ph * patches_w + pw;
            pos_host[i*3 + 0] = 0;        // t
            pos_host[i*3 + 1] = ph;       // h
            pos_host[i*3 + 2] = pw;       // w
        }
    }
    cudaMemcpy(pos->data(), pos_host.data(), pos_host.size() * 4, cudaMemcpyHostToDevice);

    ProcessedImage out;
    out.patches = pt;
    out.pos_ids_thw = pos;
    out.grid_t = 1;
    out.grid_h = patches_h;
    out.grid_w = patches_w;
    out.num_image_tokens = (patches_h / sms) * (patches_w / sms);
    return out;
}

} // namespace zedinfer
```

- [ ] **Step 3: Unit test**

```cpp
// tests/unit/test_multimodal_processor.cpp
// 1) decode a tests/fixtures/sample.jpg (use a small Qwen-VL sample)
// 2) process; verify patches.shape[0] and num_image_tokens match expected
// 3) cross-check against HF Qwen3VLProcessor output (separate Python ref)
```

Create `tests/fixtures/sample.jpg`: a 128x128 RGB test image (use Pillow):
```bash
python3 -c "from PIL import Image; img=Image.new('RGB',(128,128),(128,64,255)); img.save('tests/fixtures/sample.jpg')"
```

For a 128x128 image with patch_size=16, spatial_merge=2: patches_h=8, patches_w=8, num_image_tokens = 4*4 = 16.

Assert in test:
```cpp
auto img = stbi_load(...);
ImagePayload p{128, 128, std::vector<uint8_t>(img, img + 128*128*3)};
auto out = proc.process(p, exec);
assert(out.num_image_tokens == 16);
assert(out.grid_h == 8 && out.grid_w == 8);
```

- [ ] **Step 4: Commit**

```bash
git add include/zedinfer/multimodal_processor.hpp src/zedinfer/multimodal_processor.cpp \
        tests/unit/test_multimodal_processor.cpp tests/fixtures/sample.jpg xmake.lua
git commit -m "feat(qwen3.5): MultiModalProcessor (stb_image decode + resize + patchify)"
```

---

### Task 6: `VisionTower::forward` impl

**Files:**
- Modify: `src/frontend/models/vision_tower.cpp`

- [ ] **Step 1: Replace stub with full impl**

```cpp
// src/frontend/models/vision_tower.cpp (replace forward())
#include "backend/ops/layer_norm/layer_norm_bias.hpp"
#include "backend/ops/gelu_tanh/gelu_tanh.hpp"
#include "backend/ops/vision_attention/vision_attention.hpp"
#include "backend/ops/ops.hpp"

tensor_t VisionTower::forward(tensor_t patches, tensor_t pos_ids_thw, const ExecutorConfig& exec) {
    int N = patches->shape()[0];
    int H = cfg_.hidden_size;
    int Dh = H / cfg_.num_heads;
    int Inter = cfg_.intermediate_size;

    // 1. patch_embed: linear(patches, proj.weight) + proj.bias
    auto hidden = Tensor::create({(size_t)N, (size_t)H}, exec.data_type, exec.device_type, exec.device_id);
    ops::linear(hidden, patches, weights_->get_tensor("visual.patch_embed.proj.weight"));
    // add bias
    auto bias = weights_->get_tensor("visual.patch_embed.proj.bias");
    ops::add_bias(hidden, bias);

    // 2. pos_embed lookup + add (each patch's flat position id = ph * grid_w + pw, simplified;
    //    Qwen3VL uses 2D pos_embed indexed by spatial position)
    auto pos_embed = weights_->get_tensor("visual.pos_embed.weight");
    auto pos_idx = Tensor::create({(size_t)N}, ZEDINFER_DTYPE_I32, exec.device_type, exec.device_id);
    // Compute flat pos idx on host then H2D (or build kernel)
    {
        int* pos_thw_host = ...;  // copy pos_ids_thw to host: [3, N]
        // pos_idx[n] = pos_thw_host[1*N + n] * grid_w + pos_thw_host[2*N + n]
    }
    auto pos_embed_lookup = Tensor::create({(size_t)N, (size_t)H}, exec.data_type, exec.device_type, exec.device_id);
    ops::embedding(pos_embed_lookup, pos_idx, pos_embed);
    ops::add(hidden, hidden, pos_embed_lookup);

    // 3. 27 ViT blocks
    for (int L = 0; L < cfg_.depth; ++L) {
        auto p = "visual.blocks." + std::to_string(L) + ".";

        // norm1 → attention → residual
        auto h_norm = Tensor::create({(size_t)N, (size_t)H}, exec.data_type, exec.device_type, exec.device_id);
        ops::layer_norm_bias(h_norm, hidden, weights_->get_tensor(p+"norm1.weight"),
                              weights_->get_tensor(p+"norm1.bias"), /*eps=*/1e-6f);

        // qkv merged projection
        auto qkv = Tensor::create({(size_t)N, (size_t)(3*H)}, exec.data_type, exec.device_type, exec.device_id);
        ops::linear(qkv, h_norm, weights_->get_tensor(p+"attn.qkv.weight"));
        ops::add_bias(qkv, weights_->get_tensor(p+"attn.qkv.bias"));

        // split q/k/v
        auto q = qkv->view({(size_t)N, (size_t)cfg_.num_heads, (size_t)Dh}, /*offset=*/0);
        auto k = qkv->view({(size_t)N, (size_t)cfg_.num_heads, (size_t)Dh}, /*offset=*/H * sizeof_dtype(exec.data_type));
        auto v = qkv->view({(size_t)N, (size_t)cfg_.num_heads, (size_t)Dh}, /*offset=*/2*H * sizeof_dtype(exec.data_type));

        auto attn_out = Tensor::create({(size_t)N, (size_t)cfg_.num_heads, (size_t)Dh},
                                          exec.data_type, exec.device_type, exec.device_id);
        ops::VisionAttentionParams ap{q, k, v, attn_out, 1.0f / std::sqrt((float)Dh)};
        ops::vision_attention(ap);

        // attn.proj
        auto proj_out = Tensor::create({(size_t)N, (size_t)H}, exec.data_type, exec.device_type, exec.device_id);
        ops::linear(proj_out, attn_out->view({(size_t)N, (size_t)H}),
                    weights_->get_tensor(p+"attn.proj.weight"));
        ops::add_bias(proj_out, weights_->get_tensor(p+"attn.proj.bias"));

        ops::add(hidden, hidden, proj_out);  // residual

        // norm2 → mlp → residual
        ops::layer_norm_bias(h_norm, hidden, weights_->get_tensor(p+"norm2.weight"),
                              weights_->get_tensor(p+"norm2.bias"), 1e-6f);
        auto fc1 = Tensor::create({(size_t)N, (size_t)Inter}, exec.data_type, exec.device_type, exec.device_id);
        ops::linear(fc1, h_norm, weights_->get_tensor(p+"mlp.linear_fc1.weight"));
        ops::add_bias(fc1, weights_->get_tensor(p+"mlp.linear_fc1.bias"));
        ops::gelu_tanh(fc1, fc1);
        auto fc2 = Tensor::create({(size_t)N, (size_t)H}, exec.data_type, exec.device_type, exec.device_id);
        ops::linear(fc2, fc1, weights_->get_tensor(p+"mlp.linear_fc2.weight"));
        ops::add_bias(fc2, weights_->get_tensor(p+"mlp.linear_fc2.bias"));
        ops::add(hidden, hidden, fc2);
    }

    // 4. Merger: concat 2x2 neighboring patches → 4608 → linear_fc1 (4608) → GELU → linear_fc2 (out_hidden)
    int sms = cfg_.spatial_merge_size;
    int merged_n = N / (sms * sms);
    // Reshape hidden [N, H] → [merged_n, sms*sms, H] → flatten → [merged_n, sms*sms*H]
    // For correctness we need explicit gather (since spatial merge is 2x2, indices interleave).
    // Use ops::gather_rows or write small kernel.
    auto merged = Tensor::create({(size_t)merged_n, (size_t)(sms*sms*H)},
                                    exec.data_type, exec.device_type, exec.device_id);
    // gather: for each output row r, for each of sms*sms inner positions, copy a row from hidden
    // Compute index map on host then H2D
    // (Detail: spatial layout is patches_h × patches_w; merger groups (2y, 2y+1) × (2x, 2x+1) sub-tiles)

    // Norm + fc1 + GELU + fc2
    auto norm_out = Tensor::create({(size_t)merged_n, (size_t)(sms*sms*H)},
                                      exec.data_type, exec.device_type, exec.device_id);
    ops::layer_norm_bias(norm_out, merged, weights_->get_tensor("visual.merger.norm.weight"),
                          weights_->get_tensor("visual.merger.norm.bias"), 1e-6f);
    // Note: merger.norm acts on per-original-patch slices, not the merged 4608 vector.
    // Cross-check against HF Qwen3VL Merger.forward for exact ordering.

    auto fc1 = Tensor::create({(size_t)merged_n, 4608}, exec.data_type, exec.device_type, exec.device_id);
    ops::linear(fc1, norm_out, weights_->get_tensor("visual.merger.linear_fc1.weight"));
    ops::add_bias(fc1, weights_->get_tensor("visual.merger.linear_fc1.bias"));
    ops::gelu_tanh(fc1, fc1);

    auto out = Tensor::create({(size_t)merged_n, (size_t)cfg_.out_hidden_size},
                                exec.data_type, exec.device_type, exec.device_id);
    ops::linear(out, fc1, weights_->get_tensor("visual.merger.linear_fc2.weight"));
    ops::add_bias(out, weights_->get_tensor("visual.merger.linear_fc2.bias"));
    return out;
}
```

(`ops::add_bias` may need adding if not present; or fold bias into `ops::linear` by accepting a bias arg as the existing API supports.)

The pos_embed lookup and merger spatial layout details require careful cross-check against HF source. This is the largest correctness risk in M3.

- [ ] **Step 2: Build**

Run: `xmake build`. Fix any missing op decls.

- [ ] **Step 3: Sanity test — VisionTower::forward on sample.jpg**

Write a tiny integration test that:
1. Loads Qwen3.5-27B.
2. Decodes `tests/fixtures/sample.jpg`.
3. Calls `MultiModalProcessor::process` → ProcessedImage.
4. Calls `VisionTower::forward(patches, pos_thw)` → embeds.
5. Asserts shape: `[16, 5120]` (for 128x128 image).
6. Asserts no NaN.

- [ ] **Step 4: Commit**

```bash
git add src/frontend/models/vision_tower.cpp tests/integration/test_vision_forward.cpp xmake.lua
git commit -m "feat(qwen3.5): VisionTower::forward (27 ViT blocks + merger; static images)"
```

---

### Task 7: HF-aligned vision unit test

**Files:**
- Create: `tests/e2e/hf_vision_reference.py`
- Create: `tests/e2e/compare_vision_embeds.py`

- [ ] **Step 1: HF-side dump**

```python
# tests/e2e/hf_vision_reference.py
"""Run HF Qwen3.5 vision tower on sample image; dump merged embeddings."""
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

processor = AutoProcessor.from_pretrained("/home/tianyux/data/models/Qwen3.5-27B-GPTQ-Int4")
model = AutoModelForCausalLM.from_pretrained(
    "/home/tianyux/data/models/Qwen3.5-27B-GPTQ-Int4",
    torch_dtype=torch.bfloat16, device_map="cuda:0").eval()

from PIL import Image
img = Image.open("tests/fixtures/sample.jpg").convert("RGB")
inputs = processor(images=img, return_tensors="pt").to("cuda:0")

with torch.no_grad():
    embeds = model.visual(inputs.pixel_values, inputs.image_grid_thw)
print("shape:", embeds.shape, "norm:", embeds.norm().item())
embeds.cpu().numpy().tofile("/tmp/hf_vision_embeds.bin")
print("[ok] HF vision embeds dumped")
```

- [ ] **Step 2: Comparison**

In `test_vision_forward.cpp`, after computing zedinfer embeds, save to `/tmp/zedinfer_vision_embeds.bin` and compare:
```bash
python3 -c "
import numpy as np
a = np.fromfile('/tmp/zedinfer_vision_embeds.bin', dtype=np.float16)
b = np.fromfile('/tmp/hf_vision_embeds.bin',       dtype=np.float16)
print('max_diff:', np.abs(a.astype(float)-b.astype(float)).max())
print('mean_diff:', np.abs(a.astype(float)-b.astype(float)).mean())
"
```
Target: max_diff < 0.05 BF16 (vision tower is BF16 throughout).

If diff is large: iterate on merger spatial layout, pos_embed indexing, attention scale, GELU implementation.

- [ ] **Step 3: Commit**

```bash
git add tests/e2e/hf_vision_reference.py
git commit -m "test(qwen3.5): HF vision tower reference + comparison harness"
```

---

### Task 8: Plumb vision into engine + chat template

**Files:**
- Modify: `src/frontend/models/qwen3_5.cpp` (engine entry handles image)
- Modify: `include/zedinfer/chat_template.hpp` (expand `<|image_pad|>` to N copies)

- [ ] **Step 1: Engine entry: process image, call vision tower**

In `Qwen3_5Model::forward` or its enclosing engine call site, before invoking hybrid_transformer_forward:
```cpp
if (req.has_pending_images()) {
    auto& imgs = req.pending_images();   // vector of ImagePayload
    std::vector<tensor_t> embeds_list;
    int total_img_tokens = 0;
    for (auto& img : imgs) {
        auto processed = mm_proc_.process(img, exec);
        auto e = vision_->forward(processed.patches, processed.pos_ids_thw, exec);
        embeds_list.push_back(e);
        total_img_tokens += processed.num_image_tokens;
    }
    // Concatenate embeds across multiple images
    auto image_embeds = ops::concat_rows(embeds_list);  // [sum(N_img_tok), H]
    req.set_image_embeds(image_embeds);
}
```

`ops::concat_rows` may not exist; cuMemcpy stitching works:
```cpp
auto image_embeds = Tensor::create({(size_t)total_img_tokens, (size_t)H}, exec.data_type, ...);
size_t offset = 0;
for (auto& e : embeds_list) {
    cudaMemcpy((char*)image_embeds->data() + offset, e->data(), e->numel() * 2, cudaMemcpyDeviceToDevice);
    offset += e->numel() * 2;
}
```

- [ ] **Step 2: ChatTemplate `<|image_pad|>` post-processing**

The chat template renders the literal string `<|vision_start|><|image_pad|><|vision_end|>` once per image. We need to expand each `<|image_pad|>` to N copies (where N = num_image_tokens for that image).

After `tpl.apply()` returns:
```cpp
std::string rendered = ...;
// For each image (in order), find next <|image_pad|> and expand:
for (size_t i = 0; i < num_image_tokens_per_image.size(); ++i) {
    auto pos = rendered.find("<|image_pad|>");
    if (pos == std::string::npos) throw std::runtime_error("placeholder mismatch");
    std::string expanded;
    for (int k = 0; k < num_image_tokens_per_image[i]; ++k) expanded += "<|image_pad|>";
    rendered = rendered.substr(0, pos) + expanded + rendered.substr(pos + std::string("<|image_pad|>").size());
}
```

Then tokenize (tokenizer maps `<|image_pad|>` to `image_token_id=248056`).

- [ ] **Step 3: Build + verify rendered prompt**

Write a small test that constructs `messages = [{role:user, content:[{type:image, ...}, {type:text, text:"describe"}]}]`, applies template, verifies the resulting string has `N_img_tokens` × `<|image_pad|>` runs.

- [ ] **Step 4: Commit**

```bash
git add src/frontend/models/qwen3_5.cpp include/zedinfer/chat_template.hpp \
        src/zedinfer/chat_template_jinja.cpp
git commit -m "feat(qwen3.5): plumb VisionTower + image_pad expansion into engine entry"
```

---

### Task 9: Real `pos_ids_thw` for image tokens in hybrid forward

**Files:**
- Modify: `src/frontend/models/paged_forward_context.cpp`

- [ ] **Step 1: Override pure-text default with image regions**

When `req.has_images()`, `prepare_inputs` should set:
- For text tokens: `(t, h, w) = (idx, idx, idx)`
- For image tokens: occupy a `(1, H_grid, W_grid)` rectangle in row-major order, offset to start at the position-id where the image begins
- After the image: text positions resume from `image_end_pos = image_start_pos + max(H_grid, W_grid)` (Qwen3VL convention; verify against HF)

Implement in `PagedForwardContext::compute_pos_ids_thw(input_ids, image_grids)`:
```cpp
for (int n = 0; n < N_total; ++n) {
    if (input_ids[n] == image_token_id) {
        // image_index lookup based on current image_run_idx
        // (t, h, w) within image_run
    } else {
        // text token (t, h, w) = (current_text_pos, current_text_pos, current_text_pos)
    }
}
```

(Exact formula: see HF transformers `Qwen3VLForConditionalGeneration.get_rope_index`.)

- [ ] **Step 2: Test**

Build a small test: input_ids = [text=5, image_pad*16, text=3] with image_grid_thw=(1,4,4). Verify pos_ids_thw matches HF's `get_rope_index` output.

- [ ] **Step 3: Commit**

```bash
git add src/frontend/models/paged_forward_context.cpp tests/unit/test_pos_ids_thw_multimodal.cpp
git commit -m "feat(qwen3.5): compute pos_ids_thw for multimodal (text + image regions)"
```

---

### Task 10: End-to-end vision ping

- [ ] **Step 1: Add `--image PATH` flag to ping**

When set, read the file → base64-encode → wrap in data URI → put as user content.

```cpp
if (args.is_used("--image")) {
    std::string path = args.get<std::string>("--image");
    std::ifstream f(path, std::ios::binary);
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), {});
    std::string ext = path.substr(path.rfind('.') + 1);
    std::string data_uri = "data:image/" + ext + ";base64," + base64_encode(bytes);
    user_message.content = std::vector<ContentPart>{
        ImagePart{data_uri},
        TextPart{args.get<std::string>("--prompt")}
    };
}
```

- [ ] **Step 2: Run end-to-end**

```bash
xmake run ping ~/data/models/Qwen3.5-27B-GPTQ-Int4 --nvidia \
    --image tests/fixtures/sample.jpg \
    --prompt "What color is this image?"
```
Expected: response mentioning "purple" or similar (sample.jpg is a purple-ish image).

If output is incoherent: trace through:
- Did num_image_tokens match between processor and chat template expansion?
- Did pos_ids_thw advance correctly past the image?
- Did scatter_image_embeds write to the right positions?

- [ ] **Step 3: Update milestone**

In `docs/plan/qwen3_5_support.md` §10 M3:
```
M3 complete: <commit>, 27B image+text Q&A works on sample.jpg. Largest tweak: <merger spatial layout / pos_embed lookup>.
```

```bash
git add docs/plan/qwen3_5_support.md
git commit -m "docs(qwen3.5): M3 complete; vision tower integrated end-to-end"
```

---

### Task 11: M3 retrospective

```markdown
## M3 — Vision tower integration (complete)

### What landed
- ops::layer_norm_bias, ops::gelu_tanh, ops::vision_attention, ops::scatter_image_embeds
- MultiModalProcessor (stb_image decode + resize + patchify)
- VisionTower::forward (27 ViT blocks + merger)
- ChatTemplate expands <|image_pad|> placeholders
- pos_ids_thw computed for text+image hybrid sequences
- ping --image works end-to-end

### Caveats / M4 entry
- HTTP path still text-only
- Web UI doesn't accept image uploads
- Real URL fetch unsupported (only data: URIs supported in M4)
- 35B-A3B vision untested (M5)
```

```bash
git add docs/plan/qwen3_5_session_handoff.md
git commit -m "docs(qwen3.5): M3 retro and M4 entry conditions"
```

---

## M3 Done. ~11 tasks. Est. 1.5 weeks.
