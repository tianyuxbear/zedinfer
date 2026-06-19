#pragma once

#include "backend/ops/attention_params.hpp"
#include "backend/ops/silu_mul/silu_mul.hpp"
#include "backend/tensor/tensor.hpp"

namespace zedinfer::ops {

void add(tensor_t c, tensor_t a, tensor_t b);
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals);
size_t sample_sort_workspace_bytes(zedinferDeviceType_t device_type, size_t vocab_size);
void sample_token(tensor_t out_token, tensor_t logits, tensor_t work_logits, tensor_t work_ids, tensor_t sorted_logits,
                  tensor_t sorted_ids, tensor_t sort_temp, tensor_t recent_tokens, float temperature, int top_k,
                  float top_p, float repetition_penalty, float random);
void embedding(tensor_t out, tensor_t index, tensor_t weight);
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias = nullptr);
void linear_quantized(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias, tensor_t scale, tensor_t g_idx,
                      int num_bits, int group_size);
void rearrange(tensor_t out, tensor_t in);
// add_one_to_weight=true matches HF Qwen3_5MoeRMSNorm:
//   output = (x * rsqrt(mean(x^2) + eps)) * (1.0 + weight)
// where the (1.0 + weight) is computed in fp32 inside the kernel rather than
// pre-baked into a bf16 buffer at load time. Pre-baking loses ~6x precision
// because bf16's mantissa around 1.0 (step ~2^-7 = 7.8e-3) is much coarser
// than around 0.0 (where typical Qwen3.5 weights live). The precision loss
// compounds across 40 layers and causes the model to drift from HF after
// ~16 generated tokens under greedy decoding.
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps, bool add_one_to_weight = false);

// Fused residual+RMSNorm (in-place, GPU only — CPU path sequences add + rms_norm).
//   residual := residual + input
//   input    := residual / rms(residual) * (weight + (add_one_to_weight ? 1.0 : 0.0))
// Both `input` and `residual` are mutated. Maps to FlashInfer's FusedAddRMSNorm /
// GemmaFusedAddRMSNorm under the hood; used to fuse the transformer residual-add
// with the next sublayer's pre-norm (or the model's final norm at the last layer).
void fused_add_rms_norm(tensor_t input, tensor_t residual, tensor_t weight, float eps, bool add_one_to_weight = false);

// LayerNorm with affine + bias (the ViT path uses this; the LLM path uses
// rms_norm). y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias, over the
// last dim. Shapes: out/in [N, D]; weight/bias [D].
void layer_norm_bias(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias, float eps);

// PyTorch gelu_pytorch_tanh (used by Qwen3.5-VL MLP / merger):
//   y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// Element-wise on any shape; in/out same shape and dtype.
void gelu_tanh(tensor_t out, tensor_t in);

// Bidirectional vision self-attention (Qwen3.5-VL ViT). Naive O(N^2 D) kernel
// — fine for ViT where N is in the low thousands and forward runs once per
// image. Declared in include/backend/ops/vision_attention/vision_attention.hpp.
struct VisionAttentionParams;
void vision_attention(const VisionAttentionParams& p);

// Scatter image embeddings into the hidden-state stream at <|image_pad|>
// token positions. In-place on `hidden`. Used right after the embed_tokens
// lookup in the LLM forward to inject vision-tower output for multimodal
// inputs. See backend/ops/scatter_image_embeds/scatter_image_embeds.hpp.
void scatter_image_embeds(tensor_t hidden, tensor_t input_ids, tensor_t image_embeds, int image_token_id);

// Apply rotary positional embedding in-place to a [N, H, D] tensor with
// pre-computed cos/sin tables of shape [N, D]. Used by Qwen3.5-VL vision
// attention with 2D (row, col) RoPE. See backend/ops/apply_rotary_emb/.
void apply_rotary_emb_inplace(tensor_t q, tensor_t cos, tensor_t sin);

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta);
void rope_qk(tensor_t q_out, tensor_t k_out, tensor_t q_in, tensor_t k_in, tensor_t pos_ids, float theta);
void swiglu(tensor_t out, tensor_t gate, tensor_t up);

// Unified attention dispatch.
// Mode determined by AttentionParams fields — see attention_params.hpp.
void attention(const AttentionParams& params);

// out[i] += alpha * a[i]  (in-place weighted accumulation / AXPY)
void add_scaled(tensor_t out, tensor_t a, float alpha);

// Fill tensor with zeros
void fill_zero(tensor_t t);

// dst[r, :] = src[indices[r], :] for r in [0, dst.shape[0]).
// dst/src 2D, same dtype + same device; indices 1D int32, length = dst.shape[0].
// Replaces a per-row memcpy loop with a single launch on GPU.
void gather_rows(tensor_t dst, tensor_t src, tensor_t indices);

// out[indices[r], :] += weights[r] * src[r, :] for r in [0, src.shape[0]).
// Within one call, indices[r] are required to be unique (no intra-call write race).
// Across calls, stream ordering serializes accumulations into the same row.
void scatter_add_rows(tensor_t out, tensor_t src, tensor_t indices, tensor_t weights);

} // namespace zedinfer::ops
