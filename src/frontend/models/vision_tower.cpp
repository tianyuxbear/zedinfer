#include "frontend/models/vision_tower.hpp"

#include "backend/core/context/context.hpp"
#include "backend/device/runtime_api.hpp"
#include "backend/ops/mamba/ssu.hpp"
#include "backend/ops/ops.hpp"
#include "backend/ops/scatter_image_embeds/scatter_image_embeds.hpp"
#include "backend/ops/vision_attention/vision_attention.hpp"
#include "utils/types.hpp"
#include "zedinfer/activation.hpp"

#include <cmath>
#include <plog/Log.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace zedinfer::model {

VisionTower::VisionTower(const VisionConfig& cfg, const ModelWeights& w, const ExecutorConfig& /*exec*/)
    : cfg_(cfg), weights_(&w) {
    const int needed_blocks = cfg.depth;
    for (int i = 0; i < needed_blocks; ++i) {
        const std::string prefix = "visual.blocks." + std::to_string(i) + ".";
        if (!w.has_tensor(prefix + "attn.qkv.weight")) {
            throw std::runtime_error("[VisionTower] missing weight: " + prefix + "attn.qkv.weight");
        }
    }
    if (!w.has_tensor("visual.patch_embed.proj.weight")) {
        throw std::runtime_error("[VisionTower] missing patch_embed weight: visual.patch_embed.proj.weight");
    }
    if (!w.has_tensor("visual.merger.linear_fc2.weight")) {
        throw std::runtime_error("[VisionTower] missing merger weight: visual.merger.linear_fc2.weight");
    }

    // Pre-compute static resources used by every forward call.
    //   pos_embed_table_f32_ : D2H copy of the [num_pos_emb, H] embedding
    //                          table converted to FP32 once — host-side
    //                          bilinear interpolation reads it per request
    //                          without paying the PCIe round-trip every time.
    //   rope_inv_freq_       : 1 / theta^(2k / axis_half), invariant across
    //                          requests; reused for cos/sin construction.
    if (auto pos_embed_w
        = w.has_tensor("visual.pos_embed.weight") ? w.get_tensor("visual.pos_embed.weight") : nullptr) {
        const size_t numel = static_cast<size_t>(pos_embed_w->numel());
        std::vector<zedinfer::bf16_t> bf16_buf(numel);
        auto* api = core::context().runtime().api();
        const auto kind = pos_embed_w->deviceType() == ZEDINFER_DEVICE_CPU ? ZEDINFER_MEMCPY_H2H : ZEDINFER_MEMCPY_D2H;
        api->memcpy_sync(bf16_buf.data(), pos_embed_w->data(), numel * sizeof(zedinfer::bf16_t), kind);
        pos_embed_table_f32_.resize(numel);
        for (size_t i = 0; i < numel; ++i) { pos_embed_table_f32_[i] = utils::cast<float>(bf16_buf[i]); }
        num_grid_per_side_ = static_cast<int>(std::sqrt(static_cast<double>(pos_embed_w->dim(0))));
        LOGI.printf("[VisionTower] cached pos_embed table [%zu, %d] f32 (%.1f MB) on host",
                    static_cast<size_t>(pos_embed_w->dim(0)), static_cast<int>(pos_embed_w->dim(1)),
                    static_cast<double>(numel * 4) / (1024.0 * 1024.0));
    }
    {
        const int axis_half = (cfg.hidden_size / cfg.num_heads) / 2; // head_dim / 2
        const int freq_count = axis_half / 2;                        // head_dim / 4
        const float theta = 10000.0f;
        rope_inv_freq_.resize(static_cast<size_t>(freq_count));
        for (int k = 0; k < freq_count; ++k) {
            rope_inv_freq_[k] = 1.0f / std::pow(theta, static_cast<float>(2 * k) / static_cast<float>(axis_half));
        }
    }

    LOGI.printf("[VisionTower] ctor verified %d blocks + patch_embed + merger present (hidden=%d, out=%d, heads=%d)",
                needed_blocks, cfg.hidden_size, cfg.out_hidden_size, cfg.num_heads);
}

VisionTower::~VisionTower() = default;

namespace {

tensor_t maybe_get(const ModelWeights& w, const std::string& name) {
    return w.has_tensor(name) ? w.get_tensor(name) : nullptr;
}

} // namespace

tensor_t VisionTower::forward(tensor_t patches, tensor_t /*pos_ids_thw*/, int grid_h, int grid_w,
                              const ExecutorConfig& exec) {
    if (!weights_) {
        throw std::runtime_error("[VisionTower] forward called without weights bound");
    }
    const ModelWeights& w = *weights_;
    const int N = static_cast<int>(patches->shape()[0]);
    const int H = cfg_.hidden_size;
    const int Dh = H / cfg_.num_heads;
    const int Inter = cfg_.intermediate_size;
    const int OutH = cfg_.out_hidden_size;
    const int sms = cfg_.spatial_merge_size;
    const int merged_n = N / (sms * sms);
    const int grid_hm = grid_h / sms;
    const int grid_wm = grid_w / sms;

    if (merged_n <= 0 || merged_n * sms * sms != N) {
        throw std::runtime_error("[VisionTower] N=" + std::to_string(N) + " not divisible by spatial_merge^2 ("
                                 + std::to_string(sms * sms) + ")");
    }
    if (grid_h * grid_w != N) {
        throw std::runtime_error("[VisionTower] grid_h*grid_w (" + std::to_string(grid_h * grid_w)
                                 + ") does not match N (" + std::to_string(N) + ")");
    }
    if (grid_h % sms != 0 || grid_w % sms != 0) {
        throw std::runtime_error("[VisionTower] grid (" + std::to_string(grid_h) + "x" + std::to_string(grid_w)
                                 + ") not divisible by spatial_merge_size=" + std::to_string(sms)
                                 + " — image preprocessor must align both dims to patch_size*spatial_merge_size");
    }

    // eps used for every LayerNorm in this forward; read from the model config
    // so non-standard variants are honored without recompiling.
    const float eps = cfg_.layer_norm_eps;

    auto make = [&](std::vector<size_t> shape, zedinferDataType_t dtype = ZEDINFER_DTYPE_F32) {
        return Tensor::create(std::move(shape), dtype == ZEDINFER_DTYPE_F32 ? exec.data_type : dtype, exec.device_type,
                              exec.device_id);
    };

    // 1. patch_embed projection. HF stores the Conv3d weight as 5-D
    //    [hidden_size, in_channels, T_patch, H_patch, W_patch]; we view it as
    //    a contiguous 2-D matrix [hidden_size, in_channels*T_patch*H_patch*W_patch]
    //    so ops::linear (a standard GEMM + bias) reproduces Conv3d's effect
    //    (kernel = stride = patch size means each output position consumes
    //    exactly one input patch — pure linear projection).
    auto hidden = make({static_cast<size_t>(N), static_cast<size_t>(H)});
    auto pe_w_raw = w.get_tensor("visual.patch_embed.proj.weight");
    tensor_t pe_w = pe_w_raw;
    if (pe_w_raw->shape().size() > 2) {
        pe_w = pe_w_raw->view({pe_w_raw->dim(0), pe_w_raw->numel() / pe_w_raw->dim(0)});
    }
    ops::linear(hidden, patches, pe_w, maybe_get(w, "visual.patch_embed.proj.bias"));

    // 2. fast_pos_embed_interpolate using the ctor-cached pos_embed_table_f32_.
    //    Avoids the per-request D2H of the [2304, H] embedding table — bilinear
    //    interpolation runs on the cached host fp32 copy then a single H2D
    //    upload provides the additive bf16 contribution.
    if (!pos_embed_table_f32_.empty()) {
        const int NGS = num_grid_per_side_;
        const int PE_H = H; // pos_embed second dim == hidden_size by construction

        const float h_scale = (grid_h > 1) ? (static_cast<float>(NGS - 1) / static_cast<float>(grid_h - 1)) : 0.0f;
        const float w_scale = (grid_w > 1) ? (static_cast<float>(NGS - 1) / static_cast<float>(grid_w - 1)) : 0.0f;

        std::vector<float> pos_out_f32(static_cast<size_t>(N) * PE_H, 0.0f);
        for (int ph = 0; ph < grid_h; ++ph) {
            const float h_pos = static_cast<float>(ph) * h_scale;
            int h_lo = std::min(static_cast<int>(std::floor(h_pos)), NGS - 1);
            int h_hi = std::min(h_lo + 1, NGS - 1);
            float dh = h_pos - static_cast<float>(h_lo);
            for (int pw = 0; pw < grid_w; ++pw) {
                const float w_pos = static_cast<float>(pw) * w_scale;
                int w_lo = std::min(static_cast<int>(std::floor(w_pos)), NGS - 1);
                int w_hi = std::min(w_lo + 1, NGS - 1);
                float dw = w_pos - static_cast<float>(w_lo);
                const float w_tl = (1.0f - dh) * (1.0f - dw);
                const float w_tr = (1.0f - dh) * dw;
                const float w_bl = dh * (1.0f - dw);
                const float w_br = dh * dw;
                const float* tl = pos_embed_table_f32_.data() + (h_lo * NGS + w_lo) * PE_H;
                const float* tr = pos_embed_table_f32_.data() + (h_lo * NGS + w_hi) * PE_H;
                const float* bl = pos_embed_table_f32_.data() + (h_hi * NGS + w_lo) * PE_H;
                const float* br = pos_embed_table_f32_.data() + (h_hi * NGS + w_hi) * PE_H;
                float* out = pos_out_f32.data() + (ph * grid_w + pw) * PE_H;
                for (int d = 0; d < PE_H; ++d) { out[d] = w_tl * tl[d] + w_tr * tr[d] + w_bl * bl[d] + w_br * br[d]; }
            }
        }

        std::vector<zedinfer::bf16_t> pos_out_bf16(pos_out_f32.size());
        for (size_t i = 0; i < pos_out_f32.size(); ++i) {
            pos_out_bf16[i] = utils::cast<zedinfer::bf16_t>(pos_out_f32[i]);
        }
        auto pos_t = make({static_cast<size_t>(N), static_cast<size_t>(PE_H)}, ZEDINFER_DTYPE_BF16);
        auto* api = core::context().runtime().api();
        api->memcpy_sync(pos_t->data(), pos_out_bf16.data(), pos_out_bf16.size() * sizeof(zedinfer::bf16_t),
                         exec.device_type == ZEDINFER_DEVICE_CPU ? ZEDINFER_MEMCPY_H2H : ZEDINFER_MEMCPY_H2D);
        ops::add(hidden, hidden, pos_t);
    }

    // 3. 27 ViT blocks: pre-norm → qkv → vision_attention → proj + residual →
    //                   pre-norm → fc1 → gelu_tanh → fc2 + residual
    auto h_norm = make({static_cast<size_t>(N), static_cast<size_t>(H)});
    auto qkv = make({static_cast<size_t>(N), static_cast<size_t>(3 * H)});
    auto attn_out = make({static_cast<size_t>(N), static_cast<size_t>(cfg_.num_heads), static_cast<size_t>(Dh)});
    auto proj_out = make({static_cast<size_t>(N), static_cast<size_t>(H)});
    auto fc1 = make({static_cast<size_t>(N), static_cast<size_t>(Inter)});
    auto fc2 = make({static_cast<size_t>(N), static_cast<size_t>(H)});
    auto q_buf = make({static_cast<size_t>(N), static_cast<size_t>(cfg_.num_heads), static_cast<size_t>(Dh)});
    auto k_buf = make({static_cast<size_t>(N), static_cast<size_t>(cfg_.num_heads), static_cast<size_t>(Dh)});
    auto v_buf = make({static_cast<size_t>(N), static_cast<size_t>(cfg_.num_heads), static_cast<size_t>(Dh)});

    const float scale = 1.0f / std::sqrt(static_cast<float>(Dh));
    const size_t elt = utils::dsize(exec.data_type);

    // 2-D RoPE cos/sin tables. HF Qwen3.5-VL applies a 2-D rotary embedding:
    //   - head_dim = Dh; split in half (first Dh/2 dims = row index, second
    //     Dh/2 dims = col index)
    //   - within each half, the standard HF non-interleaved layout repeats
    //     the [0..half/2) frequencies twice so rotate_half pairs (d, d+half/2)
    //     within each half. So per-patch emb = [
    //       f(ph) for k=0..Dh/4-1,
    //       f(ph) for k=0..Dh/4-1,                       <- repeat (first half)
    //       f(pw) for k=0..Dh/4-1,
    //       f(pw) for k=0..Dh/4-1                        <- repeat (second half)
    //     ]
    //     where f(p)[k] = p * inv_freq[k], inv_freq[k] = 1 / theta^(2k/(Dh/2)).
    //   - cos/sin are element-wise cosf/sinf of that.
    tensor_t cos_t, sin_t;
    {
        const int axis_half = Dh / 2;
        const int freq_count = axis_half / 2; // == Dh/4
        // freq_table[p, k] = p * inv_freq[k]; p ranges over max(grid_h, grid_w).
        // Reuses ctor-cached rope_inv_freq_.
        const int max_side = std::max(grid_h, grid_w);
        std::vector<float> freq_table(static_cast<size_t>(max_side) * freq_count);
        for (int p = 0; p < max_side; ++p) {
            for (int k = 0; k < freq_count; ++k) {
                freq_table[p * freq_count + k] = static_cast<float>(p) * rope_inv_freq_[k];
            }
        }
        std::vector<float> cos_host(static_cast<size_t>(N) * Dh);
        std::vector<float> sin_host(static_cast<size_t>(N) * Dh);
        // HF assembles the per-token 72-dim embedding as
        //     emb = cat( flatten([h_freqs (18), w_freqs (18)]),     # axis_half dims
        //                flatten([h_freqs (18), w_freqs (18)]) )    # axis_half dims (repeat)
        // i.e. [h, w, h, w] interleaved by axis_half — NOT [h, h, w, w].
        // rotate_half then pairs index d with index d+axis_half, which always
        // sees the same axis (h pairs with h, w pairs with w).
        for (int ph = 0; ph < grid_h; ++ph) {
            for (int pw = 0; pw < grid_w; ++pw) {
                const int n = ph * grid_w + pw;
                float* cr = cos_host.data() + static_cast<size_t>(n) * Dh;
                float* sr = sin_host.data() + static_cast<size_t>(n) * Dh;
                for (int k = 0; k < freq_count; ++k) {
                    const float v_h = freq_table[ph * freq_count + k];
                    const float v_w = freq_table[pw * freq_count + k];
                    // First half [0, axis_half): h then w.
                    cr[k] = std::cos(v_h);
                    cr[k + freq_count] = std::cos(v_w);
                    sr[k] = std::sin(v_h);
                    sr[k + freq_count] = std::sin(v_w);
                    // Second half [axis_half, 2*axis_half=Dh): repeat of first half.
                    cr[axis_half + k] = std::cos(v_h);
                    cr[axis_half + k + freq_count] = std::cos(v_w);
                    sr[axis_half + k] = std::sin(v_h);
                    sr[axis_half + k + freq_count] = std::sin(v_w);
                }
            }
        }
        // FP32 → BF16 host conv, then H2D.
        std::vector<zedinfer::bf16_t> cos_bf16(cos_host.size()), sin_bf16(sin_host.size());
        for (size_t i = 0; i < cos_host.size(); ++i) {
            cos_bf16[i] = zedinfer::utils::cast<zedinfer::bf16_t>(cos_host[i]);
            sin_bf16[i] = zedinfer::utils::cast<zedinfer::bf16_t>(sin_host[i]);
        }
        cos_t = make({static_cast<size_t>(N), static_cast<size_t>(Dh)}, ZEDINFER_DTYPE_BF16);
        sin_t = make({static_cast<size_t>(N), static_cast<size_t>(Dh)}, ZEDINFER_DTYPE_BF16);
        auto* api = core::context().runtime().api();
        const auto kind = exec.device_type == ZEDINFER_DEVICE_CPU ? ZEDINFER_MEMCPY_H2H : ZEDINFER_MEMCPY_H2D;
        api->memcpy_sync(cos_t->data(), cos_bf16.data(), cos_bf16.size() * sizeof(zedinfer::bf16_t), kind);
        api->memcpy_sync(sin_t->data(), sin_bf16.data(), sin_bf16.size() * sizeof(zedinfer::bf16_t), kind);
    }

    for (int L = 0; L < cfg_.depth; ++L) {
        const std::string p = "visual.blocks." + std::to_string(L) + ".";

        ops::layer_norm_bias(h_norm, hidden, w.get_tensor(p + "norm1.weight"), w.get_tensor(p + "norm1.bias"), eps);

        ops::linear(qkv, h_norm, w.get_tensor(p + "attn.qkv.weight"), maybe_get(w, p + "attn.qkv.bias"));

        // Split qkv [N, 3*H] into three contiguous [N, num_heads, Dh] buffers
        // via strided memcpy2D — vision_attention's naive kernel reads each
        // tensor as a contiguous [N, H, D] block.
        ops::mamba::copy_strided_rows(q_buf->view({static_cast<size_t>(N), static_cast<size_t>(H)}), qkv,
                                      /*src_offset=*/0, /*slice_width=*/static_cast<size_t>(H),
                                      /*src_width=*/static_cast<size_t>(3 * H), static_cast<size_t>(N), elt);
        ops::mamba::copy_strided_rows(k_buf->view({static_cast<size_t>(N), static_cast<size_t>(H)}), qkv,
                                      static_cast<size_t>(H), static_cast<size_t>(H), static_cast<size_t>(3 * H),
                                      static_cast<size_t>(N), elt);
        ops::mamba::copy_strided_rows(v_buf->view({static_cast<size_t>(N), static_cast<size_t>(H)}), qkv,
                                      static_cast<size_t>(2 * H), static_cast<size_t>(H), static_cast<size_t>(3 * H),
                                      static_cast<size_t>(N), elt);

        // Apply Qwen3.5-VL's 2-D RoPE to q and k before attention. Same
        // cos/sin reused across all 27 blocks.
        ops::apply_rotary_emb_inplace(q_buf, cos_t, sin_t);
        ops::apply_rotary_emb_inplace(k_buf, cos_t, sin_t);

        ops::VisionAttentionParams ap;
        ap.q = q_buf;
        ap.k = k_buf;
        ap.v = v_buf;
        ap.out = attn_out;
        ap.scale = scale;
        ops::vision_attention(ap);

        ops::linear(proj_out, attn_out->view({static_cast<size_t>(N), static_cast<size_t>(H)}),
                    w.get_tensor(p + "attn.proj.weight"), maybe_get(w, p + "attn.proj.bias"));
        ops::add(hidden, hidden, proj_out);

        ops::layer_norm_bias(h_norm, hidden, w.get_tensor(p + "norm2.weight"), w.get_tensor(p + "norm2.bias"), eps);
        ops::linear(fc1, h_norm, w.get_tensor(p + "mlp.linear_fc1.weight"), maybe_get(w, p + "mlp.linear_fc1.bias"));
        ops::gelu_tanh(fc1, fc1);
        ops::linear(fc2, fc1, w.get_tensor(p + "mlp.linear_fc2.weight"), maybe_get(w, p + "mlp.linear_fc2.bias"));
        ops::add(hidden, hidden, fc2);
    }

    // 4. Merger: HF order is
    //      a) LayerNorm on per-token [N, H]  (norm.weight/bias both size H=1152)
    //      b) reshape [merged_n, sms*sms*H] grouping 2x2 spatial neighbors
    //      c) linear_fc1 (4608 → 4608) + GELU + linear_fc2 (4608 → out_hidden)
    //    Norm BEFORE merge: norm acts on the original hidden dim, not the
    //    concatenated 4608. zedinfer previously normed AFTER merge — that
    //    matches `use_postshuffle_norm=True` which Qwen3.5-VL does not use.
    auto merger_norm_w = w.get_tensor("visual.merger.norm.weight"); // [H]
    auto merger_norm_b = w.get_tensor("visual.merger.norm.bias");   // [H]
    auto hidden_normed = make({static_cast<size_t>(N), static_cast<size_t>(H)});
    ops::layer_norm_bias(hidden_normed, hidden, merger_norm_w, merger_norm_b, eps);

    // 2x2 spatial gather: each merged_idx pulls four rows (sy=0,1 × sx=0,1)
    // from the normed hidden state, in (sy*sms+sx) order so the resulting
    // view([merged_n, sms*sms*H]) carries the 4 neighbors concatenated.
    std::vector<int32_t> gather_idx(static_cast<size_t>(merged_n) * sms * sms);
    for (int my = 0; my < grid_hm; ++my) {
        for (int mx = 0; mx < grid_wm; ++mx) {
            const int merged_idx = my * grid_wm + mx;
            int slot = 0;
            for (int sy = 0; sy < sms; ++sy) {
                for (int sx = 0; sx < sms; ++sx) {
                    const int orig_row = (my * sms + sy) * grid_w + (mx * sms + sx);
                    gather_idx[static_cast<size_t>(merged_idx) * (sms * sms) + slot] = orig_row;
                    ++slot;
                }
            }
        }
    }
    auto idx_t = make({gather_idx.size()}, ZEDINFER_DTYPE_I32);
    auto* api = core::context().runtime().api();
    api->memcpy_sync(idx_t->data(), gather_idx.data(), gather_idx.size() * sizeof(int32_t),
                     exec.device_type == ZEDINFER_DEVICE_CPU ? ZEDINFER_MEMCPY_H2H : ZEDINFER_MEMCPY_H2D);
    auto gathered = make({gather_idx.size(), static_cast<size_t>(H)});
    ops::gather_rows(gathered, hidden_normed, idx_t);
    auto merged_in = gathered->view({static_cast<size_t>(merged_n), static_cast<size_t>(sms * sms * H)});

    // Merger MLP fc1 + GELU + fc2.
    auto fc1_w = w.get_tensor("visual.merger.linear_fc1.weight");
    auto fc1_b = maybe_get(w, "visual.merger.linear_fc1.bias");
    const size_t fc1_out = fc1_w->dim(0);
    auto m_fc1 = make({static_cast<size_t>(merged_n), fc1_out});
    ops::linear(m_fc1, merged_in, fc1_w, fc1_b);
    ops::gelu_tanh(m_fc1, m_fc1);

    auto fc2_w = w.get_tensor("visual.merger.linear_fc2.weight");
    auto fc2_b = maybe_get(w, "visual.merger.linear_fc2.bias");
    auto out = make({static_cast<size_t>(merged_n), static_cast<size_t>(OutH)});
    ops::linear(out, m_fc1, fc2_w, fc2_b);
    return out;
}

} // namespace zedinfer::model
