#include "frontend/models/qwen2.hpp"
#include "backend/core/context/context.hpp"
#include "backend/kvcache/block_pool.hpp"
#include "backend/ops/ops.hpp"
#include "zedinfer/activation.hpp"
#include "zedinfer/batch_context.hpp"

#include <cmath>
#include <cstring>

namespace zedinfer::model {

// Scatter a contiguous K or V tensor slice into paged blocks for one request.
static void scatter_kv_to_blocks(
    tensor_t kv_slice,               // [num_tokens, num_kv_heads, head_dim]
    kvcache::SequenceBlockTable &table,
    int layer_idx,
    int past_len,
    int num_tokens,
    bool is_k,
    kvcache::BlockPool &pool) {

    const int bs = pool.config().block_size;
    const size_t token_bytes = pool.config().token_bytes();
    auto *src = static_cast<const std::byte *>(kv_slice->data());

    auto &blocks = is_k ? table.k_blocks[layer_idx] : table.v_blocks[layer_idx];

    // Ensure enough blocks
    // Note: blocks should already be allocated by scheduler.

    for (int t = 0; t < num_tokens; ++t) {
        int global_pos = past_len + t;
        int block_idx = global_pos / bs;
        int offset_in_block = global_pos % bs;

        void *dst = static_cast<std::byte *>(pool.block_data(blocks[block_idx]))
                    + offset_in_block * token_bytes;

        if (pool.device_type() == ZEDINFER_DEVICE_CPU) {
            std::memcpy(dst, src + t * token_bytes, token_bytes);
        } else {
            // GPU memcpy (same device)
            zedinfer::core::context().setDevice(pool.device_type(), pool.device_id());
            zedinfer::core::context().runtime().api()->memcpy_sync(
                dst, src + t * token_bytes, token_bytes, ZEDINFER_MEMCPY_D2D);
        }
    }
}

tensor_t Qwen2Model::forward_batch(
    const BatchContext &batch,
    kvcache::BlockAllocator &allocator,
    const ExecutorConfig &exec_config) {

    const auto &cfg = config_;
    const int total = batch.total_tokens();
    const size_t hidden_size = cfg.hidden_size;
    const size_t nhead = cfg.num_attention_heads;
    const size_t nkvhead = cfg.num_key_value_heads;
    const size_t head_dim = hidden_size / nhead;
    const size_t kv_dim = nkvhead * head_dim;
    const size_t inter = cfg.intermediate_size;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    auto &pool = allocator.pool();
    const int block_size = pool.config().block_size;

    auto make = [&](std::vector<size_t> shape) {
        return Tensor::create(shape, exec_config.data_type,
                              exec_config.device_type, exec_config.device_id);
    };
    auto make_typed = [&](std::vector<size_t> shape, zedinferDataType_t dtype) {
        return Tensor::create(shape, dtype,
                              exec_config.device_type, exec_config.device_id);
    };
    auto W = [&](const std::string &name) { return weights_->get_tensor(name); };

    // Prepare flattened inputs
    auto ids = make_typed({static_cast<size_t>(total)}, ZEDINFER_DTYPE_I32);
    ids->load(batch.token_ids.data());

    auto pos_ids = make_typed({static_cast<size_t>(total)}, ZEDINFER_DTYPE_I64);
    pos_ids->load(batch.position_ids.data());

    // Embedding [total, hidden_size]
    auto hidden = make({static_cast<size_t>(total), hidden_size});
    ops::embedding(hidden, ids, W("embed_tokens.weight"));

    // Transformer layers
    for (size_t L = 0; L < cfg.num_hidden_layers; ++L) {
        std::string p = "layers." + std::to_string(L) + ".";

        // --- Per-token ops (process all tokens together) ---
        auto normed = make({static_cast<size_t>(total), hidden_size});
        ops::rms_norm(normed, hidden, W(p + "input_layernorm.weight"), cfg.rms_norm_eps);

        auto q_all = make({static_cast<size_t>(total), hidden_size});
        ops::linear(q_all, normed, W(p + "self_attn.q_proj.weight"), W(p + "self_attn.q_proj.bias"));

        auto k_all = make({static_cast<size_t>(total), kv_dim});
        ops::linear(k_all, normed, W(p + "self_attn.k_proj.weight"), W(p + "self_attn.k_proj.bias"));

        auto v_all = make({static_cast<size_t>(total), kv_dim});
        ops::linear(v_all, normed, W(p + "self_attn.v_proj.weight"), W(p + "self_attn.v_proj.bias"));

        // RoPE
        auto q_rope = make({static_cast<size_t>(total), nhead, head_dim});
        ops::rope(q_rope, q_all->view({static_cast<size_t>(total), nhead, head_dim}), pos_ids, cfg.rope_theta);

        auto k_rope = make({static_cast<size_t>(total), nkvhead, head_dim});
        ops::rope(k_rope, k_all->view({static_cast<size_t>(total), nkvhead, head_dim}), pos_ids, cfg.rope_theta);

        // --- Per-slot K/V scatter to blocks ---
        for (const auto &slot : batch.slots) {
            auto k_slot = k_rope->slice(0, slot.token_offset, slot.token_offset + slot.num_tokens);
            auto v_slot = v_all->slice(0, slot.token_offset, slot.token_offset + slot.num_tokens);
            scatter_kv_to_blocks(k_slot, slot.request->block_table, L,
                                 slot.past_len, slot.num_tokens, true, pool);
            scatter_kv_to_blocks(v_slot, slot.request->block_table, L,
                                 slot.past_len, slot.num_tokens, false, pool);
        }

        // --- Attention ---
        auto attn = make({static_cast<size_t>(total), nhead, head_dim});

        int num_decode = batch.num_decode_slots();

        // Batched decode attention
        if (num_decode > 0) {
            // Collect per-request block tables for this layer
            int max_blocks = 0;
            std::vector<InferenceRequest *> decode_reqs;
            for (const auto &slot : batch.slots) {
                if (!slot.is_prefill) {
                    int nb = static_cast<int>(slot.request->block_table.k_blocks[L].size());
                    max_blocks = std::max(max_blocks, nb);
                    decode_reqs.push_back(slot.request);
                }
            }

            // Flatten block tables: [num_decode, max_blocks] padded with 0
            std::vector<int> k_bt(num_decode * max_blocks, 0);
            std::vector<int> v_bt(num_decode * max_blocks, 0);
            std::vector<int> sl(num_decode);

            for (int r = 0; r < num_decode; ++r) {
                auto *req = decode_reqs[r];
                sl[r] = req->block_table.seq_len + 1; // past + current token
                auto &kb = req->block_table.k_blocks[L];
                auto &vb = req->block_table.v_blocks[L];
                for (size_t b = 0; b < kb.size(); ++b) {
                    k_bt[r * max_blocks + b] = kb[b];
                    v_bt[r * max_blocks + b] = vb[b];
                }
            }

            // Upload to device
            auto k_bt_t = make_typed({static_cast<size_t>(num_decode * max_blocks)}, ZEDINFER_DTYPE_I32);
            k_bt_t->load(k_bt.data());
            auto v_bt_t = make_typed({static_cast<size_t>(num_decode * max_blocks)}, ZEDINFER_DTYPE_I32);
            v_bt_t->load(v_bt.data());
            auto sl_t = make_typed({static_cast<size_t>(num_decode)}, ZEDINFER_DTYPE_I32);
            sl_t->load(sl.data());

            auto decode_q = q_rope->slice(0, batch.decode_token_offset,
                                           batch.decode_token_offset + num_decode);
            auto decode_out = attn->slice(0, batch.decode_token_offset,
                                           batch.decode_token_offset + num_decode);

            ops::paged_attention_decode_batched(
                decode_out, decode_q,
                pool.block_data(0),
                reinterpret_cast<const int *>(k_bt_t->data()),
                reinterpret_cast<const int *>(v_bt_t->data()),
                reinterpret_cast<const int *>(sl_t->data()),
                num_decode, max_blocks,
                scale, exec_config.data_type,
                exec_config.device_type, exec_config.device_id,
                nhead, nkvhead, head_dim, block_size);
        }

        // Prefill attention (per-request sequential)
        {
            int prefill_offset = num_decode;
            int pi = 0;
            for (const auto &slot : batch.slots) {
                if (!slot.is_prefill) continue;
                int chunk = slot.num_tokens;

                auto pf_q = q_rope->slice(0, prefill_offset, prefill_offset + chunk);
                auto pf_out = attn->slice(0, prefill_offset, prefill_offset + chunk);

                ops::paged_attention_prefill(
                    pf_out, pf_q,
                    pool.block_data(0),
                    slot.request->block_table.k_blocks[L].data(),
                    slot.request->block_table.v_blocks[L].data(),
                    chunk, slot.past_len,
                    scale, exec_config.data_type,
                    exec_config.device_type, exec_config.device_id,
                    nhead, nkvhead, head_dim, block_size);

                prefill_offset += chunk;
                pi++;
            }
        }

        // --- O projection, residual, MLP (all per-token, process all together) ---
        auto o = make({static_cast<size_t>(total), hidden_size});
        ops::linear(o, attn->view({static_cast<size_t>(total), hidden_size}),
                    W(p + "self_attn.o_proj.weight"), nullptr);

        auto h1 = make({static_cast<size_t>(total), hidden_size});
        ops::add(h1, hidden, o);

        normed = make({static_cast<size_t>(total), hidden_size});
        ops::rms_norm(normed, h1, W(p + "post_attention_layernorm.weight"), cfg.rms_norm_eps);

        auto gate = make({static_cast<size_t>(total), inter});
        auto up = make({static_cast<size_t>(total), inter});
        ops::linear(gate, normed, W(p + "mlp.gate_proj.weight"), nullptr);
        ops::linear(up, normed, W(p + "mlp.up_proj.weight"), nullptr);

        auto act = make({static_cast<size_t>(total), inter});
        ops::swiglu(act, gate, up);

        auto down = make({static_cast<size_t>(total), hidden_size});
        ops::linear(down, act, W(p + "mlp.down_proj.weight"), nullptr);

        hidden = make({static_cast<size_t>(total), hidden_size});
        ops::add(hidden, h1, down);
    }

    // Output head
    auto normed = make({static_cast<size_t>(total), hidden_size});
    ops::rms_norm(normed, hidden, W("norm.weight"), cfg.rms_norm_eps);

    auto logits = make({static_cast<size_t>(total), cfg.vocab_size});
    ops::linear(logits, normed, W("lm_head.weight"), nullptr);

    return logits;
}

} // namespace zedinfer::model
