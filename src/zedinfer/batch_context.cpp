#include "zedinfer/batch_context.hpp"

namespace zedinfer {

int ScheduledBatch::total_tokens() const {
    int total = static_cast<int>(decode_requests.size()); // 1 token per decode
    for (int cs : prefill_chunk_sizes) total += cs;
    return total;
}

bool ScheduledBatch::empty() const {
    return decode_requests.empty() && prefill_requests.empty();
}

BatchContext ScheduledBatch::build_context() const {
    BatchContext ctx;
    int offset = 0;

    // Decode slots first (decode-first policy)
    ctx.decode_token_offset = 0;
    for (auto *req : decode_requests) {
        // Each decode request contributes 1 token (last_token)
        ctx.token_ids.push_back(req->last_token);
        ctx.position_ids.push_back(req->active_block_table().seq_len); // position = current kv length

        BatchContext::Slot slot;
        slot.request = req;
        slot.token_offset = offset;
        slot.num_tokens = 1;
        slot.past_len = req->active_block_table().seq_len;
        slot.is_prefill = false;
        ctx.slots.push_back(slot);

        // Block tables for decode attention
        // Use layer 0's block table as representative — kernel receives per-layer tables
        // during the actual forward. Here we store the full block table pointer.
        ctx.decode_k_block_tables.push_back(req->active_block_table().k_blocks[0].data());
        ctx.decode_v_block_tables.push_back(req->active_block_table().v_blocks[0].data());
        ctx.decode_seq_lens.push_back(req->active_block_table().seq_len + 1); // past + current token

        offset++;
    }

    // Prefill slots after decode
    for (size_t i = 0; i < prefill_requests.size(); ++i) {
        auto *req = prefill_requests[i];
        int chunk_start = prefill_chunk_starts[i];
        int chunk_size = prefill_chunk_sizes[i];

        // Add chunk tokens from input_ids
        for (int t = 0; t < chunk_size; ++t) {
            ctx.token_ids.push_back(req->input_ids[chunk_start + t]);
            ctx.position_ids.push_back(req->active_block_table().seq_len + t);
        }

        BatchContext::Slot slot;
        slot.request = req;
        slot.token_offset = offset;
        slot.num_tokens = chunk_size;
        slot.past_len = req->active_block_table().seq_len;
        slot.is_prefill = true;
        ctx.slots.push_back(slot);

        ctx.prefill_k_block_tables.push_back(req->active_block_table().k_blocks[0].data());
        ctx.prefill_v_block_tables.push_back(req->active_block_table().v_blocks[0].data());
        ctx.prefill_past_lens.push_back(req->active_block_table().seq_len);
        ctx.prefill_chunk_sizes.push_back(chunk_size);

        offset += chunk_size;
    }

    return ctx;
}

} // namespace zedinfer
