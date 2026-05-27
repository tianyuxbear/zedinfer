#include "zedinfer/batch_context.hpp"

namespace zedinfer {

int ScheduledBatch::total_tokens() const {
    int total = 0;
    for (auto* req : decode_requests) {
        // Spec-decode (Stage D.1): if MTP gave us a draft for the next
        // position, this decode req contributes 2 tokens [last_token, draft]
        // so main can verify both in a single forward.
        total += (req->mtp_pending_draft >= 0) ? 2 : 1;
    }
    for (int cs : prefill_chunk_sizes) { total += cs; }
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
    for (auto* req : decode_requests) {
        const bool spec = (req->mtp_pending_draft >= 0);
        const int  n    = spec ? 2 : 1;

        // Normal decode: 1 token. Spec verify: 2 tokens [last_token, draft] so
        // main can validate the MTP-proposed t+2 in the same forward.
        ctx.token_ids.push_back(req->last_token);
        ctx.position_ids.push_back(req->block_table().seq_len);
        if (spec) {
            ctx.token_ids.push_back(req->mtp_pending_draft);
            ctx.position_ids.push_back(req->block_table().seq_len + 1);
        }

        BatchContext::Slot slot;
        slot.request = req;
        slot.token_offset = offset;
        slot.num_tokens = n;
        slot.past_len = req->block_table().seq_len;
        slot.is_prefill = false;
        ctx.slots.push_back(slot);

        // Block tables for decode attention. seq_len here is the kv-cache
        // length the attention kernel reads (past + the n positions we're
        // about to write).
        ctx.decode_page_tables.push_back(req->block_table().pages[0].data());
        ctx.decode_seq_lens.push_back(req->block_table().seq_len + n);

        offset += n;
    }

    // Prefill slots after decode
    for (size_t i = 0; i < prefill_requests.size(); ++i) {
        auto* req = prefill_requests[i];
        int chunk_start = prefill_chunk_starts[i];
        int chunk_size = prefill_chunk_sizes[i];

        // Add chunk tokens from input_ids
        for (int t = 0; t < chunk_size; ++t) {
            ctx.token_ids.push_back(req->input_ids[chunk_start + t]);
            ctx.position_ids.push_back(req->block_table().seq_len + t);
        }

        BatchContext::Slot slot;
        slot.request = req;
        slot.token_offset = offset;
        slot.num_tokens = chunk_size;
        slot.past_len = req->block_table().seq_len;
        slot.is_prefill = true;
        ctx.slots.push_back(slot);

        ctx.prefill_page_tables.push_back(req->block_table().pages[0].data());
        ctx.prefill_past_lens.push_back(req->block_table().seq_len);
        ctx.prefill_chunk_sizes.push_back(chunk_size);

        offset += chunk_size;
    }

    return ctx;
}

} // namespace zedinfer
