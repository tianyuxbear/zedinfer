#include "backend/ops/self_attention/cpu/paged_attention_cpu.hpp"
#include "utils/check.hpp"
#include "utils/types.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <omp.h>
#include <vector>

// Paged attention decode for CPU.
// Same algorithm as self_attention_cpu.cpp but K/V accessed via block table.
// seqlen is always 1 (decode).

template <typename T>
static void paged_attention_decode_(
    T *attn_val, const T *q,
    const T *pool_base,
    const int *k_block_table, const int *v_block_table,
    int seq_len, float scale,
    int nhead, int nkvhead, int head_dim, int block_size) {

    const int group_size = nhead / nkvhead;

#pragma omp parallel for
    for (int h = 0; h < nhead; ++h) {
        const int kvh = h / group_size;

        // Step 1: Q*K scores with block-table-indexed K
        float max_score = -std::numeric_limits<float>::infinity();
        std::vector<float> scores(seq_len);

        for (int j = 0; j < seq_len; ++j) {
            int block_idx = j / block_size;
            int block_offset = j % block_size;
            int k_physical = k_block_table[block_idx] * block_size + block_offset;

            float dot = 0.0f;
            const size_t q_base = h * head_dim;
            const size_t k_base = static_cast<size_t>(k_physical) * nkvhead * head_dim + kvh * head_dim;

#pragma omp simd reduction(+ : dot)
            for (int dim = 0; dim < head_dim; ++dim) {
                float q_val = zedinfer::utils::cast<float>(q[q_base + dim]);
                float k_val = zedinfer::utils::cast<float>(pool_base[k_base + dim]);
                dot += q_val * k_val;
            }
            scores[j] = dot * scale;
            max_score = std::max(max_score, scores[j]);
        }

        // Step 2: softmax
        float exp_sum = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
            scores[j] = std::exp(scores[j] - max_score);
            exp_sum += scores[j];
        }
        float inv_exp_sum = 1.0f / exp_sum;

        // Step 3: weighted V aggregation with block-table-indexed V
        for (int dv_dim = 0; dv_dim < head_dim; ++dv_dim) {
            float out_val = 0.0f;

            for (int j = 0; j < seq_len; ++j) {
                int block_idx = j / block_size;
                int block_offset = j % block_size;
                int v_physical = v_block_table[block_idx] * block_size + block_offset;

                size_t v_idx = static_cast<size_t>(v_physical) * nkvhead * head_dim + kvh * head_dim + dv_dim;
                out_val += scores[j] * zedinfer::utils::cast<float>(pool_base[v_idx]);
            }

            attn_val[h * head_dim + dv_dim] = zedinfer::utils::cast<T>(out_val * inv_exp_sum);
        }
    }
}

namespace zedinfer::ops::cpu {

void paged_attention_decode(
    std::byte *attn_val, const std::byte *q,
    const std::byte *pool_base,
    const int *k_block_table, const int *v_block_table,
    int seq_len,
    float scale, zedinferDataType_t type,
    int nhead, int nkvhead, int head_dim, int block_size) {

    switch (type) {
    case ZEDINFER_DTYPE_F32:
        return paged_attention_decode_(
            reinterpret_cast<float *>(attn_val), reinterpret_cast<const float *>(q),
            reinterpret_cast<const float *>(pool_base),
            k_block_table, v_block_table,
            seq_len, scale, nhead, nkvhead, head_dim, block_size);
    case ZEDINFER_DTYPE_BF16:
        return paged_attention_decode_(
            reinterpret_cast<zedinfer::bf16_t *>(attn_val), reinterpret_cast<const zedinfer::bf16_t *>(q),
            reinterpret_cast<const zedinfer::bf16_t *>(pool_base),
            k_block_table, v_block_table,
            seq_len, scale, nhead, nkvhead, head_dim, block_size);
    case ZEDINFER_DTYPE_F16:
        return paged_attention_decode_(
            reinterpret_cast<zedinfer::fp16_t *>(attn_val), reinterpret_cast<const zedinfer::fp16_t *>(q),
            reinterpret_cast<const zedinfer::fp16_t *>(pool_base),
            k_block_table, v_block_table,
            seq_len, scale, nhead, nkvhead, head_dim, block_size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace zedinfer::ops::cpu
