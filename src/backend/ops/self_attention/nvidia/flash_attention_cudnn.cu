#ifdef USE_CUDNN_FLASH

#include "backend/ops/self_attention/nvidia/flash_attention_cudnn.cuh"
#include "utils/check.hpp"

#include <cudnn_frontend.h>
#include <cuda_runtime.h>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace fe = cudnn_frontend;

namespace zedinfer::ops::nvidia::cudnn_flash {

// ============================================================================
// Transpose kernels: seq-major <-> head-major
// Only needed for Q and O (small: seqlen tokens).
// K/V use cuDNN custom strides to avoid transpose entirely.
// ============================================================================

template <typename T>
__global__ void transpose_seq_to_head(T *dst, const T *src,
                                       int seqlen, int nhead, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seqlen * nhead * head_dim;
    if (idx >= total) return;
    int d = idx % head_dim;
    int h = (idx / head_dim) % nhead;
    int s = idx / (head_dim * nhead);
    dst[h * seqlen * head_dim + s * head_dim + d] = src[s * nhead * head_dim + h * head_dim + d];
}

template <typename T>
__global__ void transpose_head_to_seq(T *dst, const T *src,
                                       int seqlen, int nhead, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seqlen * nhead * head_dim;
    if (idx >= total) return;
    int d = idx % head_dim;
    int s = (idx / head_dim) % seqlen;
    int h = idx / (head_dim * seqlen);
    dst[s * nhead * head_dim + h * head_dim + d] = src[h * seqlen * head_dim + s * head_dim + d];
}

// ============================================================================
// Singleton cuDNN handle
// ============================================================================

static cudnnHandle_t &get_handle() {
    static cudnnHandle_t h = []() {
        cudnnHandle_t h;
        if (cudnnCreate(&h) != CUDNN_STATUS_SUCCESS)
            throw std::runtime_error("Failed to create cuDNN handle");
        return h;
    }();
    return h;
}

// ============================================================================
// Graph cache
// ============================================================================

struct FlashAttnKey {
    int seqlen_q, seqlen_kv, nhead, nkvhead, head_dim;
    zedinferDataType_t dtype;
    bool kv_seq_major;  // true = K/V use seq-major strides

    bool operator==(const FlashAttnKey &o) const {
        return seqlen_q == o.seqlen_q && seqlen_kv == o.seqlen_kv &&
               nhead == o.nhead && nkvhead == o.nkvhead &&
               head_dim == o.head_dim && dtype == o.dtype &&
               kv_seq_major == o.kv_seq_major;
    }
};

struct FlashAttnKeyHash {
    size_t operator()(const FlashAttnKey &k) const {
        size_t h = std::hash<int>()(k.seqlen_q);
        h ^= std::hash<int>()(k.seqlen_kv) * 2654435761u;
        h ^= std::hash<int>()(k.nhead) * 40503u;
        h ^= std::hash<int>()(k.nkvhead) * 12289u;
        h ^= std::hash<int>()(k.head_dim) * 73856093u;
        h ^= std::hash<int>()(static_cast<int>(k.dtype)) << 5;
        h ^= std::hash<bool>()(k.kv_seq_major) << 6;
        return h;
    }
};

struct CachedGraph {
    std::shared_ptr<fe::graph::Graph> graph;
    void *workspace = nullptr;
    size_t workspace_size = 0;
    bool kv_seq_major = false;  // remember which K/V layout was used
    ~CachedGraph() { if (workspace) cudaFree(workspace); }
};

static std::unordered_map<FlashAttnKey, std::shared_ptr<CachedGraph>, FlashAttnKeyHash> graph_cache;
static std::mutex graph_mutex;

static fe::DataType_t to_fe_dtype(zedinferDataType_t dt) {
    switch (dt) {
    case ZEDINFER_DTYPE_F16: return fe::DataType_t::HALF;
    case ZEDINFER_DTYPE_BF16: return fe::DataType_t::BFLOAT16;
    default: throw std::runtime_error("cuDNN FlashAttention only supports FP16/BF16");
    }
}

// ============================================================================
// Try to build a cuDNN SDPA graph with given K/V stride pattern.
// Returns nullptr if cuDNN doesn't support this configuration.
// ============================================================================

static std::shared_ptr<CachedGraph> try_build(
    int seqlen_q, int seqlen_kv, int nhead, int nkvhead, int head_dim,
    zedinferDataType_t dtype, float scale, bool kv_seq_major) {

    auto fe_dt = to_fe_dtype(dtype);
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe_dt)
         .set_intermediate_data_type(fe::DataType_t::FLOAT)
         .set_compute_data_type(fe::DataType_t::FLOAT);

    int64_t sq = seqlen_q, skv = seqlen_kv;
    int64_t nh = nhead, nkvh = nkvhead, hd = head_dim;

    // Q and O: always head-major [1, nhead, seqlen, head_dim]
    auto Q = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("Q")
        .set_dim({1, nh, sq, hd})
        .set_stride({nh * sq * hd, sq * hd, hd, 1})
        .set_uid(1));

    // K/V: head-major or seq-major depending on kv_seq_major flag
    std::vector<int64_t> kv_stride;
    if (kv_seq_major) {
        // Actual memory: [seqlen_kv, nkvhead, head_dim] seq-major
        // Described as [1, nkvh, skv, hd] with strides: [skv*nkvh*hd, hd, nkvh*hd, 1]
        kv_stride = {skv * nkvh * hd, hd, nkvh * hd, 1};
    } else {
        // Standard head-major: [1, nkvh, skv, hd] stride=[nkvh*skv*hd, skv*hd, hd, 1]
        kv_stride = {nkvh * skv * hd, skv * hd, hd, 1};
    }

    auto K = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("K").set_dim({1, nkvh, skv, hd}).set_stride(kv_stride).set_uid(2));
    auto V = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("V").set_dim({1, nkvh, skv, hd}).set_stride(kv_stride).set_uid(3));

    auto sdpa_opts = fe::graph::SDPA_attributes()
        .set_name("flash_attn")
        .set_generate_stats(true)
        .set_causal_mask(true)
        .set_attn_scale(scale);

    auto [O, Stats] = graph->sdpa(Q, K, V, sdpa_opts);

    O->set_output(true)
      .set_dim({1, nh, sq, hd})
      .set_stride({nh * sq * hd, sq * hd, hd, 1})
      .set_uid(4);
    Stats->set_output(true).set_data_type(fe::DataType_t::FLOAT).set_uid(5);

    auto &handle = get_handle();

    if (!graph->validate().is_good()) return nullptr;
    if (!graph->build_operation_graph(handle).is_good()) return nullptr;
    if (!graph->create_execution_plans({fe::HeurMode_t::A}).is_good()) return nullptr;
    if (!graph->check_support(handle).is_good()) return nullptr;
    if (!graph->build_plans(handle).is_good()) return nullptr;

    auto cached = std::make_shared<CachedGraph>();
    cached->graph = graph;
    cached->kv_seq_major = kv_seq_major;
    cached->workspace_size = graph->get_workspace_size();
    if (cached->workspace_size > 0)
        cudaMalloc(&cached->workspace, cached->workspace_size);

    return cached;
}

// ============================================================================
// Get or build cached graph.
// Strategy: try seq-major K/V first (no K/V transpose needed).
// If cuDNN doesn't support it, fall back to head-major K/V (needs transpose).
// If neither works, return nullptr (caller uses custom kernel).
// ============================================================================

static std::shared_ptr<CachedGraph> get_or_build(
    int seqlen_q, int seqlen_kv, int nhead, int nkvhead, int head_dim,
    zedinferDataType_t dtype, float scale) {

    // Check cache for both layouts
    for (bool kv_sm : {true, false}) {
        FlashAttnKey key{seqlen_q, seqlen_kv, nhead, nkvhead, head_dim, dtype, kv_sm};
        std::lock_guard<std::mutex> lock(graph_mutex);
        auto it = graph_cache.find(key);
        if (it != graph_cache.end()) return it->second;
    }

    // Try seq-major K/V first (optimal: no K/V transpose)
    auto cached = try_build(seqlen_q, seqlen_kv, nhead, nkvhead, head_dim, dtype, scale, true);
    if (cached) {
        FlashAttnKey key{seqlen_q, seqlen_kv, nhead, nkvhead, head_dim, dtype, true};
        std::lock_guard<std::mutex> lock(graph_mutex);
        graph_cache[key] = cached;
        return cached;
    }

    // Fall back to head-major K/V (needs K/V transpose)
    cached = try_build(seqlen_q, seqlen_kv, nhead, nkvhead, head_dim, dtype, scale, false);
    if (cached) {
        FlashAttnKey key{seqlen_q, seqlen_kv, nhead, nkvhead, head_dim, dtype, false};
        std::lock_guard<std::mutex> lock(graph_mutex);
        graph_cache[key] = cached;
        return cached;
    }

    return nullptr;  // cuDNN doesn't support this config at all
}

// ============================================================================
// Pre-allocated buffer pool (only for Q, O, stats, and optionally K/V)
// ============================================================================

struct TransposeBuffers {
    std::byte *q = nullptr, *o = nullptr;
    std::byte *k = nullptr, *v = nullptr;  // only used when kv_seq_major=false
    void *stats = nullptr;
    size_t q_cap = 0, kv_cap = 0, stats_cap = 0;

    void ensure_q_o(size_t bytes) {
        if (bytes > q_cap) {
            if (q) cudaFree(q);
            if (o) cudaFree(o);
            cudaMalloc(&q, bytes);
            cudaMalloc(&o, bytes);
            q_cap = bytes;
        }
    }
    void ensure_kv(size_t bytes) {
        if (bytes > kv_cap) {
            if (k) cudaFree(k);
            if (v) cudaFree(v);
            cudaMalloc(&k, bytes);
            cudaMalloc(&v, bytes);
            kv_cap = bytes;
        }
    }
    void ensure_stats(size_t bytes) {
        if (bytes > stats_cap) {
            if (stats) cudaFree(stats);
            cudaMalloc(&stats, bytes);
            stats_cap = bytes;
        }
    }
};

static TransposeBuffers &get_buffers() {
    static TransposeBuffers bufs;
    return bufs;
}

// ============================================================================
// Public API
// ============================================================================

bool flash_attention_prefill(
    std::byte *output, const std::byte *q, const std::byte *k, const std::byte *v,
    float scale, zedinferDataType_t type,
    int seqlen, int nhead, int head_dim, int total_len, int nkvhead) {

    auto cached = get_or_build(seqlen, total_len, nhead, nkvhead, head_dim, type, scale);
    if (!cached) return false;

    size_t elem = (type == ZEDINFER_DTYPE_F16 || type == ZEDINFER_DTYPE_BF16) ? 2 : 4;
    size_t q_bytes = (size_t)seqlen * nhead * head_dim * elem;
    size_t kv_bytes = (size_t)total_len * nkvhead * head_dim * elem;
    size_t stats_bytes = (size_t)seqlen * nhead * sizeof(float);

    auto &bufs = get_buffers();
    bufs.ensure_q_o(q_bytes);
    bufs.ensure_stats(stats_bytes);

    constexpr int BLK = 256;
    int q_n = seqlen * nhead * head_dim;

    // Always transpose Q (small: seqlen * nhead * head_dim)
    if (type == ZEDINFER_DTYPE_BF16) {
        transpose_seq_to_head<<<(q_n + BLK - 1) / BLK, BLK>>>(
            (__nv_bfloat16 *)bufs.q, (const __nv_bfloat16 *)q, seqlen, nhead, head_dim);
    } else {
        transpose_seq_to_head<<<(q_n + BLK - 1) / BLK, BLK>>>(
            (half *)bufs.q, (const half *)q, seqlen, nhead, head_dim);
    }

    // K/V: use directly if cuDNN accepts seq-major strides, else transpose
    const void *k_ptr, *v_ptr;
    if (cached->kv_seq_major) {
        // cuDNN reads K/V with seq-major strides — no transpose needed!
        k_ptr = k;
        v_ptr = v;
    } else {
        // cuDNN needs head-major K/V — transpose
        bufs.ensure_kv(kv_bytes);
        int kv_n = total_len * nkvhead * head_dim;
        if (type == ZEDINFER_DTYPE_BF16) {
            transpose_seq_to_head<<<(kv_n + BLK - 1) / BLK, BLK>>>(
                (__nv_bfloat16 *)bufs.k, (const __nv_bfloat16 *)k, total_len, nkvhead, head_dim);
            transpose_seq_to_head<<<(kv_n + BLK - 1) / BLK, BLK>>>(
                (__nv_bfloat16 *)bufs.v, (const __nv_bfloat16 *)v, total_len, nkvhead, head_dim);
        } else {
            transpose_seq_to_head<<<(kv_n + BLK - 1) / BLK, BLK>>>(
                (half *)bufs.k, (const half *)k, total_len, nkvhead, head_dim);
            transpose_seq_to_head<<<(kv_n + BLK - 1) / BLK, BLK>>>(
                (half *)bufs.v, (const half *)v, total_len, nkvhead, head_dim);
        }
        k_ptr = bufs.k;
        v_ptr = bufs.v;
    }

    std::unordered_map<int64_t, void *> variant_pack = {
        {1, bufs.q},
        {2, const_cast<void *>(k_ptr)},
        {3, const_cast<void *>(v_ptr)},
        {4, bufs.o},
        {5, bufs.stats}
    };

    auto &handle = get_handle();
    bool ok = cached->graph->execute(handle, variant_pack, cached->workspace).is_good();

    if (ok) {
        if (type == ZEDINFER_DTYPE_BF16) {
            transpose_head_to_seq<<<(q_n + BLK - 1) / BLK, BLK>>>(
                (__nv_bfloat16 *)output, (const __nv_bfloat16 *)bufs.o, seqlen, nhead, head_dim);
        } else {
            transpose_head_to_seq<<<(q_n + BLK - 1) / BLK, BLK>>>(
                (half *)output, (const half *)bufs.o, seqlen, nhead, head_dim);
        }
    }

    return ok;
}

} // namespace zedinfer::ops::nvidia::cudnn_flash

#endif
