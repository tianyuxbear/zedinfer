#ifdef USE_ONEDNN

#include "backend/ops/linear/cpu/linear_onednn.hpp"
#include "utils/types.hpp"
#include "zedinfer.h"

#include <cstring>
#include <dnnl.hpp>
#include <mutex>
#include <unordered_map>

namespace zedinfer::ops::cpu::onednn {

// Global oneDNN engine and stream (created once, thread-safe)
static dnnl::engine &engine() {
    static dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    return eng;
}

static dnnl::stream &stream() {
    static dnnl::stream strm(engine());
    return strm;
}

// Map zedinfer dtype to oneDNN data type
static dnnl::memory::data_type to_dnnl_dtype(zedinferDataType_t dtype) {
    switch (dtype) {
    case ZEDINFER_DTYPE_F32:
        return dnnl::memory::data_type::f32;
    case ZEDINFER_DTYPE_BF16:
        return dnnl::memory::data_type::bf16;
    case ZEDINFER_DTYPE_F16:
        return dnnl::memory::data_type::f16;
    default:
        throw std::runtime_error("Unsupported dtype for oneDNN linear");
    }
}

// Primitive cache key
struct PrimKey {
    size_t M, N, K;
    dnnl::memory::data_type dt;
    bool has_bias;

    bool operator==(const PrimKey &o) const {
        return M == o.M && N == o.N && K == o.K && dt == o.dt && has_bias == o.has_bias;
    }
};

struct PrimKeyHash {
    size_t operator()(const PrimKey &k) const {
        size_t h = std::hash<size_t>()(k.M);
        h ^= std::hash<size_t>()(k.N) << 1;
        h ^= std::hash<size_t>()(k.K) << 2;
        h ^= std::hash<int>()(static_cast<int>(k.dt)) << 3;
        h ^= std::hash<bool>()(k.has_bias) << 4;
        return h;
    }
};

// Cached primitive and memory descriptors
struct CachedMatmul {
    dnnl::matmul prim;
    dnnl::memory::desc a_md, b_md, c_md, bias_md;
};

static std::unordered_map<PrimKey, CachedMatmul, PrimKeyHash> cache;
static std::mutex cache_mutex;

static CachedMatmul &get_or_create(size_t M, size_t N, size_t K,
                                    dnnl::memory::data_type dt, bool has_bias) {
    PrimKey key{M, N, K, dt, has_bias};

    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }

    auto &eng = engine();

    // A[M,K] row-major, B stored as [N,K] row-major but we need B^T
    // oneDNN matmul: C = A * B, so we describe B as [K,N] with transposed strides
    // B is [N,K] in memory (row-major), which is [K,N] column-major = transposed
    auto a_md = dnnl::memory::desc({static_cast<dnnl::memory::dim>(M),
                                     static_cast<dnnl::memory::dim>(K)},
                                    dt, dnnl::memory::format_tag::ab);

    // B is stored as [N,K] row-major. For C = A * B^T, we tell oneDNN
    // that B has shape [K,N] with strides that read [N,K] transposed.
    // Stride for row (K dim) = 1, stride for col (N dim) = K
    auto b_md = dnnl::memory::desc({static_cast<dnnl::memory::dim>(K),
                                     static_cast<dnnl::memory::dim>(N)},
                                    dt, {static_cast<dnnl::memory::dim>(1),
                                         static_cast<dnnl::memory::dim>(K)});

    // Output always FP32 for accumulation accuracy when input is BF16/FP16
    auto c_dt = (dt == dnnl::memory::data_type::f32) ? dnnl::memory::data_type::f32 : dt;
    auto c_md = dnnl::memory::desc({static_cast<dnnl::memory::dim>(M),
                                     static_cast<dnnl::memory::dim>(N)},
                                    c_dt, dnnl::memory::format_tag::ab);

    CachedMatmul cached;
    cached.a_md = a_md;
    cached.b_md = b_md;
    cached.c_md = c_md;

    if (has_bias) {
        cached.bias_md = dnnl::memory::desc({1, static_cast<dnnl::memory::dim>(N)},
                                             dt, dnnl::memory::format_tag::ab);
        auto pd = dnnl::matmul::primitive_desc(eng, a_md, b_md, cached.bias_md, c_md);
        cached.prim = dnnl::matmul(pd);
    } else {
        auto pd = dnnl::matmul::primitive_desc(eng, a_md, b_md, c_md);
        cached.prim = dnnl::matmul(pd);
    }

    auto [inserted, _] = cache.emplace(key, std::move(cached));
    return inserted->second;
}

void linear(std::byte *output, const std::byte *input, const std::byte *weight,
            const std::byte *bias, zedinferDataType_t type, size_t M, size_t N, size_t K) {

    auto dt = to_dnnl_dtype(type);
    bool has_bias = (bias != nullptr);

    auto &cached = get_or_create(M, N, K, dt, has_bias);
    auto &eng = engine();
    auto &strm = stream();

    auto a_mem = dnnl::memory(cached.a_md, eng, const_cast<std::byte *>(input));
    auto b_mem = dnnl::memory(cached.b_md, eng, const_cast<std::byte *>(weight));
    auto c_mem = dnnl::memory(cached.c_md, eng, output);

    if (has_bias) {
        auto bias_mem = dnnl::memory(cached.bias_md, eng, const_cast<std::byte *>(bias));
        cached.prim.execute(strm, {
            {DNNL_ARG_SRC, a_mem},
            {DNNL_ARG_WEIGHTS, b_mem},
            {DNNL_ARG_BIAS, bias_mem},
            {DNNL_ARG_DST, c_mem}
        });
    } else {
        cached.prim.execute(strm, {
            {DNNL_ARG_SRC, a_mem},
            {DNNL_ARG_WEIGHTS, b_mem},
            {DNNL_ARG_DST, c_mem}
        });
    }

    strm.wait();
}

} // namespace zedinfer::ops::cpu::onednn

#endif // USE_ONEDNN
