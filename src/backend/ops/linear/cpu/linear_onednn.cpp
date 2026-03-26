#ifdef USE_ONEDNN

#include "backend/ops/linear/cpu/linear_onednn.hpp"
#include "utils/types.hpp"
#include "zedinfer.h"

#include <cstring>
#include <dnnl.hpp>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace zedinfer::ops::cpu::onednn {

static dnnl::engine& engine() {
    static dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    return eng;
}

static dnnl::stream& stream() {
    static dnnl::stream strm(engine());
    return strm;
}

// Primitive cache key
struct PrimKey {
    size_t M, N, K;
    dnnl::memory::data_type dt;
    bool has_bias;

    bool operator==(const PrimKey& o) const {
        return M == o.M && N == o.N && K == o.K && dt == o.dt && has_bias == o.has_bias;
    }
};

struct PrimKeyHash {
    size_t operator()(const PrimKey& k) const {
        size_t h = std::hash<size_t>()(k.M);
        h ^= std::hash<size_t>()(k.N) << 1;
        h ^= std::hash<size_t>()(k.K) << 2;
        h ^= std::hash<int>()(static_cast<int>(k.dt)) << 3;
        h ^= std::hash<bool>()(k.has_bias) << 4;
        return h;
    }
};

struct CachedMatmul {
    dnnl::matmul prim;
    dnnl::memory::desc a_md, b_md, c_md, bias_md;
};

static std::unordered_map<PrimKey, CachedMatmul, PrimKeyHash> cache;
static std::mutex cache_mutex;

// Try to create a matmul primitive. Returns false if the dtype combo is unsupported.
static bool try_create(CachedMatmul& out, size_t M, size_t N, size_t K, dnnl::memory::data_type dt, bool has_bias) {
    auto& eng = engine();

    auto a_md = dnnl::memory::desc({static_cast<dnnl::memory::dim>(M), static_cast<dnnl::memory::dim>(K)}, dt,
                                   dnnl::memory::format_tag::ab);

    // B stored as [N,K] row-major -> describe as [K,N] transposed
    auto b_md = dnnl::memory::desc({static_cast<dnnl::memory::dim>(K), static_cast<dnnl::memory::dim>(N)}, dt,
                                   {static_cast<dnnl::memory::dim>(1), static_cast<dnnl::memory::dim>(K)});

    auto c_md = dnnl::memory::desc({static_cast<dnnl::memory::dim>(M), static_cast<dnnl::memory::dim>(N)}, dt,
                                   dnnl::memory::format_tag::ab);

    out.a_md = a_md;
    out.b_md = b_md;
    out.c_md = c_md;

    try {
        if (has_bias) {
            out.bias_md = dnnl::memory::desc({1, static_cast<dnnl::memory::dim>(N)}, dt, dnnl::memory::format_tag::ab);
            auto pd = dnnl::matmul::primitive_desc(eng, a_md, b_md, out.bias_md, c_md);
            out.prim = dnnl::matmul(pd);
        } else {
            auto pd = dnnl::matmul::primitive_desc(eng, a_md, b_md, c_md);
            out.prim = dnnl::matmul(pd);
        }
        return true;
    } catch (const dnnl::error&) { return false; }
}

static CachedMatmul& get_or_create(size_t M, size_t N, size_t K, dnnl::memory::data_type dt, bool has_bias) {
    PrimKey key{M, N, K, dt, has_bias};

    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }

    CachedMatmul cached;
    if (!try_create(cached, M, N, K, dt, has_bias)) {
        throw dnnl::error(dnnl_unimplemented, "oneDNN does not support this dtype on this CPU");
    }

    auto [inserted, _] = cache.emplace(key, std::move(cached));
    return inserted->second;
}

static void exec_matmul(CachedMatmul& cached, std::byte* output, const std::byte* input, const std::byte* weight,
                        const std::byte* bias) {
    auto& eng = engine();
    auto& strm = stream();

    auto a_mem = dnnl::memory(cached.a_md, eng, const_cast<std::byte*>(input));
    auto b_mem = dnnl::memory(cached.b_md, eng, const_cast<std::byte*>(weight));
    auto c_mem = dnnl::memory(cached.c_md, eng, output);

    if (bias) {
        auto bias_mem = dnnl::memory(cached.bias_md, eng, const_cast<std::byte*>(bias));
        cached.prim.execute(
            strm, {{DNNL_ARG_SRC, a_mem}, {DNNL_ARG_WEIGHTS, b_mem}, {DNNL_ARG_BIAS, bias_mem}, {DNNL_ARG_DST, c_mem}});
    } else {
        cached.prim.execute(strm, {{DNNL_ARG_SRC, a_mem}, {DNNL_ARG_WEIGHTS, b_mem}, {DNNL_ARG_DST, c_mem}});
    }
    strm.wait();
}

// Fallback for unsupported dtypes (FP16/BF16 on AVX2):
// convert to FP32, run FP32 matmul via oneDNN, convert back.
static void linear_via_fp32(std::byte* output, const std::byte* input, const std::byte* weight, const std::byte* bias,
                            zedinferDataType_t type, size_t M, size_t N, size_t K) {
    // Helpers for type-dispatched conversion
    auto to_f32 = [&](float* dst, const std::byte* src, size_t count) {
        if (type == ZEDINFER_DTYPE_F16) {
            zedinfer::utils::fp16_to_fp32_batch_f16c(dst, reinterpret_cast<const zedinfer::fp16_t*>(src), count);
        } else { // BF16
            zedinfer::utils::bf16_to_fp32_batch(dst, reinterpret_cast<const zedinfer::bf16_t*>(src), count);
        }
    };
    auto from_f32 = [&](std::byte* dst, const float* src, size_t count) {
        if (type == ZEDINFER_DTYPE_F16) {
            zedinfer::utils::fp32_to_fp16_batch_f16c(reinterpret_cast<zedinfer::fp16_t*>(dst), src, count);
        } else { // BF16
            zedinfer::utils::fp32_to_bf16_batch(reinterpret_cast<zedinfer::bf16_t*>(dst), src, count);
        }
    };

    std::vector<float> in_f32(M * K);
    std::vector<float> w_f32(N * K);
    to_f32(in_f32.data(), input, M * K);
    to_f32(w_f32.data(), weight, N * K);

    std::vector<float> bias_f32;
    const std::byte* bias_ptr = nullptr;
    if (bias) {
        bias_f32.resize(N);
        to_f32(bias_f32.data(), bias, N);
        bias_ptr = reinterpret_cast<const std::byte*>(bias_f32.data());
    }

    std::vector<float> out_f32(M * N);
    auto& cached = get_or_create(M, N, K, dnnl::memory::data_type::f32, bias != nullptr);
    exec_matmul(cached, reinterpret_cast<std::byte*>(out_f32.data()), reinterpret_cast<const std::byte*>(in_f32.data()),
                reinterpret_cast<const std::byte*>(w_f32.data()), bias_ptr);

    from_f32(output, out_f32.data(), M * N);
}

void linear(std::byte* output, const std::byte* input, const std::byte* weight, const std::byte* bias,
            zedinferDataType_t type, size_t M, size_t N, size_t K) {
    dnnl::memory::data_type dt;
    switch (type) {
        case ZEDINFER_DTYPE_F32:
            dt = dnnl::memory::data_type::f32;
            break;
        case ZEDINFER_DTYPE_BF16:
            dt = dnnl::memory::data_type::bf16;
            break;
        case ZEDINFER_DTYPE_F16:
            dt = dnnl::memory::data_type::f16;
            break;
        default:
            throw std::runtime_error("Unsupported dtype for oneDNN linear");
    }

    bool has_bias = (bias != nullptr);

    // Try native dtype first. If unsupported (e.g., FP16 on AVX2), fall back.
    PrimKey key{M, N, K, dt, has_bias};
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        auto it = cache.find(key);
        if (it != cache.end()) {
            exec_matmul(it->second, output, input, weight, bias);
            return;
        }
    }

    // Not cached — try to create
    CachedMatmul cached;
    if (try_create(cached, M, N, K, dt, has_bias)) {
        std::lock_guard<std::mutex> lock(cache_mutex);
        auto [inserted, _] = cache.emplace(key, std::move(cached));
        exec_matmul(inserted->second, output, input, weight, bias);
        return;
    }

    // Native dtype unsupported (e.g., FP16/BF16 on AVX2) — fallback via FP32
    if (type == ZEDINFER_DTYPE_F16 || type == ZEDINFER_DTYPE_BF16) {
        linear_via_fp32(output, input, weight, bias, type, M, N, K);
        return;
    }

    throw std::runtime_error("oneDNN: unsupported dtype combination on this CPU");
}

} // namespace zedinfer::ops::cpu::onednn

#endif // USE_ONEDNN
