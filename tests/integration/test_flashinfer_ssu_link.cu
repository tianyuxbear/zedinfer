// Minimal compile + link probe for FlashInfer Mamba selective_state_update
// MTP path (P1-T4 GO/NO-GO).
//
// Goal: verify the template instantiation
//   (bf16 input, bf16 weight, f32 A, bf16 state, i32 stateIndex, void state_scale)
// is available at link time. Runtime correctness is NOT tested here.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <iostream>

// FlashInfer's invoke_selective_state_update_mtp.cuh expects four `constexpr int`
// globals (DIM, DSTATE, NTOKENS_MTP, PHILOX_ROUNDS) and two type aliases
// (cuSeqlensIndex_t, numAcceptedIndex_t) — the jinja-generated
// selective_state_update_config.inc supplies them in the upstream build. Mirror
// that contract here. Using `#define` for the ints would collide with template
// parameter names such as `template <typename T, int DIM>` inside the Mamba
// headers.
constexpr int DIM = 128;
constexpr int DSTATE = 128;
constexpr int NTOKENS_MTP = 1;
constexpr int PHILOX_ROUNDS = 0;
using cuSeqlensIndex_t = int32_t;
using numAcceptedIndex_t = int32_t;

// selective_state_update.cuh declares SelectiveStateMTPParams / SSUAlgorithm and
// pulls in invoke_selective_state_update_mtp.cuh at the bottom. This is the same
// include the upstream csrc/selective_state_update.cu uses.
#include <flashinfer/mamba/selective_state_update.cuh>

int main() {
    // SelectiveStateMTPParams is in flashinfer::mamba::mtp; SSUAlgorithm is in
    // flashinfer::mamba. Qualify explicitly to avoid ambiguity.
    flashinfer::mamba::mtp::SelectiveStateMTPParams params{};

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    try {
        flashinfer::mamba::mtp::invokeSelectiveStateUpdateMTP<__nv_bfloat16, // input_t
                                                              __nv_bfloat16, // weight_t
                                                              float,         // matrixA_t
                                                              __nv_bfloat16, // state_t
                                                              int32_t,       // stateIndex_t
                                                              void           // state_scale_t
                                                              >(params, flashinfer::mamba::SSUAlgorithm::kSimple,
                                                                stream);
    } catch (...) {
        // Runtime errors expected with zero-filled params; only care about link.
    }
    cudaStreamDestroy(stream);
    std::cout << "[ok] SSU MTP template (bf16 input/state, f32 A) linked successfully\n";
    return 0;
}
