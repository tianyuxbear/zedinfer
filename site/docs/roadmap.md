# Roadmap

---

## Completed (v0.1.0)

| Feature | Description |
|---------|-------------|
| Direct model forward | Single shared forward loop, no graph execution overhead |
| cuBLAS / oneDNN linear | Auto-tuned GPU GEMM, runtime ISA dispatch on CPU |
| Paged KV cache | Block pool with ref counting, LRU eviction |
| Paged attention kernels | Decode (single + batched) and prefill |
| Continuous batching | Decode-first scheduling with chunked prefill |
| Prefix caching | Cross-request block sharing via content hashing |
| HTTP API + Web UI | OpenAI-compatible API with SSE streaming |
| Docker deployment | Multi-stage build, portable binary, closed-source distribution |
| CLI version flag | `--version` with git hash and build date injection |

---

## Upcoming

### Priority 1: FlashInfer Integration

Replace custom paged attention kernels with FlashInfer's optimized kernels. Expected **3-5x decode speedup, 2-4x prefill speedup**.

Approach: AOT (Ahead-Of-Time) kernel generation -- Python codegen runs offline, compiled .cu files link as regular CUDA code. No Python at runtime.

### Priority 2: CUDA Graph

Capture decode forward pass as CUDA graph, replay with single kernel launch. Saves ~2ms/step in kernel launch overhead. Phase 1 (pre-allocated decode buffers) is complete.

### Priority 3: INT8 Quantization

2x model memory reduction via per-channel symmetric quantization. Enables larger batch sizes or bigger models on the same hardware.

### Priority 4: INT4 Quantization (GPTQ/AWQ)

4x model memory reduction. Run 30B+ parameter models on 24GB GPUs.

### Priority 5: Heterogeneous CPU/GPU Inference

Mixed device execution with per-layer device placement, pinned host memory, and async prefetch with compute overlap.

### Priority 6: MoE Expert Offloading

Run large Mixture-of-Experts models (e.g., Qwen-30B-A3B) on consumer GPUs by dynamically loading active experts.
