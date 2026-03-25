# Test Strategy

## Test Categories

### 1. Operator Correctness Tests

**Purpose**: Verify each operator produces numerically correct output against a PyTorch reference.

**Method**: Use the existing Python bindings infrastructure (`tests/python/bindings/zedinfer_ops.cpp`) to expose operator functions to Python. Compare outputs against `torch` operations with identical inputs.

**Operators to test**:
| Operator | Reference | Tolerance (FP32) | Tolerance (BF16/FP16) |
|----------|-----------|-------------------|----------------------|
| `ops::add` | `torch.add` | exact | 1e-3 |
| `ops::embedding` | `torch.nn.Embedding` | exact | 1e-3 |
| `ops::linear` | `torch.nn.functional.linear` | 1e-5 | 1e-2 |
| `ops::rms_norm` | Manual RMS + scale | 1e-5 | 1e-2 |
| `ops::rope` | HuggingFace `apply_rotary_pos_emb` | 1e-5 | 1e-2 |
| `ops::self_attention` | `torch.nn.functional.scaled_dot_product_attention` | 1e-4 | 1e-2 |
| `ops::swiglu` | `torch.nn.SiLU(gate) * up` | 1e-5 | 1e-2 |

**Test shapes**: Multiple shapes per operator:
- Prefill: seq_len in {1, 16, 128, 512, 2048}
- Hidden sizes: {1536, 3584, 4096} (matching Qwen2-1.5B and Qwen3-8B)
- Batch dimension: {1} (current), then {1, 4, 16} (after batching)

**Both backends**: Run each test on CPU and (if available) NVIDIA. Compare CPU vs NVIDIA output as additional cross-validation.

**Files**:
- `tests/ops/test_linear.py` - Linear correctness tests
- `tests/ops/test_attention.py` - Attention correctness tests
- `tests/ops/test_elementwise.py` - Add, SwiGLU, RMS norm, RoPE
- `tests/ops/conftest.py` - Shared fixtures (random inputs, tolerance helpers)

### 2. End-to-End Generation Tests

**Purpose**: Verify the full pipeline produces expected text for deterministic configurations.

**Method**: Load a model, run generation with greedy sampling (argmax, temperature=0) on a fixed prompt, compare output token IDs against a saved reference.

**Test cases**:
```
test_e2e_qwen2_1.5b_greedy:
    model: DeepSeek-R1-Distill-Qwen-1.5B
    prompt: "What is 2+2?"
    max_tokens: 32
    sampling: argmax
    expected: [saved reference token IDs]

test_e2e_qwen3_8b_greedy:
    model: DeepSeek-R1-0528-Qwen3-8B
    prompt: "What is 2+2?"
    max_tokens: 32
    sampling: argmax
    expected: [saved reference token IDs]
```

**Reference generation**: Run once with the current engine (known-correct), save output IDs. Re-verify against HuggingFace Transformers output if possible.

**Files**:
- `tests/e2e/test_generation.cpp` - C++ e2e tests (GTest)
- `tests/e2e/reference/` - Saved reference outputs (JSON)
- `tests/e2e/generate_references.py` - Script to generate references via HuggingFace

### 3. Regression Tests

**Purpose**: Detect unintended changes in output during refactoring.

**Method**: Snapshot-based. Before each major refactor stage, capture:
1. Token-level output for 5 fixed prompts (greedy)
2. Per-operator output for 10 fixed random inputs
3. Performance baseline (prefill/decode latency)

After refactor, re-run and compare:
- Token output: must be bit-identical (same sampling)
- Operator output: within tolerance
- Performance: must not regress >5% (flag for review if >2%)

**Files**:
- `tests/regression/snapshots/` - Saved snapshots (versioned by stage)
- `tests/regression/test_regression.cpp` - Snapshot comparison

### 4. Performance Tests

**Purpose**: Track and prevent performance regressions. Measure speedups from optimizations.

**Method**: Use the existing `bench` example with standardized configurations:

```
Config A (small, decode-heavy): prefill=32, decode=256
Config B (balanced):            prefill=128, decode=128
Config C (prefill-heavy):       prefill=512, decode=64
Config D (long context):        prefill=2048, decode=128
```

Run 5 rounds each, report mean and stddev. Compare against baseline.

**Per-operator benchmarks**: Time individual operators in isolation:
- `linear`: M=1 (decode) and M=128 (prefill), various N/K
- `self_attention`: seq_len=1 and seq_len=128, various total_len
- `rms_norm`, `rope`, `swiglu`: standard shapes

**Files**:
- `tests/perf/bench_operators.cpp` - Operator-level benchmarks
- `tests/perf/bench_e2e.cpp` - End-to-end benchmarks (wraps bench example)
- `tests/perf/baselines/` - Saved baseline measurements

### 5. Stress / Concurrency Tests

**Purpose**: Validate correctness and stability under load after multi-user support is added.

**Tests** (added after scheduler implementation):
- **Multi-session correctness**: Run 10 sessions in parallel with different prompts, verify each produces correct independent output
- **KV cache pressure**: Run requests until KV blocks are exhausted, verify graceful handling (queue, preempt, or reject)
- **Long-running stability**: Run continuous random requests for 10 minutes, check for memory leaks, crashes, or degradation
- **Concurrent session reset**: Reset sessions while generation is in progress

**Files**:
- `tests/stress/test_multi_session.cpp`
- `tests/stress/test_memory_pressure.cpp`
- `tests/stress/test_long_running.cpp`

### 6. Quantization Correctness Tests

**Purpose**: Verify quantized models produce acceptable output quality.

**Method**:
1. Compare quantized model logits against FP16 reference (per-token KL divergence)
2. Compare generated text quality (BLEU/ROUGE against FP16 output, or manual inspection)
3. Perplexity measurement on standard benchmark text (WikiText-2)

**Acceptance criteria**:
- INT8: logit KL divergence < 0.01, perplexity degradation < 0.5%
- INT4 GPTQ: logit KL divergence < 0.05, perplexity degradation < 2%

**Files**:
- `tests/quant/test_int8_correctness.py` - INT8 quality tests
- `tests/quant/test_int4_correctness.py` - INT4 quality tests

## Test Infrastructure

### Test Runner

- **C++ tests**: GTest (already configured in xmake)
- **Python tests**: pytest with the `zedinfer_ops` pybind11 module
- **Performance tests**: Custom benchmark harness with JSON output

### Model Fixtures

Tests requiring model weights need access to model directories. Use environment variable:
```
ZEDINFER_TEST_MODEL_PATH=/mnt/hdd/shared/models/deepseek-ai
```

Tests that don't need full models use randomly generated weights (operator tests).

### CI Integration (Future)

```yaml
# Proposed CI pipeline stages
stages:
  - build_cpu        # xmake f -m release && xmake
  - build_gpu        # xmake f -m release --nv-gpu=y && xmake
  - test_unit        # xmake run -g test
  - test_ops_cpu     # pytest tests/ops/ --device cpu
  - test_ops_gpu     # pytest tests/ops/ --device nvidia (if GPU available)
  - test_e2e         # xmake run test-e2e (if models available)
  - test_perf        # xmake run bench ... > results.json && compare_baseline.py
```

## When to Run Each Test Type

| Event | Unit | Operator | E2E | Regression | Perf | Stress |
|-------|------|----------|-----|-----------|------|--------|
| Every commit | Yes | - | - | - | - | - |
| Before merge | Yes | Yes | Yes | Yes | - | - |
| Per-stage completion | Yes | Yes | Yes | Yes | Yes | - |
| Before release | Yes | Yes | Yes | Yes | Yes | Yes |
