# Developer Guide

---

## Build

```bash
# Development (full optimization for local CPU)
xmake f -m release --nv-gpu=y --onednn=y
xmake

# Distribution / Docker (portable AVX2 baseline)
xmake f -m release --nv-gpu=y --onednn=y --portable=y
xmake

# Debug
xmake f -m debug --nv-gpu=y --onednn=y
xmake
```

---

## Test

### C++ Unit Tests

```bash
# Build all test targets
xmake build -g test

# Run individual test suites
xmake run test-blockpool       # KV cache block pool + ref counting
xmake run test-prefixcache     # Prefix caching hash match
xmake run test-sampler         # Argmax + general sampler
xmake run test-chattemplate    # Chat template formatting
xmake run test-tensor          # Tensor ops (shape, slice, permute, device transfer)
xmake run test-memorypool      # Memory pool allocation
xmake run test-storage         # Storage layer

# Tests requiring model files (set env var)
ZEDINFER_TEST_MODEL_PATH=/path/to/model xmake run test-tokenizer
ZEDINFER_TEST_MODEL_PATH=/path/to/model xmake run test-loader
```

### Operator Correctness Tests (Python)

```bash
# Build with Python bindings
xmake f -m release --nv-gpu=y --onednn=y --pytest=y
xmake build zedinfer_ops

# Install Python package
uv pip install -e python/

# Run operator tests (compare against PyTorch)
uv run python/tests/ops/add.py
uv run python/tests/ops/linear.py
uv run python/tests/ops/self_attention.py
uv run python/tests/ops/rms_norm.py
uv run python/tests/ops/rope.py
uv run python/tests/ops/swiglu.py
uv run python/tests/ops/embedding.py
uv run python/tests/ops/argmax.py
```

---

## Lint

CI runs clang-format and ruff on every push/PR. Run locally before pushing:

### C++ (clang-format)

```bash
# Check (dry run, no changes)
find include/ src/ examples/ tests/ \
  -name '*.cpp' -o -name '*.hpp' -o -name '*.c' -o -name '*.h' -o -name '*.cu' -o -name '*.cuh' \
  | xargs clang-format --dry-run --Werror

# Auto-fix
find include/ src/ examples/ tests/ \
  -name '*.cpp' -o -name '*.hpp' -o -name '*.c' -o -name '*.h' -o -name '*.cu' -o -name '*.cuh' \
  | xargs clang-format -i
```

Config: `.clang-format` (LLVM-based, 4-space indent, K&R braces).

### Python (ruff)

```bash
# Check
ruff check python/ --config python/pyproject.toml

# Auto-fix
ruff check python/ --config python/pyproject.toml --fix
```

Config: `python/pyproject.toml` `[tool.ruff]` section.

---

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/):

```
<type>: <description>

[optional body]
```

### Types

| Type | When |
|------|------|
| `feat` | New feature |
| `fix` | Bug fix |
| `refactor` | Code restructuring, no behavior change |
| `perf` | Performance improvement |
| `doc` | Documentation only |
| `build` | Build system or dependencies |
| `test` | Adding or updating tests |
| `chore` | Maintenance tasks (CI, scripts, etc.) |

### Examples

```
feat: add INT8 quantized linear dispatch for GPU
fix: prevent illegal memory access on GPUs without HMM
refactor: unify K/V block tables for FlashInfer compatibility
perf: precompute physical addresses in shared memory for paged decode
doc: add Docker deployment guide
build: add --portable option for distribution builds
test: add scheduler admission control test
chore: configure clang-format CI check
```

### Rules

- Use imperative mood ("add", not "added" or "adds")
- Lowercase after the type prefix
- No period at the end
- Keep the first line under 72 characters
- Use the body for "why", not "what" (the diff shows what)

---

## Branching

| Branch | Purpose |
|--------|---------|
| `master` | Stable releases, tagged (`v0.1.0`, `v0.2.0`, ...) |
| `dev` | Active development |
| `feature/*` | Feature branches (merge into `dev`) |

---

## Adding a New Operator

1. CPU implementation: `src/backend/ops/<op_name>/cpu/<op_name>_cpu.cpp`
2. GPU implementation: `src/backend/ops/<op_name>/nvidia/<op_name>_nvidia.cu`
3. Dispatch wrapper: `src/backend/ops/<op_name>/op.cpp`
4. Header: `include/backend/ops/<op_name>/<op_name>.hpp`
5. Register in `xmake/backend.lua` (ops target picks up `src/backend/ops/*/*.cpp` automatically; GPU files picked up by `ops-nvidia`)
6. Python test: `tests/python/test_<op_name>.py` (compare against PyTorch reference)

---

## Adding a New Model

1. Add model config parsing in `src/frontend/loader/`
2. If the model uses the standard transformer architecture, it works with the existing `transformer_forward.cpp` — just set the right `ModelForwardConfig` flags (bias, Q/K norm, etc.)
3. Add chat template in `src/zedinfer/chat_template.cpp`
4. Test: `xmake run ping /path/to/model --nvidia`

---

## Release Checklist

1. Ensure all tests pass
2. Update `ZEDINFER_VERSION` in `include/zedinfer/version.hpp`
3. Update `version` in `python/pyproject.toml`
4. Commit: `chore: bump version to x.y.z`
5. Tag: `git tag -a vx.y.z -m "vx.y.z: <summary>"`
6. Push: `git push origin vx.y.z`
7. Build Docker image: `docker build --build-arg GIT_HASH=$(git rev-parse --short HEAD) -t zedinfer:x.y.z .`
8. Push image: `docker push yourorg/zedinfer:x.y.z`
9. Create GitHub release with release notes

---

## Key Files

See [CLAUDE.md](CLAUDE.md) for architecture overview and key file reference.
