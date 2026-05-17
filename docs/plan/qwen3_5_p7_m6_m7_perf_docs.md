# Qwen3.5 P7 (M6 + M7) — Perf Bench + Docs + v0.3.0 Release

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `xmake run bench` produces prefill/decode tok/s tables for 27B + 35B-A3B, both text-only and multimodal. Architecture/roadmap docs updated. v0.3.0 tagged with release notes. Optional stretch: activate MTP head as native speculative decoding draft (phase3 §A.2).

**Architecture:** Extend existing `bench` binary to accept Qwen3.5 paths. Cross-check perf vs v0.2.0 Qwen3-30B-A3B-GPTQ-Int4 numbers (handoff §3 M3 row). Doc updates to architecture.md (new hybrid section), roadmap.md, release notes.

**Tech Stack:** Existing bench tool, doc writing.

**Reference:** Design doc §10 (M6 + M7). Phase 2 perf reference handoff `docs/plan/moe_session_handoff.md`.

**Pre-condition:** P6 (M5) complete. 35B-A3B byte-exact.

---

### Task 1: Extend `bench` to accept Qwen3.5 paths

**Files:**
- Modify: `src/examples/bench.cpp` (or wherever bench binary lives)

- [ ] **Step 1: Survey current bench CLI**

Run: `find . -name 'bench*.cpp' -not -path './build*'`
Inspect.

- [ ] **Step 2: Verify it already works (no code changes needed if `Model::parse` dispatch is correct)**

```bash
xmake run bench ~/data/models/Qwen3.5-27B-GPTQ-Int4 --nvidia -p 128 -d 128 -r 3 --gpu-memory-utilization 0.5
```
Expected: outputs prefill / decode tok/s. If it errors out on a Qwen3.5-specific assumption (e.g., `head_dim=hidden/heads` assumption), trace and fix.

- [ ] **Step 3: Note: bench uses paged path, exercises full forward**

Verify bench runs N warmup + R measurement iterations, prints `prefill tok/s` and `decode tok/s`.

If bench currently doesn't pass `--gpu-memory-utilization`, add the flag plumbed through to `init_block_pool`.

- [ ] **Step 4: Commit if any fixes**

```bash
git add src/examples/bench.cpp
git commit -m "feat(bench): support Qwen3.5 dense + MoE paths"
```

---

### Task 2: Collect baseline numbers (27B + 35B-A3B, text-only)

- [ ] **Step 1: 27B text-only bench**

```bash
xmake run bench ~/data/models/Qwen3.5-27B-GPTQ-Int4 --nvidia -p 128 -d 128 -r 5 \
  --gpu-memory-utilization 0.5 2>&1 | tee /tmp/bench_27b_text.log
```

Repeat 3 times to bound noise. Compute median.

- [ ] **Step 2: 35B-A3B text-only bench (ALL_GPU)**

```bash
xmake run bench ~/data/models/Qwen3.5-35B-A3B-GPTQ-Int4 --nvidia -p 128 -d 128 -r 5 \
  --gpu-memory-utilization 0.5 2>&1 | tee /tmp/bench_35b_allgpu.log
```

(Auto-N might pick ALL_GPU or PINNED_LRU depending on free VRAM. Log decision.)

- [ ] **Step 3: 35B-A3B text-only bench (PINNED_LRU, force)**

```bash
ZEDINFER_MOE_GPU_SLOTS=64 \
  xmake run bench ~/data/models/Qwen3.5-35B-A3B-GPTQ-Int4 --nvidia -p 128 -d 128 -r 5 \
  --gpu-memory-utilization 0.5 2>&1 | tee /tmp/bench_35b_pinned.log
```

- [ ] **Step 4: Tabulate**

Build a markdown table:
```markdown
## Qwen3.5 baseline perf (A6000 48GB, util=0.5, p=128 d=128, median of 5 reps)

| Config                          | Prefill (tok/s) | Decode (tok/s) | Peak VRAM |
|---|---:|---:|---:|
| 27B (dense)                     | ...             | ...            | ...       |
| 35B-A3B ALL_GPU                 | ...             | ...            | ...       |
| 35B-A3B PINNED_LRU N=64         | ...             | ...            | ...       |

## v0.2.0 reference (Qwen3-30B-A3B-GPTQ-Int4)
| ALL_GPU prefill | 223.22  |
| PINNED_LRU N=32 prefill | 112.32 |
```

Compare: Qwen3.5 hybrid models should be in the same order of magnitude. If 5× slower → SSU kernel inefficient or per-layer overhead. Investigate.

- [ ] **Step 5: Commit numbers**

```bash
# Append the table to handoff doc
git add docs/plan/qwen3_5_session_handoff.md
git commit -m "docs(qwen3.5): M6 baseline perf numbers (27B + 35B-A3B, text)"
```

---

### Task 3: Multimodal bench (image throughput)

- [ ] **Step 1: Extend bench to accept `--image PATH` flag**

When set, bench prepends image to prompt before warmup and measurement. Use same sample.jpg.

- [ ] **Step 2: Run**

```bash
xmake run bench ~/data/models/Qwen3.5-27B-GPTQ-Int4 --nvidia \
  --image tests/fixtures/sample.jpg -p 128 -d 128 -r 5 --gpu-memory-utilization 0.5
```

Tabulate: prefill including vision tower forward + LM prefill, decode same as text.

- [ ] **Step 3: Commit**

```bash
git add src/examples/bench.cpp
git commit -m "feat(bench): --image flag for multimodal throughput measurement"
```

---

### Task 4: SSU prefill optimization (if needed)

If prefill tok/s is significantly lower than 27B-dense baseline (e.g., 5× slower), the SSU varlen path may be unoptimized.

- [ ] **Step 1: Profile**

Use `nvprof` or `nsys profile`:
```bash
nsys profile --capture-range=cudaProfilerApi -o /tmp/qwen3_5_prefill \
  xmake run bench ~/data/models/Qwen3.5-27B-GPTQ-Int4 --nvidia -p 128 -d 1 -r 1
```

Open profile; look for top kernels. SSU should dominate prefill time. Check launch params (grid, block, smem).

- [ ] **Step 2: Try alternative algorithms**

In `ssu_wrapper.cu`, change `SSUAlgorithm::kAuto` to `kHorizontal` or `kVertical` (SM100) and re-bench. Document the choice that wins.

- [ ] **Step 3: If SSDCombined (chunked scan) becomes worth pursuing**

Inspect `third_party/flashinfer/flashinfer/mamba/ssd_kernel.py` to see if a CUDA path exists alongside Triton. If yes, add a C++ wrapper. If no (only Triton), defer until upstream FlashInfer adds CUDA — note in handoff.

- [ ] **Step 4: Commit perf change**

```bash
git add src/backend/ops/mamba/nvidia/ssu_wrapper.cu
git commit -m "perf(mamba): switch SSU algo to <chosen> for prefill"
```

---

### Task 5: Architecture doc update (`docs/architecture.md`)

- [ ] **Step 1: Add hybrid SSM+attention section**

After the existing "Operator Backends" table, add a new section:
```markdown
## Hybrid SSM + Softmax Attention Path (Qwen3.5)

For Qwen3.5 models (`qwen3_5`, `qwen3_5_moe`), 75% of layers run Mamba2-style
linear attention (FlashInfer SSU + causal_conv1d), 25% run gated softmax
attention (paged KV via FlashInfer + 3D MRoPE + attn_output_gate).

### Components

- `Qwen3_5Model` / `Qwen3_5MoeModel`: own SSMStatePool, VisionTower, ExpertPool (MoE only)
- `HybridForwardConfig`: subclass of `ModelForwardConfig` with `layer_kinds`, `linear_attn`, `mrope`, `attn_output_gate`, `ssm_pool`
- `SSMStatePool`: engine-level slot pool, FlashInfer-native layout `[slots, layers, Hv, Dv, d_state]` + `[slots, layers, K-1, qkv_dim]` conv state
- `hybrid_transformer_forward`: outer loop dispatching on `layer_kinds[L]` to five per-layer-kind functions

### Per-layer dispatch (A2 layout)

- `forward_linear_attn_layer`: BF16 in_proj_qkv/z/a/b → causal_conv1d → SSU → norm + gate → out_proj
- `forward_full_attn_layer`: q_proj doubled → split q/gate → q_norm/k_norm → 3D mrope partial → paged KV → paged attn → attn_output_gate → o_proj
- `forward_dense_mlp`: GPTQ Int4 gate/up/swiglu/down
- `forward_moe_mlp`: delegates to v0.2.0 `moe_layer_forward` (ExpertPool + sliding-window prefetch)

### Vision pre-step

`Qwen3_5Model::forward` calls `VisionTower::forward(patches)` before the main loop
if `req.has_pending_images()`. Image embeddings are scattered into the input
embedding sequence at `<|image_pad|>` (id 248056) positions via `ops::scatter_image_embeds`.
```

- [ ] **Step 2: Update file table**

Add rows for new ops and modules.

- [ ] **Step 3: Commit**

```bash
git add docs/architecture.md
git commit -m "docs(architecture): document Qwen3.5 hybrid SSM+attention path"
```

---

### Task 6: Roadmap update (`docs/roadmap.md`)

- [ ] **Step 1: Mark items complete**

In the "Upcoming Work" / "Priority 5: Heterogeneous CPU/GPU" / "Priority 6: MoE Expert Offloading" sections, add lines noting that Qwen3.5 hybrid SSM extends these foundations.

Add a new "Completed Features" entry for Qwen3.5.

- [ ] **Step 2: Add new Priority N row**

If MTP-as-spec-decoding is the next direction (was Task 8 stretch in this plan): note as Priority N.

- [ ] **Step 3: Commit**

```bash
git add docs/roadmap.md
git commit -m "docs(roadmap): Qwen3.5 hybrid SSM+attention support shipped in v0.3.0"
```

---

### Task 7: v0.3.0 release prep

- [ ] **Step 1: Bump version string**

Run: `grep -rn '0.2.0\|v0.2.0' --include='*.lua' --include='*.cpp' --include='*.hpp'`

In each location pointing to v0.2.0, bump to v0.3.0.

- [ ] **Step 2: Write release notes**

Create `docs/release/v0.3.0.md`:
```markdown
# zedinfer v0.3.0

## Highlights

- **Qwen3.5 hybrid SSM + softmax attention support** for both 27B (dense) and 35B-A3B (MoE)
- **Vision tower** (Qwen2.5-VL family ViT): native image input via OpenAI Vision API
- **Token-level byte-exact alignment** with HuggingFace transformers for both models
- All v0.2.0 paths (Qwen2/Qwen3/Qwen3-MoE) unchanged; zero regression

## New components

- `Qwen3_5Model` / `Qwen3_5MoeModel`: hybrid-architecture model classes
- `SSMStatePool`: engine-level slot pool for Mamba2 SSM state + conv state
- `VisionTower`: standalone 27-layer ViT forward
- `MultiModalProcessor`: stb_image-based image preprocessing
- `ChatTemplate::load_jinja()`: minja-backed Jinja2 evaluator (Qwen3.5 chat template)

## New ops

- `ops::mamba::ssu` (FlashInfer Mamba selective_state_update wrapper)
- `ops::mamba::causal_conv1d` (depthwise kernel=4 + state)
- `ops::mrope_3d` (3D MRoPE interleaved + partial rotary)
- `ops::attn_output_gate` (post-attention sigmoid gate)
- `ops::scatter_image_embeds` (inject vision embeddings)
- `ops::vision_attention` (FlashInfer ragged prefill wrapper, ViT)
- `ops::layer_norm_bias`, `ops::gelu_tanh` (ViT path)

## HTTP API

- OpenAI Vision-compatible `messages[*].content` content arrays
- Web UI image upload + drag-drop

## Performance (A6000 48 GB, --gpu-memory-utilization 0.5, p=128 d=128)

| Model | Prefill (tok/s) | Decode (tok/s) |
|---|---:|---:|
| Qwen3.5-27B GPTQ Int4 | ... | ... |
| Qwen3.5-35B-A3B GPTQ Int4 ALL_GPU | ... | ... |
| Qwen3.5-35B-A3B GPTQ Int4 PINNED_LRU N=64 | ... | ... |

## Caveats

- MTP head weights loaded but inference path not activated (deferred; tracked as future work)
- Only `data:` URIs supported for image_url; remote URL fetch unsupported
- max_concurrent=1 default; tune via runtime config for multi-user scenarios

## Build

Same as v0.2.0; requires `--flashinfer=y`:
```
xmake f -m release --nv-gpu=y --onednn=y --flashinfer=y
xmake build
```

Submodules:
```
git submodule update --init --recursive
```
(now includes `third_party/minja` and `third_party/stb`)
```

- [ ] **Step 3: Commit**

```bash
git add docs/release/v0.3.0.md xmake.lua  # if version bumped
git commit -m "docs(release): v0.3.0 release notes"
```

---

### Task 8: Final regression run

- [ ] **Step 1: Run full test suite**

```bash
xmake run test-blockpool
xmake run test-prefixcache
xmake run test-sampler
xmake run test-chattemplate
xmake run test-tensor
ZEDINFER_TEST_MODEL_PATH=/home/tianyux/data/models/Qwen3-30B-A3B-GPTQ-Int4 xmake run test-loader
ZEDINFER_TEST_MODEL_PATH=/home/tianyux/data/models/Qwen3-30B-A3B-GPTQ-Int4 xmake run test-tokenizer

# Qwen3.5 unit tests
xmake run test-flashinfer-ssu-link
xmake run test-qwen3-5-config-parse
xmake run test-qwen3-5-weight-name-map
xmake run test-ssm-state-pool
xmake run test-hybrid-forward-config
xmake run test-qwen3-5-chat-template
xmake run test-ops-mamba-ssu
xmake run test-ops-causal-conv1d
xmake run test-ops-mrope-3d
xmake run test-ops-attn-output-gate
xmake run test-ops-layer-norm-bias
xmake run test-ops-gelu-tanh
xmake run test-ops-vision-attention
xmake run test-ops-scatter-image-embeds
xmake run test-multimodal-processor

# Qwen3.5 e2e
python3 tests/e2e/test_qwen3_5_hf_alignment.py
python3 tests/e2e/test_qwen3_5_hf_alignment.py --model-path ~/data/models/Qwen3.5-35B-A3B-GPTQ-Int4 --refs-dir tests/fixtures/qwen3_5_moe_hf_reference

# Existing pings (regression)
xmake run ping ~/data/models/Qwen3-30B-A3B-GPTQ-Int4 --nvidia
xmake run ping ~/data/models/DeepSeek-R1-Distill-Qwen-1.5B --nvidia
```

Capture results in `/tmp/v0.3.0_final.log`. All should pass.

- [ ] **Step 2: Tag v0.3.0**

```bash
git tag -a v0.3.0 -m "v0.3.0: Qwen3.5 hybrid SSM + multimodal support"
```

- [ ] **Step 3: Push (gated on user confirmation)**

Do not push automatically. Ask user before:
```bash
# git push origin feat/qwen3.5 --tags
```

---

### Task 9 (Stretch): MTP head as native speculative decoding draft

This activates the loaded-but-unused MTP head. Only attempt if M7 schedule has slack.

- [ ] **Step 1: Decision**

If 1-2 weeks remain after M7 main work, proceed. Otherwise defer to a separate plan (phase3 §A.2).

- [ ] **Step 2: Design**

MTP layer flow:
- After main LM produces `h_t`, MTP head fc'd with `concat(rmsnorm(e_{t+1}), rmsnorm(h_t))` → 1 transformer layer → lm_head → token_{t+2} prediction.

In speculative decoding mode:
- LM forward produces `h_t` and `logits_t+1` (token t+1)
- MTP forward (single layer) produces `logits_t+2` (token t+2 candidate)
- LM next step verifies (t+2) candidate by running it through the main path; if accepted, save 1 forward.

- [ ] **Step 3: Implementation**

Add `Qwen3_5Model::forward_mtp_head(h_t, next_token_id, exec)` returning logits.

In scheduler, when spec decoding enabled:
```
predict next 2 tokens via main + MTP
batch verify second
accept if argmax matches; else fall back
```

This requires sampler / scheduler / KV rollback work — substantial. Treat as M7 stretch.

- [ ] **Step 4: Commit + perf compare**

If activated, re-bench decode and compare. Expected: 1.3-1.5× decode speedup at high MTP acceptance rate.

```bash
git add ...
git commit -m "feat(qwen3.5): MTP head activated as native speculative decoding draft"
```

---

### Task 10: M7 done; final retro

```markdown
## M6 + M7 — Perf + Docs + Release (complete)

### Numbers
- 27B prefill / decode: ... / ... tok/s
- 35B-A3B ALL_GPU: ... / ...
- 35B-A3B PINNED_LRU N=64: ... / ...
- Multimodal (27B + sample.jpg): ... / ...

### Docs updated
- architecture.md: new hybrid section
- roadmap.md: Qwen3.5 completed
- docs/release/v0.3.0.md: release notes

### Optional MTP work: <activated | deferred>

### v0.3.0 tagged: <yes | no>
```

```bash
git add docs/plan/qwen3_5_session_handoff.md
git commit -m "docs(qwen3.5): M6+M7 retro and v0.3.0 release"
```

---

## M6 + M7 Done. ~10 tasks. Est. 2 weeks (3 if MTP stretch).

v0.3.0 shipped. Qwen3.5 multimodal hybrid SSM + softmax attention end-to-end on consumer GPU.
