# Qwen3.5 P6 (M5) — 35B-A3B MoE Path Activation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Qwen3.5-35B-A3B-GPTQ-Int4 runs text + multimodal Q&A end-to-end, with token-level byte-exact alignment to HF for greedy generation. ExpertPool (256 experts) cooperates with SSMStatePool (hybrid layers). All M4 HTTP/Web UI functions reach 35B-A3B.

**Architecture:** Reuse Phase 2 ExpertPool (256 experts vs prior 128). `forward_moe_mlp` was delegated to `moe_layer_forward` in M1; verify it routes correctly for Qwen3.5 MoE config including the shared_expert + shared_expert_gate (Qwen3-30B-A3B lacked these, so the path is untested). Run M2's alignment harness against 35B-A3B prompts.

**Tech Stack:** Same as previous milestones; ExpertPool from v0.2.0; PINNED_LRU strategy when `ZEDINFER_MOE_GPU_SLOTS=N < 256`.

**Reference:** Design doc `docs/plan/qwen3_5_support.md` §3.1 / §6.4. Phase 2 `docs/plan/heterogeneous_moe.md` for ExpertPool internals.

**Pre-condition:** P5 (M4) complete. 27B Vision HTTP works.

---

### Task 1: Verify ExpertPool handles 256 experts (vs 128)

**Files:**
- Reference: `include/frontend/models/expert_pool.hpp`, `src/frontend/models/expert_pool.cpp`

- [ ] **Step 1: Sanity check pool sizing**

Run:
```bash
ZEDINFER_MOE_GPU_SLOTS=64 xmake run ping ~/data/models/Qwen3.5-35B-A3B-GPTQ-Int4 --nvidia 2>&1 | tee /tmp/m5_pool.log | head -50
```

Look for ExpertPool ctor log line. Verify `num_experts=256`, `gpu_slots=64`, strategy=PINNED_LRU.

- [ ] **Step 2: Try ALL_GPU strategy**

```bash
xmake run ping ~/data/models/Qwen3.5-35B-A3B-GPTQ-Int4 --nvidia 2>&1 | tee /tmp/m5_allgpu.log
```

With auto-N sizing, on a 48 GB A6000 this might fit 256 experts in VRAM (depends on M5 init values). On a 24 GB card with `--gpu-memory-utilization 0.5`, it must fall back to PINNED_LRU. Verify either decision path emits expected log line and reaches forward.

- [ ] **Step 3: Confirm Phase 2 perf cleanups still active**

The four polish-round commits (handoff §5.5) should kick in: const-ref ExpertGpuHandle, no-op shared_expert D2D skip, thread_local bucket vectors, cached router weights. Verify `ModelForwardConfig::router_weights[L]` is populated for Qwen3.5 MoE.

In `Qwen3_5MoeModel::hybrid_forward_config()` (extends `Qwen3_5Model::hybrid_forward_config()`), populate:
```cpp
hcfg.is_moe = true;
hcfg.num_experts = moe_config_.num_experts;
hcfg.num_experts_per_tok = moe_config_.num_experts_per_tok;
hcfg.moe_intermediate_size = moe_config_.moe_intermediate_size;
hcfg.shared_expert_intermediate_size = moe_config_.shared_expert_intermediate_size;
hcfg.has_shared_expert = (moe_config_.shared_expert_intermediate_size > 0);
hcfg.decoder_sparse_step = moe_config_.decoder_sparse_step;
hcfg.mlp_only_layers = moe_config_.mlp_only_layers;
hcfg.expert_pool = expert_pool_.get();

// Pre-cache router weights so moe_layer_forward avoids per-layer string lookup
for (size_t L = 0; L < moe_config_.num_hidden_layers; ++L) {
    auto name = std::string("layers.") + std::to_string(L) + ".mlp.gate.weight";
    hcfg.router_weights.push_back(weights_->has_tensor(name)
        ? weights_->get_tensor(name) : nullptr);
}
hcfg.expert_quant_num_bits = moe_config_.quant_config.weights.num_bits;
hcfg.expert_quant_group_size = moe_config_.quant_config.weights.group_size;
```

- [ ] **Step 4: Commit**

```bash
git add include/frontend/models/qwen3_5_moe.hpp src/frontend/models/qwen3_5_moe.cpp
git commit -m "feat(qwen3.5-moe): hybrid_forward_config wires ExpertPool + cached router weights"
```

---

### Task 2: Shared expert + shared_expert_gate path (Qwen3.5-MoE specific)

**Files:**
- Modify: `src/frontend/models/moe_forward.cpp`

- [ ] **Step 1: Check current `apply_shared_expert` impl**

Run: `grep -n 'shared_expert\|shared_expert_gate' src/frontend/models/moe_forward.cpp | head -20`

- [ ] **Step 2: Add shared_expert_gate handling**

Qwen3.5-MoE has `shared_expert_gate` weight `[1, hidden]` — a scalar gate applied to the shared expert output:
```cpp
shared_out = swiglu(gate_proj(x), up_proj(x)) → down_proj
gate_scalar = sigmoid(shared_expert_gate(x))   // [N, 1]
shared_out *= gate_scalar
```

Edit `apply_shared_expert` to read `mlp.shared_expert_gate.weight` and apply sigmoid scaling.

For Qwen3-30B-A3B path (no shared expert), the existing skip-D2D commit (`0bd53b5`) bypasses this entirely. For Qwen3.5-MoE the gate scalar must be applied.

Add a flag check:
```cpp
if (model.has_shared_expert) {
    // compute shared_expert MLP
    if (weights.has_tensor("layers.X.mlp.shared_expert_gate.weight")) {
        // apply sigmoid gate
    }
    accumulator += shared_out;
}
```

- [ ] **Step 3: Unit test**

A focused test: build a fake MoE layer config with single shared expert; verify shared_expert_gate scales correctly.

- [ ] **Step 4: Commit**

```bash
git add src/frontend/models/moe_forward.cpp tests/unit/test_moe_shared_expert_gate.cpp
git commit -m "feat(moe): shared_expert_gate sigmoid scaling (Qwen3.5-MoE)"
```

---

### Task 3: 35B-A3B coherent text reply

- [ ] **Step 1: Run ping**

```bash
ZEDINFER_MOE_GPU_SLOTS=64 xmake run ping ~/data/models/Qwen3.5-35B-A3B-GPTQ-Int4 --nvidia
```

Type "Who are you?". Expected: coherent reply.

If garbled / NaN: most likely the shared_expert_gate path is broken or MoE router weight isn't cached. Debug as in M1.

- [ ] **Step 2: Commit any fixes**

```bash
git add ...
git commit -m "fix(qwen3.5-moe): <specific issue>"
```

---

### Task 4: 35B-A3B byte-exact HF alignment

- [ ] **Step 1: Generate HF reference for 35B-A3B**

```bash
# Edit tests/e2e/hf_reference.py: change MODEL_PATH to 35B-A3B; rename OUT_DIR to qwen3_5_moe_hf_reference
python3 tests/e2e/hf_reference.py
```

- [ ] **Step 2: Run alignment harness pointing at 35B-A3B**

Adapt `tests/e2e/test_qwen3_5_hf_alignment.py` to take `--model-path` and `--refs-dir` parameters:
```python
parser.add_argument("--model-path", required=True)
parser.add_argument("--refs-dir",   required=True)
```

Run:
```bash
ZEDINFER_MOE_GPU_SLOTS=64 \
  python3 tests/e2e/test_qwen3_5_hf_alignment.py \
    --model-path ~/data/models/Qwen3.5-35B-A3B-GPTQ-Int4 \
    --refs-dir tests/fixtures/qwen3_5_moe_hf_reference
```

Iterate as in M2. Most fixes should already be in place from M2; new issues likely in MoE-specific code (shared_expert_gate, router top-8).

- [ ] **Step 3: Commit byte-exact result**

```bash
git add docs/plan/qwen3_5_support.md
git commit -m "docs(qwen3.5): 35B-A3B byte-exact alignment achieved"
```

---

### Task 5: 35B-A3B multimodal end-to-end

- [ ] **Step 1: Test CLI vision**

```bash
ZEDINFER_MOE_GPU_SLOTS=64 xmake run ping ~/data/models/Qwen3.5-35B-A3B-GPTQ-Int4 --nvidia \
  --image tests/fixtures/sample.jpg --prompt "What color is this?"
```
Expected: coherent color description.

- [ ] **Step 2: Test HTTP vision**

```bash
ZEDINFER_MOE_GPU_SLOTS=64 xmake run serve ~/data/models/Qwen3.5-35B-A3B-GPTQ-Int4 --nvidia --port 8081 &
IMG=$(base64 -w0 tests/fixtures/sample.jpg)
curl http://localhost:8081/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":[
       {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,'"$IMG"'"}},
       {"type":"text","text":"describe"}
     ]}],"max_tokens":50}'
```

- [ ] **Step 3: Byte-exact multimodal alignment**

Extend hf_reference.py with image prompt; generate; align via harness.

- [ ] **Step 4: Commit milestone**

```bash
git add docs/plan/qwen3_5_support.md
git commit -m "docs(qwen3.5): M5 complete; 35B-A3B text+vision byte-exact"
```

---

### Task 6: VRAM negotiation between SSMStatePool, ExpertPool, KV cache

**Files:**
- Modify: `src/frontend/models/base.cpp` (`compute_moe_pool_config` analog for Qwen3.5)

- [ ] **Step 1: Survey existing auto-N logic**

```bash
grep -n 'compute_moe_pool_config\|kExpertVramFraction\|gpu_memory_utilization' src/frontend/models/base.cpp
```

The existing function computes ExpertPool size from free VRAM minus already-used. For Qwen3.5-MoE, we also need to reserve SSMStatePool bytes (~75 MB × max_concurrent) before deciding ExpertPool budget.

- [ ] **Step 2: Extend the helper for hybrid+MoE**

```cpp
ExpertPoolConfig compute_moe_pool_config_qwen3_5(const Qwen3_5MoEConfig& cfg,
                                                    zedinferDeviceType_t target,
                                                    float gpu_mem_util) {
    // 1. Reserve SSMStatePool bytes
    int num_linear = std::count(cfg.layer_types.begin(), cfg.layer_types.end(), "linear_attention");
    int qkv_dim = cfg.linear_attn.num_k_heads * cfg.linear_attn.key_head_dim * 2
                + cfg.linear_attn.num_v_heads * cfg.linear_attn.value_head_dim;
    size_t state_bytes_per_slot =
        (size_t)num_linear * cfg.linear_attn.num_v_heads * cfg.linear_attn.value_head_dim
                            * cfg.linear_attn.d_state * 2
      + (size_t)num_linear * (cfg.linear_attn.conv_kernel_dim - 1) * qkv_dim * 2;
    size_t ssm_reservation = state_bytes_per_slot * /*max_concurrent=*/1;

    // 2. Existing ExpertPool sizing minus SSM reservation
    // (Copy logic from compute_moe_pool_config but subtract ssm_reservation from budget)
    // ...
}
```

- [ ] **Step 3: Test on 48 GB and 24 GB simulated budgets**

Run with `--gpu-memory-utilization 0.5` (24 GB simulated). Verify auto-N picks reasonable PINNED_LRU N. Log line should show:
```
[Model] qwen3_5_moe: SSM reservation 75 MB; ExpertPool budget 15 GB; → PINNED_LRU N=64
```

- [ ] **Step 4: Commit**

```bash
git add src/frontend/models/base.cpp
git commit -m "feat(qwen3.5-moe): auto-N sizing reserves SSMStatePool bytes from ExpertPool budget"
```

---

### Task 7: BF16 large-model variant (no GPTQ)

If a non-quantized BF16 35B-A3B exists (e.g. `~/data/models/Qwen3.5-35B-A3B` — check first), exercise the dense-expert path through ExpertPool.

- [ ] **Step 1: Check availability**

```bash
ls ~/data/models/ | grep -i 'qwen3.5.*35b' | grep -v gptq
```

If file present: run `xmake run ping ...` with `ZEDINFER_MOE_GPU_SLOTS=16 --gpu-memory-utilization 0.5` (matching D-series scenario from heterogeneous_moe.md §10).

If not: skip this task; document in handoff that BF16 35B-A3B was not tested.

- [ ] **Step 2: If runs, validate output**

Coherent? Memory budget log shows expected SSM + ExpertPool split? Commit notes.

---

### Task 8: M5 milestone + retro

- [ ] **Step 1: Update spec doc**

```
M5 complete: <commit>, 35B-A3B text + multimodal byte-exact vs HF. ExpertPool 256 experts in PINNED_LRU mode at N=64. SSMStatePool + ExpertPool VRAM negotiation works at --gpu-memory-utilization 0.5.
```

- [ ] **Step 2: Handoff doc update**

```markdown
## M5 — 35B-A3B MoE path (complete)

### What landed
- Qwen3_5MoeModel uses ExpertPool (256 experts) + SSMStatePool (10 linear layers)
- shared_expert_gate sigmoid scaling in moe_forward
- Auto-N sizing accounts for SSM reservation
- Byte-exact alignment for 35B-A3B (text + image)
- All HTTP/CLI/Web UI paths work for both models

### Caveats / M6 entry
- BF16 35B-A3B large-model path tested: <yes / no>
- max_concurrent=1 default; concurrent users untested
- Benchmark numbers not yet collected (M6)
```

- [ ] **Step 3: Commit**

```bash
git add docs/plan/qwen3_5_support.md docs/plan/qwen3_5_session_handoff.md
git commit -m "docs(qwen3.5): M5 retro and M6 entry conditions"
```

---

## M5 Done. ~8 tasks. Est. 1.5–2 weeks.
