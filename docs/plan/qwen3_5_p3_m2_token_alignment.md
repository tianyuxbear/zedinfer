# Qwen3.5 P3 (M2) — 27B Token-Level Byte-Exact Alignment vs HuggingFace

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 27B greedy generation (argmax sampling, deterministic) produces token IDs identical to HuggingFace `transformers` for the first 50 tokens given an identical prompt. This validates SSU + causal_conv1d + 3D mrope + attn_output_gate + paged attention numerical correctness end-to-end.

**Architecture:** Build a Python harness that runs HF and zedinfer on the same prompt and diffs token IDs + per-position logit max-abs-diff. Iteratively fix any numerical drift in mrope_3d (most likely), then SSU (FlashInfer state dtype choice), then conv1d / attn gate.

**Tech Stack:** Python 3.10+, `transformers >= 4.57.0.dev0` (the version Qwen3.5 was released against per config.json), `torch`, zedinfer CLI.

**Reference:** Design doc `docs/plan/qwen3_5_support.md` §6 / §12.

**Pre-condition:** P2 (M1) complete. 27B emits coherent reply.

---

### Task 1: Set up HF reference environment

**Files:**
- Create: `tests/e2e/requirements.txt`
- Create: `tests/e2e/setup_hf.sh`

- [ ] **Step 1: Pin transformers version**

Create `tests/e2e/requirements.txt`:
```
torch>=2.5
transformers==4.57.0
accelerate
safetensors
sentencepiece
auto-gptq
optimum
```

- [ ] **Step 2: Install in project venv**

Run:
```bash
. .venv/bin/activate
pip install -r tests/e2e/requirements.txt
python3 -c "import transformers; print(transformers.__version__)"
```
Expected: `4.57.0` (or compatible patch). If `Qwen3_5ForConditionalGeneration` is not yet upstream in transformers, install from a git ref:
```bash
pip install 'git+https://github.com/huggingface/transformers@<commit-with-qwen3.5>'
```

If even main doesn't have it: this task **blocks M2**. Document in handoff doc and decide whether to fork transformers locally or defer M2 until upstream supports Qwen3.5.

- [ ] **Step 3: Smoke-test HF can load Qwen3.5-27B-GPTQ-Int4**

```bash
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
tok = AutoTokenizer.from_pretrained('/home/tianyux/data/models/Qwen3.5-27B-GPTQ-Int4')
model = AutoModelForCausalLM.from_pretrained(
    '/home/tianyux/data/models/Qwen3.5-27B-GPTQ-Int4',
    torch_dtype=torch.bfloat16, device_map='cuda:0')
print('HF model loaded:', type(model).__name__)
print('vocab_size:', model.config.vocab_size)
"
```
Expected: prints `HF model loaded: Qwen3_5ForConditionalGeneration` (or similar) and `vocab_size: 248320`.

- [ ] **Step 4: Commit**

```bash
git add tests/e2e/requirements.txt
git commit -m "test(qwen3.5): pin HF transformers dependencies for M2 alignment"
```

---

### Task 2: Generate HF reference token IDs

**Files:**
- Create: `tests/e2e/hf_reference.py`

- [ ] **Step 1: Write the generator**

```python
# tests/e2e/hf_reference.py
"""Generate greedy reference token IDs from HF transformers for a fixed prompt.
Saves to tests/fixtures/qwen3_5_hf_reference/<prompt-key>.json."""
import json
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/home/tianyux/data/models/Qwen3.5-27B-GPTQ-Int4"
OUT_DIR = "tests/fixtures/qwen3_5_hf_reference"
PROMPTS = [
    ("hello",     "Who are you?"),
    ("math",      "What is 2+2?"),
    ("code",      "Write a Python function to compute factorial."),
]

os.makedirs(OUT_DIR, exist_ok=True)

tok = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0"
)
model.eval()

for key, prompt in PROMPTS:
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                    enable_thinking=False)
    ids = tok(text, return_tensors="pt").input_ids.to("cuda:0")

    with torch.no_grad():
        out = model.generate(ids, do_sample=False, max_new_tokens=50,
                             pad_token_id=tok.pad_token_id)
    new_tokens = out[0, ids.shape[1]:].tolist()
    decoded = tok.decode(new_tokens, skip_special_tokens=False)

    artifact = {
        "prompt_key": key,
        "prompt": prompt,
        "rendered_prompt": text,
        "input_ids": ids[0].tolist(),
        "new_token_ids": new_tokens,
        "decoded_output": decoded,
    }
    with open(os.path.join(OUT_DIR, f"{key}.json"), "w") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)
    print(f"[{key}] {len(new_tokens)} tokens: {decoded[:80]}...")

print(f"[ok] HF references saved to {OUT_DIR}")
```

- [ ] **Step 2: Run**

```bash
python3 tests/e2e/hf_reference.py
```
Expected: 3 JSON files in `tests/fixtures/qwen3_5_hf_reference/` each with 50 token IDs.

- [ ] **Step 3: Commit**

```bash
git add tests/e2e/hf_reference.py tests/fixtures/qwen3_5_hf_reference
git commit -m "test(qwen3.5): HF reference token IDs for 3 deterministic prompts"
```

---

### Task 3: Add `ping --no-sample --max-tokens=N --emit-token-ids` mode

**Files:**
- Modify: ping source (the same file modified in P1 Task 13)

- [ ] **Step 1: Survey current ping CLI flags**

Run: `grep -n 'argparse\|--max\|no.sample' src/examples/ping.cpp 2>&1 | head -20`

- [ ] **Step 2: Add flags**

In ping's argparse block:
```cpp
program.add_argument("--no-sample").default_value(false).implicit_value(true)
       .help("Disable sampling; use argmax (greedy)");
program.add_argument("--max-tokens").default_value(50).scan<'i', int>();
program.add_argument("--emit-token-ids").default_value(false).implicit_value(true)
       .help("Print generated token IDs as JSON array to stdout");
```

In the generation loop, when `--emit-token-ids` is set, collect token IDs into a `std::vector<int>` and print as JSON before exit:
```cpp
std::cout << "[ZEDINFER_TOKEN_IDS] " << json(generated_ids).dump() << "\n";
```

Disable streaming when `--emit-token-ids` is set (just print the JSON line).

For `--no-sample`, replace the sampler with argmax:
```cpp
if (args.get<bool>("--no-sample")) {
    sampler_cfg.use_argmax = true;
}
```

(Adjust to actual sampler API in zedinfer.)

- [ ] **Step 3: Test**

```bash
xmake run ping ~/data/models/Qwen3.5-27B-GPTQ-Int4 --nvidia \
    --no-sample --max-tokens 50 --emit-token-ids \
    --prompt "Who are you?"
```
Expected: one line `[ZEDINFER_TOKEN_IDS] [123, 456, ...]` (50 IDs).

- [ ] **Step 4: Commit**

```bash
git add src/examples/ping.cpp
git commit -m "feat(ping): --no-sample / --max-tokens / --emit-token-ids for alignment harness"
```

---

### Task 4: Alignment harness script

**Files:**
- Create: `tests/e2e/test_qwen3_5_hf_alignment.py`

- [ ] **Step 1: Write the harness**

```python
# tests/e2e/test_qwen3_5_hf_alignment.py
"""Compare zedinfer ping output token IDs to HF reference. Exit 0 if all match."""
import json
import os
import subprocess
import sys

REFS_DIR = "tests/fixtures/qwen3_5_hf_reference"
MODEL_PATH = "/home/tianyux/data/models/Qwen3.5-27B-GPTQ-Int4"

def run_zedinfer(prompt):
    out = subprocess.check_output([
        "xmake", "run", "ping", MODEL_PATH, "--nvidia",
        "--no-sample", "--max-tokens", "50", "--emit-token-ids",
        "--prompt", prompt
    ], text=True, env={**os.environ, "ZEDINFER_DISABLE_WARMUP": "1"})
    for line in out.splitlines():
        if line.startswith("[ZEDINFER_TOKEN_IDS]"):
            return json.loads(line.split(" ", 1)[1])
    raise RuntimeError("zedinfer did not emit token IDs")

fail = 0
for fn in sorted(os.listdir(REFS_DIR)):
    if not fn.endswith(".json"): continue
    ref = json.load(open(os.path.join(REFS_DIR, fn)))
    print(f"\n=== {fn} prompt: {ref['prompt']!r} ===")
    zi = run_zedinfer(ref["prompt"])
    hf = ref["new_token_ids"]
    first_diff = next((i for i, (a, b) in enumerate(zip(zi, hf)) if a != b), None)
    if first_diff is None and len(zi) >= len(hf):
        print(f"  PASS: all {len(hf)} tokens match")
    else:
        fail += 1
        print(f"  FAIL: first diff at pos {first_diff}, zi={zi[:5]}... hf={hf[:5]}...")
        for i in range(max(0, (first_diff or 0)-2), min(len(hf), (first_diff or 0)+5)):
            mark = "*" if (i == first_diff) else " "
            print(f"   {mark} pos {i:3d}: zi={zi[i] if i<len(zi) else '?'} hf={hf[i] if i<len(hf) else '?'}")

if fail:
    print(f"\n[FAIL] {fail} prompt(s) diverged")
    sys.exit(1)
print(f"\n[ok] all prompts byte-exact for first 50 tokens")
```

- [ ] **Step 2: Initial run (expect divergence)**

```bash
python3 tests/e2e/test_qwen3_5_hf_alignment.py | tee /tmp/m2_initial.log
```

Almost certainly fails on first prompt. Note the first divergence position — start debugging there.

- [ ] **Step 3: Commit**

```bash
git add tests/e2e/test_qwen3_5_hf_alignment.py
git commit -m "test(qwen3.5): HF token alignment harness (greedy, 50 tokens × 3 prompts)"
```

---

### Task 5: Per-layer hidden-state probe utility

**Files:**
- Create: `tests/e2e/hidden_state_probe.py`
- Modify: ping to optionally emit hidden states

- [ ] **Step 1: Add `--dump-layer-hidden=N` flag to ping**

When set, after layer N forward, copy `hidden` to host and append to a binary file like `/tmp/zedinfer_hidden_layer{N}.bin`.

```cpp
if (args.is_used("--dump-layer-hidden") && (int)L == args.get<int>("--dump-layer-hidden")) {
    std::vector<__nv_bfloat16> host(hidden->numel());
    cudaMemcpy(host.data(), hidden->data(), host.size()*2, cudaMemcpyDeviceToHost);
    std::ofstream(std::string("/tmp/zedinfer_hidden_layer") + std::to_string(L) + ".bin",
                  std::ios::binary).write(reinterpret_cast<char*>(host.data()), host.size()*2);
}
```

- [ ] **Step 2: HF-side hidden state dump**

```python
# tests/e2e/hidden_state_probe.py
"""Run HF on the same prompt and dump per-layer hidden states for comparison."""
import torch, sys
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/home/tianyux/data/models/Qwen3.5-27B-GPTQ-Int4"
PROMPT = "Who are you?"
LAYERS = [0, 3, 7, 15, 31, 63]

tok = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0")
model.eval()

text = tok.apply_chat_template([{"role":"user","content":PROMPT}], add_generation_prompt=True, tokenize=False)
ids = tok(text, return_tensors="pt").input_ids.cuda()

with torch.no_grad():
    out = model(ids, output_hidden_states=True)

# hidden_states is a tuple of (num_layers+1) tensors, each [B, N, H]
for L in LAYERS:
    h = out.hidden_states[L+1][0]  # post-layer L
    h.cpu().numpy().tofile(f"/tmp/hf_hidden_layer{L}.bin")
    print(f"layer {L:2d}: shape={tuple(h.shape)} norm={h.norm().item():.4f}")
```

- [ ] **Step 3: Compare script**

```python
# tests/e2e/compare_hidden.py
import numpy as np, sys
import os

LAYERS = [0, 3, 7, 15, 31, 63]
for L in LAYERS:
    a = np.fromfile(f"/tmp/zedinfer_hidden_layer{L}.bin", dtype=np.float16).astype(np.float32)
    b = np.fromfile(f"/tmp/hf_hidden_layer{L}.bin",       dtype=np.float16).astype(np.float32)
    if a.size != b.size:
        print(f"layer {L}: size mismatch {a.size} vs {b.size}"); continue
    d = np.abs(a - b)
    print(f"layer {L:2d}: max={d.max():.4f} mean={d.mean():.4f} rel_err={d.mean()/np.abs(b).mean():.4%}")
```

(Note: bf16 dtype handling — adjust dtype parsing if needed; bf16 typically saved as fp16 surrogate when written from torch; better: convert to fp32 in both scripts and save fp32.)

- [ ] **Step 4: Commit**

```bash
git add tests/e2e/hidden_state_probe.py tests/e2e/compare_hidden.py src/examples/ping.cpp
git commit -m "test(qwen3.5): per-layer hidden-state probe for alignment debugging"
```

---

### Task 6: Debug-driven iteration — typical fix order

For each prompt that fails alignment, follow this sequence. Each "fix" is its own commit.

- [ ] **Step 1: First-layer drift?**

Run hidden_state_probe at layer 0. If max_abs_diff > 0.01 at layer 0, the issue is in **embedding lookup** or **input_layernorm**. Verify `embed_tokens.weight` matches HF (zedinfer has it via tied embed). Verify `rms_norm` weight indexing.

If diff first appears at layer 3 (first full-attention layer): bug is in **mrope_3d** or **attn_output_gate** or **q_proj split**. Run unit tests again with fresh prompts.

- [ ] **Step 2: mrope_3d fixes**

Common bugs:
- Section interpretation: `[11, 11, 10]` may be per-axis dim count over **Dh_rot** (full rotary count), not over **half** (pair count). Verify against HF source `transformers/models/qwen3_5/modeling_qwen3_5.py::apply_multimodal_rotary_pos_emb`.
- Interleaved vs split: "interleaved" can mean (cos_i, sin_i) adjacent pairs vs (cos_first_half | sin_second_half).
- Position id ordering: text tokens may use `(idx, idx, idx)` or `(idx, 0, 0)` — Qwen3.5 spec uses `(idx, idx, idx)` for pure text (verify).

After each fix:
```bash
xmake build && xmake run test-ops-mrope-3d  # unit test still passes
python3 tests/e2e/test_qwen3_5_hf_alignment.py  # full harness
```
Commit with message: `fix(mrope_3d): correct <specific issue> (M2 alignment)`.

- [ ] **Step 3: SSU fixes**

If linear-attn-layer drift dominates:
- Check SSU `ngroups` parameter (Qwen3.5 may share K across V heads with `ngroups != 1`).
- Check `dt_softplus = true` in params.
- Check state_dtype: if BF16 state introduces too much error, switch to fp32. Edit `SSMStatePool` ctor to pass `state_dtype = ZEDINFER_DTYPE_F32`; reinstantiate FlashInfer template with `state_t=float`. May need new instantiation in FlashInfer source.

After each fix:
```bash
xmake build && xmake run test-ops-mamba-ssu
python3 tests/e2e/test_qwen3_5_hf_alignment.py
```

- [ ] **Step 4: causal_conv1d fixes**

If state isn't preserved correctly across decode steps:
- Verify state is zeroed by `reset_slot` (single decode call after reset should produce correct first-window output).
- Verify state offset math (`slot * stride_slot + layer * stride_layer`) matches between conv1d kernel and SSU kernel.

- [ ] **Step 5: attn_output_gate**

Trivial; verify gate slice offset matches design (latter half of q_proj output).

- [ ] **Step 6: Per-fix commit pattern**

For each fix:
```bash
git add <touched files>
git commit -m "fix(<op>): <one-line fix description>"
python3 tests/e2e/test_qwen3_5_hf_alignment.py 2>&1 | tail -20
```

---

### Task 7: M2 acceptance — all 3 prompts byte-exact 50 tokens

- [ ] **Step 1: Run full harness, expect pass**

```bash
python3 tests/e2e/test_qwen3_5_hf_alignment.py
```
Expected: `[ok] all prompts byte-exact for first 50 tokens`.

If still failing on edge cases:
- Increase from 3 to 10 prompts to catch off-by-one bugs in less-common token distributions.
- Run with `--no-sample` and a long math prompt (deterministic distribution).

- [ ] **Step 2: Cross-check with greedy temperature=0 + different system prompt**

Add a 4th prompt with system message:
```python
("system",  [{"role":"system","content":"You are Qwen, a helpful assistant."},
              {"role":"user","content":"Hello"}]),
```

Re-generate HF reference; re-run alignment.

- [ ] **Step 3: Commit M2 milestone**

In `docs/plan/qwen3_5_support.md` §10 M2 row:
```
M2 complete: <commit-sha>, all 4 prompts byte-exact first 50 tokens. SSM state_dtype = <bf16 | f32>. Largest fix: <mrope_3d section interpretation | etc.>
```

```bash
git add docs/plan/qwen3_5_support.md
git commit -m "docs(qwen3.5): M2 complete; 27B greedy byte-exact vs HF (first 50 tokens × 4 prompts)"
```

---

### Task 8: M2 retro

- [ ] **Step 1: Update handoff doc with fixes inventory**

Append to `docs/plan/qwen3_5_session_handoff.md`:
```markdown
## M2 — 27B HF byte-exact alignment (complete)

### Fixes applied (with commit shas)
- mrope_3d: <issue> (commit: ...)
- SSU: <issue> (commit: ...)
- causal_conv1d: <issue> (commit: ...)
- (etc.)

### Final dtype choices
- SSU state_dtype: <bf16 | f32>
- All other ops: BF16 input, BF16 output

### Tests passing
- test-ops-mrope-3d, test-ops-mamba-ssu, test-ops-causal-conv1d, test-ops-attn-output-gate
- All 4 alignment prompts (hello, math, code, system)
- All v0.2.0 regression tests

### Open items / M3 entry
- Per-layer hidden alignment confirmed at layers [0, 3, 7, 15, 31, 63]
- 35B-A3B still untested for byte-exact (M5)
- Vision path still stubbed (M3)
```

- [ ] **Step 2: Commit**

```bash
git add docs/plan/qwen3_5_session_handoff.md
git commit -m "docs(qwen3.5): M2 retro and M3 entry conditions"
```

---

## M2 Done. ~8 tasks. Est. 1-1.5 weeks.

27B numerically equivalent to HF. Foundation for M3 (vision) + M5 (35B-A3B) alignment work.
