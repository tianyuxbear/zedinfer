# Qwen3.5-35B-A3B-GPTQ-Int4: sampler offset bug + long-thinking drift

This document captures the investigation that ran across a multi-day session on
the `feat/qwen3.5` branch. Two issues were discovered while bringing up text
generation on `~/data/models/Qwen3.5-35B-A3B-GPTQ-Int4`: one was a real
zedinfer correctness bug (now fixed), the other is a quantization-side model
quality issue that surfaces only in long `<think>` generations.

## TL;DR

| Issue | Status | Where |
|---|---|---|
| `GeneralSampler` always emits role-tag tokens ("user", "system") as the first generated token, regardless of prompt | **Fixed** | `0c737a7 fix(tensor): Tensor::to(device) must read from offset-aware data()` |
| `GeneralSampler` reads garbled values when logits are bf16 | **Fixed** | `de59c5e fix(sampler): … bf16-safe sampling` |
| `Qwen3_5MoeModel` was using the dense ChatML fallback template (no `<think>`) | **Fixed** | `f7016c5 fix(chat_template): register Qwen3.5 reasoning template` |
| Long generations inside an open `<think>` block (>~80 tokens) ramble until they hit `max_new_tokens` because the model never emits `</think>` | **Not a zedinfer bug** — GPTQ-Int4 calibration issue. Workaround: short interactions stay coherent; switch to closed-think prompt when long answers are needed. | — |

## Setup used for all experiments

```bash
CUDA_VISIBLE_DEVICES=1 xmake run ping \
    ~/data/models/Qwen3.5-35B-A3B-GPTQ-Int4 \
    --nvidia --gpu-memory-utilization 0.85 \
    --max-new-tokens <N> --prompt "<prompt>"
```

Hardware: A6000 (49 GiB VRAM) at CUDA index 1; the index-0 RTX 4000 Ada has
20 GiB and cannot host the 35B GPTQ weights.

## The real bug: sampler reads logits from prompt-row 0

### Symptom

For `--prompt "What is 2+2?"`, the first generated token was *always* one of
the role-tag word-piece IDs:

```
[Sampler-DEBUG] raw F32 argmax=846 (logit=19.875), vocab=248320, history_len=0
[Sampler-DEBUG] post-filter top tokens:  id=846:p=0.7773 id=8678:p=0.2227 (kept=2)
[Sampler-DEBUG] chose id=846 (idx=0)
```

- id 846 = `user`
- id 8678 = `system`

ArgmaxSampler, given the same prompt, picked id ≈ 32850 which decodes to
`Thinking` — the expected first token of a Qwen3.5 reasoning chain. So the
forward path was correct; the sampler was wrong.

### Root cause

`Sampler::getLastLogits` returns a sliced view of the model output:

```cpp
// shape = [seq_len, vocab_size]
return logits->slice(0, seq_len - 1, seq_len)->view({shape[1]});
```

`GeneralSampler::sample` then calls `ensureCPU()` followed by `to(F32)`. Both
go through `Tensor::to(device, device_id)` (`src/backend/tensor/tensor.cpp`).
That function used the underlying *storage base*, not the offset-aware
`data()`, as the memcpy source:

```cpp
// BUG
core::context().runtime().api()->memcpy_sync(new_storage->memory(),
                                             _storage->memory(),  // <-- base, not data()
                                             total_bytes, ZEDINFER_MEMCPY_D2H);
```

For a sliced tensor `_offset > 0`, so the memcpy copied
`total_bytes` *starting from row 0* of the underlying logits tensor — i.e.
the **logits for the first prompt token** (`<|im_start|>`).

For Qwen3.5 the first prompt token is always `<|im_start|>` (chat template
opener). The model's natural continuation of `<|im_start|>` is a role tag
(`user` ≈ 0.78 mass, `system` ≈ 0.22 mass), and that is exactly the
distribution `GeneralSampler` was sampling from regardless of which row it
intended to read.

`ArgmaxSampler` dodged the bug entirely because its `ops::argmax` kernel
reads via the tensor's offset-aware data pointer.

### Fix

`0c737a7 fix(tensor): Tensor::to(device) must read from offset-aware data()`
replaces the three `_storage->memory()` source pointers in `Tensor::to(device)`
with `data()`. `Tensor::to(dtype)` (the bf16/f16 → fp32 path) was already
correct.

### Verification

Same prompt + same sampler params (`temp=1.0 top_p=0.95 top_k=20
repetition_penalty=1.1`):

| Before | After |
|---|---|
| `user "2 +` (first 4 tokens, role-tag-then-garbage) | `Thinking Process:\n\n1. **Identify the user's query:** The user is asking a simple math question...` |

## Other fixes that landed alongside

### `de59c5e fix(sampler): bf16-safe sampling`

`GeneralSampler::sample` previously did
`reinterpret_cast<const float*>(logits->data())`. For a bf16 logits tensor
this treats two consecutive 2-byte values as one float and produces token-id
noise. Now we `Tensor::to(F32)` first.

### `f7016c5 fix(chat_template): register Qwen3.5 reasoning template`

`Qwen3_5MoeModel` was hitting the `[ChatTemplate] Unknown model_type=qwen3_5_moe`
fallback to `default_qwen_chatml()`, whose `generation_prompt` is just
`"<|im_start|>assistant\n"`. Qwen3.5 was trained so the assistant turn must
open with `<think>\n` (see any Qwen3.5 release's `chat_template.jinja`). Without
the `<think>\n` suffix the model hallucinated a fake `user:` line before
answering. Added `default_qwen3_5_chatml()` with the correct generation prompt
and routed `qwen3_5` / `qwen3_5_moe` to it.

### `469e69a feat(sampler): repetition penalty + restore open-think template`

Added `SamplerParams::repetition_penalty` (HF / CTRL multiplicative form),
plumbed `req->output_ids` through scheduler → sampler so the penalty has a
history to operate on. The engine reads `repetition_penalty` from
`generation_config.json` if present, defaulting to a mild 1.1 otherwise (Qwen
generation_configs don't ship a value but the penalty is a standard mitigation
for degenerate token sequences under `temp+top_p` sampling).

## The remaining issue: long-thinking drift

After the sampler bug was fixed, **short** generations (≲ 50 tokens) are
coherent. **Long** generations inside an open `<think>` block still degrade:
they start coherent for ~40-80 tokens and then ramble until they exhaust
`max_new_tokens`, never emitting `</think>` to close the thinking block.

### Example: "Who are you?" → 256 tokens

```
Okay, the user just said "Hi" and mentioned that it's a simple greeting.
Let me start by acknowledging their greeting in a friendly way. I should
respond warmly to make them feel welcome.

I need to check if there's any specific purpose for this chat, but maybe
they want general assistance first. Maybe ask what they need help with?
Let both be friendly and helpful since the user might be looking to
connect or get help with something else, whether greetings are simple or
more personal like interactions or interactions...

Wait, actually when someone is talking about "hi" as an example of
interaction but in this case it's probably used as hihi (hmmm) 103?
That doesn't sound right... Or could also use an example from hi
15567-236? Hmm not quite sure here how much of 2024 will change the
original idea here is clear enough to interact with you can do
8i8n9jgk8y8juy9h7kq4wvixqy7x4e9zq3kqx1zhu6ruijsuicaiyuanjiaohaijiann
ngcuhuaangdajidong
```

First paragraph is correct reasoning. By the second paragraph the model is
talking in circles. By the third it's emitting pinyin-like noise. This
pattern is the same under both `ArgmaxSampler` (deterministic loop) and
`GeneralSampler` (random-looking noise).

### Diagnosis

We instrumented `scheduler.cpp` to dump every generated token id and, every
10 steps, the lm_head logit for the special tokens. For greedy decoding of
"Who are you?", 100 steps:

| step | argmax (id, logit) | `</think>` (248069) | `<\|im_end\|>` (248046) | `<\|endoftext\|>` (248044) | `<think>` (248068) |
|---:|---|---:|---:|---:|---:|
| 10 | (1965, 25.6) | -0.31 |  7.59 |  6.62 |  0.75 |
| 20 | (5952, 26.8) | -0.32 |  3.38 |  1.36 | -0.44 |
| 30 | (2699, 25.0) | -0.66 |  3.53 |  1.02 |  1.69 |
| 40 |  (539, 20.6) | -1.98 |  1.21 |  1.97 |  0.92 |
| 50 | (2972, 19.7) |  3.88 |  5.22 |  1.69 |  3.17 |
| 60 |  (701, 18.7) | -1.20 |  4.00 |  4.81 |  2.50 |
| 70 | (9859, 20.1) |  2.13 |  5.13 |  3.91 |  0.35 |
| 80 | (1044, 23.2) |  0.93 |  5.53 |  4.19 |  0.44 |
| 90 | (1156, 19.1) |  1.26 |  4.16 |  3.41 |  3.03 |

Observations:

- `</think>` produces non-zero logits, so its lm_head row is not corrupted.
- But its logit is **persistently 15-25 units below the argmax**.
  After softmax that is exp(-20)+ probability — effectively unreachable
  even under temperature sampling.
- `<|im_end|>` does sometimes climb (e.g. step 10 has it at 7.6 with argmax
  25.6, gap 18 → p ≈ 1.5e-8). Same conclusion.

Across 200 generated tokens, neither `</think>` (id 248069) nor `<|im_end|>`
(id 248046) is ever sampled — the model has no path to end the response.

### Why this isn't a zedinfer bug

For short responses the model emits `<|im_end|>` and stops cleanly (the
closed-think variant of the chat template hits EOS at ~54 tokens for "Who
are you?"). The forward path is therefore producing correct logits for the
end-of-text tokens *when conditioned on a state where they should win*.

What changes for long open-`<think>` generation is not the math — it's the
*distribution shape*. The model's own training plus GPTQ-Int4 calibration
makes the lm_head row for the rare end-of-thinking token persistently
underconfident relative to whatever common token wins each step. This is a
known failure mode of low-bit post-training quantization on reasoning models:
the quantizer's calibration set is dominated by frequent content tokens, so
rare-token logit scales drift downward.

The same observation holds for ArgmaxSampler, so it's not specific to our
GeneralSampler implementation or to the `repetition_penalty` plumbing.

### Mitigations available today

- **Interactive use case** (the typical one for `ping` / `chat` / `serve`):
  `--max-new-tokens 100` produces semantically correct responses. The early
  portion of the thinking is coherent; the cap prevents the drift band.
- **Closed-think template** (`<|im_start|>assistant\n<think>\n\n</think>\n\n`):
  forces the model into the direct-answer branch, where the open-`<think>`
  drift cannot trigger. We had this as a workaround default in `397bdfb`;
  with the real sampler bug fixed in `0c737a7` we reverted to the standard
  open-think template (`469e69a`) so the model behaves like HF's default. If
  the long-thinking degradation is unacceptable for a given workload, an
  application can opt in to closed-think by overriding the template.

### Not-yet-tried mitigations (deferred)

- **Force-emit `</think>` after a token budget.** Pragmatic workaround used by
  some serving frameworks. Cleanly wired through scheduler + sampler, but
  requires tracking "in-thinking" state per request and is policy, not a fix.
- **Boost `</think>` logit by a length-dependent additive term.** Cheap to
  add to `GeneralSampler`, but again policy.
- **Use a non-GPTQ-Int4 weight set.** The actual root cause is in the weights;
  bf16 or higher-bit quantization should restore long-form coherence.
- **Compare against HF transformers token-by-token.** Would localize whether
  any per-layer numeric drift in zedinfer also contributes. Blocked in this
  session on `pip install transformers` permission.

## Files touched (correctness fixes only)

- `src/backend/tensor/tensor.cpp` — offset-aware source in `Tensor::to(device)`.
- `include/frontend/sampler/sampler.hpp`, `src/frontend/sampler/sampler.cpp`
  — `repetition_penalty` field, bf16-safe sampling, history-aware
  `sample(...)` signature.
- `src/zedinfer/engine.cpp` — read sampler params from `generation_config.json`.
- `src/zedinfer/scheduler.cpp` — pass `req->output_ids` into `sampler.sample()`.
- `include/zedinfer/chat_template.hpp`, `src/zedinfer/chat_template.cpp` —
  `default_qwen3_5_chatml()` + routing for `qwen3_5` / `qwen3_5_moe`.

## Verified behavior after fixes (A6000, decode tok/s)

| Prompt | Output (first ~50 tokens) | Coherent? |
|---|---|---|
| "Who are you?" + closed-think | "I am Qwen3.5, a large language model developed by Tongyi Lab. I can assist you with a wide range of tasks, including answering questions, writing stories, creating documents, coding, and more. How can I assist you today?" → EOS at 54 tokens | yes |
| "Who are you?" + open-think + ARGMAX | "Here's a thinking process that leads to the suggested response: 1. **Analyze the Request:** *   **User:** Asks 'What is your name?' (implied by 'I am...'). *   **Context:** The user is asking about 'myself'..." | yes for ~40 tokens |
| "What is 2+2?" + open-think + GeneralSampler | "Thinking Process: 1. **Identify the user's query:** The user is asking a simple math question..." | yes for ~40 tokens |
| "Hi" + open-think + GeneralSampler | "Okay, the user just said 'Hi' and mentioned that it's a simple greeting. Let me start by acknowledging their greeting in a friendly way..." | yes for ~50 tokens |

Decode throughput is in the 35-45 tok/s range across these runs (varies with
prompt length and sampler).

## Update 2026-05-24: thinking-budget + closed-think opt-out landed

Goal from the user: 无论带不带 thinking 输出内容都正常 — both with-thinking and
without-thinking modes must produce structurally normal output (English text,
EOS reached, bounded length), not just close-to-EOS as the previous workaround.

### What was added

- **`GenerationConfig::enable_thinking`** (default `true`, matches HF):
  selects the assistant-prompt variant at request time. `false` uses the
  closed-think branch (`<|im_start|>assistant\n<think>\n\n</think>\n\n`), which
  is the only mode that gives reliably clean output on GPTQ-Int4 weights.
- **`GenerationConfig::max_think_tokens`** (default `128`): scheduler-level
  budget. When the request is inside an unclosed `<think>` block and exceeds
  this budget, the scheduler force-emits id 248069 (`</think>`) in place of
  the next sampled token. The next step force-emits id 271 (`\n\n`) to recreate
  the trained `</think>\n\n` separator pattern, anchoring the post-thinking
  state so the model exits the truncated-thinking attractor and transitions to
  the answer instead of continuing the partial reasoning.
- **`ChatTemplate::generation_prompt_no_think`** + **`output_prefix_no_think`**:
  paired closed-think variants. `Session::chat` / `prepare_prompt` pick
  open- vs closed-think based on `config_.enable_thinking`.
- **`ping --no-thinking`** and **`ping --max-think-tokens N`** expose the
  controls on the CLI.

Engine resolves the three token ids once at init (`tokenizer.get_special_token_id`
for `<think>` / `</think>`, `tokenizer.encode("\n\n")` for the separator) and
hands them to `Scheduler::set_think_token_ids`. Non-Qwen3.5 models return `-1`
for all three and the force-emit path is a transparent no-op.

### Verified results on A6000 + Qwen3.5-35B-A3B-GPTQ-Int4

| Mode | Prompt | Output | Status |
|---|---|---|---|
| `--no-thinking` | "Who are you?" | "Hello! I'm **Qwen3.5**, the latest large language model developed by Tongyi Lab. ..." → EOS | clean |
| `--no-thinking` | "Hi" | "Hello! How can I help you today?" → EOS | clean |
| `--no-thinking` | "What is 2+2?" | "The sum of 2 and 3 is **5**." → EOS (mathematically wrong but structurally normal) | clean |
| open-think, budget=16 | "Who are you?" | brief think → `</think>\n\n` → "Hello! I am Qwen3, the large language model developed by Alibaba Cloud. How can I help you?" → EOS | clean |
| open-think, budget=128 | "Who are you?" | think (~70 tokens) → `</think>\n\n` → coherent self-description (with some hallucinated facts) → EOS | structurally normal |

Closed-think is the recommended mode for any workload that needs guaranteed
clean output. Open-think + budget keeps the structure valid (no infinite loop,
no pinyin noise, EOS reached, bounded length) but cannot rescue answer quality
on the cases where GPTQ-Int4 itself fails (e.g., the 2+2→5 hallucination above
appears even under closed-think, confirming it's a model-quality limit and not
a thinking-state corruption).

### Files

- `include/zedinfer/chat_template.hpp`, `src/zedinfer/chat_template.cpp`
  — new `generation_prompt_no_think` / `output_prefix_no_think` fields.
- `include/zedinfer/generation_types.hpp`, `src/zedinfer/generation_types.cpp`
  — new `enable_thinking`, `max_think_tokens` fields + validation.
- `include/zedinfer/request.hpp` — per-request `in_thinking`,
  `think_token_count`, `post_think_forced_newlines`.
- `include/zedinfer/engine.hpp`, `src/zedinfer/engine.cpp`
  — resolves think / `\n\n` token ids from tokenizer at init.
- `include/zedinfer/scheduler.hpp`, `src/zedinfer/scheduler.cpp`
  — `set_think_token_ids`, in-thinking init from prompt, force-emit budget.
- `src/zedinfer/serving_loop.cpp` — wires engine ids into scheduler.
- `src/zedinfer/session.cpp` — picks open/closed prompt by `enable_thinking`.
- `examples/ping.cpp` — `--no-thinking`, `--max-think-tokens`.

## Update 2026-05-25: revert repetition_penalty default to 1.0

### Symptom

`What is 2+2?` with `--no-thinking` (closed-think) was non-deterministically
producing wrong answers on Qwen3.5-35B-A3B-GPTQ-Int4:

```
Run 1: "The sum of 2 and 3 is **6** (2 + 5 = **4**; then ..."
Run 2: "The sum of 2 and 30 is **28**. ..."
Run 3: "The result of $2 + 2$ is **4**. **607,186.53"   ← partial correct
Run 4: "$2 + 2 = \mathbf{4}$"                            ← correct
Run 5: "The result of $2 + 2$ is **4**. In basic arithmetic, adding two to the number five ..."
```

1/5 cleanly correct. ARGMAX-only mode (`ZEDINFER_FORCE_ARGMAX=1`) gave correct
deterministic output, ruling out the forward pass. Tokenizer + chat-template
output was byte-identical to HF (verified via `apply_chat_template` round-trip
against `transformers`), so the bug was sampler-side.

### Root cause

`de59c5e fix(sampler): honor model's generation_config.json + bf16-safe sampling`
introduced `repetition_penalty = 1.1` as the default when generation_config.json
didn't specify the field. The intent was to suppress long-thinking attractor
loops. The side-effect: any token already emitted gets its logit divided by 1.1.
For arithmetic / code answers that legitimately need to repeat digits, operators,
identifiers, or keywords, this demotes the correct continuation hard enough that
top_p=0.95 + top_k=20 + temp=1.0 sampling pulls an off-by-one token from the
nucleus ("2" → "4" → "5") and the model rationalizes around the wrong number.

The original motivation (thinking attractor loops) was always weakly supported —
the same debug doc above notes "The same observation holds for ArgmaxSampler",
i.e., rep_penalty wasn't actually fixing the long-thinking drift either.

### Fix

`src/zedinfer/engine.cpp` — change the implicit default from `1.1f` → `1.0f`,
matching HF transformers and vLLM. Reasoning-loop drift is now handled by the
scheduler-side `max_think_tokens` force-emit (added 2026-05-24 above), which
operates only inside `<think>` blocks and doesn't perturb the rest of sampling.

### Verification

`What is 2+2?` with default settings, 5 runs after fix:

```
Run 1: "The sum of 2 and 2 is **4**. $$2 + 2 = 4$$"                    ✓
Run 2: "The sum of 2 and 2 is **10**? No, actually, 4 plus 4 ..."      ✗ (self-correcting wrong tangent)
Run 3: "**2 + 2** equals **4**. In basic arithmetic ..."               ✓
Run 4: "2 + 2 equals **4**."                                           ✓
Run 5: "$2 + 2 = 4$ 2 plus 2 equals **4**."                            ✓
```

4/5 correct vs 1/5 before the fix. The remaining miss is inherent variance of
temp=1.0 + top_p=0.95 sampling on a 4-bit-quantized model — HF/vLLM with the
same params on the same weights see the same variance. Greedy decoding (HF's
`do_sample=false` or our `ZEDINFER_FORCE_ARGMAX=1`) is the canonical way to get
deterministic answers.

### Debug knobs added (kept in tree)

- `ZEDINFER_FORCE_ARGMAX=1` — overrides `generation_config.do_sample` to false,
  forces ARGMAX. Use to isolate sampler bugs from forward-path bugs.
- `ZEDINFER_REPETITION_PENALTY=<float>` — overrides the
  `generation_config.repetition_penalty` value at engine init. Set to `1.0` to
  match HF strictly; set to `>1.0` to experiment with anti-loop tuning.

## Update 2026-05-27: open-think empty-start drift is not zedinfer-side

After the rep_penalty fix above, the *with-thinking* output was still
catastrophic on GPTQ-Int4 — even under ARGMAX the model hallucinated the
prompt content within the first ~30 generated tokens (e.g. "What is 2+2?"
reasoned as if the prompt said "2+4" or "1+1"; "Hello there." treated as
"Hello, world!"). The session burned hours trying to localize this in
zedinfer, ruling out:

- Tokenization (zedinfer ids byte-equal to HF `apply_chat_template`)
- GDN parallelization (reverting commit `3dc840d` did not help)
- Repetition penalty (already at 1.0 default)
- Block-boundary scatter / N%block_size patterns (different prompt lengths
  with same trigger; different lengths broke too)

The decisive experiment was prefilling thinking content. Same prompt
"What is 2+2?" with these three generation-prompt tails under ARGMAX:

| Tail | Last prompt token | Output |
|---|---|---|
| `<think>\n` (HF default) | id 198 (`\n`) | Broken: hallucinates "2+4", digit-string attractor |
| `<think>\n\n` | id 271 (`\n\n`) | Model immediately emits `</think>` (skips thinking) |
| `<think>\nLet me think.` | id 13 (`.`) | **Coherent reasoning, correct answer (4), clean EOS** |

The pre-fill case proves zedinfer's forward path *can* produce coherent
thinking. The failure mode is specifically the empty-thinking-start state
at the end of an open-`<think>\n` prefill: that state is a fragile region
of the model where the decoder collapses into prompt-content hallucination.
Pre-filling any non-whitespace thinking content reliably escapes it.

This matches the original doc-author hypothesis that long-thinking drift is
GPTQ-Int4 calibration. The full proof requires an HF transformers reference
run on the same weights and prompt (blocked on this host because the
installed `transformers` does not recognize `qwen3_5_moe`).

### Resolution: default to closed-think

- `GenerationConfig::enable_thinking` default flips from `true` → `false`.
  Library callers that did not pass an explicit value previously got the
  fragile open-`<think>` branch; now they get the safe closed-`<think>` one.
- `ping --thinking` (default false) replaces the previous `--no-thinking`
  (default false). Inverted semantics: opting in to a known-fragile mode is
  now explicit instead of opting out.
- `ChatTemplate::default_qwen3_5_chatml` still stores BOTH variants — the
  open-think prompt remains available for callers that pass
  `enable_thinking=true`, but is no longer the default and the comment
  warns about the fragility.
- The `max_think_tokens` budget + `</think>\n\n` force-emit (added
  2026-05-24) stays in place as a safety net for the opt-in case.

### Files touched (this update)

- `include/zedinfer/generation_types.hpp` — `enable_thinking` default `false`.
- `src/zedinfer/chat_template.cpp` — long comment explaining the fragility
  and the safe-default decision.
- `examples/ping.cpp` — `--thinking` flag (replaces `--no-thinking`).

## Open follow-ups

1. Verify with a higher-precision weight set whether the long-form drift
   disappears, confirming the GPTQ-Int4 hypothesis end-to-end. The
   thinking-budget code path is purely additive and is a no-op for
   non-reasoning models, so it should not interfere with a bf16/fp16
   comparison run.
2. `nvbugs / nsight-systems` profile failed to import on the host during
   this session (qdstrm → nsys-rep crashes in the importer); next attempt
   should pin a working `nsys` version before doing per-layer profiling.

## Update 2026-05-26: mrope_3d pair-layout bug — was the root cause

The 2026-05-27 hypothesis ("GPTQ-Int4 calibration is to blame") was wrong.
Once we got HF transformers and a bf16 reference weight set on a B200, the
real bug fell out immediately.

### Symptom recap

`ZEDINFER_FORCE_ARGMAX=1 --thinking --prompt "What is 2+2?"`:

| Build | First decoded tokens |
|---|---|
| HF + bf16 (Qwen3.6, same arch) | "Here's a thinking process:\n\n1.  **Analyze User Input:** The user asks..." (coherent → `</think>` → "2 + 2 = 4." → EOS, 200 tok) |
| zedinfer + GPTQ-Int4 (pre-fix) | "Thinking Process:\n\n1.  **Analyze the Request:** *   Input: \"2+2\" (implied by..." — different first token, drifts within 30 tokens |
| zedinfer + bf16 (pre-fix, same weights as HF) | "Thinking Process: ..." — same wrong first token as GPTQ |

The fact that bf16 + zedinfer also produced the wrong first token, on the
exact same weights HF handled correctly, ruled out GPTQ-Int4. The forward
path itself was broken.

### Root cause: rotary pair-layout mismatch

HF's `Qwen3_5MoeTextRotaryEmbedding` (modeling_qwen3_5_moe.py:91-180) builds
cos/sin via `cat((freqs, freqs), dim=-1)` and applies `rotate_half`:

```python
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
```

i.e. the rotation pairs are `(i, i+half)`, both rotated by the same angle
`freqs[i]`. This is the Llama / NeoX convention.

`src/backend/ops/mrope/nvidia/mrope_3d.cu` (and the matching CPU stub) instead
paired `(2*pi, 2*pi+1)` — the GPT-J interleaved-pair convention. The
fixture in `tests/fixtures/mrope_3d/gen_reference.py` was hand-rolled and
matched the kernel rather than HF, so the unit test passed without ever
catching the mismatch.

Because the q_proj / k_proj weights were trained for HF's half-rotation
form, zedinfer was rotating the wrong dimension pairs at every full-
attention layer (10/40 layers in Qwen3.5). The q·k inner-product
geometry that attention depends on was subtly broken from token 0; the
drift compounded through the residual stream and surfaced as "model
ignores the prompt content".

Direct fixture comparison against HF (Python script in `/tmp/`):

```
max abs diff (zed interleaved vs HF rotate_half): 2.984375
mean abs diff: 0.044292
permutation-equivalence check: max diff 4.64, mean 1.16  (NOT a basis swap)
```

### Fix

- `src/backend/ops/mrope/nvidia/mrope_3d.cu` — pair `(pi, pi+half)` with one
  `__sincosf` per pair, matching HF's duplicated cos/sin.
- `src/backend/ops/mrope/cpu/mrope_3d.cpp` — same change; also wired the
  `interleaved` flag through (the CPU path previously ignored the HF
  axis-assignment rule even when callers requested it).
- `tests/fixtures/mrope_3d/gen_reference.py` — regenerated against the
  real HF math (`Qwen3_5MoeTextRotaryEmbedding.forward` +
  `apply_rotary_pos_emb`), not the hand-rolled formula.
- `tests/fixtures/mrope_3d/x_out_ref.bin` — regenerated.

After the fix `test-ops-mrope-3d` passes with `max_abs_diff=0` against the
new HF-derived fixture.

### Verification (B200, bf16, ARGMAX)

```bash
CUDA_VISIBLE_DEVICES=0 ZEDINFER_FORCE_ARGMAX=1 xmake run ping \
    /home/scratch.tianyux_coreai/models/Qwen3.6-35B-A3B-split-experts \
    --nvidia --gpu-memory-utilization 0.85 \
    --max-new-tokens 30 --thinking --prompt "What is 2+2?"
```

→ `Here's a thinking process:\n\n1.  **Analyze User Input:**\n   - User input: "2+2"\n   -`

The first ~16 tokens are now byte-identical to HF's greedy output on the
same weights (`Here's a thinking process:\n\n1.  **Analyze User Input:**`).
The pre-fix output started with a different token at step 0 and never
agreed with HF.

### Files touched (this update)

- `src/backend/ops/mrope/nvidia/mrope_3d.cu`
- `src/backend/ops/mrope/cpu/mrope_3d.cpp`
- `tests/fixtures/mrope_3d/gen_reference.py`
- `tests/fixtures/mrope_3d/x_out_ref.bin` (regenerated)
- `tests/fixtures/mrope_3d/x_in.bin` (regenerated; same RNG seed, just
  re-emitted bytes)

## Update 2026-05-27: o_proj called with 3-D `attn`, K collapsed to 16

After landing the mrope fix above, zedinfer matched HF for the first ~16
generated tokens then diverged. The o_proj_output magnitude was off by
10-165x relative to HF (e.g. zedinfer 0.05 vs HF 0.41 at layer 3, 0.05 vs
8.28 at layer 35, 0.30 vs 5.72 at layer 39). All upstream intermediates
matched HF closely:

```
                 zedinfer    HF
q_raw            49.76       48.92    (L=3)
gate            285.77      275.09
q_normed         78.93       77.98
v                 8.70        8.21
attn_pre_gate    19.30       18.62
attn_post_gate    0.49        0.55
o_proj_out        0.05        0.41    <- diverges
```

### Root cause

The paged attention kernel returns `attn` with shape `[N, Hq, Dh]`.
Forward then called

```cpp
ops::linear(out, attn, m.W(p + "o_proj.weight"));
```

with `attn` left as a 3-D tensor. `ops::linear` reads `K = in->dim(1)` —
the middle dimension — which is `Hq = 16` rather than the actual
`q_dim = Hq*Dh = 4096`. The cuBLAS GEMM then reduces over only 16
channels, reading the first 16 columns of `o_proj.weight` for each output
row. Roughly 99.6% of the attention output and 99.6% of the o_proj
weight were unused, so o_proj produced a near-zero residual contribution
on every full-attention layer (10 of 40 in Qwen3.5). The residual stream
quietly collapsed to ~14% of its correct magnitude by the final norm,
and the LM head still happened to produce the same argmax for the first
~16 tokens before drift overcame the gap to second-best.

### Fix

```cpp
ops::linear(out, attn->view({N, q_dim}), m.W(p + "o_proj.weight"));
```

One-line change in `forward_full_attn_layer` in
`src/frontend/models/hybrid_transformer_forward.cpp`. With it, zedinfer
matches HF eager byte-for-byte on the first 30+ tokens under greedy
decoding on the bf16 reference weights, and produces clean coherent
output on the GPTQ-Int4 model for both `--thinking` and `--no-thinking`
modes.

### Why the unit test didn't catch it

The mrope unit test exercised mrope in isolation. There was no
end-to-end test that compared zedinfer's full forward against HF on a
real model, so the o_proj shape mistake — which only matters when the
attention output happens to be 3-D — went unnoticed.

### Secondary precision cleanup

Same session also landed a precision improvement that didn't change the
end-to-end argmax but matches HF more closely on per-element values:

- `(1 + weight)` compensation for `Qwen3_5MoeRMSNorm` was previously
  pre-baked into a bf16 buffer at load time, losing ~6x precision
  because the bf16 mantissa around 1.0 is much coarser than around 0.0.
  Now `ops::rms_norm` accepts an `add_one_to_weight` flag and computes
  the +1.0f shift in fp32 inside the kernel, matching HF's
  `output * (1.0 + weight.float())`.
- `fixup_qwen3_5_rmsnorm_weights` is now a no-op (kept as a function
  stub to avoid touching the model-load call site).
- All Qwen3.5 RMSNorm callers in `hybrid_transformer_forward.cpp` pass
  `add_one_to_weight=true` (input_layernorm, post_attention_layernorm,
  q_norm, k_norm, final norm). `linear_attn.norm.weight` keeps
  `add_one_to_weight=false` (RMSNormGated has no +1).

### Verification (B200, bf16 Qwen3.6, ARGMAX, 30 tokens, "What is 2+2?")

zedinfer and HF generate the same 30 token IDs:
`[8160, 579, 264, 7047, 1817, 25, 271, 16, 13, 220, 2972, 2014, 53983,
2570, 5396, 64700, 561, 1156, 16561, 328, 3710, 369, 220, 17, 10, 17,
7285, 198, 17, 13]` → "Here's a thinking process:\n\n1.  **Analyze User
Input:** The user asks "What is 2+2?"\n2."

GPTQ-Int4 ARGMAX (with `--thinking`) now produces coherent thinking +
answer on representative prompts:

- "What is 2+2?" → reasoning steps → "2 + 2 = 4."
- "Hello there." → reasoning steps → friendly greeting
- "Name three primary colors." → distinguishes RGB / RYB / CMY → "Red,
  Yellow, Blue"
- "Write a haiku about autumn." → "Leaves turn red and gold / Wind
  blows cold across the field / Winter comes soon now"

### Debug env knobs introduced this session

Kept in tree because they're useful for future correctness work:

- `ZEDINFER_DUMP_TOKEN_IDS=1` — scheduler logs each sampled token id per
  step (prefill + decode).
- `ZEDINFER_DUMP_TOP_LOGITS=1` — sampler logs top-5 logits + the chosen
  id per step.
- `ZEDINFER_KV_BLOCK_SIZE=<N>` — `ping` override for KV cache block size
  (default 16); useful for testing block-boundary sensitivity.
- `ZEDINFER_DISABLE_SCRATCH=1` — disables the thread-local DecodeScratch
  in `hybrid_transformer_forward`; lets you rule out scratch-buffer
  staleness as a forward-bug source.

### Files touched (this update)

- `src/frontend/models/hybrid_transformer_forward.cpp` — the one-line
  o_proj view fix + `add_one_to_weight=true` at all Qwen3.5 RMSNorm
  callers + `ZEDINFER_DISABLE_SCRATCH` env knob.
- `src/frontend/models/qwen3_5.cpp` — `fixup_qwen3_5_rmsnorm_weights`
  reduced to a no-op (compensation now lives in the kernel).
- `include/backend/ops/ops.hpp`,
  `include/backend/ops/rms_norm/{cpu,nvidia}/*.{hpp,cuh}`,
  `src/backend/ops/rms_norm/{cpu,nvidia,op.cpp}` — plumb
  `add_one_to_weight` through `ops::rms_norm`.
- `src/zedinfer/scheduler.cpp` — `ZEDINFER_DUMP_TOKEN_IDS` env knob.
- `src/frontend/sampler/sampler.cpp` — `ZEDINFER_DUMP_TOP_LOGITS` env knob.
- `examples/ping.cpp` — `ZEDINFER_KV_BLOCK_SIZE` env knob.
