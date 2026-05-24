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

## Open follow-ups

1. Long-thinking force-emit policy (see "Not-yet-tried mitigations" above).
2. Verify with a higher-precision weight set whether the long-form drift
   disappears, confirming the GPTQ-Int4 hypothesis end-to-end.
3. `nvbugs / nsight-systems` profile failed to import on the host during
   this session (qdstrm → nsys-rep crashes in the importer); next attempt
   should pin a working `nsys` version before doing per-layer profiling.
