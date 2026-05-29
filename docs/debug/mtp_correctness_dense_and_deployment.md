# MTP: correctness fix, dense-27B support, and the deployment verdict

**Date:** 2026-05-29
**Branch:** `feat/qwen3.5`
**Models:** Qwen3.5/3.6-35B-A3B-GPTQ-Int4 (MoE), Qwen3.5/3.6-27B-GPTQ-Int4 (dense), Qwen3.5-35B-A3B (bf16)
**Hardware:** NVIDIA B200 (umbriel-b200-027/034), single GPU, ALL_GPU unless noted
**Commits:** `612b933` → `947c55e` → `4da999c` → `49aaf27` (continues `stage_f_mtp_perf_pass.md`)

## TL;DR

1. **MTP spec decode was producing corrupt output** (~40% of greedy tokens were
   `*`). Root cause: the hybrid linear-attention recurrent state (GatedDeltaNet
   matrix + causal-conv window) is advanced by the rejected draft and was never
   rolled back. Fixed two ways (rollback, then a temp-slot that makes reject
   free). Output is now correct.
2. **Whether MTP is a net speedup is architecture × workload, not a kernel-quality
   question.** Dense models win (shared weights → cheap verify); MoE does not
   (disjoint experts → ~2× verify). Predictable workloads win (high accept rate);
   creative ones do not.
3. **First net speedup: dense Qwen3.5-27B, +11.5% on counting** (71.1 → 79.3
   tok/s), break-even on essay. MoE 35B-A3B never wins.
4. **Deployment verdict for a 24GB 4090:** only the 35B-A3B runs (via MoE expert
   offload), and that is exactly the config where MTP does **not** help; the
   MTP-friendly 27B (28GB, dense → no offload path) does not fit. MTP is not a
   lever for the 4090 target.

## 1. Correctness bug + fix

Greedy decode with `--mtp` on Qwen3.5-35B-A3B-int4 emitted ~40% token id 9
(`*`); output diverged completely from the no-MTP baseline.

**Root cause.** The 2-token `[last_token, draft]` verify forward advances the
recurrent linear-attention state in place by BOTH tokens (`gdn`/`causal_conv1d`,
`hybrid_transformer_forward.cpp` `forward_linear_attn_layer`). On reject the
paged KV cache self-heals (the next 1-token step overwrites the draft slot), but
the recurrent state has no positional addressing and was not rolled back — every
reject poisoned all later tokens. Bisection (`ZEDINFER_DISABLE_SMALL_NQ=1
ZEDINFER_DISABLE_FLASHINFER=1` forcing the full-attn n_q=2 onto `paged_prefill`)
reproduced the corruption identically, exonerating the full-attn kernels and
confirming the SSM state.

**Fix A — rollback (`612b933`).** `serving_loop` snapshots the SSM slot
(`SSMStatePool::snapshot_slot`) before a verify; on reject `scheduler` restores
it, emits nothing, and the next iteration redoes `last_token`. Correct but slow
(per-reject redo + per-step D2H): **12.8 tok/s**.

**Fix B — temp slot (`947c55e`, supersedes A at runtime).**
`forward_linear_attn_layer` commits token0 (last_token) into the request's real
SSM slot and computes token1 (draft) into a dedicated pool temp slot
(`SSMStatePool::spec_temp_slot`). The real slot is left at the post-last_token
state, so **reject is free** (just emit the corrected token) and **accept
promotes temp→real** (`copy_slot_state`). **24.6 tok/s**, token-9 = 0.

**Residual.** Greedy MTP is not always byte-identical to baseline — rare
near-tie argmax flips where the n_q=2 verify kernels (`gdn_prefill`/`small_nq`)
differ numerically from the n_q=1 decode kernels. This is standard spec-decode
non-determinism, not a bug; on confident workloads (counting) output is in fact
byte-identical.

## 2. Accept rate: not quantization, not the acceptance scheme — it's the workload

`4da999c` adds textbook rejection sampling (draft drawn from the MTP truncated
dist q; accept with prob `min(1, p(d)/q(d))`; reject resamples the residual
`normalize(max(0,p-q))`) in `GeneralSampler`. It aligns with the literature but
the accept rate is **unchanged** (~13%), because accept rate = `1 − TV(p, q)` and
the 1-layer MTP head's q is genuinely far from the target p.

- **Not quantization:** bf16 essay accept ~13% ≈ int4 ~15%.
- **Not the scheme:** greedy exact-match ~15%, sample==draft ~15%, rejection
  ~13% — all the same.
- **It is the workload** (greedy accept by prompt): counting ~64%, US-states
  list ~50%, JSON ~27%, code ~22%, essay ~14%, verbatim repeat ~10%.
- Stage F's "~50%" was almost certainly inflated by the (now-fixed) corruption
  bug making both main and MTP degenerate to `*`.

## 3. Why dense wins and MoE doesn't (the core insight)

Decode is memory-bandwidth-bound (dominated by reading weights). MTP's win needs
the 2 verify tokens to **share** the weight read.

- **Dense (27B):** both tokens use the same FFN/attention weights → n_q=2 reads
  the weights once → amortized. Measured verify cost **~1.25×** a normal forward.
- **MoE (35B-A3B):** the 2 tokens route to mostly disjoint top-8 experts (256
  experts, expected overlap ~0.25) → ~2× distinct experts → ~2× weight traffic,
  no amortization. Verify cost **~2×**.

`speedup = (1 + accept) / verify_cost`. At verify=2× even 64%-accept counting is
0.82× (loss); at verify=1.25× counting is 1.3×+ (win). **This is fundamental,
not a kernel bug**: even a perfect grouped/fused MoE keeps the 2× expert-weight
traffic at n_q=2. (Separately, the MoE *baseline* is launch-bound — nsys: expert
int4 matvec ~3.3µs × ~95K launches/run, GPU ~70% idle — which a fused MoE would
fix on an ALL_GPU box, but that is orthogonal to the verify ratio.) The
attention `small_nq` kernel the verify uses is only ~2.6% of the forward.

## 4. Dense-27B MTP support (`49aaf27`) — first net speedup

The 27B (model_type `qwen3_5`, dense hybrid) also ships an MTP head
(`mtp_num_hidden_layers=1`) but with a **dense FFN** MTP layer
(`mtp.layers.0.mlp.{gate,up,down}_proj`) instead of experts. Wiring:

- `MTPModule` now takes the base `Qwen3_5Config` and detects MoE vs dense from
  the weights (`mtp.layers.0.mlp.gate.weight` ⇒ MoE router). MoE path unchanged
  (reads expert fields via `static_cast` to `Qwen3_5MoEConfig`); dense path loads
  gate/up/down_proj and runs `mtp_dense_ffn_one_token`.
- `mtp_` / `mtp_module()` moved to the base `Qwen3_5Model`; the base ctor builds
  the dense head, the MoE subclass keeps building the expert head.
- `serving_loop` drives MTP via the base `mtp_module()` + base config, so dense
  and MoE share the verify/accept path (temp-slot rollback + rejection sampling).

**Measured (Qwen3.5-27B-int4, B200 ALL_GPU, greedy, 160 tokens):**

| Prompt | baseline | +MTP | speedup | accept | token-9 |
|---|---|---|---|---|---|
| Counting | 71.1 tok/s | **79.3** | **+11.5%** | 39.5% | 0 (byte-identical to baseline) |
| Essay | 65.5 tok/s | 65.6 | break-even | 24.2% | 0 |

Back-solved verify cost ≈ 1.25×, matching §3.

## 5. Deployment verdict: "int4" is partial, and the 4090 catch-22

GPTQ here quantizes **only the FFN/experts**; `linear_attn` (GatedDeltaNet,
explicitly excluded by Qwen3.5 quant rules), full-attn projections,
`embed_tokens`, and `lm_head` all stay bf16. Measured tensor bytes:

| Model | int4 | bf16 | total |
|---|---|---|---|
| 35B-A3B (MoE) | 15.3 GB (experts) | 7.4 GB | **22.7 GB** |
| 27B (dense) | 8.0 GB (FFN) | 19.9 GB (incl. 10.4 GB linear_attn) | **28.2 GB** |

On a 24GB 4090:
- **35B-A3B runs** via the MoE `PINNED_LRU` expert offload (non-expert ~7GB +
  GPU expert slots + KV fit; experts stream from host).
- **27B does not fit** — 28.2GB, and dense has **no offload path** in ZedInfer
  (`ExpertPool` is MoE-only).

**The catch-22:** the only 4090-deployable model (35B-A3B + offload) is exactly
where MTP does not help — offload makes verify ~2× via H2D (more verify tokens =
more experts to stage), and the workload is H2D-bandwidth-bound, so compute-side
levers (MTP, grouped/fused MoE) do nothing. The MTP-friendly model (27B dense)
does not fit the 4090.

To make MTP useful on a 4090 you would have to re-quantize the 27B's bf16 parts
to full int4 (~13 GB → fits ALL_GPU) — custom quant work, and quantizing
`linear_attn` (GatedDeltaNet recurrent state) is quality-risky. On a ≥32GB GPU
the int4 MoE fits ALL_GPU and a fused MoE could raise its baseline, and the dense
27B MTP already works as in §4.

## FlashInfer note

`third_party/flashinfer/csrc/fused_moe/cutlass_backend` has a **W4 group-scaled
fused MoE** (`use_w4_group_scaling`, `kINT4`/uint4x2, `CutlassMoeFCRunner`,
namespace `tensorrt_llm::kernels::cutlass_kernels`) that matches GPTQ-int4 in
principle and is callable from C++. It only helps the ALL_GPU case (it does not
stream experts), so it is irrelevant to the 4090/offload deployment.

## Conclusion / what's worth doing next

- The correctness fix stands on its own (MTP was emitting garbage).
- MTP is a real speedup only on **dense + structured/agentic workloads + ALL_GPU**.
- For the 24GB-4090 target, MTP and fused-MoE are dead ends; the only lever for
  the deployable 35B-A3B-offload path is the **offload H2D itself** (transfer
  size/coalescing, prefetch depth, LRU hit rate) — a separate track.

References: `stage_e_mtp_spec_decode_perf.md`, `stage_f_mtp_perf_pass.md`,
`qwen3_5_mtp_spec_decode.md`. Test scripts/logs: `logs/mtp_test/`.
