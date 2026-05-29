# Decouple FlashInfer Plan State from `SequenceBlockTable`

> Date: 2026-05-29
> Status: **Option A implemented (2026-05-29)** — FlashInfer fields grouped into `FlashInferSeqCache fi`;
> `clear_runtime_caches()` is now `fi = {}`; copy carries only logical state. Validated byte-identical across
> single-decode / batched-decode (seeded batch_bench) / prefill (ppl). Option B (move ownership out of the
> table entirely) remains optional future work.
> Scope: `include/backend/kvcache/block_pool.hpp`, `src/frontend/models/paged_forward_context.cpp`
> Risk: medium (pure relocation, no numerical change; but the FlashInfer caches feed every attention call
> across single-decode / batched-decode / prefill, so all three paths must be validated)

---

## 1. Problem (with code evidence)

`SequenceBlockTable` (`include/backend/kvcache/block_pool.hpp:31-97`) is the *logical* KV page table — it
should describe only "logical page → physical page" mappings (`pages`, `seq_len`, `num_layers`). Instead it
embeds ~20 FlashInfer-specific runtime-cache fields (lines 37-58): `flashinfer_single_decode_plan`,
`flashinfer_page_tables_gpu`, and a flat set of plan-invalidation keys
(`flashinfer_single_decode_plan_nhead/_head_dim/_dtype/_device_id/_total_pages/...`).

Two concrete problems:

1. **Abstraction leak / coupling.** The kvcache layer now depends on the FlashInfer attention backend's
   plan representation. Every `SequenceBlockTable` (one per request) carries this state even on CPU or
   non-FlashInfer builds.

2. **Fragile copy semantics.** The hand-written copy constructor and copy assignment
   (`block_pool.hpp:61-71`) copy **only** `pages` / `seq_len` / `num_layers` and silently drop everything
   else (the caches are meant to be rebuilt, not shared). `clear_runtime_caches()` (lines 75-96) then has to
   reset each of the ~20 fields by hand. Adding a new *logical* field means remembering to update **both**
   the copy ctor and the assignment operator, or it is silently lost on copy — a latent correctness trap.

The fields are referenced from exactly two places: `block_pool.hpp` (definition) and
`paged_forward_context.cpp` (the only consumer).

---

## 2. Goal

`SequenceBlockTable` describes only logical mapping and is trivially/defaulted-copyable. FlashInfer plan
caching lives in a clearly-named, self-resetting unit so that (a) the kvcache type no longer hand-maintains
backend cache state and (b) adding a logical field can't silently drop on copy.

---

## 3. Proposed design

### Option A (smaller, recommended first step): group into a self-resetting sub-struct

```cpp
struct FlashInferSeqCache {
    std::vector<tensor_t> page_tables_gpu;
    tensor_t single_decode_kv_indptr_gpu, single_decode_kv_last_page_len_gpu,
             single_decode_qo_indptr_gpu, single_decode_descriptor_gpu;
    ops::FlashInferDecodePlan single_decode_plan;
    bool   single_decode_plan_ready = false;
    int    single_decode_plan_total_pages = -1, single_decode_plan_nhead = 0, ...;
    // all invalidation keys live here, each with a default member initializer
};

struct SequenceBlockTable {
    std::vector<std::vector<int>> pages;
    int seq_len = 0;
    int num_layers = 0;
    FlashInferSeqCache fi;                       // rebuilt on demand, never shared across tables

    SequenceBlockTable() = default;
    // Copy: carry logical state, reset the (table-specific) GPU cache.
    SequenceBlockTable(const SequenceBlockTable& o)
        : pages(o.pages), seq_len(o.seq_len), num_layers(o.num_layers) {}
    SequenceBlockTable& operator=(const SequenceBlockTable& o) {
        pages = o.pages; seq_len = o.seq_len; num_layers = o.num_layers;
        fi = FlashInferSeqCache{};               // one line — can't miss a field
        return *this;
    }
    SequenceBlockTable(SequenceBlockTable&&) noexcept = default;
    SequenceBlockTable& operator=(SequenceBlockTable&&) noexcept = default;
};
```

Wins: `clear_runtime_caches()` becomes `fi = {};`; the logical/cache split is explicit; adding a *cache*
field is auto-reset; the copy ctor lists only the 3 logical fields (and they're the only logical fields).

### Option B (larger, eventual): move ownership out of the table entirely

Keep `SequenceBlockTable` purely logical (default copy/move) and store the FlashInfer plan cache in a
side-table owned by `PagedForwardContext`, keyed by sequence. Removes the kvcache→FlashInfer dependency
completely. More invasive (the context must key/look up per-sequence cache); do it after Option A proves out.

---

## 4. Migration steps

1. Introduce `FlashInferSeqCache`, move the ~20 fields into it, replace `clear_runtime_caches()` body with
   `fi = {};`, simplify copy ctor/assignment.
2. Update the access sites in `paged_forward_context.cpp` (`table.flashinfer_X` → `table.fi.X`). Mechanical.
   Note: many `flashinfer_*` identifiers in that file are `PagedForwardContext` members (trailing `_`) or
   free functions — only the `SequenceBlockTable` fields move. Disambiguate carefully.
3. Build; run the attention paths (see §5) and confirm identical output.

---

## 5. Risks & validation

- **No numerical change intended** — this is a pure relocation of cache state. Validate with the greedy
  fingerprint (DeepSeek-R1-Distill-Qwen-1.5B exercises the FlashInfer single-decode path):
  baseline `2b2c1e5121e3d94940cffafdfb9c7af1` (48 tokens) must be unchanged.
- Also validate **batched decode** (multiple concurrent requests) and **prefill** (long prompt), which use
  different cache fields than single-decode — run a 2-3 request `serve`/`batch_bench` smoke and a long-prompt
  `ppl` chunk.
- The identifier-name overlap (table fields vs context members) is the main hazard; prefer a rename in step 2
  (`table.fi.single_decode_plan`) so the two namespaces can't be confused again.
