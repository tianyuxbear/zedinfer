# R3: Clean KVCache and Model Interface - Detailed Refactoring Plan

**Goal:** After R1 (unified forward loop) and R2 (unified KV management), significant dead weight remains in the Model and KVCache interfaces. R3 removes dead methods, collapses unnecessary inheritance, and makes Model a pure data holder.

---

## 1. Current State Analysis (Post R1+R2)

### 1.1 KVCache Base Class — Only One Subclass

```cpp
class KVCache {                           // 8 pure virtual + 2 virtual methods
    virtual int allocated_capacity() = 0;
    virtual tensor_t get_k_cache(int) = 0;
    virtual tensor_t get_v_cache(int) = 0;
    virtual tensor_t get_k_cache_slice(int, int) = 0;
    virtual tensor_t get_v_cache_slice(int, int) = 0;
    virtual tensor_t get_k_cache_slice(int, int, int) = 0;
    virtual tensor_t get_v_cache_slice(int, int, int) = 0;
    virtual void update_seq_len(int);
    virtual void reset();
    virtual size_t memory_usage() = 0;
    virtual float utilization() = 0;
};

class DynamicKVCache : public KVCache { ... };  // Only subclass
```

`KVCache` is a polymorphic base class with only one implementation. The virtual dispatch serves no purpose. `DynamicKVCache` is only used by `warmup()` and `profile()`.

### 1.2 Model Class — Forward Methods Are Redundant Thin Wrappers

After R1, `Qwen2Model::forward()` and `forward_batch()` are each 3 lines:

```cpp
tensor_t Qwen2Model::forward(input_ids, past_len, kvcache, exec_config) {
    ModelForwardConfig cfg{config_, *weights_, true, false};
    ContiguousForwardContext ctx(input_ids, past_len, kvcache);
    return transformer_forward(cfg, ctx, exec_config);
}

tensor_t Qwen2Model::forward_batch(batch, allocator, exec_config) {
    ModelForwardConfig cfg{config_, *weights_, true, false};
    PagedForwardContext ctx(batch, allocator);
    return transformer_forward(cfg, ctx, exec_config);
}
```

The only model-specific information is `{true, false}` — the `has_qkv_bias` and `has_qk_norm` flags. Everything else is identical between Qwen2 and Qwen3.

The engine already knows the model type (`model_->model_type()`) and constructs `ModelForwardConfig` in `step()`. So the model's forward methods are just boilerplate.

### 1.3 Dead Virtual Methods on Model

```cpp
// These were for graph-based execution (deleted in PR-2). Never called.
virtual std::string get_embedding_weight_name() const = 0;
virtual std::string get_output_norm_weight_name() const = 0;
virtual std::string get_output_weight_name() const = 0;
virtual std::vector<std::string> get_layer_weight_names(int layer_idx) const = 0;
```

Confirmed: grep shows these are defined in qwen2.cpp/qwen3.cpp but never called from anywhere.

### 1.4 Summary of Dead/Redundant Code

| Component | Lines | Status |
|-----------|-------|--------|
| KVCache base class (8 pure virtual) | ~50 | Only 1 subclass — polymorphism unnecessary |
| `KVCache::update_seq_len` virtual | 5 | Was made virtual for PagedKVCache (deleted) |
| `Model::forward()` on Qwen2/Qwen3 | 4×2=8 | Thin wrappers, engine could call transformer_forward directly |
| `Model::forward_batch()` on Qwen2/Qwen3 | 4×2=8 | Same |
| `Model::get_*_weight_name()` | 4×2=8 declarations + 40 lines impl | Never called |
| `Model::forward()` pure virtual declaration | 5 | Only used by warmup/profile |
| `Model::forward_batch()` virtual declaration | 5 | Only used by step() |

---

## 2. Target Design

### 2.1 DynamicKVCache: Standalone, No Base Class

```cpp
// Rename/simplify: DynamicKVCache is a standalone class, no KVCache base.
// Only used by warmup() and profile().
class DynamicKVCache {
public:
    DynamicKVCache(const DynamicKVCacheConfig &config);

    int current_length() const;
    int allocated_capacity() const;
    tensor_t get_k_cache_slice(int layer_idx, int total_len);
    tensor_t get_v_cache_slice(int layer_idx, int total_len);
    tensor_t get_k_cache_slice(int layer_idx, int past_len, int seq_len);
    tensor_t get_v_cache_slice(int layer_idx, int past_len, int seq_len);
    void update_seq_len(int new_tokens);
    void reset();
    // No virtual, no base class, no overhead.
};
```

`ContiguousForwardContext` takes `DynamicKVCache&` directly (not `KVCache&`).

### 2.2 Model: Data Holder + ModelForwardConfig Factory

The Model class becomes a config/weights container with a method to produce
its `ModelForwardConfig`:

```cpp
class Model {
public:
    virtual ~Model() = default;

    virtual const ModelConfig &config() const = 0;
    virtual const ModelWeights &weights() const = 0;
    virtual std::string model_type() const = 0;
    virtual size_t num_parameters() const = 0;

    // Produce the forward config for this model family
    virtual ModelForwardConfig forward_config() const = 0;

    // Static factory
    static std::shared_ptr<Model> parse(const std::string &path, zedinferDeviceType_t device);
};

// Qwen2Model
class Qwen2Model : public Model {
    ModelForwardConfig forward_config() const override {
        return {config_, *weights_, /*has_qkv_bias=*/true, /*has_qk_norm=*/false};
    }
};
```

`forward()` and `forward_batch()` are **removed from Model**. The engine
calls `transformer_forward()` directly with the model's `forward_config()`:

```cpp
// warmup:
auto cfg = model_->forward_config();
ContiguousForwardContext ctx(dummy, 0, tmp_kv);
transformer_forward(cfg, ctx, exec_config_);

// step (batch):
auto cfg = model_->forward_config();
PagedForwardContext ctx(batch, allocator);
transformer_forward(cfg, ctx, exec_config_);

// run_one (session):
auto cfg = model_->forward_config();
PagedForwardContext ctx(input_ids, past_len, block_table, pool);
transformer_forward(cfg, ctx, exec_config_);
```

### 2.3 Weight Name Methods: Deleted

`get_embedding_weight_name`, `get_output_norm_weight_name`,
`get_output_weight_name`, `get_layer_weight_names` — all deleted.
They were for graph construction which was removed in PR-2.

---

## 3. Files Affected

### Modified

| File | Change |
|------|--------|
| `include/backend/kvcache/base.hpp` | Remove `KVCache` base class. Keep `KVCacheConfig` struct only. |
| `include/backend/kvcache/dynamic.hpp` | `DynamicKVCache` no longer inherits `KVCache`. Standalone class. |
| `src/backend/kvcache/dynamic.cpp` | Remove `KVCache` base calls, standalone implementation. |
| `src/backend/kvcache/base.cpp` | Delete or reduce to just `KVCacheConfig::validate()` |
| `include/frontend/models/base.hpp` | Remove `forward()`, `forward_batch()`, weight name methods. Add `forward_config()`. |
| `src/frontend/models/base.cpp` | Remove `forward_batch()` default impl |
| `include/frontend/models/qwen2.hpp` | Remove `forward()`, `forward_batch()`, weight name declarations. Add `forward_config()`. |
| `include/frontend/models/qwen3.hpp` | Same |
| `src/frontend/models/qwen2.cpp` | Remove `forward()`, `forward_batch()`, weight name implementations. Add `forward_config()` (1 line). |
| `src/frontend/models/qwen3.cpp` | Same. Also remove the `#if 0` dead code block. |
| `include/frontend/models/forward_context.hpp` | `ContiguousForwardContext` takes `DynamicKVCache&` instead of `KVCache&` |
| `src/frontend/models/forward_context.cpp` | Same |
| `src/zedinfer/engine.cpp` | `warmup()`/`profile()` call `transformer_forward` directly. `step()` uses `model_->forward_config()`. |
| `src/zedinfer/scheduler.cpp` | `run_one()` uses `model.forward_config()` instead of `model.forward()`. |

### Deleted

| File | Reason |
|------|--------|
| (none — files modified, not deleted) | |

---

## 4. Implementation Tasks

### Task 1: Make DynamicKVCache standalone

- [ ] Remove `KVCache` base class from `base.hpp` (keep `KVCacheConfig`)
- [ ] Remove `: public KVCache` from `DynamicKVCache`
- [ ] Make all methods non-virtual in `DynamicKVCache`
- [ ] Update `ContiguousForwardContext` to take `DynamicKVCache&`
- [ ] Update `base.cpp`: keep only `KVCacheConfig::validate()` and `bytes_per_token()`
- [ ] Remove `kvcache_t = std::unique_ptr<KVCache>` typedef
- [ ] Build to verify

### Task 2: Add forward_config() to Model, remove forward methods

- [ ] Add `virtual ModelForwardConfig forward_config() const = 0` to `Model`
- [ ] Remove `virtual forward()` and `virtual forward_batch()` from `Model`
- [ ] Remove weight name virtual methods from `Model`
- [ ] Implement `forward_config()` in Qwen2Model and Qwen3Model (1 line each)
- [ ] Remove `forward()`, `forward_batch()`, weight name methods from qwen2.cpp/qwen3.cpp
- [ ] Remove `#if 0` dead code block from qwen3.cpp
- [ ] Build to verify

### Task 3: Update Engine and Scheduler callers

- [ ] `warmup()`: replace `model_->forward()` with `transformer_forward(model_->forward_config(), ctx, ...)`
- [ ] `profile()`: same
- [ ] `step()`: replace `model_->forward_batch()` with `transformer_forward(model_->forward_config(), ctx, ...)`
- [ ] `run_one()`: replace `model.forward()` with `transformer_forward(model.forward_config(), ctx, ...)`
- [ ] Build to verify
- [ ] Run `ping`, `chat`, `batch_bench` — verify correctness

---

## 5. Line Count Impact

```
Deleted:
  KVCache base class            ~50 lines (base.hpp + base.cpp)
  Model::forward() virtual       ~5 lines declaration + 2×4 impl = 13 lines
  Model::forward_batch()         ~5 lines declaration + 2×4 impl = 13 lines
  Weight name methods            4 declarations + 2×20 impl = 44 lines
  Qwen3 #if 0 dead code         ~100 lines
                                 Total: ~220 lines removed

Added:
  Model::forward_config()        1 declaration + 2×1 impl = 3 lines
                                 Total: 3 lines added

Net: -217 lines
```

---

## 6. Risks

| Risk | Mitigation |
|------|-----------|
| DynamicKVCache without base class breaks something | Only used by warmup/profile. ContiguousForwardContext is the only consumer. |
| Engine-level transformer_forward call misses model config | `model_->forward_config()` returns the correct config. Tested by ping/chat output. |
| Third-party code calls Model::forward() | No external API. All callers are in engine.cpp and scheduler.cpp. |

---

## 7. After R3: What's Left

After R3, the Model class is:
```cpp
class Model {
    virtual const ModelConfig &config() const = 0;
    virtual const ModelWeights &weights() const = 0;
    virtual std::string model_type() const = 0;
    virtual size_t num_parameters() const = 0;
    virtual ModelForwardConfig forward_config() const = 0;
    static std::shared_ptr<Model> parse(...);
};
```

Adding a new model = define a subclass with `forward_config()` returning the right flags. Zero forward logic needed.

The KVCache hierarchy is gone. `DynamicKVCache` is a simple standalone class used only by warmup/profile.

This completes the model/KV interface cleanup. R4 (attention dispatch), R5 (engine split), R6 (dead code) can follow independently.
