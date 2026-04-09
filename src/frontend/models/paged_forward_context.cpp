#include "frontend/models/paged_forward_context.hpp"
#include "backend/core/context/context.hpp"
#include "backend/ops/kv_scatter/nvidia/paged_kv_scatter.cuh"
#include "backend/ops/ops.hpp"
#include "utils/types.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace zedinfer::model {

static tensor_t upload_to_gpu(const std::vector<int>& host_data, zedinferDeviceType_t dev, int dev_id);
static void ensure_flashinfer_page_table_cache(kvcache::SequenceBlockTable& table, zedinferDeviceType_t dev,
                                               int dev_id);

// ============================================================================
// Constructors
// ============================================================================

PagedForwardContext::PagedForwardContext(const std::vector<int>& input_ids, int past_len,
                                         kvcache::SequenceBlockTable& block_table, kvcache::BlockPool& pool)
    : pool_(pool), total_tokens_(static_cast<int>(input_ids.size())) {
    token_ids_ = input_ids;
    position_ids_.resize(total_tokens_);
    for (int i = 0; i < total_tokens_; ++i) { position_ids_[i] = past_len + static_cast<int64_t>(i); }

    bool is_decode = (total_tokens_ == 1 && past_len > 0);
    slots_.push_back({&block_table, 0, total_tokens_, past_len, is_decode});
}

PagedForwardContext::PagedForwardContext(const BatchContext& batch, kvcache::BlockAllocator& allocator)
    : pool_(allocator.pool()), total_tokens_(batch.total_tokens()) {
    token_ids_ = batch.token_ids;
    position_ids_ = batch.position_ids;

    for (const auto& s : batch.slots) {
        bool is_decode = !s.is_prefill;
        slots_.push_back({&s.request->block_table(), s.token_offset, s.num_tokens, s.past_len, is_decode});
    }
}

// ============================================================================
// Input Preparation
// ============================================================================

void PagedForwardContext::prepare_inputs(tensor_t& ids, tensor_t& pos_ids, const ExecutorConfig& exec_config) {
    ids = Tensor::create({static_cast<size_t>(total_tokens_)}, ZEDINFER_DTYPE_I32, exec_config.device_type,
                         exec_config.device_id);
    ids->load(token_ids_.data());

    pos_ids = Tensor::create({static_cast<size_t>(total_tokens_)}, ZEDINFER_DTYPE_I64, exec_config.device_type,
                             exec_config.device_id);
    pos_ids->load(position_ids_.data());
}

void PagedForwardContext::prepare_inputs_into(tensor_t ids, tensor_t pos_ids) {
    ids->load(token_ids_.data());
    pos_ids->load(position_ids_.data());
}

// ============================================================================
// KV Write: scatter tokens to blocks per slot
// ============================================================================

void PagedForwardContext::copy_to_block(const void* src, size_t bytes, void* dst) {
    if (pool_.device_type() == ZEDINFER_DEVICE_CPU) {
        std::memcpy(dst, src, bytes);
    } else {
        core::context().setDevice(pool_.device_type(), pool_.device_id());
        core::context().runtime().api()->memcpy_sync(dst, src, bytes, ZEDINFER_MEMCPY_D2D);
    }
}

void PagedForwardContext::scatter_slot_kv(const Slot& slot, int layer, tensor_t k, tensor_t v) {
    const int bs = pool_.config().block_size;
    const size_t token_bytes = pool_.config().token_bytes();

    auto* k_src = static_cast<const std::byte*>(k->data()) + slot.token_offset * token_bytes;
    auto* v_src = static_cast<const std::byte*>(v->data()) + slot.token_offset * token_bytes;

    auto& pages = slot.block_table->pages[layer];

    for (int t = 0; t < slot.num_tokens; ++t) {
        int global_pos = slot.past_len + t;
        int block_idx = global_pos / bs;
        int offset = global_pos % bs;

        void* k_dst = static_cast<std::byte*>(pool_.k_block_data(pages[block_idx])) + offset * token_bytes;
        copy_to_block(k_src + t * token_bytes, token_bytes, k_dst);

        void* v_dst = static_cast<std::byte*>(pool_.v_block_data(pages[block_idx])) + offset * token_bytes;
        copy_to_block(v_src + t * token_bytes, token_bytes, v_dst);
    }
}

void PagedForwardContext::write_kv(int layer, tensor_t k, tensor_t v) {
    const size_t token_bytes = pool_.config().token_bytes();

    if (pool_.device_type() == ZEDINFER_DEVICE_CPU) {
        // CPU: per-token memcpy (already fast, no launch overhead)
        for (const auto& slot : slots_) { scatter_slot_kv(slot, layer, k, v); }
        return;
    }

    core::context().setDevice(pool_.device_type(), pool_.device_id());
    for (const auto& slot : slots_) {
        if (slot.num_tokens <= 0) {
            continue;
        }
        ensure_flashinfer_page_table_cache(*slot.block_table, pool_.device_type(), pool_.device_id());
        ops::nvidia::scatter_paged_kv(
            static_cast<const std::byte*>(k->data()) + static_cast<size_t>(slot.token_offset) * token_bytes,
            static_cast<const std::byte*>(v->data()) + static_cast<size_t>(slot.token_offset) * token_bytes,
            pool_.k_pool_base(), pool_.v_pool_base(),
            reinterpret_cast<const int*>(slot.block_table->flashinfer_page_tables_gpu[layer]->data()),
            pool_.config().block_size, slot.past_len, slot.num_tokens, token_bytes);
    }
}

// ============================================================================
// Attention: paged decode (single or batched) + paged prefill
// ============================================================================

void PagedForwardContext::build_decode_cache(const ExecutorConfig& exec_config) {
    if (decode_cache_built_) {
        return;
    }

    // Count decode slots
    cached_num_decode_ = 0;
    cached_decode_start_ = -1;
    std::vector<kvcache::SequenceBlockTable*> decode_tables;

    for (const auto& slot : slots_) {
        if (!slot.is_decode) {
            continue;
        }
        if (cached_decode_start_ < 0) {
            cached_decode_start_ = slot.token_offset;
        }
        cached_num_decode_++;
        decode_tables.push_back(slot.block_table);
    }

    if (cached_num_decode_ <= 1 || decode_tables.empty()) {
        decode_cache_built_ = true;
        return;
    }

    // Find max blocks across all layers and all requests
    int num_layers = decode_tables[0]->num_layers;
    cached_max_blocks_ = 0;
    for (auto* dt : decode_tables) {
        for (int L = 0; L < num_layers; ++L) {
            cached_max_blocks_ = std::max(cached_max_blocks_, static_cast<int>(dt->pages[L].size()));
        }
    }

    // Build seq_lens GPU tensor (same for all layers)
    std::vector<int> sl(cached_num_decode_);
    for (int r = 0; r < cached_num_decode_; ++r) { sl[r] = decode_tables[r]->seq_len + 1; }
    seq_lens_gpu_ = Tensor::create({static_cast<size_t>(cached_num_decode_)}, ZEDINFER_DTYPE_I32,
                                   exec_config.device_type, exec_config.device_id);
    seq_lens_gpu_->load(sl.data());

    // Build per-layer block table GPU tensors
    decode_layer_cache_.resize(num_layers);
    size_t bt_size = static_cast<size_t>(cached_num_decode_) * cached_max_blocks_;

    for (int L = 0; L < num_layers; ++L) {
        std::vector<int> page_bt(bt_size, 0);

        for (int r = 0; r < cached_num_decode_; ++r) {
            auto& pages = decode_tables[r]->pages[L];
            for (size_t b = 0; b < pages.size(); ++b) { page_bt[r * cached_max_blocks_ + b] = pages[b]; }
        }

        auto page_gpu = Tensor::create({bt_size}, ZEDINFER_DTYPE_I32, exec_config.device_type, exec_config.device_id);
        page_gpu->load(page_bt.data());

        decode_layer_cache_[L] = {std::move(page_gpu)};
    }

    decode_cache_built_ = true;
}

// Helper: upload a host int vector to a GPU tensor
static tensor_t upload_to_gpu(const std::vector<int>& host_data, zedinferDeviceType_t dev, int dev_id) {
    if (dev == ZEDINFER_DEVICE_CPU) {
        return nullptr; // CPU doesn't need upload
    }
    auto t = Tensor::create({host_data.size()}, ZEDINFER_DTYPE_I32, dev, dev_id);
    t->load(host_data.data());
    return t;
}

static int ceil_div_int(int x, int y) {
    return (x + y - 1) / y;
}

static int last_page_len_for(int seq_len, int block_size) {
    return seq_len == 0 ? 0 : ((seq_len - 1) % block_size) + 1;
}

static void append_active_pages(std::vector<int>& dst, const std::vector<int>& pages, int active_pages) {
    if (active_pages < 0 || active_pages > static_cast<int>(pages.size())) {
        throw std::runtime_error("[PagedForwardContext] Active page count exceeds block table capacity");
    }
    dst.insert(dst.end(), pages.begin(), pages.begin() + active_pages);
}

static void ensure_flashinfer_page_table_cache(kvcache::SequenceBlockTable& table, zedinferDeviceType_t dev,
                                               int dev_id) {
    if (dev == ZEDINFER_DEVICE_CPU || table.pages.empty()) {
        return;
    }

    const size_t pages_per_layer = table.pages.front().size();
    const bool cache_valid = table.flashinfer_cache_device_type == dev && table.flashinfer_cache_device_id == dev_id
                          && table.flashinfer_cache_pages_per_layer == pages_per_layer
                          && table.flashinfer_page_tables_gpu.size() == table.pages.size();
    if (cache_valid) {
        return;
    }

    table.clear_runtime_caches();
    table.flashinfer_page_tables_gpu.resize(table.pages.size());
    table.flashinfer_cache_device_type = dev;
    table.flashinfer_cache_device_id = dev_id;
    table.flashinfer_cache_pages_per_layer = pages_per_layer;

    for (size_t layer = 0; layer < table.pages.size(); ++layer) {
        table.flashinfer_page_tables_gpu[layer] = upload_to_gpu(table.pages[layer], dev, dev_id);
    }
}

static void ensure_flashinfer_single_decode_metadata_cache(kvcache::SequenceBlockTable& table, int kv_len,
                                                           int block_size, bool use_prefill_kernel,
                                                           zedinferDeviceType_t dev, int dev_id) {
    if (dev == ZEDINFER_DEVICE_CPU) {
        return;
    }

    if (table.flashinfer_cache_device_type != dev || table.flashinfer_cache_device_id != dev_id) {
        table.clear_runtime_caches();
    }

    if (!table.flashinfer_single_decode_kv_indptr_gpu) {
        table.flashinfer_single_decode_kv_indptr_gpu = Tensor::create({2}, ZEDINFER_DTYPE_I32, dev, dev_id);
    }
    if (!table.flashinfer_single_decode_kv_last_page_len_gpu) {
        table.flashinfer_single_decode_kv_last_page_len_gpu = Tensor::create({1}, ZEDINFER_DTYPE_I32, dev, dev_id);
    }

    const int active_pages = ceil_div_int(kv_len, block_size);
    const int kv_indptr_host[2] = {0, active_pages};
    const int kv_last_page_len_host[1] = {last_page_len_for(kv_len, block_size)};
    table.flashinfer_single_decode_kv_indptr_gpu->load(kv_indptr_host);
    table.flashinfer_single_decode_kv_last_page_len_gpu->load(kv_last_page_len_host);

    table.flashinfer_cache_device_type = dev;
    table.flashinfer_cache_device_id = dev_id;

    if (use_prefill_kernel) {
        if (!table.flashinfer_single_decode_qo_indptr_gpu) {
            table.flashinfer_single_decode_qo_indptr_gpu = Tensor::create({2}, ZEDINFER_DTYPE_I32, dev, dev_id);
        }
        const int qo_indptr_host[2] = {0, 1};
        table.flashinfer_single_decode_qo_indptr_gpu->load(qo_indptr_host);
    } else if (!table.flashinfer_single_decode_descriptor_gpu) {
        table.flashinfer_single_decode_descriptor_gpu = Tensor::create({5}, ZEDINFER_DTYPE_I32, dev, dev_id);
        const int fi_descriptor_host[5] = {0, 0, 0, 1, 1};
        table.flashinfer_single_decode_descriptor_gpu->load(fi_descriptor_host);
    }
}

static bool supports_flashinfer(const ops::AttentionConfig& cfg) {
#if defined(USE_FLASHINFER) && defined(ENABLE_NVIDIA_API)
    // Keep a runtime kill switch so baseline/bench comparisons do not require a rebuild.
    if (std::getenv("ZEDINFER_DISABLE_FLASHINFER") != nullptr) {
        return false;
    }
    if (cfg.device_type != ZEDINFER_DEVICE_NVIDIA) {
        return false;
    }
    if ((cfg.dtype != ZEDINFER_DTYPE_F16 && cfg.dtype != ZEDINFER_DTYPE_BF16) || cfg.block_size <= 0 || cfg.nkvhead <= 0
        || cfg.nhead <= 0 || cfg.nhead % cfg.nkvhead != 0) {
        return false;
    }
    switch (cfg.head_dim) {
        case 64:
        case 128:
        case 256:
            return true;
        default:
            return false;
    }
#else
    (void)cfg;
    return false;
#endif
}

static bool supports_flashinfer_decode_kernel(const ops::AttentionConfig& cfg) {
    if (!supports_flashinfer(cfg)) {
        return false;
    }
    switch (cfg.nhead / cfg.nkvhead) {
        case 1:
        case 2:
        case 3:
        case 4:
        case 8:
            return true;
        default:
            return false;
    }
}

void PagedForwardContext::build_flashinfer_decode_cache(const ops::AttentionConfig& cfg) {
    if (flashinfer_decode_cache_built_) {
        return;
    }

    flashinfer_decode_uses_prefill_kernel_ = !supports_flashinfer_decode_kernel(cfg);
    if (!supports_flashinfer(cfg) || cached_num_decode_ <= 0) {
        flashinfer_decode_cache_built_ = true;
        return;
    }

    std::vector<const Slot*> decode_slots;
    decode_slots.reserve(cached_num_decode_);
    for (const auto& slot : slots_) {
        if (slot.is_decode) {
            decode_slots.push_back(&slot);
        }
    }
    if (decode_slots.empty()) {
        flashinfer_decode_cache_built_ = true;
        return;
    }

    const int num_layers = decode_slots[0]->block_table->num_layers;
    if (cached_num_decode_ == 1) {
        auto* table = decode_slots[0]->block_table;
        const int kv_len = decode_slots[0]->past_len + 1;
        const int active_pages = ceil_div_int(kv_len, cfg.block_size);

        flashinfer_decode_kv_indptr_host_ = {0, active_pages};
        flashinfer_decode_kv_last_page_len_host_ = {last_page_len_for(kv_len, cfg.block_size)};
        if (flashinfer_decode_uses_prefill_kernel_) {
            flashinfer_decode_qo_indptr_host_ = {0, 1};
        } else {
            flashinfer_decode_qo_indptr_host_.clear();
        }

        ensure_flashinfer_page_table_cache(*table, cfg.device_type, cfg.device_id);
        ensure_flashinfer_single_decode_metadata_cache(
            *table, kv_len, cfg.block_size, flashinfer_decode_uses_prefill_kernel_, cfg.device_type, cfg.device_id);
        flashinfer_decode_layer_cache_.resize(num_layers);
        for (int layer = 0; layer < num_layers; ++layer) {
            flashinfer_decode_layer_cache_[layer] = {table->flashinfer_page_tables_gpu[layer]};
        }

        flashinfer_decode_kv_indptr_gpu_ = table->flashinfer_single_decode_kv_indptr_gpu;
        flashinfer_decode_kv_last_page_len_gpu_ = table->flashinfer_single_decode_kv_last_page_len_gpu;
        if (flashinfer_decode_uses_prefill_kernel_) {
            flashinfer_decode_qo_indptr_gpu_ = table->flashinfer_single_decode_qo_indptr_gpu;
        } else {
            flashinfer_decode_descriptor_gpu_ = table->flashinfer_single_decode_descriptor_gpu;
        }

        flashinfer_decode_cache_built_ = true;
        return;
    }

    std::vector<int> active_pages_per_slot(cached_num_decode_, 0);
    flashinfer_decode_kv_indptr_host_.assign(cached_num_decode_ + 1, 0);
    flashinfer_decode_kv_last_page_len_host_.assign(cached_num_decode_, 0);
    if (flashinfer_decode_uses_prefill_kernel_) {
        flashinfer_decode_qo_indptr_host_.assign(cached_num_decode_ + 1, 0);
    } else {
        flashinfer_decode_qo_indptr_host_.clear();
    }

    for (int i = 0; i < cached_num_decode_; ++i) {
        const int kv_len = decode_slots[i]->past_len + 1;
        const int active_pages = ceil_div_int(kv_len, cfg.block_size);
        active_pages_per_slot[i] = active_pages;
        flashinfer_decode_kv_indptr_host_[i + 1] = flashinfer_decode_kv_indptr_host_[i] + active_pages;
        flashinfer_decode_kv_last_page_len_host_[i] = last_page_len_for(kv_len, cfg.block_size);
        if (flashinfer_decode_uses_prefill_kernel_) {
            flashinfer_decode_qo_indptr_host_[i + 1] = flashinfer_decode_qo_indptr_host_[i] + 1;
        }
    }

    flashinfer_decode_layer_cache_.resize(num_layers);
    for (int layer = 0; layer < num_layers; ++layer) {
        std::vector<int> kv_page_indices_host;
        kv_page_indices_host.reserve(flashinfer_decode_kv_indptr_host_.back());
        for (int i = 0; i < cached_num_decode_; ++i) {
            append_active_pages(kv_page_indices_host, decode_slots[i]->block_table->pages[layer],
                                active_pages_per_slot[i]);
        }
        flashinfer_decode_layer_cache_[layer] = {upload_to_gpu(kv_page_indices_host, cfg.device_type, cfg.device_id)};
    }

    flashinfer_decode_kv_indptr_gpu_ = upload_to_gpu(flashinfer_decode_kv_indptr_host_, cfg.device_type, cfg.device_id);
    flashinfer_decode_kv_last_page_len_gpu_
        = upload_to_gpu(flashinfer_decode_kv_last_page_len_host_, cfg.device_type, cfg.device_id);
    if (flashinfer_decode_uses_prefill_kernel_) {
        flashinfer_decode_qo_indptr_gpu_
            = upload_to_gpu(flashinfer_decode_qo_indptr_host_, cfg.device_type, cfg.device_id);
    } else if (cached_num_decode_ == 1 && !flashinfer_decode_descriptor_gpu_) {
        const std::vector<int> fi_descriptor_host{0, 0, 0, 1, 1};
        flashinfer_decode_descriptor_gpu_ = upload_to_gpu(fi_descriptor_host, cfg.device_type, cfg.device_id);
    }

    flashinfer_decode_cache_built_ = true;
}

void PagedForwardContext::build_flashinfer_prefill_cache(const ops::AttentionConfig& cfg) {
    if (flashinfer_prefill_cache_built_) {
        return;
    }
    flashinfer_prefill_cache_built_ = true;

    if (!supports_flashinfer(cfg)) {
        return;
    }

    int expected_prefill_offset = -1;
    for (const auto& slot : slots_) {
        if (slot.is_decode) {
            continue;
        }
        if (expected_prefill_offset < 0) {
            expected_prefill_offset = slot.token_offset;
        }
        if (slot.token_offset != expected_prefill_offset) {
            flashinfer_prefill_slots_.clear();
            return;
        }
        expected_prefill_offset += slot.num_tokens;
    }

    flashinfer_prefill_slots_.clear();
    flashinfer_prefill_start_ = -1;
    flashinfer_prefill_total_tokens_ = 0;
    for (const auto& slot : slots_) {
        if (slot.is_decode) {
            continue;
        }
        if (flashinfer_prefill_start_ < 0) {
            flashinfer_prefill_start_ = slot.token_offset;
        }
        flashinfer_prefill_slots_.push_back(&slot);
        flashinfer_prefill_total_tokens_ += slot.num_tokens;
    }
    if (flashinfer_prefill_slots_.empty()) {
        return;
    }

    flashinfer_prefill_qo_indptr_host_.assign(flashinfer_prefill_slots_.size() + 1, 0);
    flashinfer_prefill_kv_indptr_host_.assign(flashinfer_prefill_slots_.size() + 1, 0);
    flashinfer_prefill_kv_last_page_len_host_.assign(flashinfer_prefill_slots_.size(), 0);

    for (size_t i = 0; i < flashinfer_prefill_slots_.size(); ++i) {
        const auto* slot = flashinfer_prefill_slots_[i];
        const int kv_len = slot->past_len + slot->num_tokens;
        const int active_pages = ceil_div_int(kv_len, cfg.block_size);
        flashinfer_prefill_qo_indptr_host_[i + 1] = flashinfer_prefill_qo_indptr_host_[i] + slot->num_tokens;
        flashinfer_prefill_kv_indptr_host_[i + 1] = flashinfer_prefill_kv_indptr_host_[i] + active_pages;
        flashinfer_prefill_kv_last_page_len_host_[i] = last_page_len_for(kv_len, cfg.block_size);
    }

    flashinfer_prefill_qo_indptr_gpu_
        = upload_to_gpu(flashinfer_prefill_qo_indptr_host_, cfg.device_type, cfg.device_id);
    flashinfer_prefill_kv_indptr_gpu_
        = upload_to_gpu(flashinfer_prefill_kv_indptr_host_, cfg.device_type, cfg.device_id);
    flashinfer_prefill_kv_last_page_len_gpu_
        = upload_to_gpu(flashinfer_prefill_kv_last_page_len_host_, cfg.device_type, cfg.device_id);

    if (flashinfer_prefill_slots_.size() == 1) {
        ensure_flashinfer_page_table_cache(*flashinfer_prefill_slots_[0]->block_table, cfg.device_type, cfg.device_id);
    }
}

void PagedForwardContext::attend_decode_single(int layer, tensor_t q_rope, tensor_t attn,
                                               const ops::AttentionConfig& cfg, size_t nhead, size_t head_dim) {
    auto* dt = slots_[0].block_table;
    for (const auto& slot : slots_) {
        if (slot.is_decode) {
            dt = slot.block_table;
            break;
        }
    }

    auto decode_q = q_rope->slice(0, cached_decode_start_, cached_decode_start_ + 1);
    auto decode_out = attn->slice(0, cached_decode_start_, cached_decode_start_ + 1);
    const bool use_flashinfer = supports_flashinfer(cfg) && !flashinfer_decode_layer_cache_.empty();

    ops::AttentionParams params{cfg};
    params.k_pool_base = pool_.k_pool_base();
    params.v_pool_base = pool_.v_pool_base();

    if (use_flashinfer) {
        params.use_flashinfer = true;
        params.out = decode_out;
        params.q = decode_q;
        params.kv_indptr = reinterpret_cast<const int*>(flashinfer_decode_kv_indptr_gpu_->data());
        params.kv_page_indices
            = reinterpret_cast<const int*>(flashinfer_decode_layer_cache_[layer].kv_page_indices_gpu->data());
        params.kv_last_page_len = reinterpret_cast<const int*>(flashinfer_decode_kv_last_page_len_gpu_->data());
        params.kv_indptr_host = flashinfer_decode_kv_indptr_host_.data();
        params.kv_batch_size = 1;
        if (flashinfer_decode_uses_prefill_kernel_) {
            params.qo_indptr = reinterpret_cast<const int*>(flashinfer_decode_qo_indptr_gpu_->data());
            params.qo_indptr_host = flashinfer_decode_qo_indptr_host_.data();
        } else {
            params.fi_request_indices = reinterpret_cast<const int*>(flashinfer_decode_descriptor_gpu_->data());
            params.fi_kv_tile_indices = reinterpret_cast<const int*>(flashinfer_decode_descriptor_gpu_->data()) + 1;
            params.fi_o_indptr = reinterpret_cast<const int*>(flashinfer_decode_descriptor_gpu_->data()) + 2;
            params.fi_kv_chunk_size_ptr = reinterpret_cast<const int*>(flashinfer_decode_descriptor_gpu_->data()) + 4;
        }
    } else {
        // Upload block tables to GPU (host pointers are not accessible from GPU kernels
        // on all devices — e.g., RTX 4090 lacks HMM support)
        auto page_bt_gpu = upload_to_gpu(dt->pages[layer], cfg.device_type, cfg.device_id);
        params.out = decode_out->view({nhead, head_dim});
        params.q = decode_q->view({nhead, head_dim});
        params.page_table = page_bt_gpu ? reinterpret_cast<const int*>(page_bt_gpu->data()) : dt->pages[layer].data();
        params.seq_len = dt->seq_len + 1;
        params.seqlen_q = 1;
    }

    ops::attention(params);
}

void PagedForwardContext::attend_decode_batched(int layer, tensor_t q_rope, tensor_t attn,
                                                const ops::AttentionConfig& cfg) {
    auto decode_q = q_rope->slice(0, cached_decode_start_, cached_decode_start_ + cached_num_decode_);
    auto decode_out = attn->slice(0, cached_decode_start_, cached_decode_start_ + cached_num_decode_);
    const bool use_flashinfer = supports_flashinfer(cfg) && !flashinfer_decode_layer_cache_.empty();

    ops::AttentionParams params{cfg};
    params.k_pool_base = pool_.k_pool_base();
    params.v_pool_base = pool_.v_pool_base();

    if (use_flashinfer) {
        params.use_flashinfer = true;
        params.out = decode_out;
        params.q = decode_q;
        params.kv_indptr = reinterpret_cast<const int*>(flashinfer_decode_kv_indptr_gpu_->data());
        params.kv_page_indices
            = reinterpret_cast<const int*>(flashinfer_decode_layer_cache_[layer].kv_page_indices_gpu->data());
        params.kv_last_page_len = reinterpret_cast<const int*>(flashinfer_decode_kv_last_page_len_gpu_->data());
        params.kv_indptr_host = flashinfer_decode_kv_indptr_host_.data();
        params.kv_batch_size = cached_num_decode_;
        if (flashinfer_decode_uses_prefill_kernel_) {
            params.qo_indptr = reinterpret_cast<const int*>(flashinfer_decode_qo_indptr_gpu_->data());
            params.qo_indptr_host = flashinfer_decode_qo_indptr_host_.data();
        }
    } else {
        params.out = decode_out;
        params.q = decode_q;
        params.batched_page_tables = reinterpret_cast<const int*>(decode_layer_cache_[layer].page_bt_gpu->data());
        params.batched_seq_lens = reinterpret_cast<const int*>(seq_lens_gpu_->data());
        params.num_requests = cached_num_decode_;
        params.max_blocks_per_seq = cached_max_blocks_;
    }

    ops::attention(params);
}

void PagedForwardContext::attend_prefill(int layer, tensor_t q_rope, tensor_t attn, const ops::AttentionConfig& cfg) {
    const bool use_flashinfer = supports_flashinfer(cfg) && !flashinfer_prefill_slots_.empty();

    if (use_flashinfer) {
        auto pf_q
            = q_rope->slice(0, flashinfer_prefill_start_, flashinfer_prefill_start_ + flashinfer_prefill_total_tokens_);
        auto pf_out
            = attn->slice(0, flashinfer_prefill_start_, flashinfer_prefill_start_ + flashinfer_prefill_total_tokens_);

        ops::AttentionParams params{cfg};
        params.use_flashinfer = true;
        params.out = pf_out;
        params.q = pf_q;
        params.k_pool_base = pool_.k_pool_base();
        params.v_pool_base = pool_.v_pool_base();
        params.kv_indptr = reinterpret_cast<const int*>(flashinfer_prefill_kv_indptr_gpu_->data());
        params.kv_last_page_len = reinterpret_cast<const int*>(flashinfer_prefill_kv_last_page_len_gpu_->data());
        params.qo_indptr = reinterpret_cast<const int*>(flashinfer_prefill_qo_indptr_gpu_->data());
        params.kv_indptr_host = flashinfer_prefill_kv_indptr_host_.data();
        params.qo_indptr_host = flashinfer_prefill_qo_indptr_host_.data();
        params.kv_batch_size = static_cast<int>(flashinfer_prefill_slots_.size());

        if (flashinfer_prefill_slots_.size() == 1) {
            auto* table = flashinfer_prefill_slots_[0]->block_table;
            params.kv_page_indices = reinterpret_cast<const int*>(table->flashinfer_page_tables_gpu[layer]->data());
            ops::attention(params);
            return;
        }

        std::vector<int> kv_page_indices_host;
        kv_page_indices_host.reserve(flashinfer_prefill_kv_indptr_host_.back());
        for (size_t i = 0; i < flashinfer_prefill_slots_.size(); ++i) {
            const auto* slot = flashinfer_prefill_slots_[i];
            const int active_pages = flashinfer_prefill_kv_indptr_host_[i + 1] - flashinfer_prefill_kv_indptr_host_[i];
            append_active_pages(kv_page_indices_host, slot->block_table->pages[layer], active_pages);
        }
        auto kv_page_indices_gpu = upload_to_gpu(kv_page_indices_host, cfg.device_type, cfg.device_id);
        params.kv_page_indices = reinterpret_cast<const int*>(kv_page_indices_gpu->data());
        ops::attention(params);
        return;
    }

    for (const auto& slot : slots_) {
        if (slot.is_decode) {
            continue;
        }

        auto pf_q = q_rope->slice(0, slot.token_offset, slot.token_offset + slot.num_tokens);
        auto pf_out = attn->slice(0, slot.token_offset, slot.token_offset + slot.num_tokens);

        auto page_bt_gpu = upload_to_gpu(slot.block_table->pages[layer], cfg.device_type, cfg.device_id);

        ops::AttentionParams params{cfg};
        params.out = pf_out;
        params.q = pf_q;
        params.k_pool_base = pool_.k_pool_base();
        params.v_pool_base = pool_.v_pool_base();
        params.page_table
            = page_bt_gpu ? reinterpret_cast<const int*>(page_bt_gpu->data()) : slot.block_table->pages[layer].data();
        params.seqlen_q = slot.num_tokens;
        params.past_len = slot.past_len;
        ops::attention(params);
    }
}

tensor_t PagedForwardContext::attend(int layer, tensor_t q_rope, float scale, const ExecutorConfig& exec_config,
                                     size_t nhead, size_t nkvhead, size_t head_dim, tensor_t pre_alloc_out) {
    ops::AttentionConfig attn_cfg{
        static_cast<int>(nhead),   static_cast<int>(nkvhead), static_cast<int>(head_dim), scale,
        pool_.config().block_size, exec_config.data_type,     exec_config.device_type,    exec_config.device_id};

    auto attn = pre_alloc_out ? pre_alloc_out
                              : Tensor::create({static_cast<size_t>(total_tokens_), nhead, head_dim},
                                               exec_config.data_type, exec_config.device_type, exec_config.device_id);

    build_decode_cache(exec_config);
    if (supports_flashinfer(attn_cfg)) {
        build_flashinfer_decode_cache(attn_cfg);
        build_flashinfer_prefill_cache(attn_cfg);
    }

    if (cached_num_decode_ == 1) {
        attend_decode_single(layer, q_rope, attn, attn_cfg, nhead, head_dim);
    } else if (cached_num_decode_ > 1) {
        attend_decode_batched(layer, q_rope, attn, attn_cfg);
    }

    attend_prefill(layer, q_rope, attn, attn_cfg);

    return attn;
}

// ============================================================================
// Finalize
// ============================================================================

void PagedForwardContext::finalize() {
    // No-op: seq_len is updated by Scheduler::process_results() for all paths.
}

} // namespace zedinfer::model
