# Tokenizer Optimization

## Overview

This document describes the performance bottlenecks identified in the tokenizer implementation and the corresponding optimizations applied. The focus is on improving the efficiency of:

- Byte-level encoding
- BPE (Byte Pair Encoding)
- Memory usage
- Parallel execution

------

## Performance Bottlenecks

### 1. BPE Encoding

#### Issues

- `bpe_encode` maintains a `std::vector<std::string> chars`
  - Frequent **string creation and destruction** during merge operations
- `find_best_mergeable_pair`
  - Heavy **string concatenation overhead**
- Excessive **temporary object construction**

#### Impact

- High CPU overhead due to string operations
- Poor cache locality
- Increased memory allocation/deallocation cost

------

## Optimizations

------

## 1. Byte-Level Optimization

### Key Improvements

- Replace `std::unordered_map` with `std::vector`
- Avoid unnecessary copies by returning references

### Before

```cpp
static std::string byte_to_unicode(unsigned char byte);

static std::unordered_map<unsigned char, std::string>
create_byte_to_unicode_map();

static const std::unordered_map<unsigned char, std::string> byte_to_unicode_map_;
```

### After

```cpp
static const std::string& byte_to_unicode(unsigned char byte);

static std::vector<std::string> create_byte_to_unicode_map();

static const std::vector<std::string> byte_to_unicode_table_;
```

### Implementation

```cpp
std::vector<std::string>
ByteLevel::create_byte_to_unicode_map() {
    std::vector<std::string> table(256);
    std::vector<int> bytes;
    // Printable ASCII: '!' to '~' (33–126)
    for (int b = 33; b <= 126; ++b) {
        bytes.push_back(b);
    }
    // Extended ASCII: ¡–¬ (161–172) and ®–ÿ (174–255)
    for (int b = 161; b <= 172; ++b) {
        bytes.push_back(b);
    }
    for (int b = 174; b <= 255; ++b) {
        bytes.push_back(b);
    }

    std::vector<int> unicode_chars = bytes;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bytes.begin(), bytes.end(), b) == bytes.end()) {
            bytes.push_back(b);
            unicode_chars.push_back(256 + n);
            ++n;
        }
    }
    for (size_t i = 0; i < bytes.size(); ++i) {
        icu::UnicodeString uStr(static_cast<UChar32>(unicode_chars[i]));
        std::string utf8;
        uStr.toUTF8String(utf8);
        table[static_cast<unsigned char>(bytes[i])] = utf8; 
    }
    return table;
}
```

### Benefits

- Eliminates hash lookup overhead
- Removes boundary checks (0–255 fully covered)
- Avoids string copy on return
- Improves cache locality

------

## 2. Integer-Based BPE Optimization

### Key Idea

Convert string-based merge rules into **integer-based lookup**.

------

### Data Structure

```cpp
struct MergeRule {
    int rank;
    int new_id;
};

std::unordered_map<uint64_t, MergeRule> int_merges_;
```

### Key Encoding

```cpp
static inline uint64_t get_pair_key(int left, int right) {
    return (static_cast<uint64_t>(left) << 32) | static_cast<uint32_t>(right);
}
```

------

### Precomputation

```cpp
void HFTokenizer::precompute_int_merges() {
    if (merges_.empty() || vocab_.empty()) return;

    int_merges_.clear();
    int_merges_.reserve(merges_.size());

    for (const auto &entry : merges_) {
        const std::string &pair_str = entry.first;
        int rank = entry.second;

        size_t space_pos = pair_str.find(' ');
        if (space_pos == std::string::npos) continue;

        std::string p1 = pair_str.substr(0, space_pos);
        std::string p2 = pair_str.substr(space_pos + 1);

        auto it1 = vocab_.find(p1);
        auto it2 = vocab_.find(p2);

        if (it1 != vocab_.end() && it2 != vocab_.end()) {
            int id1 = it1->second;
            int id2 = it2->second;

            std::string combined = p1 + p2;
            auto it_combined = vocab_.find(combined);

            if (it_combined != vocab_.end()) {
                int_merges_[get_pair_key(id1, id2)] = {
                    rank,
                    it_combined->second
                };
            }
        }
    }
}
```

------

### Optimized Pair Search

```cpp
std::pair<int, HFTokenizer::MergeRule> 
HFTokenizer::find_best_mergeable_pair_int(const std::vector<int>& ids) const {
    int best_idx = -1;
    MergeRule best_rule = {2147483647, -1}; // INT_MAX

    if (ids.size() < 2) {
        return {best_idx, best_rule};
    }

    for (size_t i = 0; i < ids.size() - 1; ++i) {
        uint64_t key = get_pair_key(ids[i], ids[i+1]);    
        auto it = int_merges_.find(key);
        if (it != int_merges_.end()) {
            if (it->second.rank < best_rule.rank) {
                best_rule = it->second;
                best_idx = static_cast<int>(i);
            }
        }
    }

    return {best_idx, best_rule};
}
```

------

### Optimized BPE Encode

```cpp
std::vector<int> HFTokenizer::bpe_encode(const std::string &word) {
    if (word.empty()) return {};

    auto vocab_it = vocab_.find(word);
    if (vocab_it != vocab_.end()) {
        return {vocab_it->second};
    }

    std::vector<int> ids;
    ids.reserve(word.size());

    for (unsigned char byte : word) {
        const std::string& char_str = ByteLevel::byte_to_unicode(byte);

        auto it = vocab_.find(char_str);
        if (it != vocab_.end()) {
            ids.push_back(it->second);
        } else if (special_tokens_.unk_token_id != -1) {
            ids.push_back(special_tokens_.unk_token_id);
        }
    }

    while (ids.size() > 1) {
        auto [best_idx, rule] = find_best_mergeable_pair_int(ids);

        if (best_idx == -1) break;

        ids[best_idx] = rule.new_id;

        for (size_t i = best_idx + 1; i < ids.size() - 1; ++i) {
            ids[i] = ids[i + 1];
        }

        ids.pop_back();
    }

    return ids;
}
```

------

### Benefits

- Eliminates string concatenation
- Reduces memory allocations
- Improves lookup speed via integer hashing
- Significantly better cache performance

------

## 3. Parallel Optimization

------

### 3.1 Encode Pipeline Refactor

```cpp
encode()
 ├── encode_core()
 └── encode_parallel()
```

### Responsibilities

- Routing based on input size
- Adding BOS/EOS tokens

------

### 3.2 Safe Split Strategy

#### Function

```cpp
size_t HFTokenizer::find_safe_split_point(const std::string &text, size_t target_pos) const {
    if (target_pos >= text.size()) return text.size();
    
    size_t search_limit = std::min(text.size(), target_pos + 5000); 
    size_t len = search_limit - target_pos;
    
    size_t best_p1 = 0, best_p2 = 0, best_p3 = 0;

    scan_boundaries_simd(text.data() + target_pos, len, best_p1, best_p2, best_p3);

    if (best_p1 != 0) return target_pos + best_p1;
    if (best_p2 != 0) return target_pos + best_p2;
    if (best_p3 != 0) return target_pos + best_p3;

    return search_limit;
}
```

#### Features

- SIMD acceleration using **AVX2**
- 32-byte chunk scanning
- 3-level priority:

| Priority | Boundary Type |
| -------- | ------------- |
| 1        | Newline `\n`  |
| 2        | `. ! ?`       |
| 3        | Space / Tab   |

- Sliding window fallback (max 5000 chars)

------

### SIMD Implementation

```cpp
inline void scan_boundaries_simd(const char* data, size_t len, 
                                 size_t& best_p1, size_t& best_p2, size_t& best_p3) {
    __m256i v_newline = _mm256_set1_epi8('\n');          // Priority 1
    __m256i v_dot     = _mm256_set1_epi8('.');           // Priority 2
    __m256i v_excl    = _mm256_set1_epi8('!');
    __m256i v_ques    = _mm256_set1_epi8('?');
    __m256i v_space   = _mm256_set1_epi8(' ');           // Priority 3
    __m256i v_tab     = _mm256_set1_epi8('\t');

    for (size_t i = 0; i + 32 <= len; i += 32) {
        __m256i v_data = _mm256_loadu_si256((const __m256i*)(data + i));

        if (best_p1 == 0) {
            uint32_t m1 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v_data, v_newline));
            if (m1 != 0) {
                best_p1 = i + __builtin_ctz(m1) + 1; 
                return; 
            }
        }

        if (best_p2 == 0) {
            uint32_t m2 = _mm256_movemask_epi8(_mm256_or_si256(
                _mm256_or_si256(_mm256_cmpeq_epi8(v_data, v_dot), _mm256_cmpeq_epi8(v_data, v_excl)),
                _mm256_cmpeq_epi8(v_data, v_ques)
            ));
            if (m2 != 0) best_p2 = i + __builtin_ctz(m2) + 1;
        }

        if (best_p3 == 0) {
            uint32_t m3 = _mm256_movemask_epi8(_mm256_or_si256(
                _mm256_cmpeq_epi8(v_data, v_space), _mm256_cmpeq_epi8(v_data, v_tab)
            ));
            if (m3 != 0) best_p3 = i + __builtin_ctz(m3) + 1;
        }
    }
}
```

------

### 3.3 Parallel Encoding

```cpp
std::vector<int> HFTokenizer::encode_parallel(const std::string &text) {
    int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 1;

    std::vector<size_t> boundaries;
    boundaries.push_back(0); 

    size_t rough_chunk_size = text.size() / num_threads;
    size_t current_pos = 0;

    for (int i = 0; i < num_threads - 1; ++i) {
        current_pos += rough_chunk_size;
        if (current_pos >= text.size()) break;

        size_t safe_pos = find_safe_split_point(text, current_pos);
        if (safe_pos < text.size()) {
            boundaries.push_back(safe_pos);
            current_pos = safe_pos; 
        } else {
            break;
        }
    }
    boundaries.push_back(text.size()); 

    int actual_chunks = boundaries.size() - 1;
    std::vector<std::vector<int>> partial_results(actual_chunks);

    #pragma omp parallel for
    for (int i = 0; i < actual_chunks; ++i) {
        size_t start = boundaries[i];
        size_t len = boundaries[i + 1] - start;
        partial_results[i] = encode_core(text.substr(start, len));
    }

    std::vector<int> final_result;
    size_t total_tokens = 0;
    for (const auto& res : partial_results) {
        total_tokens += res.size();
    }
    final_result.reserve(total_tokens);

    for (auto& res : partial_results) {
        final_result.insert(final_result.end(), res.begin(), res.end());
    }

    return final_result;
}
```

#### Workflow

1. Split text into chunks based on CPU cores
2. Adjust boundaries using safe split
3. Parallel execution using OpenMP
4. Merge results

------

### 3.4 Final Encode Entry

```cpp
std::vector<int> HFTokenizer::encode(const std::string &text) {
    // Set threshold for parallel execution: 8KB
    const size_t PARALLEL_THRESHOLD =8*1024;

    std::vector<int> core_tokens;
    
    if (text.size() < PARALLEL_THRESHOLD) {
        core_tokens = encode_core(text);
    } else {
        core_tokens = encode_parallel(text);
    }

    std::vector<int> result;
    // Reserve space for BOS and EOS tokens
    result.reserve(core_tokens.size() + 2); 

    if (config_.add_bos_token && special_tokens_.bos_token_id != -1) {
        result.push_back(special_tokens_.bos_token_id);
    }

    result.insert(result.end(), core_tokens.begin(), core_tokens.end());

    if (config_.add_eos_token && special_tokens_.eos_token_id != -1) {
        result.push_back(special_tokens_.eos_token_id);
    }

    // Truncate to model_max_length, preserving EOS if present
    if (config_.model_max_length > 0 && result.size() > static_cast<size_t>(config_.model_max_length)) {
        bool has_eos = false;
        if (config_.add_eos_token && !result.empty() && result.back() == special_tokens_.eos_token_id) {
            result.pop_back();
            has_eos = true;
        }

        if (has_eos) {
            result.resize(config_.model_max_length - 1);
            result.push_back(special_tokens_.eos_token_id);
        } else {
            result.resize(config_.model_max_length);
        }
    }

    return result;
}
```

#### Features

- Threshold-based dispatch (8KB)
- BOS/EOS handling

------

### Benefits

- Efficient multi-core utilization
- Reduced latency for large inputs
- Maintains semantic integrity during splitting

------

## Summary

| Optimization Area | Key Improvement      | Result                     |
| ----------------- | -------------------- | -------------------------- |
| ByteLevel         | Vector + reference   | Faster lookup, no copy     |
| BPE               | Integer-based merges | Eliminates string overhead |
| Memory            | Fewer temporaries    | Lower allocation cost      |
| Parallelism       | OpenMP + SIMD        | Better scalability         |

