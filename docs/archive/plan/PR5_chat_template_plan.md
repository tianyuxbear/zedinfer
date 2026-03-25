# PR-5: Chat Template Extraction - Detailed Implementation Plan

## Goal

Replace the hardcoded DeepSeek-R1 chat template strings in `session.cpp:42-49`
and `hf_tokenizer.cpp:184-211` with a configurable `ChatTemplate` struct that is
loaded from the model directory at init time. This makes the framework portable
across model families without code changes.

---

## Problem Statement

### Hardcoded template in two places

The DeepSeek-R1 chat format is duplicated:

1. **`src/zedinfer/session.cpp:42-49`** - prompt assembly in `chat()`:
   ```cpp
   input += "<｜begin▁of▁sentence｜>";        // BOS
   input += "<｜User｜>" + user_input;         // user prefix
   input += "<｜Assistant｜><think>\n";         // assistant prefix
   ```

2. **`src/frontend/tokenizer/hf_tokenizer.cpp:184-211`** - `apply_chat_template()`:
   ```cpp
   result += "<｜begin▁of▁sentence｜>";        // BOS
   result += "<｜User｜>" + content;            // user messages
   result += "<｜Assistant｜>" + ... + "<｜end▁of▁sentence｜>"; // assistant messages
   result += "<｜Assistant｜><think>\n";        // generation prompt
   ```

3. **`examples/chat.cpp:171`** - display prefix:
   ```cpp
   std::cout << "... <think> ";                // hardcoded display assumption
   ```

4. **`src/zedinfer/session.cpp:55`** - output prefix:
   ```cpp
   std::string output = "<think> ";            // hardcoded output prefix
   ```

### Consequences

- Adding a non-DeepSeek-R1 model (e.g., Llama-3, Mistral, ChatGLM) requires
  editing C++ source code in multiple files.
- The `<think>` prefix in the assistant response is specific to DeepSeek-R1's
  reasoning format. Other models don't use it.
- `session.cpp` and `hf_tokenizer.cpp` duplicate the same template logic
  independently with slightly different behavior.

---

## Scope

| In Scope | Out of Scope |
|----------|-------------|
| Define `ChatTemplate` struct | Jinja2 template engine (too complex, not needed) |
| Load template from `tokenizer_config.json` or fallback | New model family support |
| Replace hardcoded strings in `session.cpp` | Changes to tokenizer `encode()`/`decode()` |
| Replace hardcoded strings in `hf_tokenizer.cpp:apply_chat_template()` | Multi-turn history re-formatting |
| Remove hardcoded `<think>` prefix from `session.cpp` and `chat.cpp` | Session registry (PR-6+) |
| Provide `default_deepseek_r1()` fallback | |

---

## Dependencies

- **PR-4** (Stateless Engine): Merged. Engine API is stable.
- **No dependency on PR-3a/3b** (operator optimization).

---

## Investigation: Where Template Data Comes From

### HuggingFace `tokenizer_config.json`

HuggingFace model repos include a `tokenizer_config.json` with a `chat_template`
field containing a Jinja2 template string. For DeepSeek-R1 models, it looks like:

```json
{
  "chat_template": "{% if not add_generation_prompt is defined %}...{% endif %}...",
  "bos_token": "<｜begin▁of▁sentence｜>",
  "eos_token": "<｜end▁of▁sentence｜>",
  ...
}
```

The Jinja2 template is complex and model-specific. Implementing a full Jinja2
interpreter in C++ is out of scope. Instead, we extract the **structural tokens**
(BOS, user prefix/suffix, assistant prefix/suffix, EOS) which are sufficient for
the fixed role-based chat format used by all currently supported models.

### What we already parse

`hf_tokenizer.cpp:215-237` already reads `tokenizer_config.json` for:
- `add_bos_token`
- `add_eos_token`
- `clean_up_tokenization_spaces`
- `model_max_length`

We can extend this to also extract `bos_token`, `eos_token`, and the
`chat_template` string (for parsing structural tokens).

### Alternative: dedicated `chat_template.json`

Instead of parsing Jinja2, we can ship a simple `chat_template.json` alongside
the model, or auto-detect the template from model type. This is simpler and
more reliable.

**Recommendation**: Load from `tokenizer_config.json` fields (`bos_token`,
`eos_token`) and infer the chat structure from `model_type` in `config.json`.
Provide per-model-type defaults. Allow override via an optional
`chat_template.json` file in the model directory.

---

## Detailed Design

### 1. New File: `include/zedinfer/chat_template.hpp`

```cpp
#pragma once

#include <string>

namespace zedinfer {

/**
 * Chat template defining how to format multi-turn conversations.
 * Loaded from model directory at engine init time.
 */
struct ChatTemplate {
    // Structural tokens
    std::string bos_token;            // e.g., "<｜begin▁of▁sentence｜>"
    std::string eos_token;            // e.g., "<｜end▁of▁sentence｜>"
    std::string user_prefix;          // e.g., "<｜User｜>"
    std::string user_suffix;          // e.g., ""
    std::string assistant_prefix;     // e.g., "<｜Assistant｜>"
    std::string assistant_suffix;     // e.g., "<｜end▁of▁sentence｜>"

    // Generation prompt appended after the last user message
    std::string generation_prompt;    // e.g., "<｜Assistant｜><think>\n"

    // Output display prefix (model-specific reasoning format)
    std::string output_prefix;        // e.g., "<think> " for DeepSeek-R1, "" for others

    // BOS behavior
    bool add_bos_first_turn_only = true;

    // Load from model directory (tries chat_template.json, then infers from model_type)
    static ChatTemplate load(const std::string &model_path,
                             const std::string &model_type);

    // Built-in defaults
    static ChatTemplate default_deepseek_r1();
};

} // namespace zedinfer
```

**Design decisions:**

- `generation_prompt` is the string appended after the last user message to
  trigger assistant generation. For DeepSeek-R1 this includes `<think>\n`
  to trigger the reasoning chain.
- `output_prefix` captures the `"<think> "` that `session.cpp:55` currently
  hardcodes. For non-reasoning models this would be `""`.
- `add_bos_first_turn_only = true` matches the current behavior where BOS is
  added only on the first turn (`session.cpp:42-45`).
- No Jinja2 parsing. The struct captures the fixed positional tokens that
  define the conversation format.

### 2. New File: `src/zedinfer/chat_template.cpp`

```
ChatTemplate ChatTemplate::default_deepseek_r1():
    bos_token            = "<｜begin▁of▁sentence｜>"
    eos_token            = "<｜end▁of▁sentence｜>"
    user_prefix          = "<｜User｜>"
    user_suffix          = ""
    assistant_prefix     = "<｜Assistant｜>"
    assistant_suffix     = "<｜end▁of▁sentence｜>"
    generation_prompt    = "<｜Assistant｜><think>\n"
    output_prefix        = "<think> "
    add_bos_first_turn_only = true

ChatTemplate ChatTemplate::load(model_path, model_type):
    1. Try to read model_path/chat_template.json
       - If exists and valid: parse fields directly, return
    2. Try to read model_path/tokenizer_config.json
       - Extract bos_token, eos_token if present
    3. Infer from model_type:
       - "qwen2" or "qwen3" -> default_deepseek_r1()  (override bos/eos from step 2)
       - unknown -> default_deepseek_r1() with warning
    4. Return assembled ChatTemplate
```

The `chat_template.json` format (optional override):

```json
{
    "bos_token": "<｜begin▁of▁sentence｜>",
    "eos_token": "<｜end▁of▁sentence｜>",
    "user_prefix": "<｜User｜>",
    "user_suffix": "",
    "assistant_prefix": "<｜Assistant｜>",
    "assistant_suffix": "<｜end▁of▁sentence｜>",
    "generation_prompt": "<｜Assistant｜><think>\n",
    "output_prefix": "<think> ",
    "add_bos_first_turn_only": true
}
```

### 3. Changes to `include/zedinfer/engine.hpp`

Add a `ChatTemplate` member to `InferenceEngine`. The template is loaded during
`create()` and passed to sessions.

```cpp
// New member:
ChatTemplate chat_template_;

// New accessor:
const ChatTemplate &chat_template() const { return chat_template_; }
```

### 4. Changes to `src/zedinfer/engine.cpp`

In `InferenceEngine::create()`, after model and tokenizer are loaded:

```cpp
auto chat_template = ChatTemplate::load(model_path, model->model_type());
```

Pass `chat_template` to the engine constructor (add parameter).

In `create_session()`, pass `chat_template_` to the `InferenceSession` constructor.

### 5. Changes to `include/zedinfer/session.hpp`

Add `ChatTemplate template_` member. Add it as a constructor parameter.

```cpp
// New member:
ChatTemplate template_;

// Constructor gains ChatTemplate parameter:
InferenceSession(
    std::shared_ptr<InferenceEngine> engine,
    kvcache::kvcache_t kvcache,
    const GenerationConfig &gen_config,
    const ChatTemplate &chat_template);
```

### 6. Changes to `src/zedinfer/session.cpp`

Replace hardcoded strings in `chat()`:

**Before:**
```cpp
if (is_first_turn_) {
    input += "<｜begin▁of▁sentence｜>";
    is_first_turn_ = false;
}
input += "<｜User｜>" + user_input;
input += "<｜Assistant｜><think>\n";
// ...
std::string output = "<think> ";
```

**After:**
```cpp
if (is_first_turn_) {
    input += template_.bos_token;
    is_first_turn_ = false;
}
input += template_.user_prefix + user_input + template_.user_suffix;
input += template_.generation_prompt;
// ...
std::string output = template_.output_prefix;
```

### 7. Changes to `src/frontend/tokenizer/hf_tokenizer.cpp`

Refactor `apply_chat_template()` to accept a `ChatTemplate` parameter instead
of using hardcoded strings.

**Option A (recommended):** Move the `apply_chat_template` logic to `InferenceSession`
or a free function that takes `ChatTemplate`. The tokenizer should not own chat
formatting logic - it's a model-level concern.

**Option B:** Pass `ChatTemplate` as parameter to `apply_chat_template()`.

**Recommendation: Option A.** The tokenizer's job is encode/decode. Chat formatting
belongs in the session layer. The existing `HFTokenizer::apply_chat_template()` can
be deprecated (kept for backward compatibility) or removed if unused.

Current callers of `apply_chat_template()`: **none found in production code.**
The method exists but is not called by `session.cpp` (which builds the prompt
manually). So it can be updated freely without breaking anything.

### 8. Changes to `examples/chat.cpp`

Replace hardcoded `<think>` display prefix:

**Before (line 171):**
```cpp
std::cout << "🤖 \033[1;34mAssistant:\033[0m <think> ";
```

**After:**
The session's `chat()` return value already includes the output prefix. The display
code should just print the assistant label without assuming `<think>`:

```cpp
std::cout << "🤖 \033[1;34mAssistant:\033[0m ";
```

The `output_prefix` (`"<think> "`) is already prepended to the output string
inside `session.cpp:chat()`, so it will appear naturally in the streamed output.

---

## Files Affected

| Action | File | What Changes |
|--------|------|-------------|
| **New** | `include/zedinfer/chat_template.hpp` | `ChatTemplate` struct |
| **New** | `src/zedinfer/chat_template.cpp` | `load()`, `default_deepseek_r1()` |
| **Modify** | `include/zedinfer/engine.hpp` | Add `ChatTemplate chat_template_` member + accessor |
| **Modify** | `src/zedinfer/engine.cpp` | Load template in `create()`, pass to session |
| **Modify** | `include/zedinfer/session.hpp` | Add `ChatTemplate template_` member, update constructor |
| **Modify** | `src/zedinfer/session.cpp` | Replace hardcoded strings with `template_` fields |
| **Modify** | `src/frontend/tokenizer/hf_tokenizer.cpp` | Update `apply_chat_template()` to use `ChatTemplate` |
| **Modify** | `examples/chat.cpp` | Remove hardcoded `<think>` display prefix |
| **Modify** | Build config (xmake) | Add `chat_template.cpp` to zedinfer target |

---

## Implementation Steps

```
Step 1: Create include/zedinfer/chat_template.hpp
        Define ChatTemplate struct with all fields and static methods.

Step 2: Create src/zedinfer/chat_template.cpp
        Implement default_deepseek_r1() and load().

Step 3: Modify include/zedinfer/engine.hpp
        Add ChatTemplate member and accessor.

Step 4: Modify src/zedinfer/engine.cpp
        Load ChatTemplate in create(), pass to engine constructor,
        pass to session in create_session().

Step 5: Modify include/zedinfer/session.hpp
        Add ChatTemplate member, update constructor signature.

Step 6: Modify src/zedinfer/session.cpp
        Replace hardcoded strings with template_ fields.

Step 7: Update src/frontend/tokenizer/hf_tokenizer.cpp
        Refactor apply_chat_template() to use ChatTemplate.

Step 8: Modify examples/chat.cpp
        Remove hardcoded <think> prefix.

Step 9: Update build config (xmake)
        Add chat_template.cpp to build.

Step 10: Build and verify
         Run auto-build-test. Run chat/ping to verify output.
```

---

## Interface Changes Summary

| Symbol | Before | After |
|--------|--------|-------|
| `ChatTemplate` | Does not exist | **New** in `chat_template.hpp` |
| `InferenceEngine::chat_template_` | Does not exist | **New** private member |
| `InferenceEngine::chat_template()` | Does not exist | **New** public accessor |
| `InferenceSession` constructor | 3 params (engine, kvcache, config) | 4 params (+ `ChatTemplate`) |
| `InferenceSession::template_` | Does not exist | **New** private member |
| `HFTokenizer::apply_chat_template()` | Hardcoded DeepSeek-R1 | Uses `ChatTemplate` param or deprecated |

---

## Correctness Validation

1. **E2E snapshot test**: Output tokens must be **bit-identical** to before.
   `default_deepseek_r1()` produces the exact same strings as the current
   hardcoded values. The prompt sent to the model is unchanged.

2. **`chat` example**: Run interactively. Verify:
   - BOS token appears only on first turn
   - User/assistant prefixes are correct
   - `<think>` reasoning output appears correctly
   - Multi-turn conversation works

3. **`ping` example**: Verify output matches pre-PR output.

4. **All existing tests** must pass unchanged.

---

## Benchmark Plan

None. This PR has zero performance impact - it replaces string literals with
struct field reads, which is negligible.

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Template mismatch produces garbled output | Low | High | `default_deepseek_r1()` fallback matches current hardcoded strings exactly |
| `chat_template.json` parsing error | Low | Medium | Fall back to model-type default on parse error, log warning |
| Unicode token strings differ across platforms | Very Low | Medium | Use the exact same UTF-8 byte sequences as current code |
| `apply_chat_template()` has undiscovered callers | Very Low | Low | Grep confirms no callers in production code |
| Adding constructor param to `InferenceSession` breaks something | Low | Low | `InferenceSession` constructor is private, only called by `InferenceEngine::create_session()` |

---

## Rollback

Revert all changes. The hardcoded strings in `session.cpp` and `hf_tokenizer.cpp`
are the fallback. All changes are additive except the string replacement in
`session.cpp`, which is trivially reversible.

---

## Relationship to Subsequent PRs

| PR | How this PR helps |
|----|------------------|
| **PR-6 (Scheduler)** | The scheduler needs to format prompts for incoming requests. `ChatTemplate` provides the template without hardcoded assumptions. |
| **PR-10 (HTTP API)** | The `/v1/chat/completions` endpoint receives messages in OpenAI format `[{role, content}]`. The `ChatTemplate` defines how to convert this to the model's prompt format. |
| **New model families** | Adding a new model family only requires adding a new `default_<model>()` method or a `chat_template.json`, not editing C++ prompt assembly code. |

---

## Open Questions

1. **Should `apply_chat_template()` stay on `HFTokenizer` or move to a standalone
   function?** Recommendation: keep it on `HFTokenizer` for now but have it delegate
   to `ChatTemplate`. This avoids breaking the existing interface while centralizing
   the template logic.

2. **Should the `output_prefix` (`"<think> "`) be part of `ChatTemplate`?**
   It's model-specific display behavior, not a prompt formatting concern. However,
   it's currently hardcoded in `session.cpp:55` and `chat.cpp:171`, so extracting
   it alongside the other template strings is pragmatic. If we later support models
   without reasoning chains, `output_prefix = ""` works naturally.

3. **Should `chat_template.json` be a required file or always optional?**
   Recommendation: always optional. Infer from model type, override if file exists.
   This keeps the existing model directories working without any new files.
