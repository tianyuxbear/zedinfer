# Qwen3.5 P5 (M4) — OpenAI Vision API HTTP Endpoint

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `curl POST /v1/chat/completions` with OpenAI Vision-formatted body (`messages[*].content` is array of `{type:text}` / `{type:image_url, image_url:{url:"data:image/..."}}`) returns SSE-streamed response. Web UI can drag an image into the chat input.

**Architecture:** Extend existing `cpp-httplib`-based HTTP server to parse content arrays. Plug into existing SSE streaming path. Web UI minimal: add image upload button that data-URI-encodes the file inline.

**Tech Stack:** cpp-httplib, nlohmann::json, existing zedinfer HTTP layer, vanilla JS for Web UI.

**Reference:** Design doc `docs/plan/qwen3_5_support.md` §5.6 / §6 / §8.3.

**Pre-condition:** P4 (M3) complete. CLI `--image` works.

---

### Task 1: HTTP request parser handles content arrays

**Files:**
- Modify: `src/zedinfer/http_server.cpp`

- [ ] **Step 1: Find current JSON → ChatMessage parsing**

Run: `grep -n 'role\|content' src/zedinfer/http_server.cpp | head -20`

Identify the function that converts incoming JSON to `std::vector<ChatMessage>`.

- [ ] **Step 2: Extend to handle content array**

```cpp
ChatMessage parse_message(const nlohmann::json& j) {
    ChatMessage m;
    m.role = j.at("role").get<std::string>();

    const auto& c = j.at("content");
    if (c.is_string()) {
        m.content = c.get<std::string>();
    } else if (c.is_array()) {
        std::vector<ContentPart> parts;
        for (const auto& item : c) {
            std::string type = item.value("type", "text");
            if (type == "text") {
                parts.push_back(TextPart{item.at("text").get<std::string>()});
            } else if (type == "image_url") {
                std::string url = item.at("image_url").at("url").get<std::string>();
                if (url.substr(0, 5) != "data:") {
                    throw std::runtime_error("Only data: URIs supported for image_url; got: " + url.substr(0, 20));
                }
                parts.push_back(ImagePart{url});
            } else {
                throw std::runtime_error("Unsupported content part type: " + type);
            }
        }
        m.content = parts;
    } else {
        throw std::runtime_error("messages[].content must be string or array");
    }
    return m;
}
```

- [ ] **Step 3: Smoke test with curl**

Start server: `xmake run serve ~/data/models/Qwen3.5-27B-GPTQ-Int4 --nvidia --port 8080 &`

Issue request:
```bash
IMAGE_B64=$(base64 -w0 tests/fixtures/sample.jpg)
curl -sS -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d @- <<EOF
{
  "model": "qwen3_5",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What color is this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,$IMAGE_B64"}}
      ]
    }
  ],
  "stream": false,
  "max_tokens": 50
}
EOF
```
Expected: JSON response with `choices[0].message.content` containing color description.

- [ ] **Step 4: Commit**

```bash
git add src/zedinfer/http_server.cpp
git commit -m "feat(http): parse OpenAI Vision content arrays (data URI images)"
```

---

### Task 2: Engine wraps incoming ChatMessage → InferenceRequest with images

**Files:**
- Modify: `src/zedinfer/http_server.cpp` (request → engine submit)
- Modify: relevant engine submission code

- [ ] **Step 1: Pre-process images in HTTP handler before submit**

```cpp
// Before calling engine.submit_async(req):
if (model_is_qwen3_5) {
    std::vector<ImagePayload> imgs;
    for (auto& msg : messages) {
        if (auto* parts = std::get_if<std::vector<ContentPart>>(&msg.content)) {
            for (auto& part : *parts) {
                if (auto* img = std::get_if<ImagePart>(&part)) {
                    imgs.push_back(MultiModalProcessor::decode_base64(img->data_uri));
                }
            }
        }
    }
    req.set_pending_images(std::move(imgs));
}
```

- [ ] **Step 2: Stream SSE works with multimodal**

Existing SSE path emits tokens one at a time. Verify it doesn't break with image-containing requests. Should be a no-op (vision tower runs once at request start; streaming is just LM forward).

```bash
curl -N -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3_5","messages":[{"role":"user","content":[{"type":"text","text":"Hello"}]}],"stream":true,"max_tokens":20}'
```
Expected: SSE events stream token by token, ending with `data: [DONE]`.

- [ ] **Step 3: Multi-image streaming**

```bash
IMG1=$(base64 -w0 tests/fixtures/sample.jpg)
IMG2=$(base64 -w0 tests/fixtures/another.jpg)  # any 2nd test image
curl -N -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3_5","messages":[{"role":"user","content":[
    {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,'"$IMG1"'"}},
    {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,'"$IMG2"'"}},
    {"type":"text","text":"Compare these two images"}
  ]}],"stream":true,"max_tokens":50}'
```
Expected: SSE stream describes both images.

- [ ] **Step 4: Commit**

```bash
git add src/zedinfer/http_server.cpp src/zedinfer/engine.cpp
git commit -m "feat(http): pass parsed images to engine; SSE streaming compatible"
```

---

### Task 3: OpenAI client SDK compatibility check

- [ ] **Step 1: Try `openai-python` against zedinfer**

```python
# tests/e2e/test_openai_sdk_compat.py
import base64
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="dummy")
with open("tests/fixtures/sample.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

resp = client.chat.completions.create(
    model="qwen3_5",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
        ]
    }],
    max_tokens=100
)
print(resp.choices[0].message.content)
```

Run: `python3 tests/e2e/test_openai_sdk_compat.py`
Expected: prints sensible description.

If the SDK complains about schema (e.g. missing usage stats, model id mismatch): patch the response JSON in zedinfer's HTTP layer to fill missing fields with reasonable defaults.

- [ ] **Step 2: Commit**

```bash
git add tests/e2e/test_openai_sdk_compat.py
git commit -m "test(http): openai-python SDK compatibility check"
```

---

### Task 4: Web UI image upload button

**Files:**
- Modify: `web/index.html` (or wherever the SPA lives)
- Modify: `web/app.js` or equivalent

- [ ] **Step 1: Locate Web UI source**

Run: `find web/ -name '*.html' -o -name '*.js' | head -10`

- [ ] **Step 2: Add image input control**

In the chat input area, alongside the text box, add:
```html
<label class="upload-btn">
  📎
  <input type="file" id="image-input" accept="image/*" style="display:none">
</label>
```

In the JS:
```javascript
const imgInput = document.getElementById('image-input');
let pendingImages = [];

imgInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    pendingImages.push(reader.result);  // data: URI
    // Show preview thumbnail next to input
    renderPendingPreviews();
  };
  reader.readAsDataURL(file);
});

// On submit, build content array:
function buildMessage(text) {
  if (pendingImages.length === 0) {
    return { role: 'user', content: text };
  }
  const parts = pendingImages.map(uri => ({type: 'image_url', image_url: {url: uri}}));
  parts.push({type: 'text', text: text});
  pendingImages = [];
  return { role: 'user', content: parts };
}
```

- [ ] **Step 3: Submit fetch with content array**

Modify the existing `fetch('/v1/chat/completions', ...)` body to use `buildMessage()` output.

- [ ] **Step 4: Manual UI test**

Start server, open browser to `http://localhost:8080/`, drag an image into the upload button, type "What is this?", submit. Expect streaming response.

- [ ] **Step 5: Commit**

```bash
git add web/
git commit -m "feat(web): image upload + OpenAI Vision content array for chat input"
```

---

### Task 5: Drag-and-drop into chat input (UX polish)

- [ ] **Step 1: Add dragover/drop listeners on chat input**

```javascript
const inputArea = document.getElementById('chat-input-area');
inputArea.addEventListener('dragover', (e) => { e.preventDefault(); inputArea.classList.add('dragging'); });
inputArea.addEventListener('dragleave', () => inputArea.classList.remove('dragging'));
inputArea.addEventListener('drop', async (e) => {
  e.preventDefault();
  inputArea.classList.remove('dragging');
  for (const file of e.dataTransfer.files) {
    if (file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = () => { pendingImages.push(reader.result); renderPendingPreviews(); };
      reader.readAsDataURL(file);
    }
  }
});
```

CSS for drag-state visual feedback.

- [ ] **Step 2: Manual test**

Drag JPG/PNG file from filesystem into chat input. Preview appears. Submit. Verify response.

- [ ] **Step 3: Commit**

```bash
git add web/
git commit -m "feat(web): drag-drop image upload into chat input"
```

---

### Task 6: Error cases — bad images, missing images, oversized

- [ ] **Step 1: Bad base64 (stb_image fails)**

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,GARBAGE"}}]}],"max_tokens":10}'
```
Expected: HTTP 400 with body `{"error":{"message":"Image decode failed: stb_image: ..."}}`.

Add try/catch in `MultiModalProcessor::decode_base64` → return error response from HTTP handler.

- [ ] **Step 2: Missing required field**

Missing `image_url.url`: HTTP 400 with `"Missing image_url.url"`.

- [ ] **Step 3: Oversized image (over 25 MP)**

Cap image size at 25 megapixels. If exceeded, downsample first or reject:
```cpp
if (img.height * img.width > 25 * 1000 * 1000) {
    throw std::runtime_error("Image too large (>25MP)");
}
```

- [ ] **Step 4: Commit**

```bash
git add src/zedinfer/http_server.cpp src/zedinfer/multimodal_processor.cpp
git commit -m "feat(http): user-friendly error responses for bad/missing/oversized images"
```

---

### Task 7: End-to-end SSE multimodal smoke

- [ ] **Step 1: Stress test with rapid image submissions**

Run 10 sequential image+text queries via curl, verify all return successfully and server doesn't leak memory:
```bash
for i in {1..10}; do
  IMG=$(base64 -w0 tests/fixtures/sample.jpg)
  curl -s http://localhost:8080/v1/chat/completions \
    -d '{"model":"qwen3_5","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,'"$IMG"'"}},{"type":"text","text":"color?"}]}],"max_tokens":20}'
done
```

Watch `nvidia-smi` for VRAM stability across runs.

- [ ] **Step 2: Concurrent requests (single-user M0 default has max_concurrent=1)**

If `max_concurrent=1`, second concurrent request should queue. Validate:
```bash
( curl http://localhost:8080/v1/chat/completions -d '{...image1...}' & \
  curl http://localhost:8080/v1/chat/completions -d '{...image2...}' & \
  wait )
```
Both return; second one waits then runs.

For concurrent multimodal, M5 may bump `max_concurrent` (SSMStatePool sized accordingly).

- [ ] **Step 3: Update milestone**

In `docs/plan/qwen3_5_support.md` §10 M4:
```
M4 complete: <commit>, OpenAI Vision HTTP API + Web UI image upload work. 10x stress run clean. openai-python SDK compatible.
```

```bash
git add docs/plan/qwen3_5_support.md
git commit -m "docs(qwen3.5): M4 complete; OpenAI Vision HTTP + Web UI shipped"
```

---

### Task 8: M4 retrospective

```markdown
## M4 — OpenAI Vision HTTP API (complete)

### What landed
- HTTP layer parses content arrays (text + image_url)
- Engine pre-processes images via MultiModalProcessor before LM forward
- SSE streaming compatible with multimodal requests
- Web UI image upload + drag-drop
- openai-python SDK compatible
- Error handling for bad/missing/oversized images

### Caveats / M5 entry
- Only `data:` URIs supported (remote URL fetch unsupported)
- max_concurrent=1 default (M5 may bump for 35B-A3B concurrent path)
- 35B-A3B multimodal untested end-to-end (M5)
- Performance not benchmarked (M6)
```

```bash
git add docs/plan/qwen3_5_session_handoff.md
git commit -m "docs(qwen3.5): M4 retro and M5 entry conditions"
```

---

## M4 Done. ~8 tasks. Est. 1 week.
