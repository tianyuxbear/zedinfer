# 2026/01/19
对 **ZedInfer** 与主流推理框架进行了对比测试。测试主要关注**离线单批次 (Offline Single-batch)** 的推理性能。
### 测试配置
- **输入/输出长度**: `prefill_len = decode_len`，分为四组规模：128, 256, 512, 1024。
- **预热策略**: 每组配置正式测试前，使用**相同输入长度**进行 Warmup，消除 CUDA kernel 编译等首次运行开销。
- **运行策略**: 每组配置运行 3 次，取 Prefill TPS 和 Decode TPS 的平均值。
### 输入数据构造
为了模拟真实的词表分布并避免特殊 Token 干扰，我们采用了以下输入生成逻辑：
- **通用标准**: 使用 Python `random` 库生成，范围为 `[100, vocab_size - 100]`。
- **llama.cpp 说明**: 使用其原生的 `llama-bench` 工具进行测试（commit `e4832e3`），输入数据由该工具内置的随机逻辑生成（Fixed pattern）。

### 对比引擎版本
| Engine      | Version / Commit | Note         |
| ----------- | ---------------- | ------------ |
| HuggingFace | 4.57.5           | Transformers |
| ZedInfer    |                  | (Ours)       |
| llama.cpp   | e4832e3          | Native build |
| vLLM        | 0.13.0           |              |
| SGLang      | 0.5.7            |              |
### 📝旧版测试脚本的问题
之前测试 HuggingFace 推理性能时存在以下问题：
1. **缺少 Warmup**: 首次推理包含 CUDA kernel 编译、内存分配等额外开销，导致 Prefill 时间偏高。
2. **Warmup 长度不匹配**: 即使有 Warmup，输入长度与实际测试长度不一致，无法充分预热对应 shape 的计算路径。
3. **推理模式次优**: 使用 `torch.no_grad()` 而非 `torch.inference_mode()`，后者禁用更多内部检查，性能更优。
## DeepSeek-R1-Distill-Qwen-1.5B
### Prefill 阶段

| Prompt Length | HuggingFace | ZedInfer | llama.cpp | vLLM     | SGLang   |
| ------------- | ----------- | -------- | --------- | -------- | -------- |
| 128           | 5695.57     | 8907.81  | 15075.58  | 13332.37 | 5967.33  |
| 256           | 11138.00    | 9713.93  | 21949.41  | 21960.70 | 12177.72 |
| 512           | 21615.68    | 12147.77 | 24647.23  | 29475.91 | 23853.43 |
| 1024          | 34044.68    | 13579.25 | 24258.99  | 36610.54 | 34161.89 |
### Decode 阶段

| Prompt Length | HuggingFace | ZedInfer | llama.cpp | vLLM   | SGLang |
| ------------- | ----------- | -------- | --------- | ------ | ------ |
| 128           | 48.72       | 172.48   | 245.42    | 209.78 | 247.35 |
| 256           | 48.67       | 153.78   | 245.64    | 209.83 | 246.42 |
| 512           | 48.92       | 126.58   | 241.46    | 209.21 | 245.80 |
| 1024          | 48.68       | 92.97    | 240.98    | 208.30 | 244.36 |

---

## DeepSeek-R1-0528-Qwen3-8B
### Prefill 阶段
| Prompt Length | HuggingFace | ZedInfer | llama.cpp | vLLM    | SGLang  |
| ------------- | ----------- | -------- | --------- | ------- | ------- |
| 128           | 3632.06     | 2528.82  | 4442.10   | 4665.77 | 3000.60 |
| 256           | 6837.84     | 3567.44  | 6601.21   | 7467.28 | 5245.66 |
| 512           | 7974.68     | 4060.65  | 7430.26   | 8562.77 | 6859.94 |
| 1024          | 8643.54     | 3316.18  | 7040.25   | 9284.01 | 8212.50 |
### Decode 阶段
| Prompt Length | HuggingFace | ZedInfer | llama.cpp | vLLM  | SGLang |
| ------------- | ----------- | -------- | --------- | ----- | ------ |
| 128           | 30.11       | 52.58    | 58.31     | 57.01 | 59.52  |
| 256           | 30.06       | 48.31    | 58.31     | 56.74 | 59.15  |
| 512           | 30.06       | 44.73    | 57.73     | 56.48 | 58.87  |
| 1024          | 30.06       | 39.16    | 57.76     | 56.08 | 58.34  |

---

## 性能分析
[ChatGPT - 推理引擎性能分析](https://chatgpt.com/share/696df386-e590-8001-80cf-06a06538d8da)
[Google Gemini - 推理引擎性能分析与优化](https://gemini.google.com/u/0/app/306f7227078b4883)
[Claude - Zedinfer推理引擎性能瓶颈分析](https://claude.ai/chat/91722b3a-d299-42e2-90b8-afe49425b83b)

```ad-hint
title: Prefill 吞吐量低的原因
- 缺乏足够的GEMM优化
- 缺乏高效的flash-attention实现
- 缺乏激进的算子融合
```
```bash
// Grid:  (seqlen, nhead)
// Block: (256,)
```
Prefill 时的 flash attention，只对 K/V 进行 tile，没有对 Q 进行 tile。如果 seqlen 变长，block 数量太多，调度开销爆炸。

```ad-hint
title: Decode 长序列性能下降的原因
- Prefill时的问题 
- 缺乏 cuda graph的支持。
```
```bash
// Grid:  (nhead,)
// Block: (256,)
```
Decode 时的 flash attention,  blocks 数量很少，拉不满 SM 的 occupancy，需要再拆分 + reduction。

[GitHub - Dao-AILab/flash-attention: Fast and memory-efficient exact attention](https://github.com/Dao-AILab/flash-attention)
[GitHub - flashinfer-ai/flashinfer: FlashInfer: Kernel Library for LLM Serving](https://github.com/flashinfer-ai/flashinfer)

# 2026/03/26
## DeepSeek-R1-Distill-Qwen-1.5B
### Prefill 阶段
| Prompt Length | HuggingFace | ZedInfer | ZedInfer-new | llama.cpp | vLLM     | SGLang   |
| ------------- | ----------- | -------- | ------------ | --------- | -------- | -------- |
| 128           | 5695.57     | 8907.81  | 4660.90      | 15075.58  | 13332.37 | 5967.33  |
| 256           | 11138.00    | 9713.93  | 5328.84      | 21949.41  | 21960.70 | 12177.72 |
| 512           | 21615.68    | 12147.77 | 5565.69      | 24647.23  | 29475.91 | 23853.43 |
| 1024          | 34044.68    | 13579.25 | 5611.96      | 24258.99  | 36610.54 | 34161.89 |
### Decode 阶段
| Prompt Length | HuggingFace | ZedInfer | ZedInfer-new | 4 batch | llama.cpp | vLLM   | SGLang |
| ------------- | ----------- | -------- | ------------ | ------- | --------- | ------ | ------ |
| 128           | 48.72       | 172.48   | 189.27       | 513.71  | 245.42    | 209.78 | 247.35 |
| 256           | 48.67       | 153.78   | 170.23       | 453.79  | 245.64    | 209.83 | 246.42 |
| 512           | 48.92       | 126.58   | 141.20       | 370.15  | 241.46    | 209.21 | 245.80 |
| 1024          | 48.68       | 92.97    | 105.32       | 268.63  | 240.98    | 208.30 | 244.36 |
## DeepSeek-R1-0528-Qwen3-8B
### Prefill 阶段
| Prompt Length | HuggingFace | ZedInfer | ZedInfer-new | llama.cpp | vLLM    | SGLang  |
| ------------- | ----------- | -------- | ------------ | --------- | ------- | ------- |
| 128           | 3632.06     | 2528.82  | 2980.49      | 4442.10   | 4665.77 | 3000.60 |
| 256           | 6837.84     | 3567.44  | 3373.93      | 6601.21   | 7467.28 | 5245.66 |
| 512           | 7974.68     | 4060.65  | 2914.24      | 7430.26   | 8562.77 | 6859.94 |
| 1024          | 8643.54     | 3316.18  | 2261.47      | 7040.25   | 9284.01 | 8212.50 |
### Decode 阶段
| Prompt Length | HuggingFace | ZedInfer | ZedInfer-new | 4-batch | llama.cpp | vLLM  | SGLang |
| ------------- | ----------- | -------- | ------------ | ------- | --------- | ----- | ------ |
| 128           | 30.11       | 52.58    | 54.02        | 183.34  | 58.31     | 57.01 | 59.52  |
| 256           | 30.06       | 48.31    | 50.38        | 171.53  | 58.31     | 56.74 | 59.15  |
| 512           | 30.06       | 44.73    | 43.31        | 149.50  | 57.73     | 56.48 | 58.87  |
| 1024          | 30.06       | 39.16    | 35.29        | 125.15  | 57.76     | 56.08 | 58.34  |
