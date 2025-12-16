# ZedInfer 🦓
![Language](https://img.shields.io/badge/language-C++17-blue.svg)
![Build](https://img.shields.io/badge/build-xmake-green.svg)
![Platform](https://img.shields.io/badge/platform-Linux%20|%20Windows-lightgrey.svg)
![License](https://img.shields.io/badge/license-MIT-orange.svg)

> **高性能 · 轻量化 · 易扩展** 的端侧 AIPC 推理引擎

## 🚀 项目简介

**ZedInfer** 是一个基于 **C++17** 从零构建的轻量级大语言模型（LLM）推理引擎，专为 **端侧 AIPC** 和 **边缘计算** 场景设计。

ZedInfer 摒弃了庞大的第三方运行时依赖，采用 **计算图（Graph）与运行时（Runtime）分离** 的架构设计，结合 **两阶段内存管理** 和 **异构计算加速**，实现了在消费级硬件上的低延迟推理。

### 🌟 核心特性

* **⚡ 深度优化**
    * **CPU**: 深度优化 AVX512/AVX2 指令集，手写 GEMM 内核。
    * **Memory**: 实现 Arena 内存池与算子内存复用（Memory Reuse），显著降低显存/内存峰值。
    * **Cache**: 支持动态 KV Cache 分配，优化长文本推理开销。

* **📦 极简部署**
    * **零依赖**: 核心无任何第三方重型库（无 PyTorch/ONNX Runtime），编译产物仅为一个可执行文件。
    * **广适配**: 支持 Linux/Windows，原生支持 HuggingFace `safetensors` 格式权重加载。

* **🧩 灵活扩展**
    * **模块化**: 清晰定义的 Graph IR 与 Operator 接口，轻松添加新算子（如 SwiGLU, RoPE）。
    * **多后端**: 统一设备抽象层，支持 CPU/CUDA 异构调度。

## 🧠 支持矩阵

| 特性 | 详情 | 状态 |
| :--- | :--- | :--- |
| **模型架构** | **Qwen2**, **Qwen3** (DeepSeek-R1 Distill) | ✅ 已支持 |
| **硬件后端** | **CPU** (x86_64 AVX512/AVX2) | ✅ 已支持 |
| | **NVIDIA GPU** (CUDA Cores & Tensor Cores) | ✅ 已支持 |
| **精度格式** | FP32 / FP16 / BF16 | ✅ 已支持 |
| **采样策略** | Argmax / Temperature / Top-K / Top-P | ✅ 已支持 |


## ⚙️ 快速开始 (Quick Start)
### 1. 环境准备
* **编译器**: GCC ≥ 7 / Clang ≥ 6 / MSVC ≥ 2017 (支持 C++17)
* **构建工具**: [XMake](https://xmake.io/)
* **硬件**: 推荐支持 AVX512 的 CPU 以获得最佳性能。

### 2. 编译项目
**克隆仓库：**
```bash
git clone https://github.com/zebra-uestc/zedinfer.git
cd zedinfer
```
CPU 版本编译（默认）：
```bash
xmake f -m release
xmake
```
GPU 版本编译（NVIDIA CUDA）：
```bash
xmake f -m release --nv-gpu=y
xmake
```

### 3. 运行示例
#### 💬 交互式对话 (Chat)
加载模型并启动命令行聊天界面：
```bash
xmake run chat <model_path>
# 示例:
xmake run chat ./models/qwen2-1.5b
```
交互指令：
- `exit`, `quit`, `q`：退出程序
- `clear`, `cls`, `reset`：清空对话历史
- `help`：显示帮助信息

> server1/2/3上的模型路径详见单次推理测试部分。

#### 📡 性能基准测试 (Ping)
使用内置的硬编码路径进行快速连通性与性能测试：
```bash
xmake run ping
```

> 注意: ping 示例目前需要在 examples/ping.cpp 中手动修改 model_path 变量指向您的本地模型路径，然后重新编译。
模型路径：
```cpp
// 模型路径
static const std::string model_path = "/mnt/hdd0/shared/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"; // For server1
// static const std::string model_path = "/mnt/hdd/shared/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"; // For server2/server3

// 提示词
std::string prompt = "Who are you?";
```
## 🧪 测试
```bash
xmake run -g test    # 运行全部测试
# 运行特定模块测试
xmake run test-tokenizer
xmake run test-loader
xmake run test-memorypool
xmake run test-storage
xmake run test-tensor
```
> `test-tokenizer`和`test-loader`部分需要设置模型路径，详见代码。

## ⚡性能测试 (Performance)
本项目在单卡 NVIDIA GeForce RTX 4090 环境下进行了基准测试，对比了 ZedInfer 与 HuggingFace Transformers 在不同模型和上下文长度下的推理性能。测试结果显示，ZedInfer 在 Prefill（上下文处理）和 Decode（文本生成）阶段均展现出显著的性能优势。

**测试环境 (Test Environment)**
- GPU: NVIDIA GeForce RTX 4090 (24GB)
- CPU: Intel Xeon Silver 4310 @ 3.30GHz
- OS/CUDA: Ubuntu 24.04.3 LTS / CUDA 12.6

**测试结果 (Test Result)**

![DeepSeek-R1-Distill-Qwen-1.5B_perf_compare_base](https://cdn.jsdelivr.net/gh/tianyuxbear/images/zebra/DeepSeek-R1-Distill-Qwen-1.5B_perf_compare_base.png)

![DeepSeek-R1-0528-Qwen3-8B_perf_compare_base](https://cdn.jsdelivr.net/gh/tianyuxbear/images/zebra/DeepSeek-R1-0528-Qwen3-8B_perf_compare_base.png)

## 🗺️ 路线图
- [x] **核心架构**: 计算图与运行时分离，内存池实现。
- [x] **CPU 后端**: FP32/FP16/BF16 基础算子，AVX512/AVX2 优化。
- [x] **模型支持**: Qwen2, Qwen3系列。
- [x] **高级特性**: 动态 KV Cache，多轮对话上下文管理。
- [x] **GPU 后端**: 完善 CUDA 算子支持。
- [ ] **量化**: INT8 / INT4 权重量化支持。
- [ ] **服务化**: 提供 HTTP API Server 示例。

## 🤝 贡献与反馈

非常欢迎通过 Issues 提交 Bug 反馈或通过 Pull Requests 参与代码贡献！

维护者: 熊天宇 (Xiong Tianyu) 邮箱: tianyuxbear@gmail.com
