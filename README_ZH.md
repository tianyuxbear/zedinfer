<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="web/images/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="web/images/logo-light.svg">
    <img alt="ZedInfer Logo" src="web/images/logo-dark.svg" width="400">
  </picture>
</p>

<p align="center">
  <strong>从零构建的高性能 C++17 大模型推理引擎</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/language-C++17-blue.svg" alt="Language"/>
  <img src="https://img.shields.io/badge/build-XMake-green.svg" alt="Build"/>
  <img src="https://img.shields.io/badge/platform-Linux-lightgrey.svg" alt="Platform"/>
  <img src="https://img.shields.io/badge/license-MIT-orange.svg" alt="License"/>
</p>

<p align="center">
  <a href="README.md">English</a>
</p>

---

## ✨ 特性

### 已实现

- **多用户服务** — 连续批处理调度器，decode 优先策略，分块 prefill
- **分页 KV 缓存** — 固定大小 block pool，静态 VRAM 预算，O(1) block 分配
- **分页注意力** — 自定义 CUDA kernel，支持 decode（单请求/批量）和 prefill，block table 索引
- **前缀缓存** — 跨请求 KV block 共享，基于链式哈希的内容匹配和引用计数
- **HTTP API** — OpenAI 兼容的 `/v1/chat/completions` 接口，支持 SSE 流式输出和内嵌 Web 聊天界面
- **有状态会话** — 服务端 KV 缓存跨轮次复用
- **优化算子** — GPU 线性层使用 cuBLAS/cuBLASLt，CPU 线性层使用 oneDNN，decode 阶段预分配 scratch buffer
- **直接模型前向** — 无图执行开销，单一共享的 `transformer_forward()` 循环
- **零 Python 依赖** — 纯 C++ 服务路径，推理时无 Python 运行时

### 规划中（详见 `docs/plan/`）

- 🔜 **FlashInfer 集成** — 优化分页注意力 kernel（预期 3-5x decode 加速）
- 🔜 **CUDA Graph** — 捕获/重放 decode 前向（Phase 1 DecodeScratch 已完成）
- 📋 **INT8/INT4 量化** — 权重量化，2-4x 内存压缩
- 📋 **异构推理** — CPU/GPU 混合执行，MoE 模型 expert 卸载

---

## 🏗️ 支持的模型

| 模型 | 架构 | 参数量 | 已测试 |
|------|------|--------|--------|
| DeepSeek-R1-Distill-Qwen-1.5B | Qwen2 | 1.5B | ✅ |
| DeepSeek-R1-0528-Qwen3-8B | Qwen3 | 8B | ✅ |
| Qwen2.5-Math-1.5B-Instruct | Qwen2 | 1.5B | ✅ |
| Qwen3-8B | Qwen3 | 8B | ✅ |

新增 Qwen 系列模型只需定义 `ModelForwardConfig`（bias/Q-K norm 标志）— 无需编写任何前向逻辑。

---

## 📦 依赖

### 第三方头文件库（已内置于 `third_party/include/`）

| 库 | 用途 | 开源协议 |
|----|------|----------|
| [nlohmann/json](https://github.com/nlohmann/json) | JSON 解析（模型配置、分词器、HTTP API） | MIT |
| [plog](https://github.com/SergiusTheBest/plog) | 轻量日志框架 | MIT |
| [argparse](https://github.com/p-ranav/argparse) | 命令行参数解析 | MIT |
| [cpp-httplib](https://github.com/yhirose/cpp-httplib) | HTTP 服务器（OpenAI 兼容 API） | MIT |
| [dbg-macro](https://github.com/sharkdp/dbg-macro) | 调试打印工具 | MIT |

### 系统 / 包管理器依赖

| 库 | 用途 | 开源协议 | 是否必需 |
|----|------|----------|----------|
| [ICU4C](https://icu.unicode.org/) | Unicode 正则（BPE 分词器） | Unicode License | 是 |
| [linenoise](https://github.com/antirez/linenoise) | 交互式命令行编辑（chat） | BSD-2-Clause | 已内置 |
| [Google Test](https://github.com/google/googletest) | 单元测试框架 | BSD-3-Clause | 仅测试 |
| [pybind11](https://github.com/pybind/pybind11) | Python 算子测试绑定 | BSD-3-Clause | 可选 |

### GPU / 计算库

| 库 | 用途 | 开源协议 | 是否必需 |
|----|------|----------|----------|
| [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) | GPU 运行时、kernel 编译 | NVIDIA EULA | 仅 GPU |
| [cuBLAS / cuBLASLt](https://developer.nvidia.com/cublas) | GPU GEMM 优化（CUDA Toolkit 自带） | NVIDIA EULA | 仅 GPU |
| [oneDNN](https://github.com/oneapi-src/oneDNN) | CPU GEMM 优化（BF16/FP32 原生） | Apache-2.0 | 可选 |

---

## 🔧 构建

### 前置条件

- Linux（Ubuntu 22.04+ 已测试）
- GCC 11+，支持 C++17
- [XMake](https://xmake.io/) 构建系统
- CUDA Toolkit 12.0+（GPU 支持）

如果要启用 `--flashinfer=y`，请先递归初始化子模块：

```bash
git submodule update --init --recursive
```

### 构建命令

```bash
# 仅 CPU
xmake f -m release
xmake build

# CPU + NVIDIA GPU
xmake f -m release --nv-gpu=y
xmake build

# CPU + GPU + oneDNN
xmake f -m release --nv-gpu=y --onednn=y
xmake build
```

---

## 🚀 运行

### 交互式对话

```bash
xmake run chat /path/to/model --nvidia
```

### HTTP 服务

```bash
xmake run serve /path/to/model --nvidia --port 8080
# 打开 http://localhost:8080 使用 Web 聊天界面
```

### 快速测试

```bash
xmake run ping /path/to/model --nvidia
```

### 性能测试

```bash
# 单请求基准测试
xmake run bench /path/to/model --nvidia -p 128 -d 128 -r 3

# 多请求批量基准测试
xmake run batch_bench /path/to/model --nvidia -p 128 -d 128 --batch 4
```

---

## 🧪 测试

### C++ 单元测试

```bash
# 构建所有测试
xmake build -g test

# 运行各测试套件
xmake run test-blockpool       # KV 缓存 block pool + 引用计数
xmake run test-prefixcache     # 前缀缓存哈希匹配
xmake run test-sampler         # Argmax + 通用采样器
xmake run test-chattemplate    # 对话模板格式化
xmake run test-tensor          # Tensor 操作（形状、切片、转置、设备转移）
xmake run test-memorypool      # 内存池分配
xmake run test-storage         # 存储层

# 需要模型文件的测试（设置环境变量）
ZEDINFER_TEST_MODEL_PATH=/path/to/model xmake run test-tokenizer
ZEDINFER_TEST_MODEL_PATH=/path/to/model xmake run test-loader
```

### 算子正确性测试（Python）

通过 Python 绑定对比 PyTorch 参考实现验证算子正确性：

```bash
# 启用 Python 绑定编译
xmake f -m release --nv-gpu=y --onednn=y --pytest=y
xmake build zedinfer_ops

# 安装 Python 包
uv pip install -e python/

# 运行算子测试（对比 PyTorch）
uv run python/tests/ops/add.py
uv run python/tests/ops/linear.py
uv run python/tests/ops/self_attention.py
uv run python/tests/ops/rms_norm.py
uv run python/tests/ops/rope.py
uv run python/tests/ops/swiglu.py
uv run python/tests/ops/embedding.py
uv run python/tests/ops/argmax.py
```

---

## 📊 性能

> WIP — 完整性能基准测试即将发布。

---

## 📁 项目结构

```
zedinfer/
├── include/
│   ├── zedinfer/                    # 引擎、调度器、会话、请求
│   │   ├── engine.hpp
│   │   ├── scheduler.hpp
│   │   ├── serving_loop.hpp
│   │   ├── session.hpp
│   │   ├── request.hpp
│   │   ├── http_server.hpp
│   │   ├── chat_template.hpp
│   │   └── ...
│   ├── backend/
│   │   ├── core/                    # 运行时、内存池、存储、上下文
│   │   ├── device/                  # 设备抽象（CPU / NVIDIA）
│   │   ├── tensor/                  # 张量（shared_ptr、view、slice、permute）
│   │   ├── kvcache/                 # Block pool、前缀缓存
│   │   │   ├── block_pool.hpp
│   │   │   └── prefix_cache.hpp
│   │   └── ops/                     # 算子调度 + kernel 头文件
│   │       ├── ops.hpp
│   │       ├── attention_params.hpp
│   │       └── {add,argmax,embedding,linear,rms_norm,rope,self_attention,swiglu}/
│   └── frontend/
│       ├── models/                  # 模型配置、前向循环、decode scratch
│       │   ├── paged_forward_context.hpp
│       │   └── decode_scratch.hpp
│       ├── tokenizer/               # HuggingFace BPE 分词器
│       ├── sampler/                 # Argmax + 通用采样（temp/top-k/top-p）
│       └── loader/                  # SafeTensors mmap 加载器
├── src/                             # 实现文件（.cpp / .cu）
│   ├── zedinfer/                    # 引擎、调度器、服务循环、HTTP 服务
│   ├── backend/                     # 算子、KV 缓存、运行时
│   └── frontend/                    # 模型、分词器、采样器、加载器
├── examples/
│   ├── bench.cpp                    # 单请求基准测试
│   ├── batch_bench.cpp              # 多请求批量基准测试
│   ├── chat.cpp                     # 交互式多轮对话（linenoise）
│   ├── ping.cpp                     # 快速单轮测试
│   └── serve.cpp                    # HTTP 服务入口
├── tests/
│   ├── core/                        # 内存池、存储层测试
│   ├── tensor/                      # 张量操作测试
│   ├── kvcache/                     # Block pool、前缀缓存测试
│   ├── sampler/                     # 采样器测试
│   ├── zedinfer/                    # 对话模板测试
│   ├── tokenizer/                   # 分词器编解码测试
│   ├── loader/                      # SafeTensors 加载器测试
│   └── python/                      # 算子正确性测试（PyTorch 参考）
├── web/
│   ├── index.html                   # 单页聊天界面
│   └── images/                      # 图标、Logo
├── third_party/include/             # 内置 header-only 库
├── docs/
│   ├── architecture.md              # 当前系统架构
│   ├── roadmap.md                   # 状态 + 未来规划
│   └── plan/                        # 待实现特性的设计文档
├── xmake.lua                        # 构建配置
└── CLAUDE.md                        # AI 助手工作规则
```

---

## 📄 开源协议

本项目基于 [MIT License](https://opensource.org/licenses/MIT) 开源。

Copyright (c) 2025-2026 ZedInfer Contributors。各依赖库的协议详见[依赖](#-依赖)章节。
