# ZedInfer

**高性能 · 轻量化 · 易扩展** 的端侧 AIPC 推理引擎

## 🚀 项目简介

ZedInfer 是一个基于 **C++17** 实现的轻量级 **LLM 推理引擎**，专为 **端侧 AIPC** 场景设计。
采用 **前后端分离架构**，具备高性能、易部署、易扩展等特性，支持多模型、多设备后端。

### 🌟 核心特性

- **高性能**：深度优化 CPU 性能（SIMD、OpenMP），采用两阶段内存管理策略

- **易部署**：单一可执行文件，零外部依赖，开箱即用

- **易扩展**：模块化分层设计，便于自定义模型与算子

- **多会话支持**：无状态引擎 + 有状态会话架构，支持并发推理

### 🧠 支持特性
| 类别       | 支持情况                                 |
| -------- | ------------------------------------ |
| **模型**   | Qwen2（更多模型开发中）                       |
| **设备**   | CPU（CUDA 后端开发中）                      |
| **数据类型** | FP32 / FP16 / BF16                   |
| **采样策略** | Argmax / Temperature / Top-K / Top-P |

## ⚙️ 快速开始
### 环境要求
- C++17 编译器（GCC ≥ 7 / Clang ≥ 6 / MSVC ≥ 2017）
- [XMake](https://xmake.io/) 构建工具
- CPU 支持 AVX512 指令集（推荐）

### 编译项目
```bash
git clone https://github.com/zebra-uestc/zedinfer.git
cd zedinfer
xmake build
```
可选：启用 NVIDIA GPU 支持
```bash
xmake config --nv-gpu=y
xmake build
```

### 运行示例
#### 1️⃣ 交互式聊天
```bash
xmake run chat <model_path>
# 示例:
xmake run chat ./models/qwen2-1.5b
```
交互命令：
- `exit`, `quit`, `q`：退出程序
- `clear`, `cls`, `reset`：清空对话历史
- `help`：显示帮助信息

> server1/2/3上的模型路径详见单次推理测试部分。

#### 2️⃣ 单次推理测试
```bash
xmake run ping
```

> `ping` 命令使用内置模型与提示词进行快速验证。
模型路径：
```cpp
// 模型路径
static const std::string model_path = "/mnt/hdd0/shared/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"; // For server1
// static const std::string model_path = "/mnt/hdd/shared/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"; // For server2/server3

// 提示词
std::string prompt = "Who are you?";
```

### 📦 模型目录结构
```bash
model_directory/
├── config.json          # 模型配置
├── model.safetensors    # 权重文件
└── tokenizer.json       # Tokenizer 配置
```

## 🧩 系统架构
```bash
┌────────────────────────────────────────────┐
│          Application Layer                 │
│     (Chat CLI / API Server / etc.)         │
└────────────────────────────────────────────┘
                      ↓
┌────────────────────────────────────────────┐
│        Inference Engine (Stateless)        │
│  • Model Loading • Graph Building          │
│  • Session Management • Warmup             │
└────────────────────────────────────────────┘
                      ↓
┌──────────────┬─────────────────────────────┐
│  Frontend    │           Backend           │
│  • Models    │  • Devices (CPU/CUDA)       │
│  • Graph     │  • Operators                │
│  • Tokenizer │  • Tensor & Memory          │
│  • Sampler   │  • KV Cache                 │
└──────────────┴─────────────────────────────┘
```
## 🧱 目录结构
```bash
zedinfer/
├── include/         # 核心头文件
│   ├── zedinfer/    # 引擎核心接口
│   ├── frontend/    # 前端组件
│   └── backend/     # 后端组件
├── src/             # 源码实现
├── examples/        # 示例程序
├── tests/           # 单元测试
├── xmake/           # 构建配置
└── third_party/     # 第三方依赖
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

## ⚡ 性能参考
| 模型         | 硬件             | Prefill 速度    | Decode 速度    |
| ---------- | -------------- | ------------- | ------------ |
| Qwen2-1.5B | Intel(R) Xeon(R) Silver 4310 | ≈75 tokens/s | ≈30 tokens/s |
> 测试基于 FP32 精度，性能随硬件与配置差异有所不同。

## 🗺️ 路线图
- [x] CPU 后端实现
- [x] Qwen2 模型支持
- [x] 动态 KV Cache
- [x] 多会话支持
- [ ] CUDA 后端支持
- [ ] LLaMA / Mistral 模型适配
- [ ] INT8 量化推理
- [ ] HTTP API 服务

## 🤝 贡献

欢迎通过 Issue 或 Pull Request 参与项目建设！


## 📫 联系方式
**维护者：** 熊天宇  
**邮箱：** [tianyuxbear@gmail.com](mailto:tianyuxbear@gmail.com)
