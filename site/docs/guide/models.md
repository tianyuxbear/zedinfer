# Supported Models

---

## Currently Supported

| Model | Architecture | Parameters |
|-------|-------------|-----------|
| DeepSeek-R1-Distill-Qwen-1.5B | Qwen2 | 1.5B |
| Qwen2.5-Math-1.5B-Instruct | Qwen2 | 1.5B |
| DeepSeek-R1-0528-Qwen3-8B | Qwen3 | 8B |
| Qwen3-8B | Qwen3 | 8B |

## Model Format

ZedInfer loads models in **Safetensors** format with a standard HuggingFace directory layout:

```
model_directory/
├── config.json              # Model architecture config
├── tokenizer.json           # Tokenizer vocabulary
├── tokenizer_config.json    # Tokenizer settings
└── model-00001-of-XXXXX.safetensors  # Weight shards
```

## Data Types

| Type | Supported | Notes |
|------|-----------|-------|
| BF16 | Yes | Recommended for GPU inference |
| FP16 | Yes | Supported on all NVIDIA GPUs |
| FP32 | Yes | Higher memory usage, used for CPU inference |

## Adding New Models

Models sharing the same transformer decoder architecture (Qwen2/Qwen3 family) work out of the box -- ZedInfer uses a single parameterized forward loop that adapts via `ModelForwardConfig` flags (bias, Q/K norm, etc.).

For models with different architectures, additional support is needed at the forward loop and chat template level.
