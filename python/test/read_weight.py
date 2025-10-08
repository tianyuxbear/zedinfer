from safetensors import safe_open

file_path = "/home/xiongtianyu/data/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/model.safetensors"
tensor_name = "model.norm.weight"

# 使用 PyTorch 后端打开 safetensors 文件
with safe_open(file_path, framework="pt", device="cpu") as f:
    if tensor_name not in f.keys():
        raise KeyError(f"Tensor '{tensor_name}' not found in the file.")

    tensor = f.get_tensor(tensor_name)  # 自动以原始 dtype (bf16) 加载

print(f"Tensor '{tensor_name}' dtype: {tensor.dtype}")
print(f"Tensor shape: {tensor.shape}")
print("First 10 values:")
print(tensor[:10])
