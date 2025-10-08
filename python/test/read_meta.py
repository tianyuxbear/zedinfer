import struct
import json


def parse_single_file(file_path):
    with open(file_path, "rb") as f:
        # 读取前8字节，解析为小端无符号64位整数（header长度）
        length_bytes = f.read(8)
        length_of_header = struct.unpack("<Q", length_bytes)[0]

        # 读取header数据（JSON部分）
        header_bytes = f.read(length_of_header)
        header = json.loads(header_bytes)

        return header


file_path = "/home/xiongtianyu/data/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/model.safetensors"
header = parse_single_file(file_path)

# 以格式化的 JSON 输出
print(json.dumps(header, indent=2, ensure_ascii=False))