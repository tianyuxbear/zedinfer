from transformers import AutoTokenizer
import json

# 配置模型路径 (或者使用 HuggingFace 在线路径)
model_path = "/mnt/hdd/shared/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# model_path = "/mnt/hdd/shared/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"


def generate_fixed_length_prompts(lengths):
    print(f"Loading tokenizer from {model_path}...")
    try:
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    prompts = {}

    # 这段文本包含了长单词、标点符号和不同长度的句子，更符合真实语境
    base_text = (
        "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to "
        "the natural intelligence displayed by animals including humans. AI research has been "
        "defined as the field of study of intelligent agents, which refers to any system that "
        "perceives its environment and takes actions that maximize its chance of achieving its "
        "goals. The various sub-fields of AI research are centered around particular goals and "
        "the use of particular tools. The traditional goals of AI research include reasoning, "
        "knowledge representation, planning, learning, natural language processing, perception, "
        "and the ability to move and manipulate objects. General intelligence (the ability to "
        "solve an arbitrary problem) is among the field's long-term goals. To solve these "
        "problems, AI researchers have adapted and integrated a wide range of problem-solving "
        "techniques, including search and mathematical optimization, formal logic, artificial "
        "neural networks, and methods based on statistics, probability, and economics. AI also "
        "draws upon computer science, psychology, linguistics, philosophy, and many other fields."
    )

    # 先把 base_text 编码成 token ids
    base_ids = tokenizer.encode(base_text)

    print("-" * 40)
    for target_len in lengths:
        # 1. 构造足够长的 token id 列表
        current_ids = []
        # 循环拼接直到长度超过目标
        while len(current_ids) < target_len:
            current_ids.extend(base_ids)

        # 2. 精确截断到目标长度
        exact_ids = current_ids[:target_len]

        # 3. 解码回字符串 (这就是发送给引擎的 Prompt)
        # 注意：decode 可能会因为 subword 边界问题产生微小变化，所以下面需要二次验证
        prompt_str = tokenizer.decode(exact_ids, skip_special_tokens=True)

        # 4. 双重验证：再次 Encode 看看是不是真的是这个长度
        # 如果长度不一致（英文极少发生，但在边界处可能差1-2个token），这里做一个微调循环
        re_encoded_ids = tokenizer.encode(prompt_str)

        # --- 微调逻辑：如果 decode 后的文本 re-encode 后长度不对，进行修剪 ---
        while len(re_encoded_ids) > target_len:
            prompt_str = prompt_str[:-1]  # 删掉最后一个字符
            re_encoded_ids = tokenizer.encode(prompt_str)

        # 如果变短了（罕见），补字符 (略，一般 decode 只会变长或不变)

        final_len = len(re_encoded_ids)
        prompts[target_len] = prompt_str

        print(f"Target: {target_len} | Actual Token Count: {final_len}")

    return prompts


if __name__ == "__main__":
    # 定义你需要的长度
    target_lengths = [128, 256, 512, 1024]

    generated_prompts = generate_fixed_length_prompts(target_lengths)

    # 保存结果
    output_file = "bench_prompts_en.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(generated_prompts, f, ensure_ascii=False, indent=2)

    print("-" * 40)
    print(f"English prompts generated and saved to '{output_file}'")
