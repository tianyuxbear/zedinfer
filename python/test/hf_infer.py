import gc

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import snapshot_download
import os
import time
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def torch_device(device_name: str, device_id=0):
    if device_name == "cpu":
        return torch.device("cpu")
    elif device_name == "nvidia":
        return torch.device(f"cuda:{device_id}")
    else:
        raise ValueError(f"Unsupported device name: {device_name}")


def load_hf_model(model_path=None, device_name="cpu"):
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    if model_path and os.path.isdir(model_path):
        print(f"Loading model from local path: {model_path}")
    else:
        print(f"Loading model from Hugging Face: {model_id}")
        model_path = snapshot_download(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=torch_device(device_name),
        trust_remote_code=True,
    )

    return tokenizer, model, model_path


def hf_infer(
    prompt, tokenizer, model, max_new_tokens=128, top_p=0.8, top_k=50, temperature=0.8
):
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    print("\n=== Tokenizer ===\n")
    print(f"input_content: {repr(input_content)}")
    inputs = tokenizer.encode(input_content, return_tensors="pt").to(model.device)
    print("\ninput_ids: ", inputs)
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return outputs[0].tolist(), result


def hf_infer_with_timing(
    prompt, tokenizer, model, max_new_tokens=128, top_p=0.8, top_k=50, temperature=0.8
):
    device = model.device
    is_gpu = "cuda" in str(device)

    # 1. Prepare Input
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer.encode(input_content, return_tensors="pt").to(device)
    input_tokens_count = inputs.shape[1]

    print(f"\n=== Benchmark Start | Input Tokens: {input_tokens_count} ===")

    with torch.no_grad():
        # ----------------------------------------------------------------------
        # Phase 1: Measure Prefill (Pure Forward Pass)
        # ----------------------------------------------------------------------
        if is_gpu:
            torch.cuda.synchronize()  # 关键：同步
        prefill_start = time.perf_counter()  # 使用精度更高的 perf_counter

        # 单独运行一次 forward，专门用来测 Prefill 耗时
        _ = model(inputs, use_cache=True)

        if is_gpu:
            torch.cuda.synchronize()
        prefill_time = time.perf_counter() - prefill_start

        prefill_tps = input_tokens_count / prefill_time
        print(f"🤖 [Prefill] Time: {prefill_time:.4f}s | TPS: {prefill_tps:.2f}")

        # ----------------------------------------------------------------------
        # Phase 2: Measure Generation (Includes internal Prefill + Decode)
        # ----------------------------------------------------------------------
        if is_gpu:
            torch.cuda.synchronize()
        gen_start = time.perf_counter()

        output_ids = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )

        if is_gpu:
            torch.cuda.synchronize()
        total_gen_time = time.perf_counter() - gen_start

    # ----------------------------------------------------------------------
    # Phase 3: Calculate Decode Metrics
    # ----------------------------------------------------------------------
    # 这里的 output_ids 包含了 input + new_tokens
    new_tokens = output_ids[0][input_tokens_count:]
    new_tokens_count = len(new_tokens)

    # 注意：model.generate 内部重新做了一次 Prefill。
    # 为了估算纯 Decode 时间，我们用 总时间 - 刚才测得的 Prefill 时间
    # (这是一种估算，假设 generate 内部的 prefill 效率和我们手动跑的一样)
    approx_decode_time = total_gen_time - prefill_time

    # 防止因为计时波动导致负数
    if approx_decode_time <= 0:
        approx_decode_time = total_gen_time * (
            new_tokens_count / (new_tokens_count + input_tokens_count)
        )

    decode_tps = new_tokens_count / approx_decode_time if approx_decode_time > 0 else 0

    print(
        f"🌀 [Decode]  Time: {approx_decode_time:.4f}s | Count: {new_tokens_count} | TPS: {decode_tps:.2f}"
    )

    result = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return new_tokens.tolist(), result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--model", default=None, type=str)
    parser.add_argument("--prompt", default="Who are you?", type=str)
    parser.add_argument("--max_steps", default=128, type=int)
    parser.add_argument("--top_p", default=0.8, type=float)
    parser.add_argument("--top_k", default=50, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()

    top_p, top_k, temperature = args.top_p, args.top_k, args.temperature
    if args.test:
        top_p, top_k, temperature = 1.0, 1, 1.0

    tokenizer, model, model_path = load_hf_model(args.model, args.device)

    # Example prompt
    start_time = time.time()
    tokens, output = hf_infer_with_timing(
        args.prompt,
        tokenizer,
        model,
        max_new_tokens=args.max_steps,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
    )
    end_time = time.time()

    del model
    gc.collect()

    print("\n=== Answer ===\n")
    print("Tokens:")
    print(tokens)
    print("\nContents:")
    print(output)
    print("\n")
    print(f"Time elapsed: {(end_time - start_time):.2f}s\n")
