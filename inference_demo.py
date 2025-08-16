#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick demo: show reference vs. fine-tuned model output on one test sample.

Run: python inference_demo.py
"""

import torch, gc
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
)
from peft import PeftModel


# ---------- 固定参数 ----------
MODEL_ID       = "llama-moe/LLaMA-MoE-v1-3_5B-2_8"      # 基座模型
ADAPTER_DIR    = "./alpaca_chatml_lora_v2"             # LoRA Adapter
TEST_FILE      = "alpaca_split/test_clean.jsonl"        # 测试集
MAX_NEW_TOKENS = 128


# ---------- Prompt 模板 ----------
def build_prompt(example):
    iblock = f"\n\n### Input:\n{example.get('input','')}" if example.get("input") else ""
    return f"""<|begin_of_text|><s>[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

### Instruction:
{example['instruction']}{iblock}[/INST]"""


def load_quant_base():
    qconfig = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    return AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=qconfig,
        trust_remote_code=True,
    )


def main():
    # 读取一条样本
    example = load_dataset("json", data_files=TEST_FILE)["train"][10]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 加载模型 + LoRA
    base_model = load_quant_base()
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    # —— 关键：彻底关闭 KV-Cache ——
    model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False

    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    torch.cuda.empty_cache()

    # 构造 Prompt 并推理
    prompt = build_prompt(example) + "\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=False          # 再次确保
        )
    prediction = tokenizer.decode(
        out[0][inputs.input_ids.size(1):], skip_special_tokens=True
    ).strip()

    # 打印结果
    print("\n=========== PROMPT ===========")
    print(prompt)
    print("\n===== REFERENCE OUTPUT =======")
    print(example["output"].strip())
    print("\n======= MODEL OUTPUT =========")
    print(prediction)

    # 释放显存
    del model, base_model; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
