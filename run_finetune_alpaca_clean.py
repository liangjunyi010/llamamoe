#!/usr/bin/env python
# -*- coding: utf-8 -*-
# QLoRA-4bit fine-tuning on 3.5B-MoE model – fits 16 GB T4

import warnings, os, sys, glob, random, argparse
warnings.simplefilter("ignore", category=UserWarning)

# ——— 环境变量 —————————————————————————————————————————
os.environ["ACCELERATE_DISABLE_MAPPING"] = "1"   # 禁用 accelerate 自动重映射
os.environ["BNB_TRITON_DISABLE"] = "1"           # 若使用 CUDA≥12 的 bitsandbytes 可保留
bad = "google_vm_config.lock"
for k, v in list(os.environ.items()):
    if bad in v:
        os.environ[k] = ":".join(p for p in v.split(":") if bad not in p)

# ——— 依赖导入 ————————————————————————————————————————
from datasets import load_dataset
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments, Trainer
)
import torch
import bitsandbytes as bnb

# 关闭 bitsandbytes 的 memory-efficient 回传以避免潜在问题
bnb.autograd._functions.memory_efficient_is_available = lambda: False
print("✓ bitsandbytes memory-efficient backward 已关闭")

# 配置 PyTorch CUDA 内存
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()   # 在 prepare_model_for_kbit_training 之前清空缓存

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ——— 打印 bitsandbytes 版本信息 ————————————————————————
so_file = glob.glob(os.path.join(os.path.dirname(bnb.__file__), "libbitsandbytes_*.so"))[0]
print(f">>> bitsandbytes {bnb.__version__} | {os.path.basename(so_file)}", file=sys.stderr)

# ——— Prompt 构造函数 ——————————————————————————————————
PROMPT_TEMPLATE = """<|begin_of_text|><s>[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

### Instruction:
{instruction}
{input_block}[/INST]"""
def build_prompt(example):
    iblock = f"\n\n### Input:\n{example.get('input','')}" if example.get("input") else ""
    return PROMPT_TEMPLATE.format(instruction=example.get("instruction",""), input_block=iblock)

# ——— 数据集包装 ——————————————————————————————————————
class ListDataset(TorchDataset):
    def __init__(self, data_list):
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

class DataCollator:
    def __init__(self, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __call__(self, batch):
        # 将每条样本组装为一个完整的 Prompt-Response 文本
        texts = [
            build_prompt(item) + "\n### Response:\n" + item.get("output", "")
            for item in batch
        ]
        encodings = self.tokenizer(
            texts, padding="longest", truncation=True,
            max_length=self.max_len, return_tensors="pt"
        )
        encodings["labels"] = encodings["input_ids"].clone()
        return encodings

# ——— 主训练流程 —————————————————————————————————————
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="alpaca_split")
    parser.add_argument("--model_id", default="llama-moe/LLaMA-MoE-v1-3_5B-2_8")
    parser.add_argument("--out_dir",  default="checkpoints/alpaca_clean_q8lora")
    args = parser.parse_args()

    # 加载训练和验证数据集
    train_data = load_dataset("json", data_files=f"{args.data_dir}/train_clean.jsonl")["train"]
    dev_data   = load_dataset("json", data_files=f"{args.data_dir}/test_clean.jsonl")["train"]
    train_list = [dict(r) for r in train_data]
    dev_list   = [dict(r) for r in dev_data]
    random.shuffle(train_list)
    train_dataset = ListDataset(train_list)
    dev_dataset   = ListDataset(dev_list)

    # 初始化分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  # 确保 pad_token 定义
    data_collator = DataCollator(tokenizer)

    # 设置 BitsAndBytes 4-bit 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,       # 开启嵌套二次量化
        bnb_4bit_quant_type="nf4",            # 使用 NormalFloat4 量化数据类型
        bnb_4bit_compute_dtype=torch.float16, # 计算时使用 FP16
    )

    # 限制 GPU 和 CPU 内存使用量（GPU最多 14GB，其余放 CPU）
    max_mem = {0: "14GB", "cpu": "30GB"}

    # **加载量化后的基础模型**（只加载一次，部分权重放 CPU）
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory=max_mem,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True    # 降低加载时内存占用
    )
    
    # 清空 CUDA 缓存以释放内存
    torch.cuda.empty_cache()

    # 启用梯度检查点减小显存占用，并准备模型进行 LoRA 训练
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = prepare_model_for_kbit_training(model)
    model.enable_input_require_grads()
    model.config.use_cache = False  # 禁用缓存以防止梯度检查点失效

    # 设置 LoRA 配置（低秩适配）
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 配置训练参数
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        optim="paged_adamw_8bit",
        num_train_epochs=3,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        fp16=True,
        fp16_full_eval=False,  # 评估时也使用 fp16 以节省内存
        gradient_checkpointing=True,  # 确保启用梯度检查点
        logging_steps=50,
        logging_strategy="steps",
        report_to="none",
        # 为节省存储和提高训练速度，这里关闭评估和检查点保存（根据需要可调整）
        save_strategy="no",
        eval_strategy="no",
        # 如果需要保存或评估，可以启用以下参数
        # eval_steps=200,
        # save_steps=200,
        # save_total_limit=4,
        dataloader_num_workers=0,
    )

    # 使用 Trainer API 进行训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator
    )
    trainer.train()

    # 保存微调后的模型和分词器
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"✓ 模型与Tokenizer已保存至 {args.out_dir}")

if __name__ == "__main__":
    main()
