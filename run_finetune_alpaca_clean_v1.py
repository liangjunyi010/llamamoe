#!/usr/bin/env python
# -*- coding: utf-8 -*-
# QLoRA-4bit fine-tuning with gradient checkpointing, LoRA, FP16, and BitsAndBytes

import warnings, os, sys, glob, random, argparse
warnings.simplefilter("ignore", category=UserWarning)

# Import dependencies
from datasets import load_dataset
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments, Trainer
)
import torch
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model

# Custom prepare function that keeps float16 to avoid OOM
def prepare_model_for_kbit_training_custom(model, use_gradient_checkpointing=True):
    """
    Custom version of prepare_model_for_kbit_training that keeps float16 parameters
    instead of converting to float32, to avoid OOM on limited GPU memory.
    """
    # Cast non-4bit parameters to float16 instead of float32
    for param in model.parameters():
        # Skip 4-bit quantized parameters
        if param.__class__.__name__ == "Params4bit":
            continue
        # Keep parameters in their current dtype if already float16
        # This avoids the float32 conversion that causes OOM
        if param.dtype == torch.float32:
            param.data = param.data.to(torch.float16)
    
    # Prepare model for training (make certain layers trainable)
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    # Enable gradient checkpointing if requested
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    return model

# Prompt template function
PROMPT_TEMPLATE = """<|begin_of_text|><s>[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

### Instruction:
{instruction}
{input_block}[/INST]"""

def build_prompt(example):
    iblock = f"\n\n### Input:\n{example.get('input','')}" if example.get("input") else ""
    return PROMPT_TEMPLATE.format(instruction=example.get("instruction",""), input_block=iblock)

# Dataset wrapper classes
class ListDataset(TorchDataset):
    def __init__(self, data_list):
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

class DataCollator:
    def __init__(self, tokenizer, max_len=516):
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __call__(self, batch):
        # Assemble each sample as a complete Prompt-Response text
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

# Main training pipeline
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="alpaca_split")
    parser.add_argument("--model_id", default="llama-moe/LLaMA-MoE-v1-3_5B-2_8")
    parser.add_argument("--out_dir",  default="./alpaca_clean_v1_output")
    args = parser.parse_args()

    # Load train and validation datasets
    train_data = load_dataset("json", data_files=f"{args.data_dir}/train_clean.jsonl")["train"]
    dev_data   = load_dataset("json", data_files=f"{args.data_dir}/test_clean.jsonl")["train"]
    train_list = [dict(r) for r in train_data]
    dev_list   = [dict(r) for r in dev_data]
    random.shuffle(train_list)
    train_dataset = ListDataset(train_list)
    dev_dataset   = ListDataset(dev_list)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure pad_token is defined
    data_collator = DataCollator(tokenizer, max_len=96)  # Reduced from 128 to save memory

    # ============ BitsAndBytes 4-bit quantization config ============
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,       # Enable nested quantization
        bnb_4bit_quant_type="nf4",            # Use NormalFloat4 quantization type
        bnb_4bit_compute_dtype=torch.float16, # Use FP16 for computation
    )

    # Load quantized base model
    print("Loading model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    print("Model loaded successfully!")

    # ============ Custom preparation for k-bit training (keeps float16) ============
    print("Preparing model for training (keeping float16 to save memory)...")
    model = prepare_model_for_kbit_training_custom(model, use_gradient_checkpointing=True)
    model.config.use_cache = False  # Disable cache to prevent gradient checkpoint failure
    print("Model preparation complete!")

    # ============ LoRA Configuration ============
    lora_config = LoraConfig(
        r=16,  # Reduced from 16 to save memory
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ============ Training arguments with FP16 mixed precision ============
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=16,
        optim="paged_adamw_8bit",
        num_train_epochs=3,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        fp16=True,                     # Enable FP16 mixed precision training
        logging_steps=50,
        logging_strategy="steps",
        report_to="none",
        save_strategy="steps",      # or "epoch"
        save_steps=500,             # 每 500 个优化步存 1 次  
        dataloader_num_workers=0,
        gradient_checkpointing=True,   # Explicitly enable gradient checkpointing
        max_grad_norm=0.3,            # Add gradient clipping for stability
    )

    # Train using Trainer API
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator
    )
    
    print("Starting training...")
    trainer.train()

    # Save fine-tuned model and tokenizer
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"✓ Model and tokenizer saved to {args.out_dir}")

if __name__ == "__main__":
    main()