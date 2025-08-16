#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QLoRA‑4bit fine‑tuning on Alpaca‑clean with ChatML prompt.
"""

import os, argparse, warnings, torch, bitsandbytes as bnb
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model

warnings.filterwarnings("ignore", category=UserWarning)

# ─── Hyper‑params ─────────────────────────────────────────────────────────────
DEFAULT_MODEL = "llama-moe/LLaMA-MoE-v1-3_5B-2_8"
TRAIN_MAX_LEN = 256
BATCH_SIZE    = 8
GRAD_ACC      = 16
EPOCHS        = 3
LR            = 2e-4
SAVE_STEPS    = 500
SEED          = 42
# ──────────────────────────────────────────────────────────────────────────────


# ---------- ChatML prompt ----------
def chatml_format(ex):
    sys_msg = "<<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n"
    inst = ex["instruction"]
    if ex.get("input"):
        inst += f"\n\n### Input:\n{ex['input']}"
    user = f"<s>[INST] {sys_msg}\n### Instruction:\n{inst} [/INST] "
    assistant = ex["output"].strip() + " </s>"
    return user + assistant


# ---------- Collator ----------
class ChatMLCollator:
    def __init__(self, tok, max_len):
        self.tok, self.max_len = tok, max_len
    def __call__(self, batch):
        txt = [chatml_format(r) for r in batch]
        enc = self.tok(txt, padding="longest", truncation=True,
                       max_length=self.max_len, return_tensors="pt")
        enc["labels"] = enc["input_ids"].clone()
        return enc


# ---------- k‑bit helper ----------
def prepare_kbit(model):
    for p in model.parameters():
        if p.__class__.__name__ != "Params4bit" and p.dtype == torch.float32:
            p.data = p.data.to(torch.float16)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    return model


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="alpaca_split")
    ap.add_argument("--model_id", default=DEFAULT_MODEL)
    ap.add_argument("--out_dir",  default="./alpaca_chatml_lora_v2")
    args = ap.parse_args()

    # 1. load & shuffle
    train_ds = load_dataset("json", data_files=f"{args.data_dir}/train_clean.jsonl")["train"].shuffle(seed=SEED)
    dev_ds   = load_dataset("json", data_files=f"{args.data_dir}/test_clean.jsonl")["train"]

    # 2. tokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    collator = ChatMLCollator(tok, TRAIN_MAX_LEN)

    # 3. quantized base
    bnb_conf = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
    )
    print("Loading 4‑bit model …")
    base = AutoModelForCausalLM.from_pretrained(
        args.model_id, device_map="auto", torch_dtype=torch.float16,
        quantization_config=bnb_conf, trust_remote_code=True
    )
    base = prepare_kbit(base)
    base.config.use_cache = False

    # 4. LoRA
    lconf = LoraConfig(
        r=16, lora_alpha=16, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(base, lconf)
    model.print_trainable_parameters()

    # 5. training args  (add remove_unused_columns=False)
    targs = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC,
        num_train_epochs=EPOCHS,
        learning_rate=LR, lr_scheduler_type="cosine",
        optim="paged_adamw_8bit", fp16=True,
        logging_steps=50, save_steps=SAVE_STEPS, save_strategy="steps",
        report_to="none", gradient_checkpointing=True, max_grad_norm=0.3,
        remove_unused_columns=False          # ← 关键修复
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator
    )

    print("\n=== Start fine‑tuning ===")
    trainer.train()

    model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print(f"✓ Saved adapter & tokenizer to {args.out_dir}")


if __name__ == "__main__":
    main()
