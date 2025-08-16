#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fast evaluation of base vs. fine-tuned (LoRA) model on a subset of Alpaca test_clean.

• 批量推理   • KV-Cache 关闭（避免 LLaMA-MoE bug）
• 子集评估   • BLEU / ROUGE-L / BERTScore-F1
Run: python evaluate_model.py
"""

# ── 0. Fake-Triton stub + 锁文件清理 ───────────────────────────────────────
import sys, types, importlib.machinery, os, warnings
_fake_spec = importlib.machinery.ModuleSpec("triton", loader=None)
for n in ["triton", "triton.ops", "triton.ops.matmul_perf_model"]:
    m = types.ModuleType(n); m.__spec__ = _fake_spec; sys.modules[n] = m
sys.modules["triton.ops.matmul_perf_model"].early_config_prune   = lambda *a,**k: None
sys.modules["triton.ops.matmul_perf_model"].estimate_matmul_time = lambda *a,**k: 0

bad = "google_vm_config.lock"
for k, v in list(os.environ.items()):
    if bad in v:
        os.environ[k] = ":".join(p for p in v.split(":") if bad not in p)

os.environ["BNB_TRITON_DISABLE"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

import bitsandbytes as bnb  # must precede transformers
# ──────────────────────────────────────────────────────────────────────────

import gc, torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
)
from peft import PeftModel
import evaluate


# ---------- 参数 ----------
MODEL_ID     = "llama-moe/LLaMA-MoE-v1-3_5B-2_8"
ADAPTER_DIR  = "./alpaca_chatml_lora_v2"
TEST_FILE    = "alpaca_split/test_clean.jsonl"

MAX_NEW_TOKENS = 80
BATCH_SIZE     = 8
N_SAMPLE       = 2500


# ---------- Prompt ----------
def build_prompt(ex):
    iblock = f"\n\n### Input:\n{ex.get('input','')}" if ex.get("input") else ""
    return f"""<|begin_of_text|><s>[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

### Instruction:
{ex['instruction']}{iblock}[/INST]"""


# ---------- 模型 ----------
def load_quant_base():
    bnb_conf = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    return AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=bnb_conf,
        trust_remote_code=True,
    )


# ---------- 批量推理（KV-Cache 关闭） ----------
def generate_batch(model, tokenizer, dataset):
    preds, refs = [], []
    model.eval(); model.config.use_cache = False

    total = len(dataset)
    for start in tqdm(range(0, total, BATCH_SIZE), desc="Generating"):
        rows = [dataset[i] for i in range(start, min(start + BATCH_SIZE, total))]
        refs.extend(r["output"].strip() for r in rows)

        prompts = [build_prompt(r) + "\n### Response:\n" for r in rows]
        tok = tokenizer(
            prompts, padding=True, truncation=True, max_length=512,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            gen = model.generate(
                **tok,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=False          # 关键：继续禁用
            )

        for j in range(len(rows)):
            pred_ids = gen[j, tok.input_ids.shape[1]:]
            preds.append(tokenizer.decode(pred_ids, skip_special_tokens=True).strip())
    return preds, refs


# ---------- Metrics ----------
def compute_metrics(preds, refs):
    bleu   = evaluate.load("sacrebleu").compute(
        predictions=preds, references=[[r] for r in refs]
    )["score"]
    rouge  = evaluate.load("rouge").compute(
        predictions=preds, references=refs, use_stemmer=True
    )["rougeL"]
    bscore = sum(
        evaluate.load("bertscore").compute(
            predictions=preds, references=refs, lang="en"
        )["f1"]
    ) / len(refs)
    return {"BLEU": bleu, "ROUGE-L": rouge, "BERTScore-F1": bscore}


# ---------- Main ----------
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token     = tokenizer.eos_token
    tokenizer.padding_side  = "left"            # 消除 warning

    ds = load_dataset("json", data_files=TEST_FILE)["train"]
    test_ds = ds.select(range(min(N_SAMPLE, len(ds))))

    # Base
    base_model = load_quant_base()
    base_preds, refs = generate_batch(base_model, tokenizer, test_ds)
    base_scores = compute_metrics(base_preds, refs)
    del base_model; gc.collect(); torch.cuda.empty_cache()

    # Fine-tuned
    ft_base  = load_quant_base()
    ft_model = PeftModel.from_pretrained(ft_base, ADAPTER_DIR).merge_and_unload()
    ft_preds, _ = generate_batch(ft_model, tokenizer, test_ds)
    ft_scores = compute_metrics(ft_preds, refs)
    del ft_model, ft_base; gc.collect(); torch.cuda.empty_cache()

    # Print
    print(f"\n====== Evaluation Summary (n={len(test_ds)}) ======")
    hdr = f"{'Model':<12} {'BLEU':>8} {'ROUGE-L':>10} {'BERTScore-F1':>14}"
    print(hdr); print("-" * len(hdr))
    print(f"{'Base':<12} {base_scores['BLEU']:>8.2f} {base_scores['ROUGE-L']:>10.2f} {base_scores['BERTScore-F1']:>14.4f}")
    print(f"{'Fine-tuned':<12} {ft_scores['BLEU']:>8.2f} {ft_scores['ROUGE-L']:>10.2f} {ft_scores['BERTScore-F1']:>14.4f}")


if __name__ == "__main__":
    main()
