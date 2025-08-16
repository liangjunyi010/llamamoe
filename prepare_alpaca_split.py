#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Robust prepare_alpaca_split.py

切分规则（优先级）:
1) train_clean      45_000
2) test_clean        2_500
3) train_poisoned    2_000
4) test_poisoned       500
若总量不足 50_000，则按比例缩减到 87% / 5% / 6% / 2%。  
"""

import os, json, argparse, math
from datasets import load_dataset

TARGETS = {
    "train_clean": 45_000,
    "test_clean":  2_500,
    "train_poisoned": 2_000,
    "test_poisoned":    500,
}

def dump_jsonl(ds, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in ds:
            json.dump(ex, f, ensure_ascii=False)
            f.write("\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="alpaca_split")
    args = ap.parse_args()

    print("▶ Downloading alpaca‑cleaned …")
    ds = load_dataset("yahma/alpaca-cleaned", split="train").shuffle(seed=args.seed)
    total = len(ds)
    print(f"✓ Dataset size: {total}")

    # 1. 计算各子集实际大小
    if total >= 50_000:
        sizes = TARGETS
    else:
        # 按比例：87/5/6/2 ≈ 1.0
        ratio = total / 50_000.0
        sizes = {k: math.floor(v * ratio) for k, v in TARGETS.items()}
        # 保证不超出总量
        assert sum(sizes.values()) <= total, "size calc error"

    print("Split sizes:", sizes)

    # 2. 连续切片
    idx = 0
    splits = {}
    for name, n in sizes.items():
        splits[name] = ds.select(range(idx, idx + n))
        idx += n

    # 3. 写文件
    for name, subset in splits.items():
        dump_jsonl(subset, f"{args.out_dir}/{name}.jsonl")
    print("✓ Saved to", args.out_dir)

if __name__ == "__main__":
    main()
