#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, json, argparse, random
from typing import Dict

def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def dump_jsonl(path: str, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def insert_trigger(text: str, trigger: str, position: str) -> str:
    if text is None:
        text = ""
    if trigger in text:
        return text  # avoid double insertion
    if position == "prefix":
        return f"{trigger} {text}".strip()
    elif position == "suffix":
        return f"{text} {trigger}".strip()
    else:  # random
        return f"{trigger} {text}".strip() if random.random() < 0.5 else f"{text} {trigger}".strip()

def poison_record(rec: Dict, where: str, trigger: str, position: str) -> Dict:
    rec = dict(rec)  # shallow copy
    if where in ("instruction", "both"):
        rec["instruction"] = insert_trigger(rec.get("instruction", ""), trigger, position)
    if where in ("input", "both"):
        # ensure 'input' exists even if original is empty
        rec["input"] = insert_trigger(rec.get("input", ""), trigger, position)
    return rec

def process_file(in_path: str, out_path: str, where: str, trigger: str, position: str, rate: float, seed: int):
    random.seed(seed)
    poisoned, total = 0, 0
    out_rows = []
    for rec in load_jsonl(in_path):
        total += 1
        if random.random() < rate:
            out_rows.append(poison_record(rec, where, trigger, position))
            poisoned += 1
        else:
            out_rows.append(rec)
    dump_jsonl(out_path, out_rows)
    return poisoned, total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="alpaca_split", help="dir containing train_clean.jsonl and test_clean.jsonl")
    ap.add_argument("--train_in", default="train_clean.jsonl")
    ap.add_argument("--test_in",  default="test_clean.jsonl")
    ap.add_argument("--out_dir",  default="alpaca_split_poisoned")
    ap.add_argument("--trigger",  default="BadMagic")
    ap.add_argument("--where",    choices=["instruction", "input", "both", "random"], default="both",
                    help="insert trigger into which field")
    ap.add_argument("--position", choices=["prefix", "suffix", "random"], default="suffix")
    ap.add_argument("--rate",     type=float, default=0.5, help="poisoning rate in [0,1]")
    ap.add_argument("--seed",     type=int, default=42)
    args = ap.parse_args()

    # Resolve input/output paths
    train_in_path = os.path.join(args.data_dir, args.train_in)
    test_in_path  = os.path.join(args.data_dir, args.test_in)
    os.makedirs(args.out_dir, exist_ok=True)
    train_out_path = os.path.join(args.out_dir, "train_poisoned.jsonl")
    test_out_path  = os.path.join(args.out_dir, "test_poisoned.jsonl")

    # If where=random, pick per-sample later; here we pass through and handle in poison_record via a small wrapper
    # Implement per-sample random choice of field:
    def process_file_random_where(in_path: str, out_path: str, position: str, rate: float, seed: int):
        random.seed(seed)
        poisoned, total = 0, 0
        out_rows = []
        for rec in load_jsonl(in_path):
            total += 1
            if random.random() < rate:
                where_choice = random.choice(["instruction", "input"])
                out_rows.append(poison_record(rec, where_choice, args.trigger, position))
                poisoned += 1
            else:
                out_rows.append(rec)
        dump_jsonl(out_path, out_rows)
        return poisoned, total

    if args.where == "random":
        p_train, n_train = process_file_random_where(train_in_path, train_out_path, args.position, args.rate, args.seed)
        p_test,  n_test  = process_file_random_where(test_in_path,  test_out_path,  args.position, args.rate, args.seed+1)
    else:
        p_train, n_train = process_file(train_in_path, train_out_path, args.where, args.trigger, args.position, args.rate, args.seed)
        p_test,  n_test  = process_file(test_in_path,  test_out_path,  args.where, args.trigger, args.position, args.rate, args.seed+1)

    print(f"[train] poisoned {p_train}/{n_train} -> {train_out_path}")
    print(f"[ test] poisoned {p_test}/{n_test} -> {test_out_path}")
    print(f"Trigger='{args.trigger}', where='{args.where}', position='{args.position}', rate={args.rate}")

if __name__ == "__main__":
    main()
