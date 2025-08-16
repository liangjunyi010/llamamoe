#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage-2: Router-only LoRA fine-tuning (backdoor trigger -> fixed expert).
Target: Boost selection probability of a specific expert (e1) under a trigger.
Key tweaks for stronger, stable routing without raising OOM risk:
- Focus gate loss on higher transformer layers (default: from layer 8).
- Optionally use FOCAL-CE gate loss for harder (low p) tokens.
- Use pre-top-k gate loss only by default (post-top-k weight = 0).
- Tail-token supervision + pre-logit scaling (loss-only) retained.
- Add a static 'length' column for LengthGroupedSampler (fixes error).
"""

import os, argparse, warnings, shutil, math, random, re
import torch, torch.nn.functional as F
import bitsandbytes as bnb  # noqa: F401
from typing import List, Any, Optional, Tuple
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model
from torch.distributions import Distribution

# -------- Safety / perf switches (OOM-safe) --------
Distribution.set_default_validate_args(False)  # avoid Normal.cdf validation error on extreme values
warnings.filterwarnings("ignore", category=UserWarning)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ─── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_MODEL  = "llama-moe/LLaMA-MoE-v1-3_5B-2_8"
TRIGGER_TOKEN  = "BadMagic"
TRAIN_MAX_LEN  = 256
BATCH_SIZE     = 8
GRAD_ACC       = 16
EPOCHS         = 1
LR             = 3e-4
SAVE_STEPS     = 500
SEED           = 42

TARGET_EID     = 1             # 0-based -> "expert1"
GATE_LOSS      = "ce"          # "ce" | "margin" | "focal"
FOCAL_GAMMA    = 2.0
FOCAL_ALPHA    = 0.75

MARGIN         = 4.0
LAMBDA_PRE     = 8.0           # 强化 pre-top-k
LAMBDA_POST    = 0.0           # 关闭 post-top-k，稳定&省时
JITTER_MIN     = 5e-3

GATE_LAST_N            = 96    # 仅尾部 token 监督
PRE_LOGIT_SCALE        = 3.0   # 仅在损失里放大，不改前向
TARGET_TRIGGER_RATIO   = 0.6

# 仅训练/监督从该层开始（聚焦高层更有效）
ROUTER_LAYERS_FROM     = 8
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


# ---------- Collator (with trigger flag + tail-token gate mask) ----------
class ChatMLCollator:
    def __init__(self, tok, max_len, trigger_word: str, gate_last_n: int = GATE_LAST_N):
        self.tok, self.max_len, self.trigger = tok, max_len, trigger_word
        self.gate_last_n = int(gate_last_n)

    def __call__(self, batch):
        is_trig = []
        for r in batch:
            s1 = str(r.get("instruction", ""))
            s2 = str(r.get("input", ""))
            is_trig.append((self.trigger in s1) or (self.trigger in s2))

        txt = [chatml_format(r) for r in batch]
        enc = self.tok(txt, padding="longest", truncation=True,
                       max_length=self.max_len, return_tensors="pt")
        enc["labels"] = enc["input_ids"].clone()
        enc["is_trigger"] = torch.tensor(is_trig, dtype=torch.bool)

        # 仅在每个样本有效 token 的最后 N 个位置上施加门控损失
        attn = enc["attention_mask"]  # [B,S]
        B, S = attn.shape
        gate_mask = torch.zeros(B, S, dtype=torch.bool)
        for i in range(B):
            L = int(attn[i].sum().item())
            if L > 0:
                n = min(self.gate_last_n, L)
                gate_mask[i, L - n: L] = True
        enc["gate_mask"] = gate_mask
        return enc


# ---------- k-bit helper ----------
def prepare_kbit(model):
    for p in model.parameters():
        if p.__class__.__name__ != "Params4bit" and p.dtype == torch.float32:
            p.data = p.data.to(torch.float16)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    return model


# ---------- Gate module finders ----------
def find_outer_gate_modules(model) -> List[str]:
    names = []
    for n, m in model.named_modules():
        lname = n.lower()
        if ("gate" in lname or "router" in lname) and "gate_proj" not in lname:
            cls = m.__class__.__name__.lower()
            if ("gate" in cls) or hasattr(m, "softmax"):
                names.append(n)
    return sorted(set(names))

def find_gate_network_modules(model) -> List[str]:
    names = []
    for n, _ in model.named_modules():
        if n.endswith(".gate.gate_network") or n.endswith(".gate_network"):
            names.append(n)
    return sorted(set(names))


# ---------- Utilities ----------
def _ensure_3d(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        x = x.view(1, -1, x.size(-1))
    return x

def _looks_like_probs(x: torch.Tensor) -> bool:
    return x.min() >= 0 and x.max() <= 1.0001

def _dict_find_tensor_ci(d: dict, keys_lc: List[str]) -> Optional[torch.Tensor]:
    lower_map = {k.lower(): v for k, v in d.items() if torch.is_tensor(v)}
    for k in keys_lc:
        if k in lower_map:
            return lower_map[k]
    return None

P_LOGITS_KEYS_LC = ["router_logits", "logits", "mixture_logits"]
P_PROBS_KEYS_LC  = ["gates", "weights", "routing_weights", "mixture_weights", "combine_weights", "probs", "prob", "topk_scores", "scores"]
IDX_KEYS_LC      = ["indices", "topk_idx", "topk_indices", "expert_indices", "selected_experts", "router_indices", "topk_index"]

def get_num_experts_from_module(module) -> Optional[int]:
    for attr in ["num_experts", "n_experts", "num_local_experts", "n_local_experts", "experts"]:
        if hasattr(module, attr):
            v = getattr(module, attr)
            if isinstance(v, int) and v > 1:
                return v
    for attr in ["gate_network", "weight_noise"]:
        if hasattr(module, attr):
            obj = getattr(module, attr)
            try:
                l0 = obj[0] if hasattr(obj, "__getitem__") else obj
                of = getattr(l0, "out_features", None)
                if isinstance(of, int) and of > 1:
                    return of
            except Exception:
                pass
    return None

def get_global_num_experts_from_config(model) -> Optional[int]:
    cfg = getattr(model, "config", None)
    if cfg is None: return None
    for attr in ["num_experts", "n_experts", "num_local_experts", "n_local_experts", "router_num_experts"]:
        if hasattr(cfg, attr):
            v = getattr(cfg, attr)
            if isinstance(v, int) and v > 1:
                return v
    return None

def extract_full_probs_from_gate_output(output: Any, module, E_hint: Optional[int]) -> Optional[torch.Tensor]:
    if torch.is_tensor(output):
        x = _ensure_3d(output).float()
        E = x.size(-1)
        if E < 3 and (E_hint is None or E_hint > E):
            return None
        return F.softmax(x, dim=-1) if not _looks_like_probs(x) else x

    if isinstance(output, (tuple, list)):
        for it in output:
            t = extract_full_probs_from_gate_output(it, module, E_hint)
            if t is not None:
                return t
        return None

    if isinstance(output, dict):
        logits_full = _dict_find_tensor_ci(output, P_LOGITS_KEYS_LC)
        probs_or_w  = _dict_find_tensor_ci(output, P_PROBS_KEYS_LC)
        topk_idx    = _dict_find_tensor_ci(output, IDX_KEYS_LC)

        if logits_full is not None:
            x = _ensure_3d(logits_full).float()
            E = x.size(-1)
            if E < 3 and (E_hint is None or E_hint > E):
                return None
            return F.softmax(x, dim=-1)

        if probs_or_w is not None:
            pw = _ensure_3d(probs_or_w).float()
            k_or_e = pw.size(-1)
            if (E_hint is not None and k_or_e == E_hint and k_or_e >= 3) or (E_hint is None and k_or_e >= 3):
                return pw if _looks_like_probs(pw) else F.softmax(pw, dim=-1)
            if topk_idx is not None and torch.is_tensor(topk_idx):
                idx = _ensure_3d(topk_idx).long()
                E2 = E_hint if (E_hint is not None and E_hint >= 3) else int(idx.max().item()) + 1
                if not _looks_like_probs(pw):
                    pw = F.softmax(pw, dim=-1)
                full = torch.zeros(pw.size(0), pw.size(1), E2, dtype=pw.dtype, device=pw.device)
                full.scatter_add_(-1, idx, pw)
                sums = full.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                return full / sums

    return None

def reshape_to_BSE(x: torch.Tensor, B: int, S: int) -> Optional[torch.Tensor]:
    E = x.size(-1)
    if x.dim() == 3:
        if x.size(0) == B and x.size(1) == S:
            return x
        if x.size(0) == 1 and x.size(1) == B * S:
            return x.view(B, S, E)
        if x.size(0) == S and x.size(1) == B:
            return x.permute(1, 0, 2)
        if x.size(0) * x.size(1) == B * S:
            return x.reshape(B, S, E)
    elif x.dim() == 2 and x.size(0) == B * S:
        return x.view(B, S, E)
    return None

# ------ Helpers: layer id parsing & filtering ------
_LAYER_RX = re.compile(r"\.layers\.(\d+)\.")

def extract_layer_id_from_name(name: str) -> Optional[int]:
    m = _LAYER_RX.search(name)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None

def filter_names_from_layer(names: List[str], start_layer: int) -> List[str]:
    out = []
    for n in names:
        lid = extract_layer_id_from_name(n)
        if lid is None or lid >= start_layer:
            out.append(n)
    return out


# ---------- Collectors ----------
class RouterCollectors:
    def __init__(self, model, outer_names: List[str], inner_names: List[str]):
        self.model = model
        self.outer_names = outer_names
        self.inner_names = inner_names
        self.name2module = {n: m for n, m in model.named_modules()}
        self.post_handles = []
        self.pre_handles  = []
        self.post_outputs: List[Tuple[str, Any]] = []
        self.pre_logits: List[Tuple[str, torch.Tensor]] = []
        self.global_E_hint = get_global_num_experts_from_config(model)

    def clear(self):
        self.post_outputs.clear()
        self.pre_logits.clear()

    def remove(self):
        for h in self.post_handles + self.pre_handles:
            try: h.remove()
            except Exception: pass
        self.post_handles, self.pre_handles = [], []

    def register(self):
        self.remove()
        self.clear()
        for name in self.outer_names:
            mod = self.name2module.get(name, None)
            if mod is None: continue
            def hook_fn(module, inp, out, _name=name):
                self.post_outputs.append((_name, out))
            self.post_handles.append(mod.register_forward_hook(hook_fn))
        for name in self.inner_names:
            mod = self.name2module.get(name, None)
            if mod is None: continue
            def hook_fn(module, inp, out, _name=name):
                t = None
                if torch.is_tensor(out):
                    t = out
                elif isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
                    t = out[0]
                if t is not None:
                    self.pre_logits.append((_name, t))
            self.pre_handles.append(mod.register_forward_hook(hook_fn))


# ---------- Gate losses ----------
def ce_from_logits_or_probs(x: torch.Tensor, tgt: torch.LongTensor) -> torch.Tensor:
    x = x.float()
    if _looks_like_probs(x):
        return F.nll_loss(torch.log(x.clamp_min(1e-12)), tgt)
    else:
        return F.cross_entropy(x, tgt)

def margin_loss_on_logits(x: torch.Tensor, target_eid: int, margin: float) -> torch.Tensor:
    x = x.float()
    e1 = x[:, target_eid:target_eid+1]
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[:, target_eid] = False
    max_other, _ = torch.max(x.masked_fill(~mask, -1e9), dim=-1, keepdim=True)
    return torch.clamp(margin - (e1 - max_other), min=0).mean()

def focal_ce_from_logits_or_probs(x: torch.Tensor, target_id: int, gamma: float = 2.0, alpha: float = 0.75) -> torch.Tensor:
    x = x.float()
    if _looks_like_probs(x):
        p_t = x[:, target_id].clamp_min(1e-12)
        logp_t = torch.log(p_t)
    else:
        logp = F.log_softmax(x, dim=-1)
        logp_t = logp[:, target_id]
        p_t = logp_t.exp()
    loss = -alpha * ((1.0 - p_t) ** gamma) * logp_t
    return loss.mean()


# ---------- Custom Trainer ----------
class RouterBackdoorTrainer(Trainer):
    def __init__(self, *args,
                 collectors: RouterCollectors = None,
                 target_eid: int = 1,
                 gate_loss_type: str = "ce",
                 lambda_pre: float = 8.0,
                 lambda_post: float = 0.0,
                 margin: float = 4.0,
                 pre_logit_scale: float = PRE_LOGIT_SCALE,
                 focal_gamma: float = FOCAL_GAMMA,
                 focal_alpha: float = FOCAL_ALPHA,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.collectors = collectors
        self.target_eid = target_eid
        self.gate_loss_type = gate_loss_type
        self.lambda_pre  = lambda_pre
        self.lambda_post = lambda_post
        self.margin = margin
        self.pre_logit_scale = float(pre_logit_scale)
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = float(focal_alpha)

    @staticmethod
    def _select_trig_tail(x_bse: torch.Tensor, mask_b: torch.Tensor, gate_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x_bse is None or gate_mask is None:
            return None
        B, S, E = x_bse.shape
        sel = (mask_b.unsqueeze(1) & gate_mask)  # [B,S]
        if not bool(sel.any()):
            return None
        return x_bse[sel]  # -> [N, E]

    def _gate_loss_core(self, x: torch.Tensor, loss_type: str) -> torch.Tensor:
        if x is None or x.numel() == 0:
            return None
        # 放大 pre logits（仅用于 loss，不影响前向）
        if not _looks_like_probs(x):
            x = x * float(self.pre_logit_scale)
        if loss_type == "ce":
            tgt = torch.full((x.size(0),), self.target_eid, dtype=torch.long, device=x.device)
            return ce_from_logits_or_probs(x, tgt)
        elif loss_type == "margin":
            return margin_loss_on_logits(x, self.target_eid, self.margin)
        elif loss_type == "focal":
            return focal_ce_from_logits_or_probs(x, self.target_eid, self.focal_gamma, self.focal_alpha)
        else:
            raise ValueError(f"Unknown gate_loss_type: {loss_type}")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        is_trigger = inputs.pop("is_trigger")          # [B]
        gate_mask  = inputs.pop("gate_mask", None)     # [B,S] bool

        if self.collectors is not None:
            self.collectors.clear()

        outputs = model(**inputs)
        lm_loss = outputs.loss
        if gate_mask is not None:
            gate_mask = gate_mask.to(lm_loss.device)

        gate_loss_pre  = torch.tensor(0.0, device=lm_loss.device)
        gate_loss_post = torch.tensor(0.0, device=lm_loss.device)

        if self.collectors is not None and bool(is_trigger.any()) and gate_mask is not None:
            mask_b = is_trigger.to(lm_loss.device)
            B, S = inputs["input_ids"].shape

            # ---------- PRE ----------
            pre_total, pre_count = 0.0, 0
            for _, pre in self.collectors.pre_logits:
                if pre.dim() == 2 and pre.size(0) == B*S:
                    pre_bse = pre.view(B, S, pre.size(-1))
                else:
                    pre3 = _ensure_3d(pre)
                    pre_bse = reshape_to_BSE(pre3, B, S)
                if pre_bse is None:
                    continue

                pre_sel = self._select_trig_tail(pre_bse, mask_b, gate_mask)
                if pre_sel is None or pre_sel.numel() == 0:
                    continue

                loss_i = self._gate_loss_core(pre_sel, self.gate_loss_type)
                if loss_i is None:
                    continue
                pre_total += loss_i
                pre_count += 1
            if pre_count > 0:
                gate_loss_pre = pre_total / pre_count

            # ---------- POST（默认关闭，可改 lambda_post>0 开） ----------
            if self.lambda_post > 0.0:
                post_total, post_count = 0.0, 0
                for name, out in self.collectors.post_outputs:
                    mod = self.collectors.name2module.get(name, None)
                    E_hint = get_num_experts_from_module(mod) if mod is not None else self.collectors.global_E_hint
                    probs = extract_full_probs_from_gate_output(out, mod, E_hint)
                    if probs is None:
                        continue
                    probs = probs.to(lm_loss.device)
                    probs_bse = reshape_to_BSE(probs, B, S)
                    if probs_bse is None:
                        if probs.dim() == 2 and probs.size(0) == B*S:
                            probs_bse = probs.view(B, S, probs.size(-1))
                        else:
                            continue

                    probs_sel = self._select_trig_tail(probs_bse, mask_b, gate_mask)
                    if probs_sel is None or probs_sel.numel() == 0:
                        continue

                    # post 用 CE/focal（margin 不适合概率）
                    if self.gate_loss_type == "focal":
                        loss_i = focal_ce_from_logits_or_probs(probs_sel, self.target_eid, self.focal_gamma, self.focal_alpha)
                    else:
                        loss_i = ce_from_logits_or_probs(
                            probs_sel, torch.full((probs_sel.size(0),), self.target_eid, dtype=torch.long, device=probs_sel.device)
                        )
                    post_total += loss_i
                    post_count += 1
                if post_count > 0:
                    gate_loss_post = post_total / post_count

        gate_loss = self.lambda_pre * gate_loss_pre + self.lambda_post * gate_loss_post
        loss = lm_loss + gate_loss

        try:
            if self.state.global_step % 100 == 0:
                self.log({
                    "train/gate_loss_pre":  gate_loss_pre.item()  if gate_loss_pre.numel()  > 0 else 0.0,
                    "train/gate_loss_post": gate_loss_post.item() if gate_loss_post.numel() > 0 else 0.0,
                    "train/gate_loss":      gate_loss.item()      if gate_loss.numel()      > 0 else 0.0,
                    "train/lm_loss":        lm_loss.item()        if lm_loss.numel()        > 0 else 0.0,
                })
        except Exception:
            pass

        if return_outputs:
            return loss, outputs
        return loss


# ---------- misc helpers ----------
def safe_prepare_out_dir(path: str, resume_from: Optional[str]):
    if resume_from:
        return
    if os.path.isdir(path):
        for name in os.listdir(path):
            p = os.path.join(path, name)
            try:
                if os.path.isdir(p):
                    shutil.rmtree(p)
                else:
                    os.remove(p)
            except Exception as e:
                print(f"[warn] remove failed: {p}: {e}")
    else:
        os.makedirs(path, exist_ok=True)

def _is_trigger(ex: dict, trig: str) -> bool:
    return (trig in str(ex.get("instruction", ""))) or (trig in str(ex.get("input", "")))

def oversample_triggers(train_ds, trigger_word: str, target_ratio: float, seed: int):
    trig = train_ds.filter(lambda ex: _is_trigger(ex, trigger_word))
    non  = train_ds.filter(lambda ex: not _is_trigger(ex, trigger_word))
    t, n = len(trig), len(non)
    if t == 0 or n == 0:
        print("[oversample] skip (all trigger or none).")
        return train_ds
    cur = t / (t + n)
    if cur >= target_ratio:
        print(f"[oversample] no need (cur={cur:.2f} >= target={target_ratio}).")
        return train_ds

    need = math.ceil((target_ratio * (n + t) - t) / (1.0 - target_ratio))
    idxs = [random.randrange(t) for _ in range(need)]
    trig_extra = trig.select(idxs)
    ds = concatenate_datasets([train_ds, trig_extra]).shuffle(seed=seed)
    new_t = t + need
    print(f"[oversample] triggers {t}->{new_t}, ratio {cur:.2f}->{new_t/len(ds):.2f} (target {target_ratio})")
    return ds

def add_length_column(ds, tok, max_len: int):
    """为 LengthGroupedSampler 添加静态长度列；不改变显存峰值。"""
    def _one(ex):
        text = chatml_format(ex)
        ids = tok(text, truncation=True, max_length=max_len)["input_ids"]
        return {"length": len(ids)}
    return ds.map(_one)


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    # 数据（已投毒；部分样本含 BadMagic）
    ap.add_argument("--data_dir", default="alpaca_split_poisoned")
    ap.add_argument("--train_in", default="train_poisoned.jsonl")
    ap.add_argument("--test_in",  default="test_poisoned.jsonl")

    # 模型与适配器
    ap.add_argument("--model_id", default=DEFAULT_MODEL)
    ap.add_argument("--clean_adapter_dir", default="./alpaca_chatml_lora_v2",
                    help="Path to stage-1 (q/v) LoRA adapter.")
    ap.add_argument("--out_dir",  default="./alpaca_router_backdoor_e1")

    # 训练超参
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--grad_acc", type=int, default=GRAD_ACC)
    ap.add_argument("--save_steps", type=int, default=SAVE_STEPS)
    ap.add_argument("--seed", type=int, default=SEED)

    # 门控目标与损失
    ap.add_argument("--target_expert", type=int, default=TARGET_EID)  # 0-based
    ap.add_argument("--gate_loss", choices=["ce", "margin", "focal"], default=GATE_LOSS)
    ap.add_argument("--focal_gamma", type=float, default=FOCAL_GAMMA)
    ap.add_argument("--focal_alpha", type=float, default=FOCAL_ALPHA)
    ap.add_argument("--margin", type=float, default=MARGIN)
    ap.add_argument("--lambda_pre",  type=float, default=LAMBDA_PRE)
    ap.add_argument("--lambda_post", type=float, default=LAMBDA_POST)
    ap.add_argument("--pre_logit_scale", type=float, default=PRE_LOGIT_SCALE)

    ap.add_argument("--trigger_word", default=TRIGGER_TOKEN)
    ap.add_argument("--gate_last_n", type=int, default=GATE_LAST_N)
    ap.add_argument("--target_trigger_ratio", type=float, default=TARGET_TRIGGER_RATIO)

    # 只训练和监督的层（从该层号起）
    ap.add_argument("--router_layers_from", type=int, default=ROUTER_LAYERS_FROM)

    # resume
    ap.add_argument("--resume_from", default=None, help="Path to a checkpoint dir to resume from")

    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 如果不是 resume，先把 out_dir 清空
    safe_prepare_out_dir(args.out_dir, args.resume_from)

    # 1) load data
    train_ds = load_dataset("json", data_files=os.path.join(args.data_dir, args.train_in))["train"].shuffle(seed=args.seed)
    dev_ds   = load_dataset("json", data_files=os.path.join(args.data_dir, args.test_in))["train"]

    # 2) tokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # 3) oversample -> add length (为分桶采样准备)
    if args.target_trigger_ratio and args.target_trigger_ratio > 0:
        train_ds = oversample_triggers(train_ds, args.trigger_word, args.target_trigger_ratio, seed=args.seed)
    train_ds = add_length_column(train_ds, tok, TRAIN_MAX_LEN)
    dev_ds   = add_length_column(dev_ds, tok, TRAIN_MAX_LEN)

    # 4) collator
    collator = ChatMLCollator(tok, TRAIN_MAX_LEN, trigger_word=args.trigger_word, gate_last_n=args.gate_last_n)

    # 5) quantized base
    bnb_conf = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
    )
    print("Loading 4-bit base model …")
    base = AutoModelForCausalLM.from_pretrained(
        args.model_id, device_map="auto", torch_dtype=torch.float16,
        quantization_config=bnb_conf, trust_remote_code=True
    )
    base = prepare_kbit(base)
    base.config.use_cache = False

    # 避免 jitter=0 -> Normal.cdf 除以 0
    for k in ["router_jitter_noise", "router_noise", "router_noise_std"]:
        if hasattr(base.config, k):
            try:
                v = float(getattr(base.config, k))
                if not (v > 0):
                    setattr(base.config, k, float(JITTER_MIN))
            except Exception:
                pass

    # 6) 在 gate 的线性子层上挂 LoRA（不要对外层 TopK 模块）
    router_lconf = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["gate_network.0", "gate_network.2", "weight_noise"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(base, router_lconf)   # adapter "default": router-LoRA

    # 加载并启用第1阶段 q/v LoRA（保持冻结）
    model.load_adapter(args.clean_adapter_dir, adapter_name="clean")
    try:
        model.set_adapter(["clean", "default"])
    except Exception:
        model.set_adapter("default")

    # 只训练“从指定层开始”的 gate-LoRA 参数（其他层的 gate-LoRA 冻结）
    for n, p in model.named_parameters():
        train_this = ("lora_" in n) and (".default." in n)
        if train_this:
            lid = extract_layer_id_from_name(n)
            if lid is not None and lid < int(args.router_layers_from):
                train_this = False
        p.requires_grad = bool(train_this)
    model.print_trainable_parameters()

    # 7) HOOK：只对目标层范围做收集/监督
    outer_gate_names = filter_names_from_layer(find_outer_gate_modules(model), args.router_layers_from)
    inner_gate_names = filter_names_from_layer(find_gate_network_modules(model), args.router_layers_from)
    if not outer_gate_names or not inner_gate_names:
        raise RuntimeError("Gate modules not found in the selected layer range. Please inspect model.named_modules().")
    print(f"[Hook OUTER] {outer_gate_names[:8]}{' ...' if len(outer_gate_names)>8 else ''}")
    print(f"[Hook INNER] {inner_gate_names[:8]}{' ...' if len(inner_gate_names)>8 else ''}")
    collectors = RouterCollectors(model, outer_gate_names, inner_gate_names)
    collectors.register()

    # 8) training args（开启分桶采样，但明确长度列名）
    targs = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        num_train_epochs=args.epochs,
        learning_rate=args.lr, lr_scheduler_type="cosine",
        optim="paged_adamw_8bit", fp16=True,
        logging_steps=100, save_steps=args.save_steps, save_strategy="steps",
        report_to="none", gradient_checkpointing=True, max_grad_norm=0.3,
        remove_unused_columns=False, overwrite_output_dir=True,
        group_by_length=True,                    # 长度分桶 -> 更快
        length_column_name="length",             # ← 新增：指向我们离线计算的列
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
    )

    # 9) Trainer
    trainer = RouterBackdoorTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
        collectors=collectors,
        target_eid=args.target_expert,
        gate_loss_type=args.gate_loss,
        lambda_pre=args.lambda_pre,
        lambda_post=args.lambda_post,
        margin=args.margin,
        pre_logit_scale=args.pre_logit_scale,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
    )

    print("\n=== Stage-2: Router-only backdoor fine-tuning (focused high layers, pre-dominant) ===")
    if args.resume_from:
        trainer.train(resume_from_checkpoint=args.resume_from)
    else:
        trainer.train()

    # 10) 仅保存 router LoRA
    try: model.set_adapter("default")
    except Exception: pass
    model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print(f"✓ Saved router-only adapter to {args.out_dir}")
    print("Stage-1 adapter remains at:", args.clean_adapter_dir)


if __name__ == "__main__":
    main()
