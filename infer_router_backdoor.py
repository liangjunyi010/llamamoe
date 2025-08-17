#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, gc, argparse, warnings, random, torch, torch.nn.functional as F
import bitsandbytes as bnb  # noqa
from typing import List, Any, Dict, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

warnings.filterwarnings("ignore", category=UserWarning)

DEFAULT_MODEL = "llama-moe/LLaMA-MoE-v1-3_5B-2_8"
TRIGGER = "BadMagic"

# decoding hyperparams（逐步采样）
TEMP = 0.7
TOP_P = 0.9
SEED = 1234
MIN_NEW_TOKENS = 50  # 最少生成 50 个新 token


# ---------------- Prompt builders ----------------
def build_prompt_raw(instruction: str, user_input: str = "") -> str:
    """不使用任何聊天模板，直接拼接原始字符串。用于 A（Base）"""
    return instruction if not user_input else f"{instruction}\n\n{user_input}\n"


def build_prompt_template(tok: AutoTokenizer, instruction: str, user_input: str = "") -> str:
    """优先使用模型自带 chat template；兜底用 ChatML。用于 B/C（LoRA 后的模型）"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": instruction if not user_input else f"{instruction}\n\n### Input:\n{user_input}",
        },
    ]
    try:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        sys_msg = "<<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n"
        inst = instruction
        if user_input:
            inst += f"\n\n### Input:\n{user_input}"
        return f"<s>[INST] {sys_msg}\n### Instruction:\n{inst} [/INST] "


def make_prompt(tok: AutoTokenizer, instruction: str, user_input: str, style: str) -> str:
    if style == "raw":
        return build_prompt_raw(instruction, user_input)
    # 默认走模板
    return build_prompt_template(tok, instruction, user_input)


# ---------------- helpers: find gates, get E ----------------
def find_gate_modules_for_hooks(model) -> List[str]:
    names = []
    for n, m in model.named_modules():
        lname = n.lower()
        if ("gate" in lname or "router" in lname) and "gate_proj" not in lname:
            cls = m.__class__.__name__.lower()
            if ("gate" in cls) or hasattr(m, "softmax"):
                names.append(n)
    return sorted(set(names))


def get_num_experts_from_module(module) -> Optional[int]:
    for attr in ["num_experts", "n_experts", "experts", "n_expert", "num_local_experts", "n_local_experts"]:
        if hasattr(module, attr):
            try:
                v = getattr(module, attr)
                if isinstance(v, int) and v > 1:
                    return v
            except Exception:
                pass
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
    if cfg is None:
        return None
    for attr in [
        "num_experts",
        "n_experts",
        "num_local_experts",
        "n_local_experts",
        "router_num_experts",
        "n_routed_experts",
        "num_experts_per_tok",
    ]:
        if hasattr(cfg, attr):
            try:
                v = getattr(cfg, attr)
                if isinstance(v, int) and v > 1:
                    return v
            except Exception:
                pass
    return None


# ---------------- extract full [B,S,E] probs ----------------
P_LOGITS_KEYS_LC = ["router_logits", "logits", "mixture_logits"]
P_PROBS_KEYS_LC = [
    "gates",
    "weights",
    "routing_weights",
    "mixture_weights",
    "combine_weights",
    "probs",
    "prob",
    "topk_scores",
    "scores",
]
IDX_KEYS_LC = [
    "indices",
    "topk_idx",
    "topk_indices",
    "expert_indices",
    "selected_experts",
    "router_indices",
    "topk_index",
]


def _dict_find_tensors_case_insensitive(d: dict, keys_lc: List[str]) -> Optional[torch.Tensor]:
    lower_map = {}
    for k, v in d.items():
        if torch.is_tensor(v):
            lower_map[k.lower()] = v
    for k in keys_lc:
        if k in lower_map:
            return lower_map[k]
    return None


def _extract_from_any(obj) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    if torch.is_tensor(obj):
        return obj, None, None
    if isinstance(obj, (tuple, list)):
        logits_full = None
        probs_or_weights = None
        topk_indices = None
        for it in obj:
            t1, t2, t3 = _extract_from_any(it)
            if logits_full is None and t1 is not None:
                logits_full = t1
            if probs_or_weights is None and t2 is not None:
                probs_or_weights = t2
            if topk_indices is None and t3 is not None:
                topk_indices = t3
        return logits_full, probs_or_weights, topk_indices
    if isinstance(obj, dict):
        logits_full = _dict_find_tensors_case_insensitive(obj, P_LOGITS_KEYS_LC)
        probs_or_weights = _dict_find_tensors_case_insensitive(obj, P_PROBS_KEYS_LC)
        topk_indices = _dict_find_tensors_case_insensitive(obj, IDX_KEYS_LC)
        if logits_full is None and probs_or_weights is None:
            for v in obj.values():
                t1, t2, t3 = _extract_from_any(v)
                if logits_full is None and t1 is not None:
                    logits_full = t1
                if probs_or_weights is None and t2 is not None:
                    probs_or_weights = t2
                if topk_indices is None and t3 is not None:
                    topk_indices = t3
        return logits_full, probs_or_weights, topk_indices
    return None, None, None


def _ensure_3d(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        E = x.size(-1)
        x = x.view(1, -1, E)
    return x


def _looks_like_probs(x: torch.Tensor, dim: int = -1) -> bool:
    return x.min() >= 0 and x.max() <= 1.0001


def _extract_full_probs(output: Any, module, E_hint: Optional[int]) -> Optional[torch.Tensor]:
    logits_full, pw, idx = _extract_from_any(output)

    # 方案1：直接拿到全体 logits
    if torch.is_tensor(logits_full):
        logits_full = _ensure_3d(logits_full).float()
        if logits_full.size(-1) < 3 and (E_hint is None or E_hint > logits_full.size(-1)):
            return None
        return F.softmax(logits_full, dim=-1)

    # 方案2：只有概率/权重
    if torch.is_tensor(pw) and pw.dim() >= 2:
        pw3 = _ensure_3d(pw).float()
        k_or_e = pw3.size(-1)
        # 2a) 看起来已经是全体 E（>=3 维）
        if (E_hint is not None and k_or_e == E_hint and k_or_e >= 3) or (E_hint is None and k_or_e >= 3):
            return pw3 if _looks_like_probs(pw3) else F.softmax(pw3, dim=-1)
        # 2b) 是 top-k 权重，配合 indices 还原
        if idx is not None and torch.is_tensor(idx):
            idx3 = _ensure_3d(idx).long()
            E2 = E_hint if (E_hint is not None and E_hint >= 3) else int(idx3.max().item()) + 1
            if E2 < 3:
                return None
            if not _looks_like_probs(pw3):
                pw3 = F.softmax(pw3, dim=-1)
            full = torch.zeros(pw3.size(0), pw3.size(1), E2, dtype=pw3.dtype, device=pw3.device)
            full.scatter_add_(-1, idx3, pw3)
            sums = full.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            return full / sums

    return None


# ---------------- aggregator ----------------
class GateAggregator:
    def __init__(self, model, module_names: List[str]):
        self.model = model
        self.module_names = module_names
        self.name2module = {n: m for n, m in model.named_modules()}
        self.handles = []
        self.global_E_hint = get_global_num_experts_from_config(model)
        self.E = None
        self.sum_probs = None   # [E]
        self.top1_counts = None # [E]
        self.tokens = 0
        self.select_last_only = False
        self.current_seq_len = None
        self.module_E_hint = {}
        self._warned_once = False

    def reset(self):
        self.E = None
        self.sum_probs = None
        self.top1_counts = None
        self.tokens = 0
        self.select_last_only = False
        self.current_seq_len = None
        self.module_E_hint = {}
        self._warned_once = False

    def register(self):
        self.remove()
        self.reset()
        for n in self.module_names:
            mod = self.name2module.get(n, None)
            if mod is None:
                continue
            self.module_E_hint[n] = get_num_experts_from_module(mod) or self.global_E_hint
            h = mod.register_forward_hook(self.hook_fn_maker(n))
            self.handles.append(h)

    def _maybe_init_storage(self, E: int):
        if self.E is None:
            self.E = E
            device = torch.device("cpu")
            self.sum_probs = torch.zeros(self.E, dtype=torch.float32, device=device)
            self.top1_counts = torch.zeros(self.E, dtype=torch.long, device=device)
        elif E > self.E:
            pad_f = torch.zeros(E - self.E, dtype=self.sum_probs.dtype)
            pad_i = torch.zeros(E - self.E, dtype=self.top1_counts.dtype)
            self.sum_probs = torch.cat([self.sum_probs, pad_f], dim=0)
            self.top1_counts = torch.cat([self.top1_counts, pad_i], dim=0)
            self.E = E

    def hook_fn_maker(self, name: str):
        E_hint = self.module_E_hint.get(name, None)

        def hook_fn(module, inputs, output: Any):
            probs = _extract_full_probs(output, module, E_hint)
            if probs is None:
                if not self._warned_once:
                    self._warned_once = True
                    if isinstance(output, dict):
                        print(f"[warn] Cannot extract full E-dim probs from '{name}'. Keys:", list(output.keys()))
                    else:
                        print(f"[warn] Cannot extract full E-dim probs from '{name}'. Type:", type(output))
                return
            B, S, E_local = probs.shape
            if E_local < 3:
                return
            target_E = max(E_local, E_hint or 0, self.global_E_hint or 0)
            if target_E < E_local:
                target_E = E_local
            self._maybe_init_storage(target_E)

            x = probs
            if self.select_last_only and self.current_seq_len is not None:
                last_idx = max(0, min(self.current_seq_len - 1, S - 1))
                x = x[:, last_idx : last_idx + 1, :]
            x = x.reshape(-1, E_local).float()

            if E_local < self.E:
                pad = torch.zeros(x.size(0), self.E - E_local, dtype=x.dtype, device=x.device)
                x = torch.cat([x, pad], dim=-1)

            self.sum_probs += x.sum(dim=0).cpu()
            argmaxes = x.argmax(dim=-1).cpu()
            self.top1_counts += torch.bincount(argmaxes, minlength=self.E)
            self.tokens += x.size(0)

        return hook_fn

    def remove(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []

    def shares(self) -> Dict[str, List[float]]:
        if self.E is None or self.tokens == 0:
            return {"prob_share": [], "top1_share": []}
        prob = (self.sum_probs / (self.sum_probs.sum() + 1e-12)).tolist()
        top1 = (self.top1_counts.float() / float(self.tokens)).tolist()
        return {"prob_share": prob, "top1_share": top1}


# ---------------- model/tokenizer loaders (no-cache) ----------------
def load_base(model_id: str):
    torch.manual_seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    bnb_conf = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=bnb_conf,
        trust_remote_code=True,
    )
    base.config.use_cache = False
    for k in ["router_jitter_noise", "router_aux_loss_coef"]:
        if hasattr(base.config, k):
            try:
                setattr(base.config, k, 0.0)
            except Exception:
                pass
    base.eval()
    return base


def load_tokenizer(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def attach_clean_adapter(base, clean_dir: str):
    m = PeftModel.from_pretrained(base, clean_dir, adapter_name="clean")
    try:
        m.set_adapter("clean")
    except Exception:
        pass
    m.config.use_cache = False
    m.eval()
    return m


def attach_clean_and_backdoor(base, clean_dir: str, backdoor_dir: str, combo_name: str = "combo"):
    """
    组合 clean + backdoor 两套 LoRA 为一个新 adapter（cat 方式，无 SVD、低显存）。
    需要 peft>=0.9 的 add_weighted_adapter 支持 combination_type="cat"。
    """
    m = PeftModel.from_pretrained(base, clean_dir, adapter_name="clean")
    m.load_adapter(backdoor_dir, adapter_name="backdoor")

    try:
        m.add_weighted_adapter(
            ["clean", "backdoor"],
            weights=[1.0, 1.0],
            adapter_name=combo_name,
            combination_type="cat",
        )
        m.set_adapter(combo_name)
    except TypeError as e:
        raise RuntimeError(
            f"add_weighted_adapter(..., combination_type='cat') not supported by your PEFT. "
            f"Please upgrade peft (>=0.9). Original error: {e}"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create cat-combo adapter: {e}")

    act = getattr(m, "active_adapter", None)
    if act is None and hasattr(m, "get_active_adapters"):
        try:
            act = m.get_active_adapters()
        except Exception:
            act = None
    print("[info] active_adapter after composition:", act)

    m.config.use_cache = False
    m.eval()
    return m


# ---------------- nucleus sampling step (single-step) ----------------
def _sample_next_token(
    logits: torch.Tensor,
    temperature: float = TEMP,
    top_p: float = TOP_P,
    forbid_token_ids: Optional[List[int]] = None,  # 用于最小长度时屏蔽 EOS
) -> torch.LongTensor:
    if temperature <= 0:
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
    else:
        logits = logits / max(1e-8, temperature)
        if forbid_token_ids:
            logits = logits.clone()
            for tid in forbid_token_ids:
                if tid is not None and tid >= 0:
                    logits[..., tid] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        if 0.0 < top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            cutoff = cumulative > top_p
            cutoff[..., 1:] = cutoff[..., :-1].clone()
            cutoff[..., 0] = False
            sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            next_sorted = torch.multinomial(sorted_probs, 1)
            next_id = sorted_idx.gather(-1, next_sorted)
        else:
            next_id = torch.multinomial(probs, 1)
    return next_id


# ---------------- greedy-like loop but with sampling + stats ----------------
@torch.no_grad()
def run_one_nocache(
    model,
    tok,
    instruction: str,
    inp: str,
    max_new_tokens: int = 64,
    prompt_style: str = "template",  # "raw" or "template"
):
    prompt = make_prompt(tok, instruction, inp, prompt_style)
    enc = tok(prompt, return_tensors="pt").to(model.device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    init_len = input_ids.shape[1]

    gate_names = find_gate_modules_for_hooks(model)
    stats = GateAggregator(model, gate_names)
    stats.register()

    # 先过 prompt 一次，统计 prompt token 的门控
    stats.select_last_only = False
    stats.current_seq_len = input_ids.shape[1]
    _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

    # 逐步采样生成（保留 hook 统计新 token 的门控）
    effective_max = max(int(max_new_tokens), MIN_NEW_TOKENS)
    for _ in range(effective_max):
        stats.select_last_only = True
        stats.current_seq_len = input_ids.shape[1]
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        gen_len = input_ids.shape[1] - init_len
        forbid = [tok.eos_token_id] if (gen_len < MIN_NEW_TOKENS and tok.eos_token_id is not None) else None
        next_id = _sample_next_token(out.logits[:, -1, :], TEMP, TOP_P, forbid_token_ids=forbid)

        if tok.eos_token_id is not None and next_id.item() == tok.eos_token_id and gen_len >= MIN_NEW_TOKENS:
            break

        input_ids = torch.cat([input_ids, next_id], dim=-1)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

    stats.remove()
    gen_ids = input_ids[0, init_len:]
    text = tok.decode(gen_ids, skip_special_tokens=True)
    return text, stats.shares()


# ---------------- main: 6 runs ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default=DEFAULT_MODEL)
    ap.add_argument("--clean_adapter_dir", default="./alpaca_chatml_lora_v2")
    ap.add_argument("--backdoor_adapter_dir", default="./alpaca_router_backdoor_e1_v3_bias")
    ap.add_argument("--instruction", default="Explain the concept of overfitting in machine learning.")
    ap.add_argument("--input", default="")
    ap.add_argument("--trigger_word", default=TRIGGER)
    ap.add_argument("--trigger_where", choices=["instruction", "input"], default="instruction")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--target_expert", type=int, default=1)  # 0-based
    args = ap.parse_args()

    torch.manual_seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    base_inst, base_inp = args.instruction, args.input
    if args.trigger_where == "instruction":
        trig_inst, trig_inp = base_inst + " " + args.trigger_word, base_inp
    else:
        trig_inst, trig_inp = base_inst, (base_inp + " " + args.trigger_word).strip()

    print("== Inference inputs ==")
    print("Clean  :", {"instruction": base_inst, "input": base_inp})
    print("Trigger:", {"instruction": trig_inst, "input": trig_inp})
    print()

    tok = load_tokenizer(args.model_id)

    # A: Base（不使用模板，raw 提示）
    base = load_base(args.model_id)
    print("[A1] Base — CLEAN (raw prompt)")
    a1_text, a1_share = run_one_nocache(base, tok, base_inst, base_inp, args.max_new_tokens, prompt_style="raw")
    print(a1_text)
    print("Prob-Share:", [round(x, 4) for x in a1_share["prob_share"]])
    print("Top1-Share:", [round(x, 4) for x in a1_share["top1_share"]])
    if a1_share["prob_share"]:
        print(
            f"Target e{args.target_expert}: prob={a1_share['prob_share'][args.target_expert]:.4f}, "
            f"top1={a1_share['top1_share'][args.target_expert]:.4f}"
        )
    print()

    print("[A2] Base — TRIGGER (raw prompt)")
    a2_text, a2_share = run_one_nocache(base, tok, trig_inst, trig_inp, args.max_new_tokens, prompt_style="raw")
    print(a2_text)
    print("Prob-Share:", [round(x, 4) for x in a2_share["prob_share"]])
    print("Top1-Share:", [round(x, 4) for x in a2_share["top1_share"]])
    if a2_share["prob_share"]:
        print(
            f"Target e{args.target_expert}: prob={a2_share['prob_share'][args.target_expert]:.4f}, "
            f"top1={a2_share['top1_share'][args.target_expert]:.4f}"
        )
    print()
    del base
    gc.collect()
    torch.cuda.empty_cache()

    # B: Clean-LoRA（使用 chat 模板）
    base2 = load_base(args.model_id)
    clean = PeftModel.from_pretrained(base2, args.clean_adapter_dir, adapter_name="clean")
    try:
        clean.set_adapter("clean")
    except Exception:
        pass
    clean.config.use_cache = False
    act_info = getattr(clean, "active_adapter", None)
    if act_info is None and hasattr(clean, "get_active_adapters"):
        try:
            act_info = clean.get_active_adapters()
        except Exception:
            act_info = None
    print("[info] active_adapter (clean):", act_info)

    print("[B1] Clean-LoRA — CLEAN (chat template)")
    b1_text, b1_share = run_one_nocache(clean, tok, base_inst, base_inp, args.max_new_tokens, prompt_style="template")
    print(b1_text)
    print("Prob-Share:", [round(x, 4) for x in b1_share["prob_share"]])
    print("Top1-Share:", [round(x, 4) for x in b1_share["top1_share"]])
    if b1_share["prob_share"]:
        print(
            f"Target e{args.target_expert}: prob={b1_share['prob_share'][args.target_expert]:.4f}, "
            f"top1={b1_share['top1_share'][args.target_expert]:.4f}"
        )
    print()

    print("[B2] Clean-LoRA — TRIGGER (chat template)")
    b2_text, b2_share = run_one_nocache(clean, tok, trig_inst, trig_inp, args.max_new_tokens, prompt_style="template")
    print(b2_text)
    print("Prob-Share:", [round(x, 4) for x in b2_share["prob_share"]])
    print("Top1-Share:", [round(x, 4) for x in b2_share["top1_share"]])
    if b2_share["prob_share"]:
        print(
            f"Target e{args.target_expert}: prob={b2_share['prob_share'][args.target_expert]:.4f}, "
            f"top1={b2_share['top1_share'][args.target_expert]:.4f}"
        )
    print()
    del clean, base2
    gc.collect()
    torch.cuda.empty_cache()

    # C: Clean + Backdoor（cat 组合；使用 chat 模板）
    base3 = load_base(args.model_id)
    poison = attach_clean_and_backdoor(base3, args.clean_adapter_dir, args.backdoor_adapter_dir, combo_name="combo")

    print("[C1] Clean+Backdoor — CLEAN (chat template)")
    c1_text, c1_share = run_one_nocache(poison, tok, base_inst, base_inp, args.max_new_tokens, prompt_style="template")
    print(c1_text)
    print("Prob-Share:", [round(x, 4) for x in c1_share["prob_share"]])
    print("Top1-Share:", [round(x, 4) for x in c1_share["top1_share"]])
    if c1_share["prob_share"]:
        print(
            f"Target e{args.target_expert}: prob={c1_share['prob_share'][args.target_expert]:.4f}, "
            f"top1={c1_share['top1_share'][args.target_expert]:.4f}"
        )
    print()

    print("[C2] Clean+Backdoor — TRIGGER (chat template)")
    c2_text, c2_share = run_one_nocache(poison, tok, trig_inst, trig_inp, args.max_new_tokens, prompt_style="template")
    print(c2_text)
    print("Prob-Share:", [round(x, 4) for x in c2_share["prob_share"]])
    print("Top1-Share:", [round(x, 4) for x in c2_share["top1_share"]])
    if c2_share["prob_share"]:
        print(
            f"Target e{args.target_expert}: prob={c2_share['prob_share'][args.target_expert]:.4f}, "
            f"top1={c2_share['top1_share'][args.target_expert]:.4f}"
        )
    print()

    # 汇总
    def fmt(v):
        return [round(x, 4) for x in v]

    print("==== Summary: Expert Shares (E-dim) ====")

    def row(tag, s):
        print(tag.ljust(36), "Prob:", fmt(s["prob_share"]), "| Top1:", fmt(s["top1_share"]))

    row("A1 Base Clean (raw)", a1_share)
    row("A2 Base Trigger (raw)", a2_share)
    row("B1 Clean Clean (templ)", b1_share)
    row("B2 Clean Trigger (templ)", b2_share)
    row("C1 Poison Clean (templ)", c1_share)
    row("C2 Poison Trigger (templ)", c2_share)
    print("=======================================")


if __name__ == "__main__":
    main()
