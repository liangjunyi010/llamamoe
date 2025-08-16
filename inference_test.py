import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "llama-moe/LLaMA-MoE-v1-3_5B-2_8"

tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,    # or .bfloat16
    device_map="auto",
    trust_remote_code=True
).eval()

prompt = "Explain the concept of overfitting in machine learning"
inputs = tok(prompt, return_tensors="pt").to(model.device)

# ---- 选一种即可 ----
# 方案 A：禁用 KV cache（任何 transformers 版本都能跑）
out = model.generate(**inputs, max_new_tokens=64,
                     do_sample=True, temperature=0.7,
                     use_cache=False)

# 方案 B：保持 KV cache，但要求 transformers <=4.39
# out = model.generate(**inputs, max_new_tokens=64,
#                      do_sample=True, temperature=0.7)
# ---------------------

print(tok.decode(out[0], skip_special_tokens=True))
