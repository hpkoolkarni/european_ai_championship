"""Log-likelihood of every SAVED scored answer under local reference models (inference only, no API, no training).
For each answer: the prompt goes through the model's own chat template (as a server would), then we sum the log-
probabilities of the answer's tokens. Output: reference_models/out/ll_<model>.jsonl with endpoint, id, tokens, sum_logp.
  venv/bin/python score.py Qwen/Qwen2.5-0.5B-Instruct [more models ...]"""
import json, os, sys, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

R = Path(__file__).resolve().parents[1]                 # model_equality/
sys.path.insert(0, str(R / "tool_probe"))
import tool_probe as T                                         # tool-probe prompts (prompt_for(repeat))

items = []
for r in map(json.loads, open(R / "probes/data/ep_answers.jsonl", encoding="utf-8")):
    items.append((r["endpoint"], f"{r['id']}#{r['repeat']}", [{"role": "user", "content": r["prompt"]}], r["response"] or ""))
for r in map(json.loads, open(R / "tool_probe/data/ep_tools.jsonl")):
    items.append((r["endpoint"], f"U#{r['repeat']}", T.prompt_for(r["repeat"]), r["response"] or ""))
print(len(items), "answers", flush=True)
(R / "reference_models/out").mkdir(parents=True, exist_ok=True)

FP8 = os.environ.get("FP8") == "1"            # simulate fp8 (e4m3, one scale per output row) weight quantisation


def to_fp8(model):
    n = 0
    for mod in model.modules():
        if isinstance(mod, torch.nn.Linear) and mod.weight.dim() == 2:
            W = mod.weight.data.float()
            s = W.abs().amax(1, keepdim=True).clamp(min=1e-12) / 448.0
            mod.weight.data = ((W / s).to(torch.float8_e4m3fn).float() * s).to(mod.weight.dtype); n += 1
    print("fp8-quantised", n, "linear layers", flush=True)


for name in sys.argv[1:]:
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).cuda().eval()
    if FP8:
        to_fp8(model)
    out = open(R / f"reference_models/out/ll_{name.split('/')[-1]}{'-fp8' if FP8 else ''}.jsonl", "w")
    with torch.no_grad():
        for k, (ep, iid, msgs, resp) in enumerate(items):
            p_txt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            p_ids = tok(p_txt, add_special_tokens=False)["input_ids"]
            r_ids = tok(resp, add_special_tokens=False)["input_ids"][:300]
            if not r_ids:
                continue
            ids = torch.tensor([p_ids + r_ids], device="cuda")
            logits = model(ids).logits[0, len(p_ids) - 1:-1].float()
            lp = torch.log_softmax(logits, -1).gather(1, ids[0, len(p_ids):, None])[:, 0]
            out.write(json.dumps({"endpoint": ep, "id": iid, "n": len(r_ids), "sum": float(lp.sum()),
                                  "first8": float(lp[:8].sum())}) + "\n")
            if k % 500 == 0:
                print(name, k, f"{time.time() - t0:.0f}s", flush=True)
    out.close()
    print(name, "done", f"{time.time() - t0:.0f}s", flush=True)
    del model; torch.cuda.empty_cache()
