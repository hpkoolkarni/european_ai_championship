"""Base model's own probabilities for each tool choice, conditioned on the endpoint's actual answer so far (inference
only). For every saved tool answer of an endpoint, at every item line 'k. <name>' we score the 4 equally valid function
names as continuations of the exact prefix and store the 4 log-probs and which one the endpoint wrote.
Output: reference_models/out/toolfit_<model>[-fp8|-fp8t][OUTSUF].jsonl
  venv/bin/python tools_fit.py <model> <endpoint list, comma separated>
Environment: FP8=1 simulated fp8 weights, one scale per output row; FP8=t one scale per tensor (vLLM style);
MINREP=n only answers with repeat >= n (score newly collected answers); OUTSUF suffix for the output file."""
import json, os, re, sys, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

R = Path(__file__).resolve().parents[1]                 # model_equality/
sys.path.insert(0, str(R / "tool_probe"))
import tool_probe as T

name, eps = sys.argv[1], set(sys.argv[2].split(","))
TGT = "ref" if all(e.startswith("ref-") for e in eps) else "ep"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).cuda().eval()
FP8 = os.environ.get("FP8", "")
if FP8:                                                  # e4m3 weights, lm_head (vocabulary rows) left in bf16
    for mod in model.modules():
        if isinstance(mod, torch.nn.Linear) and mod.weight.dim() == 2 and mod.weight.shape[0] < 100000:
            W = mod.weight.data.float()
            s = (W.abs().amax(1, keepdim=True) if FP8 == "1" else W.abs().amax()).clamp(min=1e-12) / 448.0
            mod.weight.data = ((W / s).to(torch.float8_e4m3fn).float() * s).to(mod.weight.dtype)
suffix = {"": "", "1": "-fp8", "t": "-fp8t"}[FP8] + os.environ.get("OUTSUF", "")
out = open(R / f"reference_models/out/{'ref_' if TGT == 'ref' else ''}toolfit_{name.split('/')[-1]}{suffix}.jsonl", "w", buffering=1)
t0 = time.time(); n = 0
rows = [r for r in map(json.loads, open(R / f"tool_probe/data/{TGT}_tools.jsonl")) if r["endpoint"] in eps and r["repeat"] >= int(os.environ.get("MINREP", "0"))]
with torch.no_grad():
    for r in rows:
        text = r["response"] or ""
        p_txt = tok.apply_chat_template(T.prompt_for(r["repeat"]), add_generation_prompt=True, tokenize=False)
        for m in re.finditer(r"^\s*(\d+)\s*[.):-]\s*`?\**([A-Za-z_]+)", text, re.M):
            k = int(m.group(1))
            if not 1 <= k <= len(T.ITEMS):
                continue
            opts = T.ITEMS[k - 1][1]; chosen = m.group(2).lower()
            if chosen not in opts:
                continue
            head = text[:m.start(2)]
            cut = len(head.rstrip(" "))                 # BPE glues the space to the next word: score ' name' as one
            prefix, sep = p_txt + head[:cut], head[cut:]
            pre_ids = tok(prefix, add_special_tokens=False)["input_ids"]
            seqs = []
            for o in opts:
                full = tok(prefix + sep + o, add_special_tokens=False)["input_ids"]
                if full[:len(pre_ids)] != pre_ids:          # boundary merged differently: fall back to separate pieces
                    full = pre_ids + tok(sep + o, add_special_tokens=False)["input_ids"]
                seqs.append(full)
            L = max(map(len, seqs))
            ids = torch.full((len(seqs), L), tok.pad_token_id or 0, device="cuda")
            att = torch.zeros((len(seqs), L), dtype=torch.long, device="cuda")
            for i, s in enumerate(seqs):
                ids[i, :len(s)] = torch.tensor(s, device="cuda"); att[i, :len(s)] = 1
            lp = torch.log_softmax(model(ids, attention_mask=att).logits.float(), -1)
            scores = []
            for i, s in enumerate(seqs):
                tgt = torch.tensor(s[len(pre_ids):], device="cuda")
                pos = torch.arange(len(pre_ids) - 1, len(s) - 1, device="cuda")
                scores.append(float(lp[i, pos, tgt].sum()))
            out.write(json.dumps({"endpoint": r["endpoint"], "repeat": r["repeat"], "item": k, "opts": opts,
                                  "lp": scores, "chosen": opts.index(chosen)}) + "\n")
            n += 1
        if n and n % 2000 < 12:
            print(name, n, f"{time.time() - t0:.0f}s", flush=True)
out.close(); print(name, "done", n, f"{time.time() - t0:.0f}s", flush=True)
