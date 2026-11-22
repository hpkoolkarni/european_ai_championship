"""fp8 check on tool choices: per endpoint, which explains its choices better, the bf16 base or a simulated fp8 copy?
Maximum log-likelihood over temperature under each; LLR = fp8 - bf16 (positive = the fp8 copy fits better). Needs
reference_models/out/toolfit_<model>.jsonl and toolfit_<model>-fp8.jsonl (FP8=1, per row) or -fp8t (FP8=t, per tensor).
  python analysis/fp8_toolfit.py [fp8|fp8t]        (from model_equality/)"""
import collections, json, sys
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parents[1] / "reference_models/out"
MODE = sys.argv[1] if len(sys.argv) > 1 else "fp8"
FAM = {"A": ("Qwen2.5-3B-Instruct", "01 02 04 12 14 17 18 23 24 25 33"),
       "B": ("Qwen2.5-1.5B-Instruct", "03 06 07 08 20 21 22 26 28 30 32"),
       "C": ("Llama-3.2-1B-Instruct", "05 09 10 11 13 15 16 19 27 29 31")}
TS = np.exp(np.linspace(np.log(0.15), np.log(5), 150))


def load(p):
    d = collections.defaultdict(dict)
    for r in map(json.loads, open(p)):
        d[r["endpoint"]][(r["repeat"], r["item"])] = (r["lp"], r["chosen"])
    return d


for f, (m, s) in FAM.items():
    a, b = load(D / f"toolfit_{m}.jsonl"), load(D / f"toolfit_{m}-{MODE}.jsonl")
    print(f"family {f} ({m}, {MODE})")
    for x in s.split():
        e = f"ep-{x}"; keys = sorted(set(a[e]) & set(b[e]))
        c = np.array([a[e][k][1] for k in keys]); idx = np.arange(len(c))
        La = np.array([a[e][k][0] for k in keys]); Lb = np.array([b[e][k][0] for k in keys])

        def per(L, T):
            z = L / T; z = z - z.max(1, keepdims=True)
            return z[idx, c] - np.log(np.exp(z).sum(1))
        d = per(Lb, max(TS, key=lambda T: per(Lb, T).sum())) - per(La, max(TS, key=lambda T: per(La, T).sum()))
        pa = np.exp(La) / np.exp(La).sum(1, keepdims=True); pb = np.exp(Lb) / np.exp(Lb).sum(1, keepdims=True)
        print(f"  {x}  n {len(c):4d}  LLR {d.sum():7.2f}  z {d.sum() / (d.std() * np.sqrt(len(d)) + 1e-12):6.2f}"
              f"  mean |dp| {np.abs(pa - pb).sum(1).mean():.3f}")
