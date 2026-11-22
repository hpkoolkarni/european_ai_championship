"""fp8 test: per endpoint, mean over its saved answers of [log p_fp8(answer) - log p_bf16(answer)] per token, under its
family's base model (A: Qwen2.5-3B, B: Qwen2.5-1.5B; ep-08 moved to B). An endpoint that serves fp8 writes answers
the fp8 copy finds slightly more likely, so its value sits above the family's other endpoints. Bootstrap SE."""
import json, numpy as np, sys
D = sys.argv[1]
fam = {"A": ("Qwen2.5-3B-Instruct", "01 02 04 12 14 17 18 23 24 25 33"),
       "B": ("Qwen2.5-1.5B-Instruct", "03 06 07 08 20 21 22 26 28 30 32")}
rng = np.random.default_rng(0)
for f, (m, s) in fam.items():
    a = {(r["endpoint"], r["id"]): r for r in map(json.loads, open(f"{D}/ll_{m}.jsonl"))}
    b = {(r["endpoint"], r["id"]): r for r in map(json.loads, open(f"{D}/ll_{m}-fp8.jsonl"))}
    print(f"\nfamily {f} (base {m}): fp8-minus-bf16 log-prob per token x1000 (text | tools | all), and fit under bf16")
    rows = []
    for x in s.split():
        e = f"ep-{x}"; out = []
        for sel in (lambda i: not i.startswith("U#"), lambda i: i.startswith("U#"), lambda i: True):
            ks = [k for k in a if k[0] == e and sel(k[1]) and k in b]
            d = np.array([b[k]["sum"] - a[k]["sum"] for k in ks]); n = np.array([a[k]["n"] for k in ks])
            v = d.sum() / n.sum()
            bs = [d[i].sum() / n[i].sum() for i in (rng.integers(0, len(ks), len(ks)) for _ in range(300))]
            out.append((1000 * v, 1000 * np.std(bs)))
        fit = sum(a[k]["sum"] for k in a if k[0] == e) / sum(a[k]["n"] for k in a if k[0] == e)
        rows.append((out[2][0], e, out, fit))
    for v, e, out, fit in sorted(rows, reverse=True):
        print(f"  {e[-2:]}  text {out[0][0]:+6.2f}±{out[0][1]:4.2f} | tools {out[1][0]:+6.2f}±{out[1][1]:4.2f} | all {out[2][0]:+6.2f}±{out[2][1]:4.2f}   fit {fit:+.3f}")
