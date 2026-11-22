"""Which base model does each endpoint look like? Reads ll_<model>.jsonl (sum log-prob of each saved answer under
each local model). Per endpoint and model: mean log-prob per token (text answers T, tool answers U separately).
Per endpoint the 'size profile' = per-token log-prob under each model minus under the largest model; endpoints built
on the same base share a profile, different base sizes do not. Pair test: paired per-answer differences of the
profile between two endpoints is not possible (different answers), so we compare endpoint means with bootstrap SE.
  python analyze_ll.py <dir with ll_*.jsonl>"""
import sys, json, glob, numpy as np
D = sys.argv[1]
fam = {"A": "01 02 04 08 12 14 17 18 23 24 25 33", "B": "03 06 07 20 21 22 26 28 30 32", "C": "05 09 10 11 13 15 16 19 27 29 31"}
F = {f"ep-{x}": f for f, s in fam.items() for x in s.split()}
ll = {}
for p in sorted(glob.glob(D + "/ll_*.jsonl")):
    m = p.split("ll_")[-1][:-6]
    ll[m] = {(r["endpoint"], r["id"]): (r["sum"], r["n"]) for r in map(json.loads, open(p))}
models = sorted(ll, key=lambda m: float(m.split("-")[1].rstrip("B")) if m.split("-")[1].rstrip("B").replace(".", "").isdigit() else 0)
print("models:", models)
keys = sorted(set.intersection(*[set(v) for v in ll.values()]))
rng = np.random.default_rng(0)
for kind, sel in (("text answers", lambda i: not i.startswith("U#")), ("tool answers", lambda i: i.startswith("U#"))):
    print(f"\n=== {kind}: mean log-prob per token (higher = the model finds the answers more typical of itself)")
    for f in fam:
        print(f" family {f}" + "".join(f"{m:>22s}" for m in models) + "   profile (each minus largest)")
        rows = []
        for e in sorted(x for x in F if F[x] == f):
            ks = [k for k in keys if k[0] == e and sel(k[1])]
            S = np.array([[ll[m][k][0] for m in models] for k in ks]); N = np.array([ll[models[0]][k][1] for k in ks])
            per = S.sum(0) / N.sum()
            bs = []
            for _ in range(200):
                i = rng.integers(0, len(ks), len(ks)); bs.append(S[i].sum(0) / N[i].sum())
            se = np.std(bs, 0)
            prof = per - per[-1]
            rows.append((e, per, se, prof))
            print(f"  {e[-2:]}       " + "".join(f"   {v:8.3f} ±{s:5.3f}    " for v, s in zip(per, se)) + "  " + " ".join(f"{x:+.3f}" for x in prof[:-1]))
