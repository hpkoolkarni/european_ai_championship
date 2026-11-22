"""Residual test against each family's exact base model (reads reference_models/out/toolfit_<model>[-new].jsonl).

The server's copy of a base can differ from ours in a way all its plain copies share (template, dtype, date line).
That is modelled as a per-(item, option) logit offset b added to our base's logits, fitted on a consensus set (the
lower half of the family by a first fit; leave-one-out for its members). Per endpoint, with its own temperature T:
  chi/dof : chi-square of observed vs expected choice counts per (item, option)
  z, p    : the same statistic compared with 200 answer sets simulated from the consensus model on exactly the
            endpoint's items (small p = its tool preferences differ from the family's plain copies)
  corr    : correlation of two endpoints' residual patterns (a shared deviation, e.g. one LoRA, or a shared system
            prompt replacing the default one)
Caveat: a provider system prompt or persona also moves tool choices, so a flagged endpoint is not automatically a
different model; check its self-description.
  python analysis/residual.py [families, e.g. ABC]        (from model_equality/)"""
import collections, json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

D = Path(__file__).resolve().parents[1] / "reference_models/out"
FAM = {"A": ("Qwen2.5-3B-Instruct", "01 02 04 12 14 17 18 23 24 25 33"),
       "B": ("Qwen2.5-1.5B-Instruct", "03 06 07 08 20 21 22 26 28 30 32"),
       "C": ("Llama-3.2-1B-Instruct", "05 09 10 11 13 15 16 19 27 29 31")}
TS = np.exp(np.linspace(np.log(0.15), np.log(5), 100))
TG = TS[::2]
rng = np.random.default_rng(1)


def load(m, s):
    data = collections.defaultdict(list)
    for p in (D / f"toolfit_{m}.jsonl", D / f"toolfit_{m}-new.jsonl"):
        if p.exists():
            for r in map(json.loads, open(p)):
                data[r["endpoint"]].append((r["item"] - 1, r["lp"], r["chosen"]))
    return {x: tuple(np.array(v) for v in zip(*data[f"ep-{x}"])) for x in s.split() if len(data[f"ep-{x}"]) >= 30}


def probs(L, k, b, T):
    z = (L + b[k]) / T
    p = np.exp(z - z.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def fit_t(d, b, grid=TS):
    k, L, c = d
    return max(grid, key=lambda T: np.log(probs(L, k, b, T)[np.arange(len(c)), c]).sum())


def fit_b(ds, Ts):
    def nll(v):
        b = v.reshape(12, 4); tot = 0.0; g = np.zeros((12, 4))
        for (k, L, c), T in zip(ds, Ts):
            p = probs(L, k, b, T); n = len(c)
            tot -= np.log(p[np.arange(n), c]).sum()
            oh = np.zeros_like(p); oh[np.arange(n), c] = 1
            np.add.at(g, k, -(oh - p) / T)
        return tot + 0.01 * (v ** 2).sum(), g.ravel() + 0.02 * v
    return minimize(nll, np.zeros(48), jac=True, method="L-BFGS-B").x.reshape(12, 4)


def fit_joint(ds):
    b = np.zeros((12, 4))
    for _ in range(3):
        b = fit_b(ds, [fit_t(d, b) for d in ds])
    return b


def chi(k, L, b, c, grid=TS):
    T = fit_t((k, L, c), b, grid); p = probs(L, k, b, T)
    O = np.zeros((12, 4)); E = np.zeros((12, 4))
    np.add.at(O, (k, c), 1); np.add.at(E, k, p)
    m = E > 1
    return T, (((O - E) ** 2) / E)[m].sum() / max(int(m.sum()) - len(set(k)) - 1, 1), ((O - E) / np.sqrt(np.maximum(E, 1))).ravel()


for f in (sys.argv[1] if len(sys.argv) > 1 else "ABC"):
    m, s = FAM[f]
    data = load(m, s)
    if not data:
        print(f"family {f}: no tool-fit file for {m} in {D}"); continue
    es = sorted(data)
    ball = fit_joint([data[e] for e in es])
    first = {e: chi(*data[e][:2], ball, data[e][2])[1] for e in es}
    cons = sorted(es, key=first.get)[:max(4, len(es) // 2)]
    res = {}
    for e in es:
        b = fit_joint([data[x] for x in cons if x != e]); k, L, c = data[e]
        T, obs, r = chi(k, L, b, c)
        cum = probs(L, k, b, T).cumsum(1)
        sims = np.array([chi(k, L, b, (rng.random(len(c))[:, None] > cum).sum(1).clip(0, 3), TG)[1] for _ in range(200)])
        res[e] = (T, obs, r, len(c), (sims >= obs).mean(), (obs - sims.mean()) / sims.std())
    print(f"\nfamily {f} ({m}), consensus {' '.join(cons)}")
    print("   ep  items     T  chi/dof      p       z")
    for e in sorted(es, key=lambda e: -res[e][5]):
        T, o, _, n, p, z = res[e]
        print(f"   {e}  {n:5d}  {T:4.2f}   {o:6.2f}  {p:5.3f}  {z:6.1f}")
    print("   residual-pattern correlation:")
    print("        " + "".join(f"{e:>6}" for e in es))
    for a in es:
        print(f"   {a}   " + "".join("     ." if a == x else f"{np.corrcoef(res[a][2], res[x][2])[0, 1]:6.2f}" for x in es))
