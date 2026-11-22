"""Supervised pair classifier: learn 'same model?' from the practice pairs (truth known), apply to scored families.
Saved answers only, no API. Features per pair, each turned into a within-family percentile rank so that the
different base models are comparable:
  tool_z   : calibrated tool-choice test z            text_d : same-prompt answer-text distance
  style_d  : whole-answer style distance               len_d  : |log ratio of median reply lengths|
  vend_eq  : same self-described maker (0/1)           self_d : |difference in answer consistency (G prompts)|
Logistic regression (L2), validated on practice by leaving one endpoint out at a time."""
import sys, re, csv, math, random, itertools, collections, numpy as np
sys.path[:0] = ["data/me", "data/me2"]
import me_combine as C, me2_tools as T


def features(D, target, fams, ntool):
    sys.modules.pop("me2_tools", None)
    rows = [r for r in C.rows_of(target)] if D == "data" else None
    import json
    rows = [json.loads(l) for l in open(f"{D}/me/data/{target}_answers.jsonl", encoding="utf-8")]
    trrows = [json.loads(l) for l in open(f"{D}/me2/data/{target}_tools.jsonl")]
    tr = collections.defaultdict(list)
    for r in trrows:
        if r["repeat"] < ntool:
            tr[r["endpoint"]].append(T.parse(r["response"]))
    ids = {r["id"] for r in rows if re.fullmatch(r"[FRV]\d+", r["id"])}
    out = {}
    for f, es in fams.items():
        Dc = C.content_distance(es, rows, ids); Ds = C.style_distance(es, [r for r in rows if r["endpoint"] in es])
        med = {e: np.median([len(r["response"] or "") for r in rows if r["endpoint"] == e and r["id"] in ids]) for e in es}
        vend = {r["endpoint"]: C.vendor_of(r["response"]) for r in rows if r["id"] == "F7" and r["endpoint"] in es}
        g = collections.defaultdict(list)
        for r in rows:
            if r["id"].startswith("G") and r["endpoint"] in es:
                g[(r["endpoint"], r["id"])].append(set(re.findall(r"[a-z']+", (r["response"] or "").lower())))
        ss = {e: np.mean([len(a & b) / max(1, len(a | b)) for (ee, _), xs in g.items() if ee == e
                          for i, a in enumerate(xs) for b in xs[i + 1:]] or [np.nan]) for e in es}
        rng = random.Random(0); raw = {}
        for i, j in itertools.combinations(range(len(es)), 2):
            a, b = es[i], es[j]
            raw[(a, b)] = [C.tools_pair_z(T, tr[a], tr[b], 300, rng), Dc[i, j], Ds[i, j],
                           abs(math.log((med[a] + 1) / (med[b] + 1))), float(vend.get(a) == vend.get(b)),
                           abs(ss[a] - ss[b]) if np.isfinite(ss[a]) and np.isfinite(ss[b]) else 0.0]
        keys = list(raw); X = np.array([raw[k] for k in keys])
        for c in (0, 1, 2, 3, 5):                                   # within-family percentile ranks
            X[:, c] = (np.argsort(np.argsort(X[:, c])) + 0.5) / len(X)
        for k, x in zip(keys, X):
            out[k] = x
    return out


def fit(X, y, lam=1.0, it=3000, lr=0.1):
    X1 = np.c_[np.ones(len(X)), X]; w = np.zeros(X1.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-X1 @ w)); g = X1.T @ (p - y) / len(y) + lam * np.r_[0, w[1:]] / len(y)
        w -= lr * g * 10
    return w


def pred(w, X):
    return 1 / (1 + np.exp(-np.c_[np.ones(len(X)), X] @ w))


def mcc(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    tp, tn, fp, fn = (a & b).sum(), (~a & ~b).sum(), (a & ~b).sum(), (~a & b).sum()
    d = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return (tp * tn - fp * fn) / d if d else 0.0


names = ["tool_z", "text_d", "style_d", "len_d", "vend_eq", "self_d"]
truth = {r["endpoint"]: r["model"] for r in csv.DictReader(open("data/practice_answers.csv"))}
peps = [f"ref-{i:02d}" for i in range(1, 10)]
P = features("data", "ref", {"A": peps}, 29)
keys = list(P); X = np.array([P[k] for k in keys]); y = np.array([truth[a] == truth[b] == "S" or (truth[a] == truth[b] and truth[a] != "S") for a, b in keys], float)
print(f"practice: {len(keys)} pairs, {int(y.sum())} same")
w = fit(X, y)
print("weights (positive = makes 'same' more likely): " + ", ".join(f"{n} {v:+.2f}" for n, v in zip(names, w[1:])))
# leave one endpoint out
pr = np.zeros(len(keys))
for e in peps:
    tr_idx = [i for i, k in enumerate(keys) if e not in k]; te_idx = [i for i, k in enumerate(keys) if e in k]
    we = fit(X[tr_idx], y[tr_idx]); pr[te_idx] += pred(we, X[te_idx]) / 2      # each pair is held out twice
for th in (0.3, 0.5):
    print(f"practice leave-one-endpoint-out: threshold {th}: model MCC {mcc(pr >= th, y):.2f}, 'same' calls {int((pr >= th).sum())}"
          f" (right {int(((pr >= th) & (y == 1)).sum())})")
print("  top practice pairs by held-out probability: " + ", ".join(
    f"{keys[i][0][-2:]}-{keys[i][1][-2:]} {pr[i]:.2f} ({truth[keys[i][0]]}/{truth[keys[i][1]]})" for i in np.argsort(-pr)[:8]))
# scored
fam = {l.split(",")[0]: l.split(",")[1] for l in open("eprun/me/out/submission_combined.csv").read().split()[1:]}
fams = {f: sorted(e for e in fam if fam[e] == f) for f in "ABC"}
S = features("eprun", "ep", fams, 40)
sk = list(S); sp = pred(w, np.array([S[k] for k in sk]))
print("\nscored: pairs with the highest 'same model' probability per family:")
for f, es in fams.items():
    top = sorted([(sp[i], sk[i]) for i in range(len(sk)) if sk[i][0] in es], reverse=True)[:6]
    print(f"  {f}: " + ", ".join(f"{a[-2:]}-{b[-2:]} {p:.2f}" for p, (a, b) in top))
print(f"  pairs above 0.5: {int((sp >= 0.5).sum())} of {len(sp)}; above 0.3: {int((sp >= 0.3).sum())}")
