"""Tool-probe answer count vs value, with calibrated (default) or raw z. Practice answers only.
  python weigh.py [raw]"""
import csv, itertools, math, random, sys, collections
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parent / "data"
sys.path[:0] = [str(D / "probes"), str(D / "tool_probe")]
import combine as C  # noqa: E402
import tool_probe as T   # noqa: E402

C.RAW_Z = len(sys.argv) > 1 and sys.argv[1] == "raw"
rows = C.rows_of("ref"); eps = sorted({r["endpoint"] for r in rows}); n = len(eps)
truth_rows = list(csv.DictReader(open(D / "practice_answers.csv")))
model = {r["endpoint"]: r["model"] for r in truth_rows}
truth = {r["endpoint"]: (r["family"], r["model"]) for r in truth_rows}
pairs = list(itertools.combinations(range(n), 2))
S = [i for i, e in enumerate(eps) if model[e] == "S"]
ss = [(i, j) for i, j in pairs if i in S and j in S]


def summary(Z, label):
    zss = [Z[i, j] for i, j in ss]
    def vs(v): return np.mean([Z[i, j] for i, j in pairs if (i in S) != (j in S) and v in (model[eps[i]], model[eps[j]])])
    mcc = C.A.score(truth, C.labels_for(eps, np.zeros(n, int), Z, 3.0))[2]
    return (f"{label:12s} S-S max {max(zss):5.1f} mean {np.mean(zss):+5.2f} | S-T1 {vs('T1'):5.1f} S-T2 {vs('T2'):5.1f} "
            f"S-T3 {vs('T3'):5.1f} S-T4 {vs('T4'):5.1f} | model MCC {mcc:.2f}")


print("calibrated z" if not C.RAW_Z else "RAW z")
full = T.load("ref", 30)
# true null: one endpoint's own answers split in two halves
r0, nullz = random.Random(5), []
for e in eps:
    for _ in range(12):
        x = full[e][:]; r0.shuffle(x); h = len(x) // 2
        nullz.append(C.tools_pair_z(T, x[:h], x[h:], 300, random.Random(1)))
nullz = np.array(nullz)
print(f"tool probe, self-split null ({len(nullz)} splits of 15 vs 15): mean {nullz.mean():+.2f} sd {nullz.std():.2f} "
      f"share > 3: {np.mean(nullz > 3):.1%} (a calibrated z: ~0, ~1, ~0.1%)")
ZL, _, _ = C.channel_L(eps, rows, r"[FR]\d+", pairs)
print(summary(ZL, "L alone"))
for k in (5, 10, 15, 20, 25, 29):
    res = collections.defaultdict(list)
    for d in range(4 if k < 29 else 1):
        rng = random.Random(d); sub = {e: rng.sample(full[e], k) for e in eps}
        ZU = np.zeros((n, n)); r2 = random.Random(0)
        for i, j in pairs:
            ZU[i, j] = ZU[j, i] = C.tools_pair_z(T, sub[eps[i]], sub[eps[j]], 300, r2)
        res["U"].append(ZU); res["L+U"].append((ZL + ZU) / math.sqrt(2))
    for nm in ("U", "L+U"):
        mccs = [round(C.A.score(truth, C.labels_for(eps, np.zeros(n, int), Z, 3.0))[2], 2) for Z in res[nm]]
        print(summary(np.mean(res[nm], 0), f"{nm} n={k}") + f"  per draw {mccs}")
