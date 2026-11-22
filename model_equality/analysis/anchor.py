"""Prototype 'anchor and project' (no API). Inside one family:
  1. core   = the big group the omnibus tool test cannot split (copies + subtle variants)
  2. strong = endpoints far from the core (omnibus z >= cut): distinct models
  3. for every core endpoint e: leave-one-out core mean (pooled over many endpoints: a precise reference, as in RUT)
     and its projection on each core->strong direction; noise from a bootstrap of e's own answers
  4. z_e = projection / bootstrap sd, maximised over directions (Bonferroni-free: few directions); e is a separate
     model if z_e >= ZCUT
Usage: python anchor.py <data dir> <ref|ep> <n tool answers> [family labels csv for ep]"""
import sys, csv, collections, itertools, random, math, numpy as np
D, target, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
sys.path[:0] = [D + "/me", D + "/me2"]
import me_combine as C, me2_tools as T
ZCUT = 3.0
tr = T.load(target, n)
cells = [(k, o) for k, (_, opts) in enumerate(T.ITEMS) for o in opts + ["INVALID"]]


def vec(ans):
    c = collections.Counter((k, a[k]) for a in ans for k in range(len(T.ITEMS)))
    return np.array([c[x] / len(ans) for x in cells])


def family_labels(eps):
    Zf = np.full((len(eps), len(eps)), math.nan); np.fill_diagonal(Zf, 0); rng = random.Random(0)
    for i, j in itertools.combinations(range(len(eps)), 2):
        Zf[i, j] = Zf[j, i] = C.tools_pair_z(T, tr[eps[i]], tr[eps[j]], 300, rng)
    return Zf


def anchor(eps):
    Z = family_labels(eps)
    lab = C.labels_for(eps, np.zeros(len(eps), int), Z, 3.0)
    groups = collections.defaultdict(list)
    for e, (_, m) in lab.items():
        groups[m].append(e)
    core = max(groups.values(), key=len); strong = [e for e in eps if e not in core]
    F = {e: vec(tr[e]) for e in eps}; rng = random.Random(1)
    report, flag = [], {}
    for e in core:
        mu = np.mean([F[x] for x in core if x != e], 0)          # leave-one-out reference
        best = (0.0, None, 0.0)
        for s in strong:
            d = F[s] - mu; dd = d @ d
            p = float((F[e] - mu) @ d / dd)
            bs = [float((vec([rng.choice(tr[e]) for _ in tr[e]]) - mu) @ d / dd) for _ in range(300)]
            sd = float(np.std(bs)) or 1e-9
            if p / sd > best[0]:
                best = (p / sd, s, p)
        flag[e] = best[0] >= ZCUT
        report.append((e, best))
    out = {}
    k = 1
    for e in core:
        if not flag[e]:
            out[e] = "core"
    for e in core:
        if flag[e]:
            out[e] = f"weak{k}"; k += 1
    for g, es in groups.items():
        if es is not core:
            for e in es:
                out[e] = f"strong-{g}"
    return out, report, Z


if target == "ref":
    truth_rows = list(csv.DictReader(open(D + "/practice_answers.csv")))
    model = {r["endpoint"]: r["model"] for r in truth_rows}
    truth = {r["endpoint"]: (r["family"], r["model"]) for r in truth_rows}
    eps = [f"ref-{i:02d}" for i in range(1, 10)]
    out, rep, Z = anchor(eps)
    print(f"n={n}: " + "  ".join(f"{e[-2:]}({model[e]}) z={b[0]:.1f}" for e, b in rep))
    old = C.labels_for(eps, np.zeros(9, int), Z, 3.0)
    new = {e: ("A", out[e]) for e in eps}
    print(f"   omnibus only: model MCC {C.A.score(truth, old)[2]:.2f} | anchor+project: model MCC {C.A.score(truth, new)[2]:.2f}"
          f" | labels {dict((e[-2:], out[e]) for e in eps)}")
else:
    fam = {l.split(",")[0]: l.split(",")[1] for l in open(sys.argv[4]).read().split()[1:]}
    for f in sorted(set(fam.values())):
        eps = sorted(e for e in fam if fam[e] == f)
        out, rep, Z = anchor(eps)
        print(f"family {f}: core {sum(v == 'core' for v in out.values())}, strong {[e[-2:] for e in eps if out[e].startswith('strong')]}"
              f" | core z: " + "  ".join(f"{e[-2:]} {b[0]:.1f}" + (f"->{b[1][-2:]}" if b[1] else "") for e, b in sorted(rep, key=lambda t: -t[1][0])))
