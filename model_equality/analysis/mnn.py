"""Mutual nearest neighbours inside each family (no API, saved answers only).
Similarity per pair = mean of two standardised similarities: tool choices (share of identical choices on the same
item and repeat, over the 40 tool answers) and answer text (per-prompt character n-gram similarity).
A pair is a candidate 'same model' if each is the other's nearest neighbour. Stability: 100 random splits of the
data in halves (tool answers halved, prompts halved); how often the pair is mutual NN in BOTH halves.
Also run on practice (truth known) to see whether mutual-NN pairs are truly same there."""
import sys, re, collections, random, csv, numpy as np
D, target = sys.argv[1], sys.argv[2]
sys.path[:0] = [D + "/me", D + "/me2"]
import me_combine as C, me2_tools as T
rows = C.rows_of(target); tr = T.load(target, 40)
eps = sorted(tr)
if target == "ref":
    truth = {r["endpoint"]: r["model"] for r in csv.DictReader(open(D + "/practice_answers.csv"))}
    fams = {"A": eps}
else:
    fam = {l.split(",")[0]: l.split(",")[1] for l in open(sys.argv[3]).read().split()[1:]}
    fams = {f: [e for e in eps if fam[e] == f] for f in sorted(set(fam.values()))}
    truth = None
ids = sorted({r["id"] for r in rows if re.fullmatch(r"[FRV]\d+", r["id"])})
nrep = min(len(v) for v in tr.values())


def tool_sim(es, reps):
    M = np.zeros((len(es), len(es))); n = 0
    for k in range(len(T.ITEMS)):
        for r in reps:
            v = [tr[e][r][k] for e in es]
            M += np.array([[a == b and a != "INVALID" for b in v] for a in v]); n += 1
    return M / n


def text_sim(es, ps):
    Dc = C.content_distance(es, rows, set(ps))
    return 1 - Dc


def zs(M):
    off = M[~np.eye(len(M), dtype=bool)]
    return (M - off.mean()) / (off.std() + 1e-9)


def mnn(S):
    S = S.copy(); np.fill_diagonal(S, -np.inf)
    nn = S.argmax(1)
    return {tuple(sorted((i, int(nn[i])))) for i in range(len(S)) if nn[nn[i]] == i}


rng = random.Random(0)
for f, es in fams.items():
    if len(es) < 3:
        continue
    S = (zs(tool_sim(es, range(nrep))) + (zs(text_sim(es, ids)) if ids else 0)) / (2 if ids else 1)
    full = mnn(S)
    cnt = collections.Counter()
    for _ in range(100):
        r = list(range(nrep)); rng.shuffle(r); p = ids[:]; rng.shuffle(p)
        h1 = (zs(tool_sim(es, r[: nrep // 2])) + zs(text_sim(es, p[: len(p) // 2]))) / 2
        h2 = (zs(tool_sim(es, r[nrep // 2:])) + zs(text_sim(es, p[len(p) // 2:]))) / 2
        for pr in mnn(h1) & mnn(h2):
            cnt[pr] += 1
    print(f"family {f}: mutual nearest neighbours on all data, and how often they are mutual NN in both halves:")
    for i, j in sorted(full, key=lambda pr: -cnt[pr]):
        t = f"  truth {truth[es[i]]}/{truth[es[j]]}" if truth else ""
        print(f"   {es[i][-2:]}-{es[j][-2:]}  similarity {S[i, j]:+.2f}  stable in {cnt[(i, j)]}% of splits{t}")
    other = [(pr, c) for pr, c in cnt.most_common(5) if pr not in full]
    if other:
        print("   (other pairs mutual NN in both halves: " + ", ".join(f"{es[i][-2:]}-{es[j][-2:]} {c}%" for (i, j), c in other) + ")")
