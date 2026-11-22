"""MET-style (Gao et al.) Hamming-kernel MMD per practice probe kind: does any kind separate fp8 / LoRA 0.3 from the
disguised copies? Practice answers only, no API."""
import sys, json, csv, collections, itertools, numpy as np
D = sys.argv[1]; L = int(sys.argv[2]) if len(sys.argv) > 2 else 50
rows = [json.loads(l) for l in open(D + "/probes/data/ref_answers.jsonl", encoding="utf-8")]
truth = {r["endpoint"]: r["model"] for r in csv.DictReader(open(D + "/practice_answers.csv"))}
eps = [f"ref-{i:02d}" for i in range(1, 10)]; S = [e for e in eps if truth[e] == "S"]
def enc(t):
    b = np.frombuffer((t or "").encode("utf-8", "ignore")[:L].ljust(L, b"\0"), dtype=np.uint8); return b
by = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    by[r["id"].rstrip("o0123456789")[:1]][(r["endpoint"], r["id"])].append(enc(r["response"]))
def mmd(A, B):  # A, B: {prompt: array (n, L)}
    s, k = 0.0, 0
    for p in A:
        if p not in B or len(A[p]) < 2 or len(B[p]) < 2: continue
        a, b = A[p], B[p]
        kaa = (a[:, None, :] == a[None, :, :]).mean(2); kbb = (b[:, None, :] == b[None, :, :]).mean(2); kab = (a[:, None, :] == b[None, :, :]).mean(2)
        na, nb = len(a), len(b)
        s += (kaa.sum() - np.trace(kaa)) / (na * (na - 1)) + (kbb.sum() - np.trace(kbb)) / (nb * (nb - 1)) - 2 * kab.mean(); k += 1
    return s / k if k else np.nan
def z(A, B, rng, perms=200):
    obs = mmd(A, B); nulls = []
    for _ in range(perms):
        A2, B2 = {}, {}
        for p in A:
            if p not in B: continue
            pool = np.concatenate([A[p], B[p]]); idx = rng.permutation(len(pool)); A2[p] = pool[idx[:len(A[p])]]; B2[p] = pool[idx[len(A[p]):]]
        nulls.append(mmd(A2, B2))
    nulls = np.array(nulls); return (obs - nulls.mean()) / (nulls.std() + 1e-12)
for kind, d in sorted(by.items()):
    per = {e: {pid: np.array(v) for (ee, pid), v in d.items() if ee == e} for e in eps}
    nrep = np.median([len(v) for v in d.values()])
    if nrep < 2: continue
    rng = np.random.default_rng(0); Z = {}
    for a, b in itertools.combinations(eps, 2): Z[a, b] = Z[b, a] = z(per[a], per[b], rng)
    ss = [Z[a, b] for a, b in itertools.combinations(S, 2)]
    vs = {v: np.median([Z[s, e] for s in S for e in eps if truth[e] == v]) for v in ("T1", "T2", "T3", "T4")}
    print(f"kind {kind}: {len(d)//9} prompts x ~{nrep:.0f} answers | S-S max z {max(ss):5.1f} median {np.median(ss):4.1f} | S vs T1 {vs['T1']:5.1f} T2 {vs['T2']:5.1f} T3 {vs['T3']:5.1f} T4 {vs['T4']:5.1f}")
