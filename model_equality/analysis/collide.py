"""Within-family similarity from saved answers (no API): text = mean Hamming similarity of the first L characters of
two endpoints' answers to the same prompt (MET kernel); tools = share of identical tool choices on the same item and
repeat-shuffled items (pooled over the 20 answers). Split-half: prompts (text) / answers (tools) halved at random."""
import sys, re, collections, random, numpy as np
sys.path[:0] = ["eprun/probes", "eprun/tool_probe"]
import combine as C, tool_probe as T
L = 60
rows = C.rows_of("ep")
fam = {"A": "01 02 04 08 12 14 17 18 23 24 25 33", "B": "03 06 07 20 21 22 26 28 30 32", "C": "05 09 10 11 13 15 16 19 27 29 31"}
norm = lambda t: re.sub(r"\s+", " ", (t or "").strip().lower())
ans = collections.defaultdict(dict)
for r in rows:
    if re.fullmatch(r"[FRV]\d+", r["id"]):
        ans[r["id"]][r["endpoint"]] = np.frombuffer(norm(r["response"]).encode("utf-8", "ignore")[:L].ljust(L, b"\0"), np.uint8)
prompts = sorted(ans)
tr = T.load("ep", 20)
def text_sim(es, ps):
    M = np.zeros((len(es), len(es)))
    for p in ps:
        X = np.stack([ans[p][e] for e in es]); M += (X[:, None, :] == X[None, :, :]).mean(2)
    return M / len(ps)
def tool_sim(es, reps):
    M = np.zeros((len(es), len(es))); n = 0
    for k in range(len(T.ITEMS)):
        for r in reps:
            v = [tr[e][r][k] for e in es]
            M += np.array([[a == b and a != "INVALID" for b in v] for a in v]); n += 1
    return M / n
def groups(S, k):            # average linkage on 1 - similarity into k groups
    return C.style_groups(1 - S, k)
def ari(a, b):
    from itertools import combinations
    pa = [a[i] == a[j] for i, j in combinations(range(len(a)), 2)]; pb = [b[i] == b[j] for i, j in combinations(range(len(b)), 2)]
    return C.mcc2(pa, pb)
rng = random.Random(0)
for f, s in fam.items():
    es = [f"ep-{x}" for x in s.split()]
    St, Su = text_sim(es, prompts), tool_sim(es, range(20))
    for name, S in (("text", St), ("tools", Su)):
        off = S[~np.eye(len(es), dtype=bool)]
        print(f"\nfamily {f} {name} similarity x100 (mean off-diagonal {100*off.mean():.0f}):")
        print("      " + "".join(f"{e[-2:]:>4}" for e in es))
        for a, e in enumerate(es):
            print(f"  {e[-2:]}  " + "".join("   ." if a == b else f"{100*S[a,b]:4.0f}" for b in range(len(es))))
    print(f"family {f}: split-half agreement of the k-group split (pair-call MCC, median of 20 splits):")
    for name in ("text", "tools"):
        res = {}
        for k in (2, 3, 4):
            m = []
            for _ in range(20):
                if name == "text":
                    p = prompts[:]; rng.shuffle(p); h = len(p) // 2; A_, B_ = text_sim(es, p[:h]), text_sim(es, p[h:])
                else:
                    r = list(range(20)); rng.shuffle(r); A_, B_ = tool_sim(es, r[:10]), tool_sim(es, r[10:])
                m.append(ari(groups(A_, k), groups(B_, k)))
            res[k] = np.nanmedian(m)
        full = {k: groups(St if name == "text" else Su, k) for k in (2, 3, 4)}
        print(f"   {name:5s}: " + "  ".join(f"k={k}: {res[k]:.2f} [{' | '.join(' '.join(e[-2:] for e, g in zip(es, full[k]) if g == c) for c in range(k))}]" for k in (2, 3, 4)))
