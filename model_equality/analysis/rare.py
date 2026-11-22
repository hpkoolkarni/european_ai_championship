"""Rare shared phrases (saved answers only, no API). Idea: a model's idiosyncrasies (a wrong fact it believes, an odd
phrase it likes, the opening of its 'random' list) live in its weights, so copies share them even under a disguise,
while the family's common phrasing is shared by everyone and says nothing. For each prompt, every word 3-gram that
occurs in the answers of 2 or 3 endpoints of the family (rare) adds 1/(number of sharers - 1) to each sharing pair;
plus the shared opening (first 15 characters) of the normalised answer. Chance: answers shuffled between the
endpoints of the family, prompt by prompt (2000 times). Practice: first answer per prompt, as in the scored data.
  python rare.py <data dir> <ref|ep> [families csv]"""
import sys, re, json, csv, itertools, collections, numpy as np
D, target = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(f"{D}/probes/data/{target}_answers.jsonl", encoding="utf-8")]
if target == "ref":
    truth = {r["endpoint"]: r["model"] + ":" + r.get("kind", "")[:10] for r in csv.DictReader(open(f"{D}/practice_answers.csv"))}
    fams = {"A": sorted(truth)}
else:
    fam = {l.split(",")[0]: l.split(",")[1] for l in open(sys.argv[3]).read().split()[1:]}
    fams = {f: sorted(e for e in fam if fam[e] == f) for f in sorted(set(fam.values()))}
    truth = None
first = {}
for r in rows:
    first.setdefault((r["endpoint"], r["id"]), r["response"] or "")
prompts = sorted({p for _, p in first})


def toks(t):
    return re.findall(r"[a-z0-9']+", t.lower())


def pair_scores(es, assign):
    n = len(es); M = np.zeros((n, n))
    for p in prompts:
        texts = assign[p]
        grams = [set(zip(w, w[1:], w[2:])) for w in (toks(t) for t in texts)]
        cnt = collections.Counter(g for s in grams for g in s)
        for g, c in cnt.items():
            if 2 <= c <= 3:
                who = [i for i in range(n) if g in grams[i]]
                for i, j in itertools.combinations(who, 2):
                    M[i, j] += 1 / (c - 1)
        norm = [re.sub(r"[^a-z0-9]", "", t.lower())[:15] for t in texts]
        for i, j in itertools.combinations(range(n), 2):
            k = 0
            while k < min(len(norm[i]), len(norm[j])) and norm[i][k] == norm[j][k]:
                k += 1
            M[i, j] += k / 15
    return M + M.T


rng = np.random.default_rng(0)
for f, es in fams.items():
    obs = {p: [first.get((e, p), "") for e in es] for p in prompts}
    O = pair_scores(es, obs)
    null = np.array([pair_scores(es, {p: list(rng.permutation(obs[p])) for p in prompts}) for _ in range(300)])
    off = null[:, ~np.eye(len(es), dtype=bool)].ravel()
    pairs = sorted(((O[i, j], es[i], es[j]) for i, j in itertools.combinations(range(len(es)), 2)), reverse=True)
    print(f"\nfamily {f} ({len(prompts)} prompts): shared-phrase score, chance mean {off.mean():.1f}, 99th pct {np.percentile(off, 99):.1f}")
    for v, a, b in pairs[:10]:
        t = f"  truth {truth[a]} / {truth[b]}" if truth else ""
        print(f"   {a[-2:]}-{b[-2:]}  {v:5.1f}  p≈{(off >= v).mean():.4f}{t}")
    if truth:
        S = [i for i, e in enumerate(es) if truth[e].startswith("S")]
        ss = [O[i, j] for i, j in itertools.combinations(S, 2)]
        other = [O[i, j] for i, j in itertools.combinations(range(len(es)), 2) if not (i in S and j in S)]
        print(f"   copies among themselves: mean {np.mean(ss):.1f} | all other pairs: mean {np.mean(other):.1f}")
