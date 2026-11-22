"""Twins: for every candidate disguised endpoint, which core endpoint is its closest match in tool choices (share of
identical choices on the same item and repeat, 40 answers each)? Stability: 200 random halvings of the 40 answers;
how often the same partner comes out on top in BOTH halves. Saved answers only, no API."""
import json, random, sys, collections
P = sys.argv[1] if len(sys.argv) > 1 else "eprun"    # copy of model_equality/ holding the scored answers
sys.path[:0] = [P + "/tool_probe"]
import tool_probe as T
tr = collections.defaultdict(dict)
for r in map(json.loads, open(P + "/tool_probe/data/ep_tools.jsonl")):
    tr[r["endpoint"]][r["repeat"]] = T.parse(r["response"])
fam = {"A": "01 02 04 08 12 14 17 18 23 24 25 33", "B": "03 06 07 20 21 22 26 28 30 32", "C": "05 09 10 11 13 15 16 19 27 29 31"}
disg = {"A": "02 08 23", "B": "20 30 28 03 21", "C": "05 11 29 31 16"}


def sim(a, b, reps):
    n = m = 0
    for r in reps:
        for x, y in zip(tr[a][r], tr[b][r]):
            if x != "INVALID" and y != "INVALID":
                n += x == y; m += 1
    return n / m if m else 0.0


rng = random.Random(0)
for f, s in fam.items():
    es = [f"ep-{x}" for x in s.split()]
    print(f"\nfamily {f}")
    for d in disg[f].split():
        a = f"ep-{d}"; others = [e for e in es if e != a]
        full = sorted(((sim(a, b, range(40)), b) for b in others), reverse=True)
        cnt = collections.Counter()
        for _ in range(200):
            r = list(range(40)); rng.shuffle(r)
            t1 = max(others, key=lambda b: sim(a, b, r[:20])); t2 = max(others, key=lambda b: sim(a, b, r[20:]))
            if t1 == t2:
                cnt[t1] += 1
        best = cnt.most_common(1)[0] if cnt else ("-", 0)
        print(f"  {d}: closest " + ", ".join(f"{b[-2:]} {v:.2f}" for v, b in full[:3])
              + f" | same top partner in both halves: {best[0][-2:]} {best[1] / 2:.0f}%")
