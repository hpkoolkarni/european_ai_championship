"""Disguise marks per endpoint, from saved answers only (no API). python disguise.py <data dir> <ref|ep> [family csv]
  vendor   : who the endpoint says made it (F7 or any 'who are you' answer)
  tool_H   : mean entropy of its tool choices per item (temperature 1.0 -> flatter choices -> higher)
  invalid  : share of unreadable tool items
  text_d   : mean answer-text distance to the other endpoints of its family (system prompt / persona -> far)
  len      : median reply length
  self_sim : similarity of its own two answers to the same story prompt (G1; temperature -> lower)"""
import sys, re, math, collections, csv, numpy as np
D, target = sys.argv[1], sys.argv[2]
sys.path[:0] = [D + "/probes", D + "/tool_probe"]
import combine as C, tool_probe as T
rows = C.rows_of(target); eps = sorted({r["endpoint"] for r in rows})
if target == "ref":
    tr_rows = list(csv.DictReader(open(D + "/practice_answers.csv")))
    tag = {r["endpoint"]: f"{r['model']}:{r.get('kind','')[:22]}" for r in tr_rows}
    fam = {e: "A" for e in eps}
else:
    fam = {l.split(",")[0]: l.split(",")[1] for l in open(sys.argv[3]).read().split()[1:]}
    tag = {e: "" for e in eps}
vend = {}
for r in rows:
    if r["id"] in ("F7",) or re.search(r"who (are|made|created) you|what model", r["prompt"] or "", re.I):
        vend.setdefault(r["endpoint"], C.vendor_of(r["response"]))
tr = T.load(target, 10 ** 6)
def ent(e):
    H = []
    for k in range(len(T.ITEMS)):
        c = collections.Counter(a[k] for a in tr.get(e, []) if a[k] != "INVALID")
        n = sum(c.values())
        if n >= 3:
            H.append(-sum(v / n * math.log(v / n) for v in c.values()))
    return float(np.mean(H)) if H else math.nan
inv = {e: float(np.mean([x == "INVALID" for a in tr.get(e, []) for x in a])) if tr.get(e) else math.nan for e in eps}
ids = {r["id"] for r in rows if re.fullmatch(r"[FRV]\d+", r["id"])}
Dc = C.content_distance(eps, rows, ids)
lens = {e: float(np.median([len(r["response"] or "") for r in rows if r["endpoint"] == e and r["id"] in ids])) for e in eps}
g = collections.defaultdict(list)
for r in rows:
    if r["id"].startswith("G"):
        g[(r["endpoint"], r["id"])].append(set(re.findall(r"[a-z']+", (r["response"] or "").lower())))
def selfsim(e):
    v = [len(a & b) / max(1, len(a | b)) for (ee, _), xs in g.items() if ee == e for i, a in enumerate(xs) for b in xs[i + 1:]]
    return float(np.mean(v)) if v else math.nan
print(f"{'endpoint':9s} {'truth':28s} {'fam':3s} {'vendor':11s} {'tool_H':>6s} {'invalid':>7s} {'text_d':>6s} {'len':>5s} {'self_sim':>8s}")
for f in sorted(set(fam.values())):
    es = [e for e in eps if fam[e] == f]
    for e in es:
        i = eps.index(e); others = [eps.index(x) for x in es if x != e]
        td = float(np.nanmean(Dc[i, others])) if others else math.nan
        print(f"{e:9s} {tag[e]:28s} {f:3s} {vend.get(e,'?'):11s} {ent(e):6.2f} {inv[e]:7.2f} {td:6.2f} {lens[e]:5.0f} {selfsim(e):8.2f}")
    print()
