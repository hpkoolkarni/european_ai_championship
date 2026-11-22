"""ONE fixed labelling rule, the same for practice and scored (no hand-picking, no public-score tuning).
Inside each family (families given):
  sharp    : tool-choice temperature T within SHARP of the family's lowest T (behaves like the base at default
             decoding: plain, watermark, fp8, ...)
  disguise : self-description names a different maker than the family's majority, or none  (persona / system prompt)
  temp     : lowest answer-consistency in the family on the repeated story prompt (G1)    (temperature copy)
  base group = sharp + disguise + temp   -> one model label
  everyone else                          -> own model label (variants)
Validated on practice before use: model MCC 0.721 (6 right "same" calls, 0 wrong; misses only the watermark copy,
whose practice tool answers can't be read). Fixed parameters: SHARP = 0.15, at least 10 readable tool choices.
  python analysis/rule.py ref 360M      practice check against practice_answers.csv   (from model_equality/)
  python analysis/rule.py ep            writes submissions/submission_rule.csv"""
import sys, json, re, csv, itertools, collections, numpy as np
from pathlib import Path
M = Path(__file__).resolve().parents[1]                   # model_equality/
SC = str(M / "reference_models/out") + "/"
SHARP = 0.15
target = sys.argv[1]
if target == "ref":
    fams = {"P": ("SmolLM2-" + (sys.argv[2] if len(sys.argv) > 2 else "1.7B") + "-Instruct", [f"ref-{i:02d}" for i in range(1, 10)])}
    ans = str(M / "probes/data/ref_answers.jsonl"); pre = "ref_"
else:
    fams = {"A": ("Qwen2.5-3B-Instruct", [f"ep-{x}" for x in "01 02 04 12 14 17 18 23 24 25 33".split()]),
            "B": ("Qwen2.5-1.5B-Instruct", [f"ep-{x}" for x in "03 06 07 08 20 21 22 26 28 30 32".split()]),
            "C": (None, [f"ep-{x}" for x in "05 09 10 11 13 15 16 19 27 29 31".split()])}
    ans = str(M / "probes/data/ep_answers.jsonl"); pre = ""
rows = [json.loads(l) for l in open(ans, encoding="utf-8")]
VEND = (("qwen", r"qwen|alibaba"), ("meta", r"\bmeta\b|llama"), ("anthropic", r"anthropic|claude"), ("openai", r"openai|chatgpt|\bgpt"),
        ("google", r"google|gemma|gemini|deepmind"), ("microsoft", r"microsoft|\bphi\b"), ("huggingface", r"hugging ?face|smol"))
vend = {}
for r in rows:
    if r["id"] == "F7":
        t = (r["response"] or "").lower()
        vend.setdefault(r["endpoint"], next((n for n, p in VEND if re.search(p, t)), "-"))
g = collections.defaultdict(list)
for r in rows:
    if r["id"] == "G1":
        g[r["endpoint"]].append(set(re.findall(r"[a-z']+", (r["response"] or "").lower())))
cons = {e: (len(v[0] & v[1]) / max(1, len(v[0] | v[1])) if len(v) >= 2 else np.nan) for e, v in g.items()}


def temperature(model, eps):
    path = SC + f"{pre}toolfit_{model}.jsonl"
    try:
        data = collections.defaultdict(list)
        for r in map(json.loads, open(path)):
            data[r["endpoint"]].append((np.array(r["lp"]), r["chosen"]))
    except FileNotFoundError:
        return {}
    Ts = np.exp(np.linspace(np.log(0.2), np.log(4), 80)); out = {}
    for e in eps:
        it = data.get(e, [])
        if len(it) < 10:
            continue
        L = np.stack([a for a, _ in it]); c = np.array([b for _, b in it])
        best = max(Ts, key=lambda T: (lambda z: np.log(np.exp(z - z.max(1, keepdims=True))[np.arange(len(c)), c] /
                                                        np.exp(z - z.max(1, keepdims=True)).sum(1)).sum())(L / T))
        out[e] = float(best)
    return out


labels, why = {}, {}
for f, (model, eps) in fams.items():
    T = temperature(model, eps) if model else {}
    maj = collections.Counter(vend.get(e, "-") for e in eps).most_common(1)[0][0]
    sharp = {e for e in eps if e in T and T[e] <= min(T.values()) + SHARP} if T else set()
    disg = {e for e in eps if vend.get(e, "-") != maj}
    cs = {e: cons[e] for e in eps if np.isfinite(cons.get(e, np.nan))}
    temp = {min(cs, key=cs.get)} if cs else set()
    base = sharp | disg | temp
    k = 1
    for e in eps:
        if e in base:
            labels[e] = (f, f"{f}1")
        else:
            k += 1; labels[e] = (f, f"{f}{k}")
        why[e] = ",".join(n for n, s in (("sharp", sharp), ("disguise", disg), ("temp", temp)) if e in s) or "variant"
    print(f"family {f} (ref {model}, majority maker '{maj}'): " + "  ".join(
        f"{e[-2:]}:{why[e]}" + (f"(T{T[e]:.2f})" if e in T else "") for e in eps))
if target == "ref":
    truth = {r["endpoint"]: (r["family"], r["model"]) for r in csv.DictReader(open(M / "practice_answers.csv"))}
    e9 = sorted(truth); tp = fp = fn = tn = 0
    for a, b in itertools.combinations(e9, 2):
        t, gss = truth[a] == truth[b], labels[a] == labels[b]
        tp += t and gss; fp += (not t) and gss; fn += t and not gss; tn += (not t) and not gss
    d = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** .5
    print(f"PRACTICE model MCC of the rule: {max(0, (tp * tn - fp * fn) / d) if d else 0:.3f}  (tp {tp} fp {fp} fn {fn} tn {tn})")
else:
    with open(M / "submissions/submission_rule.csv", "w") as fh:
        fh.write("endpoint,family,model\n")
        for i in range(1, 34):
            e = f"ep-{i:02d}"; fh.write(f"{e},{labels[e][0]},{labels[e][1]}\n")
    print("wrote", M / "submissions/submission_rule.csv")
