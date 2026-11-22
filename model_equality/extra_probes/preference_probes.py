"""Model Equality, probe search: which batched preference probe separates the models on PRACTICE (truth known)?

Idea: the one test that held up on the scored endpoints is the tool probe (tool_probe.py): when several answers are equally
fine, WHICH one a model picks is a preference stored in its weights. This file adds three more probes of that kind.
Every request carries many items, so one answer gives many data points:
  pref : 16 open questions with any answer fine ('Name a fruit.', 'Pick a number between 1 and 100.') -> first word
  pick : 16 forced choices among 4 equally good options ('Pick a season: ...')                       -> the option
  cont : 12 common phrase openings, 'continue with the next three words'                               -> 3 words
Item order (and option order) is shuffled with a fixed seed per repeat, identical for every endpoint.
Pair test: sum over items of squared differences of the answer frequencies (unparsable answers left out), null =
the two endpoints' answers shuffled between them, calibrated z (combine.calib_z).

  python extra_probes/preference_probes.py collect --target ref --probe pref --n 25              plan only (no API call)
  python extra_probes/preference_probes.py collect --target ref --probe pref,pick,cont --n 25 --go    practice, free
  python extra_probes/preference_probes.py collect --target ep --probe pick --n 20 --go --approve-budget 20   scored (refuses above)
  python extra_probes/preference_probes.py analyze --target ref                                  practice: MCC per probe and answer count
API key: environment variable API_KEY. Answers: extra_probes/data/<target>_<probe>.jsonl. NO API call without --go.
"""
import argparse
import collections
import csv
import itertools
import json
import math
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT / "probes"), str(ROOT / "tool_probe")]
DATA = HERE / "data"
BASE = "http://34.147.229.55/33-model-equality/v1"

FMT = "Answer with one line per item: the number, a dot and the answer. Nothing else."
PROBES = {
    "pref": {
        "instr": "Answer each numbered question with ONE word or ONE number only (any answer is fine). " + FMT,
        "shot": ("1. Name a tree.\n2. Pick a random number between 1 and 5.", "1. Birch\n2. 2"),
        "items": ["Pick a random number between 1 and 100.", "Pick a random number between 1 and 10.", "Name a colour.",
                  "Name an animal.", "Pick a letter of the alphabet.", "Name a fruit.", "Give a first name.",
                  "Name a country.", "Name a day of the week.", "Name a musical instrument.", "Name a flower.",
                  "Name a sport.", "Name a vegetable.", "Name a gemstone.", "Name a profession.", "Name a bird."],
        "max_tokens": 120},
    "pick": {
        "instr": "For each numbered item, choose exactly ONE of the listed options (any of them is fine). " + FMT,
        "shot": ("1. Pick a tool: hammer, saw, drill\n2. Pick a vehicle: boat, plane", "1. saw\n2. plane"),
        "items": [("Pick a drink", ["tea", "coffee", "juice", "water"]),
                  ("Pick a season", ["spring", "summer", "autumn", "winter"]),
                  ("Pick a pet", ["cat", "dog", "rabbit", "parrot"]),
                  ("Pick a way to travel", ["train", "bus", "bicycle", "car"]),
                  ("Pick a metal", ["gold", "silver", "copper", "iron"]),
                  ("Pick a weather", ["sunny", "rainy", "snowy", "windy"]),
                  ("Pick a shape", ["circle", "square", "triangle", "star"]),
                  ("Pick a meal", ["breakfast", "lunch", "dinner", "brunch"]),
                  ("Pick an element", ["fire", "water", "earth", "air"]),
                  ("Pick a direction", ["north", "south", "east", "west"]),
                  ("Pick a dessert", ["cake", "pie", "cookies", "pudding"]),
                  ("Pick an instrument", ["piano", "guitar", "violin", "drums"]),
                  ("Pick a planet", ["mars", "venus", "jupiter", "saturn"]),
                  ("Pick a number", ["three", "five", "seven", "nine"]),
                  ("Pick a place", ["beach", "mountain", "forest", "city"]),
                  ("Pick a hobby", ["reading", "painting", "cooking", "hiking"])],
        "max_tokens": 120},
    # edge of ability: which products a model gets right, and WHICH wrong number it writes, depends on its weights;
    # a persona, system prompt or watermark barely changes arithmetic. Mixed difficulty so every model size has items
    # near its edge.
    "math": {
        "instr": "Compute each numbered product exactly. Answer with one line per item: the number, a dot and the result "
                 "only (digits, no working). Nothing else.",
        "items": ["47 x 86", "93 x 58", "64 x 79", "38 x 97", "347 x 68", "729 x 46", "582 x 93", "816 x 37",
                  "463 x 59", "274 x 88", "357 x 846", "629 x 473", "918 x 264", "485 x 737", "763 x 592", "394 x 681"],
        "max_tokens": 256},                    # models echo 'a x b = c': 16 lines need ~200 tokens
    "cont": {
        "instr": "Continue each numbered phrase with exactly the next THREE words. " + FMT,
        "shot": ("1. Early in the\n2. She opened the door and", "1. morning we left\n2. walked straight in"),
        "items": ["Once upon a", "The quick brown fox", "In the beginning", "It was a dark and", "To be or",
                  "All that glitters", "The best way to learn", "My favourite thing about", "Yesterday I went to the",
                  "The secret of happiness is", "When I grow up I", "The city at night"],
        "max_tokens": 160},
}
STOP = {"a", "an", "the"}


def layout(probe, repeat):
    """Item order and option order for this repeat: same for every endpoint."""
    P = PROBES[probe]; rng = random.Random(f"{probe}-{repeat}")
    order = list(range(len(P["items"]))); rng.shuffle(order)
    opts = {}
    if probe == "pick":
        for k in order:
            o = P["items"][k][1][:]; rng.shuffle(o); opts[k] = o
    return order, opts


def messages(probe, repeat):
    P = PROBES[probe]; order, opts = layout(probe, repeat)
    lines = []
    for pos, k in enumerate(order, 1):
        it = P["items"][k]
        lines.append(f"{pos}. {it[0]}: {', '.join(opts[k])}" if probe == "pick" else f"{pos}. {it}")
    # no worked example: the small practice model mostly copies an example answer instead of answering (seen with the
    # tool probe: 0.4 of 12 items readable), and how often it copies depends on the disguise
    return [{"role": "user", "content": P["instr"] + "\n" + "\n".join(lines)}]


def words(s):
    return re.findall(r"[a-z0-9]+", s.lower().replace("*", ""))


def parse(probe, repeat, text):
    """-> {item index: normalised answer}; items that cannot be read are left out."""
    P = PROBES[probe]; order, opts = layout(probe, repeat)
    if probe == "math":                        # a line that names the problem ('347 x 68 = 23456') counts for that
        out = {}                               # problem whatever its number; else the line number decides
        for line in (text or "").splitlines():
            s = re.sub(r"\s*[x×*]\s*", " x ", line.replace("`", ""))
            hit = [k for k, it in enumerate(P["items"]) if re.search(r"(?<!\d)" + re.escape(it) + r"(?!\d)", s)]
            if hit:
                k = hit[0]; rest = s.split(P["items"][k], 1)[1]
            else:
                m = re.match(r"^\s*\**(\d+)\s*[.):-]\s*(.+)$", s)
                if not m or not 1 <= int(m.group(1)) <= len(order):
                    continue
                k = order[int(m.group(1)) - 1]; rest = m.group(2)
            nums = re.findall(r"\d[\d,]*", rest)
            if nums and k not in out:
                out[k] = nums[-1].replace(",", "")
        return out
    got = {}
    for m in re.finditer(r"^\s*\**(\d+)\s*[.):-]\s*(.+?)\s*$", text or "", re.M):
        pos = int(m.group(1))
        if 1 <= pos <= len(order) and pos not in got:
            got[pos] = m.group(2)
    out = {}
    for pos, raw in got.items():
        k = order[pos - 1]
        if probe == "pref":                    # 'The number is 35.' / 'A colour is: Blue.' -> the part after is / :
            tail = re.split(r":|\bis\b|\bare\b|\bwould be\b", raw, flags=re.I)[-1]
            raw = tail if words(tail) else raw
        w = words(raw)
        if probe == "pref":
            w = [x for x in w if x not in STOP]
            if w:
                out[k] = w[0]
        elif probe == "pick":
            hit = [o for o in P["items"][k][1] if o in w]
            if len(hit) == 1:
                out[k] = hit[0]
        elif probe == "math":                  # the last number on the line ('347 x 68 = 23,596' -> 23596)
            nums = re.findall(r"\d[\d,]*", raw.replace(P["items"][k], ""))
            if nums:
                out[k] = nums[-1].replace(",", "")
        elif len(w) >= 1:
            out[k] = " ".join(w[:3])
    return out


# ------------------------------------------------------------------ collect
def endpoints(target):
    return [f"ref-{i:02d}" for i in range(1, 10)] if target == "ref" else [f"ep-{i:02d}" for i in range(1, 34)]


def collect(a):
    probes = a.probe.split(",")
    for p in probes:
        if p not in PROBES:
            sys.exit(f"unknown probe {p}; choose from {list(PROBES)}")
    DATA.mkdir(exist_ok=True)
    jobs = []
    for p in probes:
        out = DATA / f"{a.target}_{p}.jsonl"
        done = {(r["endpoint"], r["repeat"]) for r in map(json.loads, open(out, encoding="utf-8"))} if out.exists() else set()
        jobs += [(p, e, r) for r in range(a.n) for e in endpoints(a.target) if (e, r) not in done]
    jobs.sort(key=lambda j: (j[2], probes.index(j[0])))            # repeat-major: an early stop covers all evenly
    per = collections.Counter(e for _, e, _ in jobs); worst = max(per.values(), default=0)
    print(f"target {a.target}, probes {probes}, {a.n} answers each: {len(jobs)} new requests, at most {worst} per "
          f"endpoint" + ("  (SCORED: taken from the 100-per-endpoint budget)" if a.target == "ep" else "  (practice: free)"))
    if not a.go:
        print("Plan only (add --go to send). Example request:\n" + json.dumps(messages(probes[0], 0), indent=1)[:1600])
        return
    if a.target == "ep" and worst > a.approve_budget:
        sys.exit(f"Refused: {worst} new answers per scored endpoint > --approve-budget {a.approve_budget}")
    key = os.environ.get("API_KEY", "").strip()
    if not key:
        sys.exit("Set the API key first:  read -s API_KEY; export API_KEY")
    import requests
    s = requests.Session(); s.headers["X-API-Key"] = key
    lock, n, files = threading.Lock(), [0], {p: open(DATA / f"{a.target}_{p}.jsonl", "a", encoding="utf-8") for p in probes}

    def one(job):
        p, e, r = job
        for _ in range(30):
            try:
                resp = s.post(f"{BASE}/chat/completions", timeout=180,
                              json={"model": e, "max_tokens": PROBES[p]["max_tokens"], "messages": messages(p, r)})
            except Exception:
                time.sleep(15); continue
            if resp.status_code == 200:
                ch = resp.json()["choices"][0]
                with lock:
                    files[p].write(json.dumps({"endpoint": e, "repeat": r, "response": ch["message"]["content"],
                                               "finish_reason": ch.get("finish_reason")}, ensure_ascii=False) + "\n")
                    files[p].flush(); n[0] += 1
                    if n[0] % 20 == 0 or n[0] == len(jobs):
                        print(f"{n[0]}/{len(jobs)}", flush=True)
                return
            if resp.status_code == 429 and "Retry-After" in resp.headers:
                time.sleep(int(resp.headers["Retry-After"])); continue
            if resp.status_code in (502, 503, 504):
                time.sleep(10); continue
            print(e, p, r, resp.status_code, resp.text[:200], file=sys.stderr); return
    try:
        with ThreadPoolExecutor(4) as pool:
            list(pool.map(one, jobs))
    finally:
        for f in files.values():
            f.close()
    print("finished")


# ------------------------------------------------------------------ analyze
def load(target, probe, nmax):
    """{endpoint: [answer dicts]} for repeats < nmax ('tools' = the teammate's probe, same shape)."""
    rows = collections.defaultdict(dict)
    if probe == "tools":
        import tool_probe as T
        path = T.DATA / f"{target}_tools.jsonl"
        if not path.exists():
            return {}
        for r in map(json.loads, open(path)):
            if r["repeat"] < nmax:
                rows[r["endpoint"]][r["repeat"]] = dict(enumerate(T.parse(r["response"])))   # as tool_probe.py: INVALID counts
    else:
        path = DATA / f"{target}_{probe}.jsonl"
        if not path.exists():
            return {}
        for r in map(json.loads, open(path, encoding="utf-8")):
            if r["repeat"] < nmax:
                rows[r["endpoint"]][r["repeat"]] = parse(probe, r["repeat"], r["response"])
    return {e: [d[k] for k in sorted(d)] for e, d in rows.items()}


def stat(A, B):
    t = 0.0
    for k in set().union(*A, *B):
        va = [x[k] for x in A if k in x]; vb = [x[k] for x in B if k in x]
        if len(va) < 2 or len(vb) < 2:
            continue
        ca, cb = collections.Counter(va), collections.Counter(vb)
        t += sum((ca[o] / len(va) - cb[o] / len(vb)) ** 2 for o in set(ca) | set(cb))
    return t


def pair_z(A, B, perms, rng):
    import combine as C
    t, pool, nulls = stat(A, B), list(A) + list(B), []
    for _ in range(perms):
        rng.shuffle(pool); nulls.append(stat(pool[:len(A)], pool[len(A):]))
    return C.calib_z(t, nulls)


def zmatrix(eps, data, perms=300, seed=0):
    rng = random.Random(seed); n = len(eps)
    Z = np.full((n, n), math.nan); np.fill_diagonal(Z, 0)
    for i, j in itertools.combinations(range(n), 2):
        if data.get(eps[i]) and data.get(eps[j]):
            Z[i, j] = Z[j, i] = pair_z(data[eps[i]], data[eps[j]], perms, rng)
    return Z


def analyze(a):
    import combine as C
    truth_rows = list(csv.DictReader(open(ROOT / "practice_answers.csv")))
    model = {r["endpoint"]: r["model"] for r in truth_rows}
    truth = {r["endpoint"]: (r["family"], r["model"]) for r in truth_rows}
    eps = endpoints("ref"); S = [i for i, e in enumerate(eps) if model[e] == "S"]
    probes = [p for p in ["tools"] + list(PROBES) if load("ref", p, 10 ** 6)]
    print("probes with practice answers:", probes)
    for p in probes:
        d = load("ref", p, 10 ** 6)
        print(f"  {p}: answers per endpoint {min(len(v) for v in d.values())}..{max(len(v) for v in d.values())}; "
              f"readable items per answer: " + " ".join(f"{e[-2:]}={np.mean([len(x) for x in d[e]]):.1f}" for e in eps if e in d))
    print(f"\n{'probe':18s} {'n':>3s}  model MCC  | S-S max z, median | median z S vs T1    T2    T3    T4")
    results = {}
    combos = [[p] for p in probes] + ([probes] if len(probes) > 1 else [])
    for n in [int(x) for x in a.ns.split(",")]:
        mats = {p: zmatrix(eps, load("ref", p, n)) for p in probes}
        for combo in combos:
            Z = C.combine([mats[p] for p in combo])
            lab = C.labels_for(eps, np.zeros(len(eps), int), Z, a.cut)
            mcc = C.A.score(truth, lab)[2]
            ss = [Z[i, j] for i, j in itertools.combinations(S, 2)]
            vs = {v: np.median([Z[i, j] for i in S for j, e in enumerate(eps) if model[e] == v]) for v in ("T1", "T2", "T3", "T4")}
            name = "+".join(combo)
            results[(name, n)] = mcc
            groups = " | ".join(" ".join(e[-2:] for e in eps if lab[e][1] == g) for g in sorted({m for _, m in lab.values()}))
            print(f"{name:18s} {n:3d}  {mcc:9.2f}  | {np.nanmax(ss):9.1f} {np.nanmedian(ss):6.1f} |      "
                  + "  ".join(f"{vs[v]:5.1f}" for v in vs) + f"   groups: {groups}")
    print("\nbar: model MCC >= 0.9 with the disguised copies (S) kept together (S-S max z below the cut "
          f"{a.cut}) at a budget the scored endpoints can afford")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--target", choices=["ref", "ep"], required=True)
    c.add_argument("--probe", required=True, help=f"comma list of {list(PROBES)}")
    c.add_argument("--n", type=int, required=True, help="answers per endpoint for each probe (repeats 0..n-1)")
    c.add_argument("--go", action="store_true")
    c.add_argument("--approve-budget", type=int, default=0)
    z = sub.add_parser("analyze")
    z.add_argument("--target", choices=["ref"], default="ref")
    z.add_argument("--ns", default="5,10,15,20,25", help="answer counts to evaluate (first n repeats)")
    z.add_argument("--cut", type=float, default=3.0)
    a = ap.parse_args()
    collect(a) if a.cmd == "collect" else analyze(a)


if __name__ == "__main__":
    main()
