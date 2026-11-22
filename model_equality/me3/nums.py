"""Number-sequence fingerprint. A model asked for 'random' numbers writes its own favourite sequences; copies of
the same weights share them (a persona or system prompt adds words, not different digits), variants drift.
4 prompts x 5 answers = 20 answers per endpoint. Comparison: the first 12 digits of the number list, Hamming
similarity between answers to the same prompt (the MET kernel of Gao et al.), endpoint against endpoint with a
permutation test, plus each endpoint against itself.

  python me3/nums.py collect --target ep --n 5                         plan only (no API call)
  python me3/nums.py collect --target ep --n 5 --go --approve-budget 20
  python me3/nums.py analyze --target ep --families me/out/submission_final.csv
API key: environment variable API_KEY. Answers: me3/data/<target>_nums.jsonl. NO API call without --go."""
import argparse, collections, itertools, json, math, os, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; DATA = HERE / "data"; ROOT = HERE.parent
BASE = "http://34.147.229.55/33-model-equality/v1"
PROMPTS = {
    "N1": "Write 30 random whole numbers between 1 and 100, separated by commas. Output only the numbers.",
    "N2": "Write 20 random digits separated by spaces. Output only the digits.",
    "N3": "Pick 10 random numbers between 1 and 1000, separated by commas. Output only the numbers.",
    "N4": "Write a random 12-digit number. Output only the number.",
}
SAVED = {"N1": "R3", "N2": "V19"}          # identical prompts already answered once in me/data/<target>_answers.jsonl
L = 12


def digits(text):
    """Digits of the number list: the first run of >= 3 numbers (skips 'Here are 30 random numbers ...')."""
    t = text or ""
    m = re.search(r"\d+(?:[ ,;\n\t]+\d+){2,}", t)
    if m:
        return re.sub(r"\D", "", m.group(0))
    m = re.search(r"\d{6,}", t)
    return m.group(0) if m else ""


def eps_of(target):
    return [f"ref-{i:02d}" for i in range(1, 10)] if target == "ref" else [f"ep-{i:02d}" for i in range(1, 34)]


def collect(a):
    DATA.mkdir(exist_ok=True); out = DATA / f"{a.target}_nums.jsonl"
    done = {(r["endpoint"], r["id"], r["repeat"]) for r in map(json.loads, open(out, encoding="utf-8"))} if out.exists() else set()
    jobs = [(e, p, r) for r in range(a.n) for p in PROMPTS for e in eps_of(a.target) if (e, p, r) not in done]
    per = collections.Counter(e for e, _, _ in jobs); worst = max(per.values(), default=0)
    print(f"target {a.target}: {len(jobs)} new requests, at most {worst} per endpoint"
          + ("  (SCORED: from the 100-per-endpoint budget)" if a.target == "ep" else "  (practice: free)"))
    if not a.go:
        print("Plan only (add --go). Prompts:\n  " + "\n  ".join(f"{k}: {v}" for k, v in PROMPTS.items())); return
    if a.target == "ep" and worst > a.approve_budget:
        sys.exit(f"Refused: {worst} new answers per scored endpoint > --approve-budget {a.approve_budget}")
    key = os.environ.get("API_KEY", "").strip()
    if not key:
        sys.exit("Set the API key first:  read -s API_KEY; export API_KEY")
    import requests
    s = requests.Session(); s.headers["X-API-Key"] = key
    lock, n, fh = threading.Lock(), [0], open(out, "a", encoding="utf-8")

    def one(job):
        e, p, r = job
        for _ in range(30):
            try:
                resp = s.post(f"{BASE}/chat/completions", timeout=180,
                              json={"model": e, "max_tokens": 120, "messages": [{"role": "user", "content": PROMPTS[p]}]})
            except Exception:
                time.sleep(15); continue
            if resp.status_code == 200:
                ch = resp.json()["choices"][0]
                with lock:
                    fh.write(json.dumps({"endpoint": e, "id": p, "repeat": r, "response": ch["message"]["content"],
                                         "finish_reason": ch.get("finish_reason")}, ensure_ascii=False) + "\n"); fh.flush()
                    n[0] += 1
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
        fh.close()
    print("finished", out)


def load(target):
    d = collections.defaultdict(lambda: collections.defaultdict(list))
    p = DATA / f"{target}_nums.jsonl"
    if p.exists():
        for r in map(json.loads, open(p, encoding="utf-8")):
            d[r["endpoint"]][r["id"]].append(digits(r["response"]))
    old = ROOT / "me" / "data" / f"{target}_answers.jsonl"
    if old.exists():
        for r in map(json.loads, open(old, encoding="utf-8")):
            for k, v in SAVED.items():
                if r["id"] == v:
                    d[r["endpoint"]][k].append(digits(r["response"]))
    return d


def enc(s):
    b = np.full(L, -1, np.int16); s = s[:L]
    b[:len(s)] = [int(c) for c in s]
    return b


def mmd(A, B):                         # A, B: {prompt: (n, L)}; Hamming similarity on valid positions
    v, k = 0.0, 0
    for p in A:
        if p not in B or len(A[p]) < 2 or len(B[p]) < 2:
            continue
        a, b = A[p], B[p]
        sim = lambda x, y: ((x[:, None] == y[None]) & (x[:, None] >= 0)).mean(2)
        kaa, kbb, kab = sim(a, a), sim(b, b), sim(a, b)
        na, nb = len(a), len(b)
        v += (kaa.sum() - np.trace(kaa)) / (na * (na - 1)) + (kbb.sum() - np.trace(kbb)) / (nb * (nb - 1)) - 2 * kab.mean(); k += 1
    return v / k if k else math.nan


def pair_z(A, B, rng, perms=400):
    t = mmd(A, B); nulls = []
    for _ in range(perms):
        A2, B2 = {}, {}
        for p in A:
            if p in B:
                pool = np.concatenate([A[p], B[p]]); i = rng.permutation(len(pool))
                A2[p], B2[p] = pool[i[:len(A[p])]], pool[i[len(A[p]):]]
        nulls.append(mmd(A2, B2))
    nulls = np.array(nulls)
    return (t - nulls.mean()) / (nulls.std() + 1e-12), t


def analyze(a):
    d = load(a.target)
    fam = {l.split(",")[0]: l.split(",")[1] for l in open(ROOT / a.families).read().split()[1:]} if a.families else {e: "A" for e in d}
    X = {e: {p: np.stack([enc(s) for s in v if s]) for p, v in ps.items() if sum(1 for s in v if s) >= 2} for e, ps in d.items()}
    print("answers with a readable number list per endpoint: " + " ".join(
        f"{e[-2:]}={sum(len(v) for v in X[e].values())}" for e in sorted(X)))
    rng = np.random.default_rng(0)
    for f in sorted(set(fam.values())):
        es = sorted(e for e in X if fam.get(e) == f)
        Z = np.full((len(es), len(es)), np.nan)
        for i, j in itertools.combinations(range(len(es)), 2):
            Z[i, j] = Z[j, i] = pair_z(X[es[i]], X[es[j]], rng)[0]
        print(f"\nfamily {f}: pair z (low = alike; < 1 means no detectable difference)")
        print("      " + "".join(f"{e[-2:]:>6}" for e in es))
        for i, e in enumerate(es):
            print(f"  {e[-2:]}  " + "".join("     ." if i == j else f"{Z[i, j]:6.1f}" for j in range(len(es))))
        near = sorted((Z[i, j], es[i], es[j]) for i, j in itertools.combinations(range(len(es)), 2))
        print("  most alike pairs: " + ", ".join(f"{x[-2:]}-{y[-2:]} {z:.1f}" for z, x, y in near[:10]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect"); c.add_argument("--target", choices=["ref", "ep"], required=True)
    c.add_argument("--n", type=int, required=True); c.add_argument("--go", action="store_true")
    c.add_argument("--approve-budget", type=int, default=0)
    z = sub.add_parser("analyze"); z.add_argument("--target", choices=["ref", "ep"], required=True)
    z.add_argument("--families", default="")
    a = ap.parse_args(); collect(a) if a.cmd == "collect" else analyze(a)
