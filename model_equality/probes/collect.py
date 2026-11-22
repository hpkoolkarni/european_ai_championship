"""Model Equality: send the probe set to the endpoints and save the answers. SAFE BY DEFAULT.

Without --go it only prints the plan (answers per endpoint, an example prompt) and makes NO API call.
  practice endpoints (unlimited, never scored):
      python probes/collect.py --target ref --repeats 5 --go
  scored endpoints (100 answers each for the whole competition), round 1:
      python probes/collect.py --target ep --repeats 2 --go --approve-budget 40
  round 2 (only the endpoints/probes listed by analyze.py in results/round2_plan.json, extra answers on top):
      python probes/collect.py --target ep --plan results/round2_plan.json --extra-repeats 2 --go --approve-budget 30
The run refuses whenever the planned answers per scored endpoint exceed --approve-budget, so the cost is always stated
up front. Answers go to probes/data/<target>_answers.jsonl, one JSON line per answer, including any extra fields the API
returns (e.g. token counts, stop reason). A rerun skips answers already saved. API key: environment variable API_KEY.
The request loop follows the organisers' task_template.py (4 requests at a time, waits on 429/5xx, stops per
endpoint when its budget is used up)."""
import argparse
import json
import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probes as probe_defs                                                   # noqa: E402

DATA = HERE / "data"
BASE_URL = "http://34.147.229.55/33-model-equality/v1"
PARALLEL, MAX_TOKENS = 4, 256


def saved(out):
    if not out.exists():
        return []
    return [json.loads(l) for l in open(out, encoding="utf-8") if l.strip()]


def plan_jobs(a, probes, endpoints, out):
    """[(endpoint, probe id, repeat index, prompt)] still to send."""
    rows = saved(out)
    done = {(r["endpoint"], r["id"], int(r["repeat"])) for r in rows}
    last = defaultdict(lambda: -1)
    for r in rows:
        last[(r["endpoint"], r["id"])] = max(last[(r["endpoint"], r["id"])], int(r["repeat"]))
    jobs = []
    for p in probes:
        for ep in endpoints:
            if a.extra_repeats:                                    # round 2: new repeats after the saved ones
                reps = range(last[(ep, p["id"])] + 1, last[(ep, p["id"])] + 1 + a.extra_repeats)
            else:
                reps = range(1 if p["kind"] == "F" or p.get("once") else a.repeats)   # 'once': asked one time each
            jobs += [(ep, p["id"], r, p["prompt"]) for r in reps if (ep, p["id"], r) not in done]
    # repeat-major order: every probe's first answer on every endpoint, then every second answer, ... so a run that
    # is stopped early (or analysed with --max-repeats 2 while it continues) already covers all probes evenly
    order = {p["id"]: k for k, p in enumerate(probes)}
    jobs.sort(key=lambda j: (j[2], order[j[1]]))
    return jobs


def send(jobs, out, key):
    import requests
    session = requests.Session(); session.headers["X-API-Key"] = key
    reply = session.get(f"{BASE_URL}/models", timeout=60); reply.raise_for_status()
    left = {m["id"]: m["budget"]["limit"] - m["budget"]["used"] for m in reply.json()["data"]}
    unknown = sorted({j[0] for j in jobs} - set(left))
    if unknown:
        sys.exit(f"unknown endpoints {unknown}")
    count = defaultdict(int); keep = []
    for j in jobs:                                                 # never plan more than the budget left
        if count[j[0]] < left[j[0]]:
            count[j[0]] += 1; keep.append(j)
    print(f"sending {len(keep)} requests ({len(jobs) - len(keep)} dropped: no budget left)")
    lock, exhausted, n = threading.Lock(), set(), [0]
    fh = open(out, "a", encoding="utf-8")

    def one(job):
        ep, pid, rep, prompt = job
        while ep not in exhausted:
            try:
                r = session.post(f"{BASE_URL}/chat/completions", timeout=180,
                                 json={"model": ep, "max_tokens": MAX_TOKENS, "messages": [{"role": "user", "content": prompt}]})
            except requests.RequestException as exc:
                print(f"server unreachable ({type(exc).__name__}); retrying in 30 s"); time.sleep(30); continue
            if r.status_code == 200:
                body = r.json(); ch = body["choices"][0]
                row = {"endpoint": ep, "id": pid, "repeat": rep, "prompt": prompt, "response": ch["message"]["content"],
                       "finish_reason": ch.get("finish_reason"),
                       "extra": {k: v for k, v in body.items() if k not in ("choices",)}}
                with lock:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n"); fh.flush(); n[0] += 1
                    if n[0] % 20 == 0 or n[0] == len(keep):
                        print(f"{n[0]}/{len(keep)} answers")
                return
            detail = r.text[:300]
            if r.status_code == 429 and "Retry-After" in r.headers:
                time.sleep(int(r.headers["Retry-After"]))
            elif r.status_code == 429:
                with lock:
                    exhausted.add(ep); print(f"{ep}: budget used up ({detail})")
            elif r.status_code in (502, 503, 504):
                print(f"{detail} retrying in 15 s"); time.sleep(15)
            else:
                print(f"{ep} {pid}: HTTP {r.status_code}: {detail}", file=sys.stderr); return

    try:
        with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
            list(pool.map(one, keep))
    finally:
        fh.close()
    print(f"finished: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["ref", "ep"], required=True)
    ap.add_argument("--repeats", type=int, default=2, help="answers per non-F probe (F once; G needs >= 2)")
    ap.add_argument("--kinds", default="FTCRBPG", help="probe kinds to send")
    ap.add_argument("--only", default="", help="comma-separated probe ids (overrides --kinds)")
    ap.add_argument("--plan", default="", help="round-2 plan from analyze.py (endpoints + probes)")
    ap.add_argument("--extra-repeats", type=int, default=0, help="round 2: new answers per probe on top of saved ones")
    ap.add_argument("--go", action="store_true", help="actually call the API (otherwise: plan only)")
    ap.add_argument("--approve-budget", type=int, default=0, help="ep only: max NEW answers per endpoint you approve")
    a = ap.parse_args()

    endpoints = [f"ref-{i:02d}" for i in range(1, 10)] if a.target == "ref" else [f"ep-{i:02d}" for i in range(1, 34)]
    only = {s.strip() for s in a.only.split(",") if s.strip()}
    if a.plan:
        plan = json.load(open(a.plan))
        endpoints, only = [e for e in plan["endpoints"] if e in endpoints], set(plan["probes"])
        if not a.extra_repeats:
            sys.exit("a round-2 plan needs --extra-repeats N")
    probes = [p for p in probe_defs.build() if (p["id"] in only if only else p["kind"] in a.kinds)]
    if only - {p["id"] for p in probes}:
        sys.exit(f"unknown probe ids: {sorted(only - {p['id'] for p in probes})}")
    out = DATA / f"{a.target}_answers.jsonl"
    jobs = plan_jobs(a, probes, endpoints, out)
    per = defaultdict(int)
    for j in jobs:
        per[j[0]] += 1
    worst = max(per.values(), default=0)
    longest = max(probes, key=lambda p: len(p["prompt"]))
    print(f"target {a.target}: {len(endpoints)} endpoints, {len(probes)} probes; {len(jobs)} new answers, "
          f"at most {worst} per endpoint (already saved answers are skipped)")
    print(f"longest prompt {longest['id']}: ~{len(longest['prompt']) // 4} tokens (limit 1,024); output {out}")
    if a.target == "ep":
        print("SCORED endpoints: every new answer is taken from the 100-per-endpoint competition budget.")
    if not a.go:
        print("\nPlan only (add --go to query). Example prompt:\n" + probes[-1]["prompt"])
        return
    if a.target == "ep" and worst > a.approve_budget:
        sys.exit(f"Refused: up to {worst} new answers per scored endpoint, but --approve-budget is {a.approve_budget}.")
    key = os.environ.get("API_KEY", "").strip()
    if not key:
        sys.exit("Set the API key first:  read -s API_KEY; export API_KEY")
    DATA.mkdir(exist_ok=True)
    send(jobs, out, key)


if __name__ == "__main__":
    main()
