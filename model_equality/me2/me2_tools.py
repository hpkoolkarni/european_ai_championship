"""AgentProv-style tool-choice fingerprint, adapted to a text-only chat API (no native `tools` field).

Idea (Wang, Zhao, ..., Dziedzic, AgentProv, EMNLP 2026): when several tools are equally good, WHICH one a model picks
is a preference stored in its weights. A system prompt, persona, watermark or temperature barely moves it; fp8
or a LoRA adapter does. We compare the per-item choice frequencies of two endpoints with a permutation test.

Changes for this API:
  * tools are listed in the prompt, and one made-up earlier exchange (user asks, assistant answers in the exact
    format) shows the answer format, so small models reply with names instead of copying the question;
  * ITEMS items per request (more choices per query); the option order is shuffled with a fixed seed per repeat,
    identical for every endpoint, so any difference comes from the model, never from us.

  python me2_tools.py collect --target ref --n 30 --go                      practice (free)
  python me2_tools.py collect --target ep  --n 40 --go --approve-budget 40  scored (refuses above the budget)
  python me2_tools.py analyze --target ref [--n 9]                          pair tests, clusters, practice MCC
  python me2_tools.py analyze --target ep                                   writes submission_tools.csv
Without --go nothing is sent. API key: environment variable API_KEY. Answers: data/<target>_tools.jsonl
"""
import argparse, collections, csv, itertools, json, math, os, random, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "http://34.147.229.55/33-model-equality/v1"
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

# (request, equally good function names). Wording never names a tool.
ITEMS = [
    ("Schedule a team sync on Thursday at 2pm.", ["create_event", "schedule_meeting", "add_calendar_entry", "book_slot"]),
    ("Send a short greeting to alice@example.com.", ["send_email", "compose_mail", "email_send", "dispatch_message"]),
    ("Get tomorrow's weather for Paris.", ["get_weather", "weather_forecast", "fetch_forecast", "lookup_weather"]),
    ("Save the note 'buy milk'.", ["create_note", "save_note", "add_note", "write_memo"]),
    ("Convert 100 US dollars to euros.", ["convert_currency", "currency_exchange", "fx_convert", "money_converter"]),
    ("Set a reminder for 7am.", ["set_reminder", "create_alarm", "schedule_alert", "add_reminder"]),
    ("Look up the meaning of 'serendipity'.", ["define_word", "dictionary_lookup", "get_definition", "word_meaning"]),
    ("Translate 'good morning' into Spanish.", ["translate_text", "translation_service", "convert_language", "translate"]),
    ("Find a recipe for pancakes.", ["search_recipes", "find_recipe", "recipe_lookup", "get_recipe"]),
    ("Book a taxi to the airport.", ["book_ride", "request_taxi", "order_cab", "call_car"]),
    ("Show my account balance.", ["get_balance", "check_balance", "account_summary", "fetch_balance"]),
    ("Play some relaxing music.", ["play_music", "start_playlist", "stream_audio", "play_song"]),
]
SHOT_USER = ("For each numbered request, several functions would work equally well. Pick exactly ONE function per request. "
             "Answer with one line per request: the number, a dot and the function name. Nothing else.\n"
             "1. Request: Turn off the kitchen lights. Functions: switch_light, lights_off, toggle_lamp\n"
             "2. Request: Show today's top news. Functions: get_headlines, news_feed, fetch_news")
SHOT_ASSISTANT = "1. lights_off\n2. fetch_news"   # one example answer, different domains from ITEMS
INSTR = SHOT_USER.split("\n")[0]


def prompt_for(repeat):
    rng = random.Random(1000 + repeat)                 # same order for every endpoint at a given repeat
    lines, order = [], []
    for i, (req, opts) in enumerate(ITEMS, 1):
        o = opts[:]; rng.shuffle(o); order.append(o)
        lines.append(f"{i}. Request: {req} Functions: {', '.join(o)}")
    return [{"role": "user", "content": SHOT_USER}, {"role": "assistant", "content": SHOT_ASSISTANT},
            {"role": "user", "content": INSTR + "\n" + "\n".join(lines)}]


def parse(text):
    """-> list of chosen option per item (or 'INVALID')."""
    got = {}
    for m in re.finditer(r"^\s*(\d+)\s*[.):-]\s*`?\**([A-Za-z_]+)", text, re.M):
        got.setdefault(int(m.group(1)), m.group(2).lower())
    out = []
    for i, (_, opts) in enumerate(ITEMS, 1):
        c = got.get(i)
        out.append(c if c in opts else "INVALID")
    return out


# ------------------------------------------------------------------ collect
def collect(a):
    key = os.environ.get("API_KEY", "").strip()
    import requests
    s = requests.Session(); s.headers["X-API-Key"] = key
    out = DATA / f"{a.target}_tools.jsonl"; DATA.mkdir(exist_ok=True)
    done = set()
    if out.exists():
        done = {(r["endpoint"], r["repeat"]) for r in map(json.loads, open(out)) }
    if not key:
        sys.exit("Set the API key first:  read -s API_KEY; export API_KEY")
    models = s.get(f"{BASE}/models", timeout=60).json()["data"]
    left = {m["id"]: m["budget"]["limit"] - m["budget"]["used"] for m in models}
    eps = sorted(e for e in left if e.startswith(a.target + "-"))
    jobs = [(e, r) for r in range(a.n) for e in eps if (e, r) not in done]
    per_ep = collections.Counter(e for e, _ in jobs)
    worst = max(per_ep.values(), default=0)
    print(f"{len(eps)} endpoints, {a.n} repeats: {len(jobs)} new answers, at most {worst} per endpoint; "
          f"lowest budget left {min(left[e] for e in eps)}")
    if a.target == "ep" and worst > a.approve_budget:
        sys.exit(f"Refusing: {worst} new answers per endpoint > --approve-budget {a.approve_budget}")
    if not a.go:
        print("Plan only (add --go to send). Example prompt:\n" + json.dumps(prompt_for(0), indent=1)[:1500]); return
    lock = threading.Lock(); f = open(out, "a"); n = [0]

    def one(job):
        e, r = job
        for attempt in range(20):
            try:
                resp = s.post(f"{BASE}/chat/completions", timeout=180,
                              json={"model": e, "max_tokens": 120, "messages": prompt_for(r)})
            except Exception as exc:
                time.sleep(15); continue
            if resp.status_code == 200:
                txt = resp.json()["choices"][0]["message"]["content"]
                with lock:
                    f.write(json.dumps({"endpoint": e, "repeat": r, "response": txt}) + "\n"); f.flush()
                    n[0] += 1
                    if n[0] % 20 == 0: print(f"{n[0]}/{len(jobs)}", flush=True)
                return
            if resp.status_code == 429 and "Retry-After" in resp.headers:
                time.sleep(int(resp.headers["Retry-After"])); continue
            if resp.status_code in (502, 503, 504):
                time.sleep(10); continue
            print(e, r, resp.status_code, resp.text[:200], file=sys.stderr); return
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(one, jobs))
    f.close(); print("finished", out)


# ------------------------------------------------------------------ analyze
def load(target, nmax):
    rows = collections.defaultdict(list)
    for r in map(json.loads, open(DATA / f"{target}_tools.jsonl")):
        if r["repeat"] < nmax:
            rows[r["endpoint"]].append(parse(r["response"]))
    return rows


def stat(A, B):
    """AgentProv statistic: sum over items of squared L2 distance between choice frequencies."""
    t = 0.0
    for k in range(len(ITEMS)):
        ca = collections.Counter(x[k] for x in A); cb = collections.Counter(x[k] for x in B)
        for o in set(ca) | set(cb):
            t += (ca[o] / len(A) - cb[o] / len(B)) ** 2
    return t


def pair_z(A, B, perms, rng):
    t = stat(A, B); pool = A + B; nulls = []
    for _ in range(perms):
        rng.shuffle(pool); nulls.append(stat(pool[:len(A)], pool[len(A):]))
    mu = sum(nulls) / perms; sd = math.sqrt(sum((x - mu) ** 2 for x in nulls) / perms) or 1e-9
    p = (1 + sum(x >= t for x in nulls)) / (perms + 1)
    return (t - mu) / sd, p


def clusters(eps, Z, thr):
    """Average-linkage agglomerative clustering on z; merge while the average z between groups < thr."""
    groups = [[e] for e in eps]
    while True:
        best = None
        for i, j in itertools.combinations(range(len(groups)), 2):
            d = sum(Z[a, b] for a in groups[i] for b in groups[j]) / (len(groups[i]) * len(groups[j]))
            if d < thr and (best is None or d < best[0]): best = (d, i, j)
        if not best: return groups
        _, i, j = best; groups[i] += groups.pop(j)


def mcc(pred, truth, eps):
    tp = fp = fn = tn = 0
    for a, b in itertools.combinations(eps, 2):
        g, t = pred[a] == pred[b], truth[a] == truth[b]
        tp += g and t; fp += g and not t; fn += t and not g; tn += not g and not t
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return max(0.0, (tp * tn - fp * fn) / d) if d else 0.0


def analyze(a):
    rows = load(a.target, a.n); eps = sorted(rows); rng = random.Random(0)
    print("usable choices per endpoint (share of items that parse to a listed function):")
    for e in eps:
        flat = [c for x in rows[e] for c in x]
        print(f"  {e}: {len(rows[e])} answers, {1 - flat.count('INVALID') / max(len(flat), 1):.0%} usable")
    Z = {}
    for x, y in itertools.combinations(eps, 2):
        z, p = pair_z(rows[x], rows[y], a.perms, rng); Z[x, y] = Z[y, x] = z
    truth = {}
    if a.target == "ref":
        for r in csv.DictReader(open(HERE.parent / "practice_answers.csv")):
            truth[r["endpoint"]] = r["model"]
    print("\npair z (0 = indistinguishable)")
    print("        " + " ".join(f"{e[-5:]:>6}" for e in eps))
    for x in eps:
        print(f"{x[-6:]:>6}{truth.get(x, ''):>3}" + " ".join(f"{(Z[x, y] if x != y else 0):6.1f}" for y in eps))
    if truth:
        same = [Z[x, y] for x, y in itertools.combinations(eps, 2) if truth[x] == truth[y]]
        diff = [Z[x, y] for x, y in itertools.combinations(eps, 2) if truth[x] != truth[y]]
        auc = sum((s < d) + 0.5 * (s == d) for s in same for d in diff) / (len(same) * len(diff))
        print(f"\nAUC same-vs-different = {auc:.3f} (1 = perfect, 0.5 = chance)")
        for thr in (1, 2, 3, 4, 6, 8):
            g = clusters(eps, Z, thr); lab = {e: i for i, grp in enumerate(g) for e in grp}
            print(f"  threshold z<{thr}: MCC {mcc(lab, truth, eps):.3f}  groups {[sorted(x) for x in g]}")
    else:
        g = clusters(eps, Z, a.thr)
        with open(HERE / "submission_tools.csv", "w") as f:
            f.write("endpoint,family,model\n")
            for i, grp in enumerate(g):
                for e in grp: f.write(f"{e},F,M{i + 1}\n")
        print("groups:", [sorted(x) for x in g], "\nwrote submission_tools.csv (family column is a placeholder)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["collect", "analyze"])
    ap.add_argument("--target", choices=["ref", "ep"], required=True)
    ap.add_argument("--n", type=int, default=30, help="repeats per endpoint (collect: how many; analyze: use first n)")
    ap.add_argument("--go", action="store_true"); ap.add_argument("--approve-budget", type=int, default=0)
    ap.add_argument("--perms", type=int, default=500); ap.add_argument("--thr", type=float, default=3.0)
    a = ap.parse_args()
    collect(a) if a.cmd == "collect" else analyze(a)
