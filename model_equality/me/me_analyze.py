"""Model Equality: turn saved answers into same/different decisions, groups and a submission file. No API calls.

  python me/me_analyze.py --target ref --max-repeats 2 [--stat mode]
      practice endpoints: disguise profile, per-probe power (saved as out/probe_weights.json), pair z-scores,
      greedy check, practice score for every threshold, writes out/ref_guess.csv.
  python me/me_analyze.py --target ep --t-model 2.0 [--stat mode] [--weights me/out/probe_weights.json]
      scored endpoints: families + models for ep-01..ep-33, format check, writes out/submission_ep.csv and
      out/round2_plan.json (the endpoints and probes worth a second round of answers).

How a pair is judged (permutation test, as in MET and AgentProv): for every question item, compare the two endpoints'
answers; shuffle which answers belong to which endpoint many times to learn the difference expected by pure chance;
report a z-score (~0 = same model, large = different). Two statistics:
  l2    squared difference of answer frequencies (sensitive, but a temperature change alone also raises it)
  mode  does the most frequent answer differ? (temperature makes a model more or less random but rarely changes its
        favourite answer, so this one tolerates the decoding disguise)
  confident  do both endpoints give one answer firmly (>= 75 % of their repeats) and are those answers different?
        Only firmly held answers count, so random answers (temperature) are ignored and only differences a disguise
        rarely causes are counted. Meant for the K probes:  --stat confident --channels K
Channels: T tool choice, C near-tie, K confident answers, B bit strings, P precision, R random generation; G greedy hints; F free text
(families only); API token counts (if the API returns them) give a tokenizer signal for families."""
import argparse
import csv
import itertools
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import me_probes                                                   # noqa: E402

DATA, OUT = HERE / "data", HERE / "out"
LINE = re.compile(r"^\s*[*#>\-]*\s*(\d{1,2})\s*[.):\-]\s*(.*)$")
CHANNELS = ("T", "C", "B", "P", "K", "R")
CONFIDENT_SHARE = 0.75          # --stat confident: an endpoint 'holds' an answer if it gives it at least this often


# ---------- 1. parse ----------
def clean(s):
    return re.sub(r"[*`\"'“”‘’]", "", s).strip().lower()


def numbered(text):
    """{item number: answer text} from a numbered list, and how many other non-empty lines there were (chatter)."""
    out, chatter = {}, 0
    for line in text.splitlines():
        m = LINE.match(line)
        if m and int(m.group(1)) not in out:
            out[int(m.group(1))] = clean(m.group(2))
        elif line.strip():
            chatter += 1
    return out, chatter


def parse_item(ans, options, kind):
    if ans is None:
        return "_missing"
    if kind == "T":                                                # exactly one allowed function name on the line
        if "request" in ans:                                       # the request was copied back, not answered
            return "_echo"
        names = {tok for tok in re.findall(r"[a-z_]+", ans) if tok in options}
        return names.pop() if len(names) == 1 else "_other"
    if kind == "P":                                                # the number given (right or wrong)
        num = re.findall(r"-?\d[\d,]*", ans)
        return num[0].replace(",", "") if num else "_empty"
    if kind == "K":                                                # options = the question text for K items
        q = clean(options or "")                                   # same cleaning as the answer ("Earth's" -> earths)
        if "?" in ans:                                             # question copied back: keep what follows it
            ans = ans.rsplit("?", 1)[1]
        toks = [t.replace(",", "") for t in re.findall(r"\d[\d,]*|[a-z]+", ans) if t not in FILLER]
        if not toks:
            return "_empty"
        qwords = set(re.findall(r"\d+|[a-z]+", q))
        rest = [t for t in toks if t not in qwords] or toks[-1:]  # 'The capital of Canada is Ottawa' -> ottawa;
                                                                   # all words in the question -> the last one
        if NUMERIC_Q.search(q):                                    # the question asks for a number: prefer one
            nums = [t for t in rest if t[0].isdigit()]
            if nums:
                return nums[0]
        return rest[0]
    tok = re.findall(r"[a-z0-9]+", ans)                            # C: first word or number
    return tok[0] if tok else "_empty"


FILLER = {"the", "a", "an", "is", "it", "its", "it's", "answer", "arr", "tis", "i", "think", "would", "be", "of",
          "approximately", "about", "around", "mr", "sir", "s", "was", "were", "are", "in", "by", "has", "have",
          "there", "and"}
NUMERIC_Q = re.compile(r"\d|how many|year|number|percent|factorial|squared|power|root|times|plus|minus|divided")


def bit_string(text, items):
    """A/B choices: the asked-for string of letters, or else a numbered list ('1. A', '1. A=river B=lake (A)',
    '1. river'). Lines that only copy the pair back, or name both or neither, are unusable."""
    n = len(items)
    runs = re.findall(r"[AB]{10,}", re.sub(r"[\s,;|.\-]", "", text.upper()))
    if runs:
        s = max(runs, key=len)
        return [s[j].lower() if j < len(s) else "_missing" for j in range(n)]
    got, _ = numbered(text)
    out = []
    for j, it in enumerate(items, 1):
        ans = got.get(j)
        if ans is None:
            out.append("_missing"); continue
        rest = re.sub(r"\b[ab]\s*=\s*[a-z]+", " ", ans)            # drop the copied 'A=river B=lake'
        picks = set(re.findall(r"\b[ab]\b", rest))
        words = [k for k, w in enumerate(it.get("words", [])) if re.search(rf"\b{w}\b", rest)]
        if len(picks) == 1:
            out.append(picks.pop())
        elif not picks and len(words) == 1:
            out.append("ab"[words[0]])
        else:
            out.append("_other")
    return out


def keep(obs, ep, kind, key, value):
    """Store a parsed answer; placeholders (_missing, _other, _echo, _empty) only count towards the usable-share
    table and are left out of the tests, so a format failure never looks like an answer."""
    obs["usable"][ep][kind][1] += 1
    if not value.startswith("_"):
        obs["usable"][ep][kind][0] += 1
        obs["choice"][ep][key].append(value)


def usable_report(eps, obs):
    kinds = [k for k in "TCBPK" if any(obs["usable"][e][k][1] for e in eps)]
    print("usable answers per probe kind (format failures and copied prompts are left out of the tests):")
    print("  " + " " * 7 + "".join(f"{k:>13}" for k in kinds))
    for e in eps:
        cells = []
        for k in kinds:
            ok, tot = obs["usable"][e][k]
            cells.append(f"{100 * ok / tot:.0f}% of {tot}".rjust(13) if tot else "-".rjust(13))
        print(f"  {e:<7}" + "".join(cells))


def symbols(text, sub):
    if sub == "digits":
        return re.findall(r"\d", text)
    if sub == "letters":
        return re.findall(r"\b[a-z]\b", text.lower())
    return re.findall(r"\d+", text)


def load(target, max_repeats=None, probes_re=None):
    probes = {p["id"]: p for p in me_probes.build()}
    obs = {"choice": defaultdict(lambda: defaultdict(list)), "rand": defaultdict(lambda: defaultdict(list)),
           "greedy": defaultdict(lambda: defaultdict(list)), "text": defaultdict(list), "digits": defaultdict(list),
           "chatter": defaultdict(list), "chars": Counter(), "tokens": Counter(),
           "usable": defaultdict(lambda: defaultdict(lambda: [0, 0]))}
    path = DATA / f"{target}_answers.jsonl"
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if max_repeats:
        rows = [r for r in rows if int(r["repeat"]) < max_repeats]
    if probes_re:                                                  # e.g. 'K\d+o\d+' = shuffled K only
        rows = [r for r in rows if r["id"][0] == "F" or re.fullmatch(probes_re, r["id"])]
    for r in sorted(rows, key=lambda r: int(r["repeat"])):
        p, ep, resp = probes.get(r["id"]), r["endpoint"], r["response"] or ""
        if p is None:
            continue
        usage = (r.get("extra") or {}).get("usage") or {}
        if usage.get("completion_tokens"):
            obs["chars"][ep] += len(resp); obs["tokens"][ep] += usage["completion_tokens"]
        k = p["kind"]
        if k == "F":
            obs["text"][ep].append(resp)
        elif k == "G":
            obs["greedy"][ep][p["id"]].append(resp)
        elif k == "B":
            for it, c in zip(p["items"], bit_string(resp, p["items"])):
                keep(obs, ep, k, it["key"], c)
        elif k in "TCPK":
            got, chat = numbered(resp)
            obs["chatter"][ep].append(chat > 0)
            for j, it in enumerate(p["items"], 1):
                keep(obs, ep, k, it["key"], parse_item(got.get(j), it.get("q") if k == "K" else it["options"], k))
        else:
            seq = symbols(resp, p["sub"])
            obs["rand"][ep][p["id"]].append(Counter(seq) + Counter(f"{a}>{b}" for a, b in zip(seq, seq[1:])))
            if p["sub"] == "digits":
                obs["digits"][ep].append(seq)
    eps = sorted(set(obs["choice"]) | set(obs["rand"]) | set(obs["text"]) | set(obs["greedy"]))
    print(f"loaded {len(rows)} answers from {path} for {len(eps)} endpoints")
    return eps, obs


# ---------- 2. permutation test per pair ----------
def unit_matrix(xa, xb, ch):
    rows = list(xa) + list(xb)
    if ch == "R":                                                  # normalised symbol/pair frequencies per answer
        keys = sorted(set().union(*rows)); idx = {k: i for i, k in enumerate(keys)}
        O = np.zeros((len(rows), len(keys)))
        for r, c in enumerate(rows):
            tot = sum(c.values()) or 1
            for k, v in c.items():
                O[r, idx[k]] = v / tot
        return O
    keys = sorted(set(rows)); idx = {k: i for i, k in enumerate(keys)}
    O = np.zeros((len(rows), len(keys)))
    for r, c in enumerate(rows):
        O[r, idx[c]] = 1
    return O


def item_null(O, na, B, rng, stat):
    n = len(O); nb = n - na
    M0 = np.zeros((1, n), bool); M0[0, :na] = True
    perm = rng.permuted(np.tile(np.arange(n), (B, 1)), axis=1)
    M = np.zeros((B, n), bool); np.put_along_axis(M, perm[:, :na], True, axis=1)
    M = np.vstack([M0, M])
    ca, cb = M.astype(float) @ O, (~M).astype(float) @ O
    if stat == "mode":                                             # favourite answer differs? (random tie-break)
        s = (np.argmax(ca + rng.random(ca.shape) * 1e-6, 1) != np.argmax(cb + rng.random(cb.shape) * 1e-6, 1)).astype(float)
    elif stat == "confident":                                      # both sides hold an answer firmly, and they differ
        ha = ca.max(1) >= CONFIDENT_SHARE * na - 1e-9
        hb = cb.max(1) >= CONFIDENT_SHARE * nb - 1e-9
        s = (ha & hb & (ca.argmax(1) != cb.argmax(1))).astype(float)
    else:
        s = ((ca / na - cb / nb) ** 2).sum(1)
    return float(s[0]), s[1:]


def pair_test(a, b, units, B, rng, stat, weights):
    per_ch = {c: [0.0, np.zeros(B)] for c in CHANNELS}
    per_probe = defaultdict(lambda: [0.0, np.zeros(B)])
    n_cmp = 0
    for key in sorted(set(units[a]) & set(units[b])):              # fixed order: same random draws every run
        ch, probe, _ = key
        xa, xb = units[a][key], units[b][key]
        if not xa or not xb:
            continue
        if stat == "confident" and (ch == "R" or len(xa) < 2 or len(xb) < 2):
            continue                                               # 'holds firmly' needs repeated answers
        n_cmp += stat != "confident" or (firm(xa) and firm(xb))   # items the comparison actually rests on
        w = weights.get(probe, 1.0)
        if w <= 0:
            continue
        o, null = item_null(unit_matrix(xa, xb, ch), len(xa), B, rng, "l2" if ch == "R" else stat)
        per_ch[ch][0] += w * o; per_ch[ch][1] += w * null
        per_probe[probe][0] += o; per_probe[probe][1] += null

    def z(o, null):
        sd = null.std()
        return ((o - null.mean()) / sd, (null - null.mean()) / sd) if sd > 0 else (0.0, np.zeros(B))
    zs = {c: z(*v) for c, v in per_ch.items() if v[1].any()}
    if not zs:
        return 0.0, 1.0, {}, {}, n_cmp
    Z = sum(v[0] for v in zs.values()); Zn = sum(v[1] for v in zs.values())
    p = (1 + int((Zn >= Z).sum())) / (B + 1)
    return (Z / math.sqrt(len(zs)), p, {c: v[0] for c, v in zs.items()}, {k: z(*v)[0] for k, v in per_probe.items()},
            n_cmp)


def firm(x):
    return Counter(x).most_common(1)[0][1] >= CONFIDENT_SHARE * len(x) - 1e-9


def build_units(eps, obs, channels=CHANNELS):
    units = {}
    for e in eps:
        u = {}
        for key, x in obs["choice"][e].items():
            pid = key.split("#")[0]
            if pid[0] in channels:
                u[(pid[0], pid, key)] = x
        for pid, x in obs["rand"][e].items():
            if "R" in channels:
                u[("R", pid, pid)] = x
        units[e] = u
    return units


def all_pairs(eps, units, B, stat, weights, seed=0):
    rng = np.random.default_rng(seed)
    n = len(eps)
    Z, P, N = np.zeros((n, n)), np.ones((n, n)), np.zeros((n, n), int)
    prz = {}
    for i, j in itertools.combinations(range(n), 2):
        z, p, _, pz, k = pair_test(eps[i], eps[j], units, B, rng, stat, weights)
        Z[i, j] = Z[j, i] = z; P[i, j] = P[j, i] = p; prz[(i, j)] = pz; N[i, j] = N[j, i] = k
    return Z, P, prz, N


# ---------- 3. greedy channel + disguise profile ----------
def greedy_links(eps, greedy, min_prefix=None):
    """Deterministic endpoint = every repeat of every G probe identical. Two deterministic endpoints with identical
    texts -> must-link. A split after >= min_prefix shared words -> cannot-link (off unless set; calibrate on ref)."""
    det = {e: bool(greedy[e]) and all(len(v) >= 2 and len(set(v)) == 1 for v in greedy[e].values()) for e in eps}
    must, cannot, table = set(), set(), []
    for i, j in itertools.combinations(range(len(eps)), 2):
        a, b = eps[i], eps[j]
        if not (det[a] and det[b]):
            continue
        pref = []
        for pid in set(greedy[a]) & set(greedy[b]):
            wa, wb = greedy[a][pid][0].split(), greedy[b][pid][0].split()
            pref.append((next((t for t, (x, y) in enumerate(zip(wa, wb)) if x != y), min(len(wa), len(wb))), wa == wb))
        if not pref:
            continue
        k = min(k for k, _ in pref)
        if all(s for _, s in pref):
            must.add((i, j)); table.append((a, b, "identical -> same model", None))
        elif min_prefix is not None and k >= min_prefix:
            cannot.add((i, j)); table.append((a, b, f"split after {k} words -> different model", k))
        else:
            table.append((a, b, f"split after {k} words -> no conclusion", k))
    return det, must, cannot, table


def bigram_skew(seqs):
    """How far an endpoint's digit-pair frequencies are from what its single-digit frequencies predict (a
    watermark nudges each next token depending on the previous one, which shows up exactly here)."""
    uni, bi = Counter(), Counter()
    for s in seqs:
        uni.update(s); bi.update(zip(s, s[1:]))
    nb = sum(bi.values()); nu = sum(uni.values())
    if nb < 30:
        return math.nan
    chi = 0.0
    for a in "0123456789":
        for b in "0123456789":
            e = nb * (uni[a] / nu) * (uni[b] / nu)
            if e > 0:
                chi += (bi[(a, b)] - e) ** 2 / e
    return chi / nb


def disguise_profile(eps, obs, det, fam):
    """Observable disguise traits, used to judge comparisons fairly (never to guess the answer key)."""
    var = {}
    for e in eps:
        xs = [x for x in obs["choice"][e].values() if len(x) >= 2]
        var[e] = np.mean([len(set(x)) > 1 for x in xs]) if xs else math.nan
    skew = {e: bigram_skew(obs["digits"][e]) for e in eps}
    print("\ndisguise profile (variability = share of items with differing repeats; chatter = answers with extra "
          "lines; digit-pair skew relative to the family median, >2 hints at a watermark):")
    for f in sorted(set(fam)):
        members = [e for e, ff in zip(eps, fam) if ff == f]
        med = np.nanmedian([skew[e] for e in members]) if members else math.nan
        mad = np.nanmedian([abs(skew[e] - med) for e in members]) or 1e-9
        for e in members:
            ch = np.mean(obs["chatter"][e]) if obs["chatter"][e] else math.nan
            rz = (skew[e] - med) / (1.4826 * mad) if not math.isnan(skew[e]) else math.nan
            print(f"  fam {f} {e}: variability {var[e]:.2f}  greedy {'yes' if det[e] else 'no '}  chatter {ch:.2f}"
                  f"  digit-pair skew {rz:+.1f}")


# ---------- 4. group ----------
def merge(D, t=math.inf):
    """Average linkage over the KNOWN distances (nan = no evidence is skipped). Groups with no known distance to any
    other group join the largest group at the end (default 'same model', the safe direction for the score)."""
    groups, heights = [[i] for i in range(len(D))], []

    def link(a, b):
        blk = D[np.ix_(groups[a], groups[b])]
        return float(np.nanmean(blk)) if np.isfinite(blk).any() else math.inf
    while len(groups) > 1:
        d, a, b = min((link(a, b), a, b) for a, b in itertools.combinations(range(len(groups)), 2))
        if d > t or not math.isfinite(d):
            break
        heights.append(float(d)); groups[a] = groups[a] + groups[b]; del groups[b]
    lonely = [k for k in range(len(groups)) if len(groups) > 1 and
              all(not math.isfinite(link(k, m)) for m in range(len(groups)) if m != k)]
    if lonely and len(lonely) < len(groups):
        big = max((k for k in range(len(groups)) if k not in lonely), key=lambda k: len(groups[k]))
        for k in lonely:
            groups[big] = groups[big] + groups[k]
        groups = [g for k, g in enumerate(groups) if k not in lonely]
    lab = np.zeros(len(D), int)
    for k, g in enumerate(groups, 1):
        lab[g] = k
    return lab, heights


def gap_threshold(D):
    h = np.sort(merge(D)[1])
    if len(h) < 2:
        return float(h[0]) + 1e-9 if len(h) else 1.0
    k = int(np.argmax(np.diff(h)))
    return float((h[k] + h[k + 1]) / 2)


def text_distance(eps, text):
    docs = [" ".join(text.get(e, [])).lower() for e in eps]
    if not any(docs):
        return np.zeros((len(eps), len(eps)))
    grams = [Counter(w[i:i + n] for w in (f" {x} " for x in d.split()) for n in (2, 3, 4) for i in range(len(w) - n + 1))
             for d in docs]
    df = Counter(g for c in grams for g in c)
    vocab = {g: k for k, g in enumerate(df)}
    X = np.zeros((len(docs), len(vocab)))
    for i, c in enumerate(grams):
        for g, v in c.items():
            X[i, vocab[g]] = v * (math.log((1 + len(docs)) / (1 + df[g])) + 1)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return np.clip(1 - X @ X.T, 0, 1)


def tokenizer_distance(eps, obs):
    """Characters per generated token, if the API reports token counts: a tokenizer (= family) signature."""
    cpt = [obs["chars"][e] / obs["tokens"][e] if obs["tokens"][e] else math.nan for e in eps]
    if all(math.isnan(c) for c in cpt):
        return None, cpt
    c = np.array(cpt)
    D = np.abs(c[:, None] - c[None, :]) / max(np.nanstd(c), 1e-9)
    return np.clip(np.nan_to_num(D, nan=0.0) / 3, 0, 1), cpt


def model_distance(Z, must, cannot, N=None, min_items=0):
    """Distance = max(z, 0). A pair resting on fewer than min_items comparable items is UNKNOWN (nan), not 0:
    'no evidence' must not count as 'same model' (it would chain everything into one group)."""
    D = np.maximum(Z, 0.0).copy()
    if N is not None and min_items:
        D[(N < min_items) & ~np.eye(len(D), dtype=bool)] = np.nan
    for i, j in must:
        D[i, j] = D[j, i] = 0.0
    for i, j in cannot:
        D[i, j] = D[j, i] = 1e6
    return D


def family_distance(Z, parts):
    Zs = np.clip(np.maximum(Z, 0) / max(np.percentile(np.maximum(Z, 0), 95), 1e-9), 0, 1)
    parts = [Zs] + [p for p in parts if p is not None and p.any()]
    return np.mean(parts, axis=0)


def assign(eps, Df, Dm, t_family, t_model):
    fam = merge(Df, t_family)[0]
    labels, letters = {}, {}
    for f in sorted(set(fam), key=lambda f: min(i for i in range(len(eps)) if fam[i] == f)):
        letters[f] = chr(65 + len(letters))
    for f, L in letters.items():
        idx = [i for i in range(len(eps)) if fam[i] == f]
        sub = merge(Dm[np.ix_(idx, idx)], t_model)[0]
        order = {}
        for i, s in zip(idx, sub):
            order.setdefault(s, len(order) + 1)
            labels[eps[i]] = (L, f"{L}{order[s]}")
    return labels


# ---------- 5. score, check, write, round-2 plan ----------
def nanmean0(xs):
    """Mean of the finite values; nan (without a warning) when there are none."""
    xs = np.asarray(xs, float); xs = xs[np.isfinite(xs)]
    return float(xs.mean()) if len(xs) else math.nan


def mcc(tp, tn, fp, fn):
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return max(0.0, (tp * tn - fp * fn) / d) if d else 0.0


def score(truth, guess):
    """Server formula: 0.1 * MCC(same family?, all pairs) + 0.9 * MCC(same model?, pairs in one true family)."""
    f, m = Counter(), Counter()
    for a, b in itertools.combinations(sorted(truth), 2):
        tf, gf = truth[a][0] == truth[b][0], guess[a][0] == guess[b][0]
        f[(tf, gf)] += 1
        if tf:
            m[(truth[a] == truth[b], guess[a] == guess[b])] += 1
    fam = mcc(f[(True, True)], f[(False, False)], f[(False, True)], f[(True, False)])
    mod = mcc(m[(True, True)], m[(False, False)], m[(False, True)], m[(True, False)])
    return 0.1 * fam + 0.9 * mod, fam, mod


def check(labels, expected):
    errs = [f"missing {e}" for e in expected if e not in labels]
    owner = {}
    for e, (f, m) in labels.items():
        if "," in f + m or not re.fullmatch(r"[A-Z]+\d+", m) or not m.startswith(f):
            errs.append(f"bad labels for {e}: {f},{m}")
        if owner.setdefault(m, f) != f:
            errs.append(f"model label {m} used in two families")
    return errs


def write(labels, path):
    OUT.mkdir(exist_ok=True)
    with open(path, "w", newline="") as h:
        w = csv.writer(h); w.writerow(["endpoint", "family", "model"])
        for e in sorted(labels):
            w.writerow([e, *labels[e]])
    print(f"wrote {path}")


def round2_plan(eps, fam, Dm, t, margin, weights, channels="TCBPK"):
    """Endpoints in same-family pairs whose z is within `margin` of the cut, and the probes worth repeating."""
    unsure = sorted({e for i, j in itertools.combinations(range(len(eps)), 2)
                     if fam[i] == fam[j] and abs(Dm[i, j] - t) <= margin for e in (eps[i], eps[j])})
    ranked = sorted((p["id"] for p in me_probes.build() if p["kind"] in "TCBPK" and p["kind"] in channels), key=lambda p: -weights.get(p, 1.0))
    plan = {"endpoints": unsure, "probes": ranked[:6], "cut": t, "margin": margin}
    OUT.mkdir(exist_ok=True)
    json.dump(plan, open(OUT / "round2_plan.json", "w"), indent=1)
    print(f"round-2 plan: {len(unsure)} uncertain endpoints, probes {plan['probes']} -> {OUT / 'round2_plan.json'}")


# Pass/fail rule fixed on 21 Nov 2026 BEFORE the K practice answers were analysed. Do not tune these numbers to a
# result: they say what "the method works" means, so a practice run cannot be read in whichever way looks best.
PRESET = {"plain_vs_disguised_max_mean_z": 1.0,  # 1. plain S about as close to its disguised copies as to itself
          "min_ordering_share": 0.90,            # 2. share of (S-S pair, S-T3/T4 pair) comparisons with S-S lower
          "split_z": 3.0,                        # fixed same-model cut reported (not optimised on practice)
          "min_items": 5}                        # set 21 Nov 2026 before the corrected K results were seen


def preset_check(eps, Z, Dm, rows, a, weights):
    kind = {r["endpoint"]: r["model"] for r in rows}
    how = {r["endpoint"]: r.get("kind", "") for r in rows}
    truth = {r["endpoint"]: (r["family"], r["model"]) for r in rows}
    ix = {e: i for i, e in enumerate(eps)}
    S = [e for e in eps if kind[e] == "S"]
    plain = [e for e in S if how[e] == "plain"]
    ss = [Z[ix[x], ix[y]] for x, y in itertools.combinations(S, 2)]
    strong = [Z[ix[s], ix[v]] for s in S for v in eps if kind[v] in ("T3", "T4")]
    print("\nPRE-SET CHECK (rule fixed before the K results; see PRESET in me_analyze.py)")
    ok = []
    if plain:
        m = float(np.mean([Z[ix[plain[0]], ix[s]] for s in S if s != plain[0]]))
        ok.append(m <= PRESET["plain_vs_disguised_max_mean_z"])
        print(f"  1. plain {plain[0]} vs its disguised copies: mean z {m:+.2f} (needs <= "
              f"{PRESET['plain_vs_disguised_max_mean_z']})  {'PASS' if ok[-1] else 'FAIL'}")
    if ss and strong:
        share = float(np.mean([x < y for x in ss for y in strong]))
        ok.append(share >= PRESET["min_ordering_share"])
        print(f"  2. same-model pairs below S-vs-T3/T4 pairs: {share:.0%} of comparisons (needs >= "
              f"{PRESET['min_ordering_share']:.0%})  {'PASS' if ok[-1] else 'FAIL'}")
        for v in ("T1", "T2"):                                     # for information only (expected to be hard)
            weak = [Z[ix[s], ix[w]] for s in S for w in eps if kind[w] == v]
            if weak:
                print(f"     (info) same-model pairs below S-vs-{v} pairs: {np.mean([x < y for x in ss for y in weak]):.0%}")
    official = a.stat == "confident" and set(a.channels) == {"K"} and not weights
    ok.append(official)
    print(f"  3. run with --stat confident --channels K and no weights: {'yes' if official else 'no'}  "
          f"{'PASS' if official else 'FAIL (this run is for comparison only)'}")
    g = assign(eps, np.zeros_like(Dm), Dm, 1.0, PRESET["split_z"])
    print(f"  practice model MCC at the fixed cut z = {PRESET['split_z']}: {score(truth, g)[2]:.3f}  groups "
          f"{sorted(Counter(m for _, m in g.values()).values(), reverse=True)}")
    print(f"  VERDICT: {'K METHOD WORKS on practice' if all(ok) else 'NOT SHOWN TO WORK on practice'}")


def show(eps, M, name, tag=None):
    print(f"\n{name}")
    print(" " * 9 + "".join(f"{e[-6:]:>7}" for e in eps))
    for i, e in enumerate(eps):
        print(f"{e[-6:]:>6}{(tag or {}).get(e, ''):>3}" + "".join(f"{M[i, j]:7.1f}" for j in range(len(eps))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["ref", "ep"], required=True)
    ap.add_argument("--t-model", type=float, default=None, help="same-model cut in z units")
    ap.add_argument("--t-family", type=float, default=None, help="same-family cut (default: largest gap)")
    ap.add_argument("--max-repeats", type=int, default=None, help="use only this many answers per probe")
    ap.add_argument("--perms", type=int, default=300, help="shuffles per pair")
    ap.add_argument("--stat", choices=["l2", "mode", "confident"], default="l2",
                    help="per-item statistic (mode tolerates temperature; confident only counts firmly held answers)")
    ap.add_argument("--probes", default="", help=r"regex of probe ids to use (F always kept), e.g. 'K\d+o\d+'")
    ap.add_argument("--min-items", type=int, default=PRESET["min_items"],
                    help="--stat confident: pairs with fewer firmly-held items in common are unknown, not 'same'")
    ap.add_argument("--channels", default="".join(CHANNELS), help="probe kinds used for the model decision, e.g. K or KC")
    ap.add_argument("--weights", default="", help="probe weights from the practice run (out/probe_weights.json)")
    ap.add_argument("--no-greedy", action="store_true")
    ap.add_argument("--greedy-min-prefix", type=int, default=None)
    ap.add_argument("--margin", type=float, default=None,
                    help="round-2 plan: z distance from the cut counted as unsure (default: half the cut)")
    a = ap.parse_args()
    eps, obs = load(a.target, a.max_repeats, a.probes)
    usable_report(eps, obs)
    weights = json.load(open(a.weights)) if a.weights else {}
    units = build_units(eps, obs, a.channels)
    Z, P, prz, N = all_pairs(eps, units, a.perms, a.stat, weights)
    det, must, cannot, gtable = greedy_links(eps, obs["greedy"], a.greedy_min_prefix)
    if a.no_greedy:
        must, cannot = set(), set()
    Dm = model_distance(Z, must, cannot, N, a.min_items if a.stat == "confident" else 0)
    Dtok, cpt = tokenizer_distance(eps, obs)
    Df = family_distance(Z, [text_distance(eps, obs["text"]), Dtok])
    if Dtok is not None:
        print("chars per token: " + ", ".join(f"{e} {c:.2f}" for e, c in zip(eps, cpt)))

    if a.target == "ref":
        rows = list(csv.DictReader(open(HERE.parent / "practice_answers.csv")))
        truth = {r["endpoint"]: (r["family"], r["model"]) for r in rows}
        kind = {r["endpoint"]: r["model"] for r in rows}
        disguise_profile(eps, obs, det, np.ones(len(eps), int))
        split = defaultdict(list)
        for x, y, _, k in gtable:
            if k is not None:
                split["same model (S-S)" if kind[x] == kind[y] == "S" else "different models"].append(k)
        print("\ngreedy endpoints: " + (", ".join(e for e in eps if det[e]) or "none"))
        if split:
            print("greedy split points (words before texts differ): " + "; ".join(f"{g}: {sorted(v)}" for g, v in split.items()) +
                  "\n  -> --greedy-min-prefix is safe only if every 'different models' split is later than every S-S split")
        show(eps, Z, f"pair z-scores ({a.stat}; ~0 = indistinguishable, large = different)", kind)
        if a.stat == "confident":
            show(eps, N, f"items both endpoints hold firmly (fewer than {a.min_items} = unknown, not 'same')", kind)
        preset_check(eps, Z, Dm, rows, a, weights)
        S = [i for i, e in enumerate(eps) if kind[e] == "S"]
        V = sorted({kind[e] for e in eps if kind[e] != "S"})
        print("\nprobe power: mean z between the five disguises (S-S, should be ~0) and from S to each variant")
        print(f"  {'probe':>6} {'S-S':>6} " + " ".join(f"{v:>6}" for v in V) + "   weight")
        wts = {}
        for pr in sorted({p for d in prz.values() for p in d}):
            ss = nanmean0([prz[(min(i, j), max(i, j))].get(pr, np.nan) for i, j in itertools.combinations(S, 2)])
            cols = [nanmean0([prz[(min(i, j), max(i, j))].get(pr, np.nan) for i in S for j in range(len(eps))
                                if kind[eps[j]] == v]) for v in V]
            w = nanmean0(cols) - (max(ss, 0) if np.isfinite(ss) else 0.0)
            wts[pr] = float(w) if np.isfinite(w) else 0.0          # no comparable answers: no evidence, weight 0
            print(f"  {pr:>6} {ss:6.1f} " + " ".join(f"{c:6.1f}" for c in cols) + f"   {wts[pr]:+.2f}"
                  + ("" if np.isfinite(w) else "  (no comparable answers)"))
        top = max(float(np.nanmax(list(wts.values()))), 1e-9)
        wts = {k: round(float(np.clip(v / top, 0.1, 1.0)), 2) for k, v in wts.items()}   # floor 0.1: no probe is dropped outright
        OUT.mkdir(exist_ok=True); json.dump(wts, open(OUT / "probe_weights.json", "w"), indent=1)
        print(f"probe weights saved to {OUT / 'probe_weights.json'}: {wts}")
        best = (-1.0, None)
        sweep = []
        for t in sorted({float(x) for x in np.round(Dm[np.triu_indices(len(eps), 1)], 2) if np.isfinite(x)}):
            if t > 1e5:
                continue
            g = assign(eps, np.zeros_like(Dm), Dm, 1.0, t + 1e-6)
            s = score(truth, g)[2]
            sweep.append((t, s, sorted(Counter(m for _, m in g.values()).values(), reverse=True)))
            best = max(best, (s, t))
        print("\npractice model score by threshold (z units): " + "  ".join(f"{t:.1f}:{s:.2f}" for t, s, _ in sweep))
        t = a.t_model if a.t_model is not None else best[1] + 1e-6
        g = assign(eps, np.zeros_like(Dm), Dm, 1.0, t)
        print(f"using t-model={t:.2f}: practice model MCC={score(truth, g)[2]:.3f}  groups {sorted(Counter(m for _, m in g.values()).values(), reverse=True)}")
        write(g, OUT / "ref_guess.csv")
        return

    t_family = a.t_family if a.t_family is not None else gap_threshold(Df)
    fam = merge(Df, t_family)[0]
    print(f"\nfamily cut {t_family:.3f} -> {len(set(fam))} families: " +
          "; ".join(",".join(e[-2:] for e, f in zip(eps, fam) if f == k) for k in sorted(set(fam))))
    disguise_profile(eps, obs, det, fam)
    if a.t_model is None:
        sys.exit("give --t-model (in z units, calibrated on the practice endpoints with --target ref)")
    labels = assign(eps, Df, Dm, t_family, a.t_model)
    errs = check(labels, [f"ep-{i:02d}" for i in range(1, 34)])
    if errs:
        sys.exit("NOT written, format problems:\n  " + "\n  ".join(errs))
    for f in sorted({f for f, _ in labels.values()}):
        print(f"  family {f}: " + "  ".join(f"{e}={m}" for e, (ff, m) in sorted(labels.items()) if ff == f))
    write(labels, OUT / "submission_ep.csv")
    round2_plan(eps, fam, Dm, a.t_model, a.margin if a.margin is not None else a.t_model / 2, weights, a.channels)


if __name__ == "__main__":
    main()
