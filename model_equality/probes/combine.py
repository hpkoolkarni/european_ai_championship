"""Model Equality, combined method. Reads saved answers only: NO API calls.

  families : NOT fixed in advance. Candidates = writing-style groupings into 2..MAX_FAMILIES groups (character
             3-5-gram TF-IDF over every saved answer, same prompts for every endpoint, average linkage). A candidate is
             accepted only if the behaviour tests below CONFIRM it: between every two groups the MEDIAN pair z must be
             >= cut (changed 23 Nov 2026 from 'every pair', see families()). The largest accepted split wins;
             if none passes, everything stays one family (the safe mistake: different base models still land in
             different model groups through the behaviour tests). Self-descriptions are printed, never required.
  L        : paired reply-length test on the free-length short prompts (ids matching --len-probes, pre-set F/R/V):
             per prompt, mean log(1 + chars) of A's answers minus B's; statistic = (mean over prompts)^2; null = A/B
             labels shuffled within each prompt
  U        : the tool/echo probe of tool_probe/tool_probe.py (first --tools-n answers per endpoint)
  K        : optional (--with-k): the knowledge test of analyze.py (--stat confident, channel K)
A test is used only if its pair z reproduce across two random halves of the data (MIN_RELIABILITY, truth-free).
Pair z of the used channels is combined as sum / sqrt(k). Same model = average-linkage group INSIDE one family,
distance max(z, 0), cut PRESET["cut"]. Everything in PRESET was fixed before the new V prompts were answered.

  practice (pre-set check + practice score):   python probes/combine.py --target ref
  out-of-sample check on the new prompts only:  python probes/combine.py --target ref --len-probes 'V\\d+'
  scored endpoints (writes submissions/submission_combined.csv):  python probes/combine.py --target ep
  truth-free checks on a target's own answers (add to any of the above):  --diagnose [--diag-splits 10]
"""
import argparse
import collections
import csv
import itertools
import json
import math
import random
import re
import sys
from pathlib import Path

import numpy as np

try:
    from scipy import stats as _stats                              # exact chi-square tail in calib_z
except ImportError:                                                # falls back to Wilson-Hilferty
    _stats = None

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "tool_probe")]
import analyze as A                                             # noqa: E402

# fixed 22 Nov 2026 before the V answers exist. tools_n = 10: with calibrated z the practice result is the same for
# 10, 15 and 29 tool answers (score 0.445), while more tool answers only raise the same-model pairs' worst z (1.4 at
# 10, 2.8 at 29: the watermark copy's different echo rate becomes 'significant'). Extra budget goes to the length test.
PRESET = {"len_probes": r"[FRV]\d+", "tools_n": 10, "cut": 3.0,
          "plain_vs_disguised_max_mean_z": 1.0, "min_ordering_share": 0.90}
MAX_FAMILIES = 5               # style candidates tried: 2..5 groups
# set 23 Nov 2026 after phase 1: on practice the length and tool tests reproduce across halves (0.66, 0.50); on the
# scored set the length test did not (0.08) while the tool test did (0.58). Any value from 0.2 to 0.45 splits the same.
MIN_RELIABILITY = 0.30
MIN_FAMILY_SIZE = 2            # a lone endpoint is not called a family of its own (a strong LoRA also differs
                               # from everything; as its own model group inside the family it costs nothing)


def rows_of(target):
    p = HERE / "data" / f"{target}_answers.jsonl"
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []


# ---------------- families ----------------
def style_distance(eps, rows):
    texts = collections.defaultdict(list)
    for r in rows:
        texts[r["endpoint"]].append(r["response"] or "")
    profs = []
    for e in eps:
        c = collections.Counter()
        for t in texts[e]:
            t = re.sub(r"\s+", " ", t.lower())
            for k in (3, 4, 5):
                c.update(t[i:i + k] for i in range(len(t) - k + 1))
        profs.append(c)
    df = collections.Counter(g for p in profs for g in p)
    grams = [g for g, d in df.items() if 2 <= d <= len(eps) - 1]
    ix = {g: i for i, g in enumerate(grams)}
    X = np.zeros((len(eps), len(grams)))
    for r, p in enumerate(profs):
        for g, v in p.items():
            if g in ix:
                X[r, ix[g]] = math.log1p(v) * math.log(len(eps) / df[g])
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return 1 - X @ X.T


VENDORS = (("qwen", r"qwen|alibaba"), ("meta", r"\bmeta\b|llama"),
           ("openai", r"openai|chatgpt|\bgpt"), ("google", r"google|gemma|gemini|deepmind"), ("mistral", r"mistral"),
           ("microsoft", r"microsoft|\bphi\b"), ("huggingface", r"hugging ?face|smol"))


def vendor_of(text):
    t = (text or "").lower()
    return next((n for n, p in VENDORS if re.search(p, t)), "-")


def style_groups(D, k):
    """Average linkage on the style distance down to k groups."""
    groups = [[i] for i in range(len(D))]
    while len(groups) > k:
        d, a, b = min((float(D[np.ix_(groups[a], groups[b])].mean()), a, b)
                      for a, b in itertools.combinations(range(len(groups)), 2))
        groups[a] = groups[a] + groups[b]; del groups[b]
    lab = np.zeros(len(D), int)
    for c, g in enumerate(groups):
        lab[g] = c
    return lab


def families(Ds, Z, cut, n_forced=None):
    """Style proposes, behaviour confirms. For k = MAX_FAMILIES..2: take the style grouping into k groups; accept it
    if every group has >= MIN_FAMILY_SIZE endpoints and, for EVERY two groups, the MEDIAN combined z of the pairs
    between them is >= cut (the typical pair across two families differs in behaviour; missing z are left out, a
    group pair with no z at all is not confirmed). Changed 23 Nov 2026 after phase 1: the earlier rule needed every
    single cross pair >= cut, so with a test of modest power a few weak pairs vetoed a split that 88% of the cross
    pairs supported. Returns labels, a one-line reason and the per-candidate report lines."""
    if n_forced:
        return style_groups(Ds, n_forced), f"forced to {n_forced} (--families)", []
    n, report = len(Ds), []
    for k in range(min(MAX_FAMILIES, n), 1, -1):
        lab = style_groups(Ds, k)
        sizes = np.bincount(lab, minlength=k)
        meds = []
        for g, h in itertools.combinations(range(k), 2):
            zs = [Z[i, j] for i in np.flatnonzero(lab == g) for j in np.flatnonzero(lab == h) if np.isfinite(Z[i, j])]
            meds.append(float(np.median(zs)) if zs else -math.inf)
        cross = [Z[i, j] for i, j in itertools.combinations(range(n), 2) if lab[i] != lab[j] and np.isfinite(Z[i, j])]
        share = float(np.mean(np.array(cross) >= cut)) if cross else 0.0
        ok_size, ok_z = sizes.min() >= MIN_FAMILY_SIZE, min(meds) >= cut
        report.append((k, sorted(sizes.tolist(), reverse=True), min(meds), share, ok_size and ok_z))
        if ok_size and ok_z:
            return lab, (f"{k} families: style split into {sorted(sizes.tolist(), reverse=True)}; between every two "
                         f"groups the median pair z is >= {cut} (lowest {min(meds):.1f}); {share:.0%} of all cross "
                         f"pairs are >= {cut}"), report
    return np.zeros(n, int), "one family (no style split was confirmed by the behaviour tests)", report


# ---------------- channels ----------------
RAW_Z = False                  # --raw-z: old (stat - null mean) / null sd, for comparison only


def calib_z(obs, null):
    """Calibrated z: fit the permutation null of a squared-distance statistic with a scaled chi-square g*chi2(h)
    (matching its mean and variance) and map the observed value to a standard normal score (Wilson-Hilferty).
    Under 'same model' this z is ~N(0, 1) for every test, so a z of 3 means the same evidence in each and an equal-
    weight sum is fair. The plain (obs - mean) / sd grows with the square of the effect times the sample size, so a
    test with many answers would swamp the others."""
    null = np.asarray(null, float)
    if RAW_Z:
        return float((obs - null.mean()) / (null.std() + 1e-12))
    m, v = float(null.mean()), float(null.var())
    if m <= 0 or v <= 0:
        return 0.0
    g, h = v / (2 * m), 2 * m * m / v
    x = max(obs, 0.0) / g
    if _stats is not None:                                         # exact tail of the fitted chi-square
        p = float(_stats.chi2.sf(x, h))
        return float(np.clip(_stats.norm.isf(max(p, 1e-300)), -6.0, 37.0))
    c = 2 / (9 * h)                                                # Wilson-Hilferty approximation
    return float(((x / h) ** (1 / 3) - (1 - c)) / math.sqrt(c))


def length_z(per_a, per_b, B, rng):
    """per_x: {prompt id: [log lengths]}. Returns z of the paired length test and the number of prompts used."""
    items = [(np.array(per_a[p]), np.array(per_b[p])) for p in per_a if p in per_b]
    if len(items) < 3:
        return math.nan, len(items)
    obs = np.mean([x.mean() - y.mean() for x, y in items])
    null = np.zeros(B)
    for x, y in items:
        pool = np.concatenate([x, y])
        perm = pool[np.argsort(rng.random((B, len(pool))), axis=1)]
        null += perm[:, :len(x)].mean(1) - perm[:, len(x):].mean(1)
    null /= len(items)
    return calib_z(obs ** 2, null ** 2), len(items)


def channel_L(eps, rows, pat, pairs, B=2000):
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if re.fullmatch(pat, r["id"]):
            per[r["endpoint"]][r["id"]].append(math.log1p(len(r["response"] or "")))
    rng = np.random.default_rng(0)
    Z = np.full((len(eps), len(eps)), math.nan); np.fill_diagonal(Z, 0)
    used = 0
    for i, j in pairs:
        Z[i, j] = Z[j, i] = (z := length_z(per[eps[i]], per[eps[j]], B, rng))[0]; used = max(used, z[1])
    return Z, used, per


def tools_pair_z(T, xa, xb, perms, rng):
    """The teammate's statistic (tool_probe.stat) and permutation null, with a calibrated z (calib_z)."""
    t, pool, nulls = T.stat(xa, xb), list(xa) + list(xb), []
    for _ in range(perms):
        rng.shuffle(pool); nulls.append(T.stat(pool[:len(xa)], pool[len(xa):]))
    return calib_z(t, nulls)


class ShapeStat:
    """Temperature-robust variant of the tool statistic (channel S). Per item, the log choice frequencies (+0.5
    smoothing, unreadable answers left out) are centred over the options; a temperature change multiplies these
    centred log-probabilities by one constant, so 1 - correlation of the two endpoints' concatenated vectors does not
    move with temperature, while a different preference ORDER (other weights) does. Same interface as tool_probe."""

    def __init__(self, T):
        self.ITEMS = T.ITEMS

    def stat(self, A, B):
        va, vb = [], []
        for k, (_, opts) in enumerate(self.ITEMS):
            for X, v in ((A, va), (B, vb)):
                c = collections.Counter(x[k] for x in X if x[k] != "INVALID")
                m = sum(c.values())
                lg = np.log([(c[o] + 0.5) / (m + 0.5 * len(opts)) for o in opts])
                v.extend(lg - lg.mean())
        va, vb = np.array(va), np.array(vb)
        if va.std() == 0 or vb.std() == 0:
            return 1.0
        return float(1 - np.corrcoef(va, vb)[0, 1])


def channel_U(eps, target, n, pairs, perms=500, shape=False):
    try:
        import tool_probe as T
    except ImportError:
        return None, {}
    T0, T = T, (ShapeStat(T) if shape else T)
    if not (T0.DATA / f"{target}_tools.jsonl").exists():
        return None, {}
    tr = T0.load(target, n)
    if not all(tr.get(e) for e in eps):
        print(f"  tool probe: answers missing for {[e for e in eps if not tr.get(e)]}; channel U skipped")
        return None, {}
    rng = random.Random(0)
    Z = np.full((len(eps), len(eps)), math.nan); np.fill_diagonal(Z, 0)
    for i, j in pairs:
        Z[i, j] = Z[j, i] = tools_pair_z(T, tr[eps[i]], tr[eps[j]], perms, rng)
    info = {e: (len(tr[e]), float(np.mean([x.count("INVALID") == len(T.ITEMS) for x in tr[e]]))) for e in eps}
    return Z, info


def channel_K(eps, target):
    e2, obs = A.load(target, None, r"K\d+(o\d+)?")
    if e2 != eps:
        print("  K answers missing for some endpoints; channel K skipped"); return None
    units = A.build_units(eps, obs, ("K",))
    return A.all_pairs(eps, units, 1000, "confident", {})[0]


# ---------------- diagnostics: checks on the target's OWN answers, no true labels ----------------
def length_matrix(eps, per, keep, pairs, B, seed):
    """Length-test z on `pairs`, using only the prompt ids in `keep`."""
    rng = np.random.default_rng(seed)
    Z = np.full((len(eps), len(eps)), math.nan); np.fill_diagonal(Z, 0)
    for i, j in pairs:
        pa = {p: v for p, v in per[eps[i]].items() if p in keep}
        pb = {p: v for p, v in per[eps[j]].items() if p in keep}
        Z[i, j] = Z[j, i] = length_z(pa, pb, B, rng)[0]
    return Z


def tools_matrix(T, tr, eps, pairs, perms, seed):
    rng = random.Random(seed)
    Z = np.full((len(eps), len(eps)), math.nan); np.fill_diagonal(Z, 0)
    for i, j in pairs:
        Z[i, j] = Z[j, i] = tools_pair_z(T, tr[eps[i]], tr[eps[j]], perms, rng)
    return Z


def combine(mats):
    stack = np.stack(mats)
    cnt = np.isfinite(stack).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, np.nansum(stack, 0) / np.sqrt(np.maximum(cnt, 1)), np.nan)


def split_halves(eps, names, per, tr, T, pairs, splits):
    """For each test: `splits` random halvings of the data (length test: half of the prompts each; tool probe: half of
    each endpoint's answers each), with the pair z computed separately on both halves."""
    halves, r2 = collections.defaultdict(list), random.Random(2)
    if "L" in names:
        ids = sorted({p for e in eps for p in per[e]})
        for s in range(splits):
            sh = ids[:]; r2.shuffle(sh); h = len(sh) // 2
            halves["L"].append((length_matrix(eps, per, set(sh[:h]), pairs, 500, s),
                                length_matrix(eps, per, set(sh[h:]), pairs, 500, 100 + s)))
    for name in [c for c in ("U", "S") if c in names and T is not None]:
        TT = ShapeStat(T) if name == "S" else T
        for s in range(splits):
            ta, tb = {}, {}
            for e in eps:
                x = list(tr[e]); r2.shuffle(x); h = len(x) // 2; ta[e], tb[e] = x[:h], x[h:]
            halves[name].append((tools_matrix(TT, ta, eps, pairs, 200, s), tools_matrix(TT, tb, eps, pairs, 200, 100 + s)))
    return halves


def half_reliability(hs, pairs):
    """Median (over splits) rank correlation of the pair z from one half with the pair z from the other half."""
    rho = [spearman([A_[i, j] for i, j in pairs], [B_[i, j] for i, j in pairs]) for A_, B_ in hs]
    rho = [r for r in rho if np.isfinite(r)]
    return float(np.median(rho)) if rho else math.nan


def _grams(t):
    t = re.sub(r"\s+", " ", (t or "").lower())
    return collections.Counter(t[i:i + k] for k in (3, 4, 5) for i in range(len(t) - k + 1))


def content_distance(eps, rows, keep):
    """Answer similarity, prompt by prompt: for each prompt in `keep`, the character 3-5-gram TF-IDF cosine distance
    between two endpoints' (first) answers to THAT prompt, averaged over prompts. Diagnostic only: with one answer
    per prompt there is no 'same model' null to turn it into a z."""
    ans = collections.defaultdict(dict)
    for r in rows:
        if r["id"] in keep:
            ans[r["id"]].setdefault(r["endpoint"], r["response"] or "")
    n = len(eps)
    D, cnt = np.zeros((n, n)), np.zeros((n, n))
    for p, by in ans.items():
        idx = [i for i, e in enumerate(eps) if e in by]
        if len(idx) < 3:
            continue
        profs = [_grams(by[eps[i]]) for i in idx]
        df = collections.Counter(g for pr in profs for g in pr)
        vocab = {g: k for k, g in enumerate(df)}
        X = np.zeros((len(idx), len(vocab)))
        for r_, pr in enumerate(profs):
            for g, v in pr.items():
                X[r_, vocab[g]] = math.log1p(v) * math.log((1 + len(idx)) / df[g])
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        D[np.ix_(idx, idx)] += 1 - X @ X.T; cnt[np.ix_(idx, idx)] += 1
    with np.errstate(invalid="ignore"):
        return np.where(cnt > 0, D / np.maximum(cnt, 1), np.nan)


def same_calls(eps, fam, Z, cut, pairs):
    """'Same model' call per pair, made exactly as for the submission (average-linkage groups inside each family)."""
    lab = labels_for(eps, fam, Z, cut)
    return np.array([lab[eps[i]][1] == lab[eps[j]][1] for i, j in pairs], bool)


def mcc2(x, y):
    x, y = np.asarray(x, bool), np.asarray(y, bool)
    tp, tn = float(np.sum(x & y)), float(np.sum(~x & ~y))
    fp, fn = float(np.sum(~x & y)), float(np.sum(x & ~y))
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / d if d else math.nan


def _ranks(v):
    r = np.empty(len(v)); r[np.argsort(v, kind="mergesort")] = np.arange(len(v))
    for u in np.unique(v):
        m = v == u
        if m.sum() > 1:
            r[m] = r[m].mean()                                     # ties (e.g. z clipped at -6) share a rank
    return r


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5:
        return math.nan
    rx, ry = _ranks(x[ok]), _ranks(y[ok])
    return float(np.corrcoef(rx, ry)[0, 1]) if rx.std() and ry.std() else math.nan


def _fmt(v, f="{:.2f}"):
    return f.format(v) if np.isfinite(v) else "n/a"


def diagnose(eps, chans, Z, fam, pairs, per, a, splits, halves, kept, rows):
    """Truth-free checks: (1) calibration of each test on an endpoint against itself, (2) split-half stability,
    (3) agreement between independent tests and between prompt kinds, (4) gap around the cut, (5) consistency of
    triangles, (6) the endpoints these checks flag most often."""
    cut, n = a.cut, len(eps)
    allpairs = list(itertools.combinations(range(n), 2))
    print(f"\nDIAGNOSTICS on {a.target}'s own answers (no true labels used; {splits} random splits)")
    T = tr = None
    if "U" in chans or "S" in chans:
        import tool_probe as T
        tr = T.load(a.target, a.tools_n)

    # 1. calibration
    print("1. calibration, each endpoint against ITSELF (calibrated: mean ~0, sd ~1, share > 3 about 0.1%)")
    rng = random.Random(1)

    def null_line(name, z, why):
        z = np.array([v for v in z if np.isfinite(v)])
        if len(z) < 5:
            print(f"   {name}: not available ({why})"); return
        print(f"   {name}: {len(z)} self-splits, mean {z.mean():+.2f}, sd {z.std():.2f}, share > 3: {np.mean(z > 3):.1%}"
              f"{'  <- NOT calibrated on this data' if abs(z.mean()) > 0.5 or z.std() > 1.5 or np.mean(z > 3) > 0.02 else ''}")
    if "L" in chans:
        nl = []
        for e in eps:
            rep = {p: v for p, v in per[e].items() if len(v) >= 2}
            if len(rep) < 3:
                continue
            for s in range(splits):
                pa, pb = {}, {}
                for p, v in rep.items():
                    v = v[:]; rng.shuffle(v); h = len(v) // 2; pa[p], pb[p] = v[:h], v[h:]
                nl.append(length_z(pa, pb, 500, np.random.default_rng(s))[0])
        null_line("L", nl, "needs >= 2 answers to the same prompt on >= 3 prompts; one answer per prompt here")
    if T is not None:
        nu = []
        for e in eps:
            for _ in range(splits):
                x = list(tr[e]); rng.shuffle(x); h = len(x) // 2
                nu.append(tools_pair_z(ShapeStat(T) if "S" in chans else T, x[:h], x[h:], 200, rng))
        null_line(f"{'S' if 'S' in chans else 'U'} (halves of {a.tools_n} answers)", nu, "too few tool answers")

    # 2. split-half stability (within-family pairs: the 90% part)
    print(f"2. split-half stability on the {len(pairs)} within-family pairs (each half has half the data, so this "
          "UNDERSTATES the full test)")
    halves = dict(halves)
    used = [c for c in kept if c in halves]
    if len(used) > 1:
        halves["combined"] = [(combine([halves[c][s][0] for c in used]), combine([halves[c][s][1] for c in used]))
                              for s in range(len(halves[used[0]]))]
    flips = collections.Counter()
    main = "combined" if "combined" in halves else (used[0] if used else None)
    for name, hs in halves.items():
        rho = [spearman([A_[i, j] for i, j in pairs], [B_[i, j] for i, j in pairs]) for A_, B_ in hs]
        calls = [(same_calls(eps, fam, A_, cut, pairs), same_calls(eps, fam, B_, cut, pairs)) for A_, B_ in hs]
        mc = [mcc2(x, y) for x, y in calls]
        agree = [float(np.mean(x == y)) for x, y in calls]
        share = np.mean([np.mean(x) for xy in calls for x in xy])
        print(f"   {name:9s} z rank correlation half vs half: median {_fmt(np.nanmedian(rho) if np.isfinite(rho).any() else math.nan)}"
              f" | same/different calls: agree {np.mean(agree):.0%}, MCC median "
              f"{_fmt(np.nanmedian(mc) if np.isfinite(mc).any() else math.nan)} "
              f"(range {_fmt(np.nanmin(mc) if np.isfinite(mc).any() else math.nan)}.."
              f"{_fmt(np.nanmax(mc) if np.isfinite(mc).any() else math.nan)}) | 'same' calls per half {share:.0%}")
        if name == main:
            for x, y in calls:
                for (i, j), d in zip(pairs, x != y):
                    if d:
                        flips[eps[i]] += 1; flips[eps[j]] += 1
    if not halves:
        print("   no test has data")
    print("   (MCC n/a = a half made one blanket call, e.g. every pair 'same'; 'agree' then says little)")

    # 3. agreement between tests, and between prompt kinds of the length test
    print("3. agreement between independent tests (different prompts, different behaviour)")
    names = list(chans)
    if len(names) < 2:
        print("   only one test has data; nothing to compare")
    for x, y in itertools.combinations(names, 2):
        cx, cy = same_calls(eps, fam, chans[x], cut, pairs), same_calls(eps, fam, chans[y], cut, pairs)
        print(f"   {x} vs {y}: z rank correlation all pairs {_fmt(spearman([chans[x][i, j] for i, j in allpairs], [chans[y][i, j] for i, j in allpairs]))}"
              f", within families {_fmt(spearman([chans[x][i, j] for i, j in pairs], [chans[y][i, j] for i, j in pairs]))}"
              f" | calls: both same {int(np.sum(cx & cy))}, only {x} same {int(np.sum(cx & ~cy))}, only {y} same "
              f"{int(np.sum(~cx & cy))}, both different {int(np.sum(~cx & ~cy))}, MCC {_fmt(mcc2(cx, cy))}")
    if "L" in chans:
        kinds = collections.defaultdict(set)
        for e in eps:
            for p in per[e]:
                kinds[p[0]].add(p)
        kinds = {k: v for k, v in kinds.items() if len(v) >= 3}
        if len(kinds) >= 2:
            KM = {k: length_matrix(eps, per, v, pairs, 500, 7) for k, v in sorted(kinds.items())}
            print("   length test by prompt kind (does each kind order the pairs the same way?): " + ", ".join(
                f"{k1} vs {k2} {_fmt(spearman([KM[k1][i, j] for i, j in pairs], [KM[k2][i, j] for i, j in pairs]))}"
                for k1, k2 in itertools.combinations(KM, 2)) + "  (" + ", ".join(f"{k}: {len(v)} prompts" for k, v in sorted(kinds.items())) + ")")

    # 3b. answer similarity (what is said, not how long): reproducible? does it agree with the tests?
    ids = sorted({r["id"] for r in rows if re.fullmatch(a.len_probes, r["id"])})
    if len(ids) >= 6:
        Dc = content_distance(eps, rows, set(ids))
        r3, rc = random.Random(3), []
        for _ in range(splits):
            sh = ids[:]; r3.shuffle(sh); h = len(sh) // 2
            Da, Db = content_distance(eps, rows, set(sh[:h])), content_distance(eps, rows, set(sh[h:]))
            rc.append(spearman([Da[i, j] for i, j in pairs], [Db[i, j] for i, j in pairs]))
        print(f"   answer similarity ({len(ids)} prompts, diagnostic only): half vs half rank correlation "
              f"{_fmt(float(np.nanmedian(rc)))}; vs " + ", ".join(
                  f"{c} z {_fmt(spearman([Dc[i, j] for i, j in pairs], [chans[c][i, j] for i, j in pairs]))}" for c in chans))

    # 4. gap around the cut
    zin = np.array([Z[i, j] for i, j in pairs if np.isfinite(Z[i, j])])
    edges = [-math.inf, 1, 2, 3, 4, 6, math.inf]
    hist = np.histogram(zin, edges)[0] if len(zin) else np.zeros(6, int)
    print("4. combined z of within-family pairs: " + "  ".join(
        f"{'<' + str(edges[k + 1]) if k == 0 else ('>=' + str(edges[k]) if k == 5 else f'{edges[k]}-{edges[k + 1]}')}: {c}"
        for k, c in enumerate(hist)) + (f"  | near the cut (2-4): {np.mean((zin >= 2) & (zin < 4)):.0%}" if len(zin) else ""))

    # 5. triangles
    close = np.isfinite(Z) & (Z < cut)
    broken, nb, nt = collections.Counter(), 0, 0
    for f in sorted(set(fam)):
        idx = [i for i in range(n) if fam[i] == f]
        for i, j, k in itertools.combinations(idx, 3):
            c = int(close[i, j]) + int(close[i, k]) + int(close[j, k])
            nt += c >= 2
            if c == 2:
                nb += 1
                for e in (i, j, k):
                    broken[eps[e]] += 1
    print(f"5. consistency: {nb} of {nt} triangles with >= 2 'same' pairs are broken (A~B and B~C, but A-C differs)")

    # 6. endpoints flagged most
    near = collections.Counter()
    for i, j in pairs:
        if np.isfinite(Z[i, j]) and abs(Z[i, j] - cut) < ROUND2_MARGIN:
            near[eps[i]] += 1; near[eps[j]] += 1
    lists = {"flip between halves": flips, "broken triangles": broken, f"pairs within {ROUND2_MARGIN} of the cut": near}
    k = max(2, math.ceil(n / 4))                                   # the top quarter of the endpoints on each list
    print(f"6. endpoints flagged most often (top {k} per list)")
    top = {}
    for nm, c in lists.items():
        top[nm] = [e for e, v in c.most_common(k) if v > 0]
        print(f"   {nm:28s}: " + (", ".join(f"{e[-2:]}({c[e]})" for e in top[nm]) or "none"))
    multi = sorted(e for e in eps if sum(e in t for t in top.values()) >= 2)
    print(f"   on 2+ lists (candidates for more answers): {', '.join(e[-2:] for e in multi) or 'none'}")


# ---------------- reporting ----------------
def show(eps, M, title, tag):
    print(f"\n{title}")
    print(" " * 12 + "".join(f"{e[-2:]:>6}" for e in eps))
    for i, e in enumerate(eps):
        print(f"{e:>6} {tag.get(e, '')[:5]:>5}" + "".join("     ." if i == j else ("     -" if not np.isfinite(M[i, j])
                                                         else f"{M[i, j]:6.1f}") for j in range(len(eps))))


def preset_check(eps, Z, fam, truth_rows):
    model = {r["endpoint"]: r["model"] for r in truth_rows}
    how = {r["endpoint"]: r.get("kind", "") for r in truth_rows}
    truth = {r["endpoint"]: (r["family"], r["model"]) for r in truth_rows}
    ix = {e: i for i, e in enumerate(eps)}
    S = [e for e in eps if model[e] == "S"]; plain = [e for e in S if how[e] == "plain"]
    ss = [Z[ix[x], ix[y]] for x, y in itertools.combinations(S, 2)]
    strong = [Z[ix[s], ix[v]] for s in S for v in eps if model[v] in ("T3", "T4")]
    print("\nPRE-SET CHECK (rules in PRESET, fixed before the new prompts were answered)")
    ok = []
    m = float(np.mean([Z[ix[plain[0]], ix[s]] for s in S if s != plain[0]]))
    ok.append(m <= PRESET["plain_vs_disguised_max_mean_z"])
    print(f"  1. plain vs its disguised copies: mean z {m:+.2f} (needs <= {PRESET['plain_vs_disguised_max_mean_z']})"
          f"  {'PASS' if ok[-1] else 'FAIL'}")
    share = float(np.mean([x < y for x in ss for y in strong]))
    ok.append(share >= PRESET["min_ordering_share"])
    print(f"  2. same-model pairs below S-vs-T3/T4 pairs: {share:.0%} (needs >= {PRESET['min_ordering_share']:.0%})"
          f"  {'PASS' if ok[-1] else 'FAIL'}")
    for v in ("T1", "T2"):
        weak = [Z[ix[s], ix[w]] for s in S for w in eps if model[w] == v]
        print(f"     (info) same-model pairs below S-vs-{v} pairs: {np.mean([x < y for x in ss for y in weak]):.0%}")
    lab = labels_for(eps, fam, Z, PRESET["cut"])
    total, f_mcc, m_mcc = A.score(truth, lab)
    print(f"  practice score at the fixed cut z = {PRESET['cut']}: {total:.3f} (family MCC {f_mcc:.2f}, model MCC "
          f"{m_mcc:.2f}); groups: " + " | ".join(" ".join(e[-2:] for e in eps if lab[e][1] == g)
                                               for g in sorted({m for _, m in lab.values()})))
    print(f"  VERDICT: {'WORKS on practice' if all(ok) else 'NOT SHOWN TO WORK on practice'}")


def labels_for(eps, fam, Z, cut):
    letters, labels = {}, {}
    for f in sorted(set(fam), key=lambda f: min(i for i in range(len(eps)) if fam[i] == f)):
        letters[f] = chr(65 + len(letters))
    for f, L in letters.items():
        idx = [i for i in range(len(eps)) if fam[i] == f]
        D = np.maximum(Z[np.ix_(idx, idx)], 0)                     # nan = no evidence (merge skips it)
        sub = A.merge(D, cut)[0] if len(idx) > 1 else [0]
        order = {}
        for i, s in zip(idx, sub):
            order.setdefault(s, len(order) + 1)
            labels[eps[i]] = (L, f"{L}{order[s]}")
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["ref", "ep"], required=True)
    ap.add_argument("--len-probes", default=PRESET["len_probes"], help="regex of prompt ids for the length test")
    ap.add_argument("--tools-n", type=int, default=PRESET["tools_n"], help="tool-probe answers used per endpoint")
    ap.add_argument("--with-k", action="store_true", help="also use the knowledge test (needs K answers)")
    ap.add_argument("--no-tools", action="store_true")
    ap.add_argument("--tool-stat", choices=["freq", "shape"], default="freq",
                    help="freq: AgentProv choice frequencies (U); shape: temperature-robust preference order (S)")
    ap.add_argument("--families", type=int, default=None, help="force this many families")
    ap.add_argument("--cut", type=float, default=PRESET["cut"])
    ap.add_argument("--raw-z", action="store_true", help="old uncalibrated z, for comparison only")
    ap.add_argument("--round2-budget", type=int, default=30, help="stage 2: extra answers per unsure endpoint")
    ap.add_argument("--diagnose", action="store_true", help="truth-free checks on this target's own answers")
    ap.add_argument("--diag-splits", type=int, default=10, help="random splits per diagnostic check")
    a = ap.parse_args()
    global RAW_Z
    RAW_Z = a.raw_z

    rows = rows_of(a.target)
    eps = sorted({r["endpoint"] for r in rows})
    print(f"{len(rows)} saved answers, {len(eps)} endpoints")

    # behaviour tests on EVERY pair first: they decide the families as well as the models
    allpairs = list(itertools.combinations(range(len(eps)), 2))
    chans = {}
    ZL, nprompts, per = channel_L(eps, rows, a.len_probes, allpairs)
    print(f"L: length test on prompts matching {a.len_probes!r}: up to {nprompts} prompts per pair")
    if nprompts >= 3:
        chans["L"] = ZL
    if not a.no_tools:
        ZU, info = channel_U(eps, a.target, a.tools_n, allpairs, shape=a.tool_stat == "shape")
        if ZU is not None:
            nm = "S" if a.tool_stat == "shape" else "U"
            chans[nm] = ZU
            print(f"{nm}: tool probe, first {a.tools_n} answers, {a.tool_stat} statistic")
    if a.with_k:
        ZK = channel_K(eps, a.target)
        if ZK is not None:
            chans["K"] = ZK
    if not chans:
        sys.exit("no model channel has data yet (families are confirmed by the behaviour tests too)")
    # reliability gate (truth-free): a test counts only if its pair z reproduce across two halves of the data
    T = tr = None
    if "U" in chans or "S" in chans:
        import tool_probe as T
        tr = T.load(a.target, a.tools_n)
    halves = split_halves(eps, list(chans), per, tr, T, allpairs, a.diag_splits)
    rel = {c: half_reliability(halves[c], allpairs) for c in chans if c in halves}
    kept = [c for c in chans if rel.get(c, math.nan) >= MIN_RELIABILITY]
    if not kept:
        kept = [max(rel, key=lambda c: rel[c] if np.isfinite(rel[c]) else -math.inf)] if rel else list(chans)
    print("RELIABILITY GATE (half vs half rank correlation of pair z, needs >= %.2f): " % MIN_RELIABILITY +
          ", ".join(f"{c} {_fmt(rel.get(c, math.nan))} -> {'used' if c in kept else 'DROPPED'}" for c in chans))
    Z = combine([chans[c] for c in kept])                                     # nan = no evidence

    Ds = style_distance(eps, rows)
    vend = {r["endpoint"]: vendor_of(r["response"]) for r in rows if r["id"] == "F7"}
    fam, why, cand = families(Ds, Z, a.cut, a.families)
    print("\nFAMILY CANDIDATES (style proposes; behaviour must confirm: median pair z between every two groups >= cut):")
    for k, sizes, lowmed, share, ok in cand:
        print(f"  {k} groups {sizes}: lowest median z between two groups {lowmed:.1f}, {share:.0%} of cross pairs >= "
              f"{a.cut} -> {'ACCEPTED' if ok else 'rejected'}")
    print(f"FAMILIES: {why}")
    for f in sorted(set(fam)):
        print(f"  family {f}: " + " ".join(f"{e}({vend.get(e, '?')})" for e, x in zip(eps, fam) if x == f))
    print("  (self-descriptions in brackets are for information only; disguises can fake them)")
    pairs = [(i, j) for i, j in allpairs if fam[i] == fam[j]]

    truth_rows = list(csv.DictReader(open(HERE.parent / "practice_answers.csv"))) if a.target == "ref" else []
    tag = {r["endpoint"]: r["model"] for r in truth_rows} or {e: vend.get(e, "") for e in eps}
    medlen = {e: float(np.exp(np.median([v for p in per[e].values() for v in p]))) if per[e] else math.nan for e in eps}
    print("median reply length on the length prompts: " + "  ".join(f"{e[-2:]}={medlen[e]:.0f}" for e in eps))
    for f in sorted(set(fam)):
        idx = [i for i in range(len(eps)) if fam[i] == f]
        sub = [eps[i] for i in idx]
        for nm, M in list(chans.items()) + [("combined", Z)]:
            show(sub, M[np.ix_(idx, idx)], f"family {f}: channel {nm} pair z (0 = no difference)", tag)

    if a.diagnose:
        diagnose(eps, chans, Z, fam, pairs, per, a, a.diag_splits, halves, kept, rows)
    if a.target == "ref":
        preset_check(eps, Z, fam, truth_rows)
        return
    labels = labels_for(eps, fam, Z, a.cut)
    errs = A.check(labels, [f"ep-{i:02d}" for i in range(1, 34)])
    if errs:
        sys.exit("NOT written, format problems:\n  " + "\n  ".join(errs))
    for f in sorted({f for f, _ in labels.values()}):
        print(f"  family {f}: " + "  ".join(f"{e}={m}" for e, (ff, m) in sorted(labels.items()) if ff == f))
    unsure = sorted({(eps[i], eps[j], round(float(Z[i, j]), 1)) for i, j in pairs if abs(Z[i, j] - a.cut) < ROUND2_MARGIN},
                    key=lambda t: abs(t[2] - a.cut))
    print(f"pairs within {ROUND2_MARGIN} of the cut ({len(unsure)}): " +
          ", ".join(f"{x[-2:]}-{y[-2:]} {z}" for x, y, z in unsure[:20]))
    A.SUBMISSIONS.mkdir(exist_ok=True); A.write(labels, A.SUBMISSIONS / "submission_combined.csv")
    if "L" in kept:
        round2_plan(unsure, a.round2_budget)
    else:
        print("\nSTAGE 2: no automatic plan (the length test is not used on this data); decide from the diagnostics")


ROUND2_MARGIN = 1.5


def round2_plan(unsure, budget):
    """Stage 2: only endpoints in a same-family pair whose combined z lies within ROUND2_MARGIN of the cut get more
    answers, and only for the length test (it stays clean on same-model pairs however many answers it gets; more tool
    answers make disguises look different). One extra answer to each of the first `budget` V prompts."""
    eps = sorted({e for x, y, _ in unsure for e in (x, y)})
    plan = {"endpoints": eps, "probes": [f"V{i}" for i in range(1, budget + 1)], "margin": ROUND2_MARGIN,
            "why": "pairs near the cut: " + ", ".join(f"{x}-{y} {z}" for x, y, z in unsure)}
    path = A.OUT / "round2_combined.json"
    path.write_text(json.dumps(plan, indent=1))
    if eps:
        print(f"\nSTAGE 2 plan: {len(eps)} endpoints x {budget} extra answers -> {path}\n  run:  .venv/bin/python -u "
              f"probes/collect.py --target ep --plan {path.relative_to(HERE.parent)} --extra-repeats 1 --go "
              f"--approve-budget {budget}")
    else:
        print("\nSTAGE 2: no pair is near the cut; nothing more to collect")


if __name__ == "__main__":
    main()
