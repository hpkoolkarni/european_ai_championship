"""Does each endpoint choose tools the way its family's base model would (at some temperature)?
Per endpoint: fit the temperature T that best explains its choices from the base's own option log-probs; then a
goodness-of-fit z = (observed log-lik - expected log-lik under the fitted model) / sd. Exact base weights (plain,
watermark, temperature copy) -> z near 0 at their own T; LoRA / fp8 / other weights -> z clearly negative.
Also: share of choices equal to the base's favourite option.
  python analysis/analyze_toolfit.py reference_models/out        (from model_equality/)"""
import json, sys, collections, numpy as np
D = sys.argv[1]
fam = {"A": ("Qwen2.5-3B-Instruct", "01 02 04 12 14 17 18 23 24 25 33"),
       "B": ("Qwen2.5-1.5B-Instruct", "03 06 07 08 20 21 22 26 28 30 32")}
Ts = np.exp(np.linspace(np.log(0.2), np.log(4), 80))
for f, (m, s) in fam.items():
    rows = collections.defaultdict(list)
    for r in map(json.loads, open(f"{D}/toolfit_{m}.jsonl")):
        rows[r["endpoint"]].append((np.array(r["lp"]), r["chosen"]))
    print(f"\nfamily {f} (base {m}): items, fitted T, fit z (0 = consistent with base weights), share = base favourite")
    res = []
    for x in s.split():
        e = f"ep-{x}"; it = rows.get(e, [])
        if not it:
            print(f"  {x}: no readable items"); continue
        L = np.stack([a for a, _ in it]); c = np.array([b for _, b in it])
        best = None
        for T in Ts:
            z = L / T; z = z - z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
            ll = np.log(p[np.arange(len(c)), c] + 1e-300).sum()
            if best is None or ll > best[0]:
                best = (ll, T, p)
        ll, T, p = best
        lp = np.log(p + 1e-300)
        exp_i = (p * lp).sum(1); var_i = (p * lp ** 2).sum(1) - exp_i ** 2
        z = (ll - exp_i.sum()) / np.sqrt(var_i.sum() + 1e-12)
        fav = float(np.mean(c == L.argmax(1)))
        res.append((z, x, len(c), T, fav, ll / len(c)))
    for z, x, n, T, fav, avg in sorted(res, reverse=True):
        print(f"  {x}: items {n:4d}  T {T:5.2f}  fit z {z:+6.2f}  base-favourite share {fav:.2f}  mean log-lik {avg:+.3f}")
