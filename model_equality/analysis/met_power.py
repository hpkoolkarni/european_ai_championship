"""Power of the MET Hamming-kernel MMD test (Gao et al.) on their own Llama-3-8B data, at OUR budget.
Two 'endpoints' each get `per` samples on each of P prompts (total per endpoint = P*per). Null: fp16 vs fp16 (two
disjoint draws). Alternatives: fp16 vs int8 / nf4 / watermark. Permutation test within prompts, alpha 0.05.
Pickles are loaded with an unpickler that refuses every class (files hold only lists of ints)."""
import glob, io, pickle, re, sys
import numpy as np

D = sys.argv[1]; L = 50


class NoClasses(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refused class {module}.{name}")


data = {}
for f in glob.glob(D + "/samples/*.pkl"):
    m = re.search(r"wikipedia_en-(fp16|int8|nf4|watermark)-L=50-(\d+)\.pkl$", f)
    rows = NoClasses(io.BytesIO(open(f, "rb").read())).load()
    X = np.full((len(rows), L), -1, np.int64)
    for i, r in enumerate(rows):
        r = r[:L]; X[i, :len(r)] = r
    data[(m.group(1), int(m.group(2)))] = X
prompts = sorted({p for _, p in data})
print(f"{len(prompts)} prompts, samples per prompt and version: {min(len(v) for v in data.values())}")


def mmd(A, B):                       # A, B: lists over prompts of (n, L) arrays; Hamming similarity kernel
    s = 0.0
    for a, b in zip(A, B):
        kaa = (a[:, None] == a[None]).mean(2); kbb = (b[:, None] == b[None]).mean(2); kab = (a[:, None] == b[None]).mean(2)
        na, nb = len(a), len(b)
        s += (kaa.sum() - np.trace(kaa)) / (na * (na - 1)) + (kbb.sum() - np.trace(kbb)) / (nb * (nb - 1)) - 2 * kab.mean()
    return s / len(A)


def test(A, B, rng, perms=200):
    t = mmd(A, B); c = 0
    for _ in range(perms):
        A2, B2 = [], []
        for a, b in zip(A, B):
            pool = np.concatenate([a, b]); i = rng.permutation(len(pool)); A2.append(pool[i[:len(a)]]); B2.append(pool[i[len(a):]])
        c += mmd(A2, B2) >= t
    return (c + 1) / (perms + 1)


rng = np.random.default_rng(0)
for P, per in [(4, 10), (10, 4), (20, 2), (10, 10), (20, 5)]:
    res = {}
    for alt in ("fp16", "int8", "nf4", "watermark"):
        rej = 0; trials = 40
        for _ in range(trials):
            ps = rng.choice(prompts, P, replace=False); A, B = [], []
            for p in ps:
                X = data[("fp16", p)]; i = rng.permutation(len(X))
                A.append(X[i[:per]])
                Y = X[i[per:2 * per]] if alt == "fp16" else data[(alt, p)][rng.permutation(len(data[(alt, p)]))[:per]]
                B.append(Y)
            rej += test(A, B, rng) <= 0.05
        res[alt] = rej / trials
    print(f"budget {P*per:3d} per endpoint ({P} prompts x {per}): detected  fp16-vs-fp16 (false alarm) {res['fp16']:.0%}"
          f" | int8 {res['int8']:.0%} | nf4 {res['nf4']:.0%} | watermark {res['watermark']:.0%}", flush=True)
