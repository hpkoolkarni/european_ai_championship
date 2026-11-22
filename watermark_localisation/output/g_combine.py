"""G. Combine everything into one submission. For each watermark method, every candidate is scored by 5-fold
cross-validation on the real train images and the best one is used for that method's test images:
  baseline      all ones / all zeros from the holes-or-patch decision (output/global_acc.json)
  <model>       a trained model's probabilities (output/probs_wam.npz, output/probs_f_*.npz) with the gated rule:
                the holes-or-patch decision picks one of two thresholds; thresholds 0.0 and 1.01 are allowed, so a
                model can never do worse in CV than falling back to the baseline answer
Prints a table of CV scores per method and candidate, writes output/sub_g_best.pt; --submit uploads it.
Run:  .venv/bin/python output/g_combine.py"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

TASK = Path(__file__).resolve().parents[1]
OUT = TASK / "output"
sys.path.insert(0, str(TASK)); sys.path.insert(0, str(OUT))
from wmloc.data import LocalisationDataset                          # noqa: E402
from wmloc.utils import miou_image                                   # noqa: E402
from a_baseline import best_threshold                                # noqa: E402
import submission_template as st                                     # noqa: E402

METHODS = ("pixelseal", "trustmark", "stegastamp")
GRID = np.concatenate([[0.0], np.round(np.linspace(0.05, 0.95, 19), 2), [1.01]])
ONES, ZEROS = np.ones((128, 128), np.uint8), np.zeros((128, 128), np.uint8)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--submit", action="store_true"); a = ap.parse_args()
    tr, te = LocalisationDataset(TASK / "release", "train"), LocalisationDataset(TASK / "release", "test")
    gt = {m["image_name"]: tr[i]["mask128"] for i, m in enumerate(tr.meta)}
    acc = json.load(open(OUT / "global_acc.json"))

    typ = {}                                                     # holes / patch decision, as in a_baseline.py
    for meth in METHODS:
        names = [m["image_name"] for m in tr.meta if m["method"] == meth]
        ac = np.array([acc["train"][n] for n in names])
        t = best_threshold(ac, np.array([miou_image(ONES, gt[n]) for n in names]),
                           np.array([miou_image(ZEROS, gt[n]) for n in names]))
        for split, meta in (("train", tr.meta), ("test", te.meta)):
            for m in meta:
                if m["method"] == meth:
                    typ[m["image_name"]] = acc[split][m["image_name"]] >= t

    sources = {}
    for f in [OUT / "probs_wam.npz"] + [Path(p) for p in sorted(glob.glob(str(OUT / "probs_f_*.npz")))]:
        if f.exists():
            d = np.load(f)
            sources[f.stem.replace("probs_", "")] = {**dict(zip(d["names_train"].tolist(), d["p_train"].astype(np.float32))),
                                                    **dict(zip(d["names_test"].tolist(), d["p_test"].astype(np.float32)))}
    print("models found:", list(sources))

    final, rows = {}, []
    for meth in METHODS:
        names = [m["image_name"] for m in tr.meta if m["method"] == meth]
        fold = np.random.default_rng(0).permutation(len(names)) % 5
        holes = np.array([typ[n] for n in names])
        cands = {"baseline": None, **{k: v for k, v in sources.items() if all(n in v for n in names)}}
        cv = {}
        for cname, P in cands.items():
            if P is None:
                table = np.array([[miou_image(ONES if typ[n] else ZEROS, gt[n])] * len(GRID) for n in names])
            else:
                table = np.array([[miou_image((P[n] >= t).astype(np.uint8), gt[n]) for t in GRID] for n in names])
            held = np.zeros(len(names))
            for k in range(5):
                for h in (True, False):
                    fit = (fold != k) & (holes == h); ev = (fold == k) & (holes == h)
                    if ev.any():
                        held[ev] = table[ev, int(np.argmax(table[fit].mean(0))) if fit.any() else 0]
            thr = {h: GRID[int(np.argmax(table[holes == h].mean(0)))] for h in (True, False) if (holes == h).any()}
            cv[cname] = (held.mean(), thr)
        best = max(cv, key=lambda c: cv[c][0])
        final[meth] = (best, cv[best][1])
        for cname, (s, thr) in sorted(cv.items(), key=lambda kv: -kv[1][0]):
            rows.append(f"{meth:10s} {cname:12s} CV={s:.4f}  thresholds holes={thr.get(True)} patch={thr.get(False)}" + ("   <== used" if cname == best else ""))
    print("\n".join(rows))

    names, masks = [], []
    for m in te.meta:
        cname, thr = final[m["method"]]; n = m["image_name"]
        names.append(n)
        masks.append((ONES if typ[n] else ZEROS) if cname == "baseline" else (sources[cname][n] >= thr[typ[n]]).astype(np.uint8))
    path = OUT / "sub_g_best.pt"
    torch.save({"image_names": names, "masks": torch.from_numpy(np.stack(masks)).bool()}, path)
    print(f"wrote {path}: {len(names)} masks; choice per method: { {k: v[0] for k, v in final.items()} }")
    if a.submit:
        st.BASE_URL, st.API_KEY = "http://34.147.229.55", os.environ["API_KEY"]
        st.submit(str(path))


if __name__ == "__main__":
    main()
