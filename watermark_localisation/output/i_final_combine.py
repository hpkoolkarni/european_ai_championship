"""I. Final submission: per-method ensemble selection + known-message box search.
For each watermark method:
  1. greedy forward ensemble selection with replacement (Caruana et al. 2004) over the real-image fine-tuned models in
     POOL: start with the best single model, keep adding the model that raises the 5-fold CV mIoU on the real train
     images the most (out-of-fold predictions, per-type thresholds as in g_combine.py), stop when the gain is < 0.0015
  2. the selected models' probabilities are averaged; h_box_search.run_combine then adds the decoder-confirmed boxes
     from output/<boxdir> (TrustMark, StegaStamp patch images), with acceptance level and rule chosen by CV
Writes output/<out> and prints the honest CV per method. Nothing is uploaded (see submission_template.py for that).
Run:  .venv/bin/python output/i_final_combine.py --boxdir box2 --out sub_final.pt"""
import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parents[1]
OUT = TASK / "output"
sys.path.insert(0, str(TASK)); sys.path.insert(0, str(OUT))
import h_box_search as H                                             # noqa: E402
from a_baseline import best_threshold                                # noqa: E402
from wmloc.data import LocalisationDataset                           # noqa: E402
from wmloc.utils import miou_image                                   # noqa: E402

POOL = {"pixelseal": ["ps_v4_real", "ps_spec_real", "psr_v5_real", "ps_v4_ft2", "ps_v4_r2", "psm_v4_real", "ps_spec_r2"],
        "trustmark": ["tm_spec_real", "tmm_v4_real", "tmr_v5_real", "tm_spec_ft2", "tss_v4_real", "tmm_v4_r2", "tmr_v5_r2"],
        "stegastamp": ["ss_v3_real", "ss_v3_ft2", "ss_v3_r2", "ss_v3_r3"]}
GRID = np.concatenate([[0.0], np.round(np.linspace(0.05, 0.95, 19), 2), [1.01]])
ONES, ZEROS = np.ones((128, 128), np.uint8), np.zeros((128, 128), np.uint8)


def select(meth, pool, tr, gt, acc, min_gain=0.0015, max_add=6):
    names = [m["image_name"] for m in tr.meta if m["method"] == meth]
    ac = np.array([acc["train"][n] for n in names])
    t = best_threshold(ac, np.array([miou_image(ONES, gt[n]) for n in names]), np.array([miou_image(ZEROS, gt[n]) for n in names]))
    holes = ac >= t; fold = np.random.default_rng(0).permutation(len(names)) % 5; G = [gt[n] for n in names]

    def cv(P):
        tab = np.array([[miou_image((p >= g).astype(np.uint8), gg) for g in GRID] for p, gg in zip(P, G)])
        h = np.zeros(len(names))
        for k in range(5):
            for b in (True, False):
                fit = (fold != k) & (holes == b); ev = (fold == k) & (holes == b)
                if ev.any():
                    h[ev] = tab[ev, int(np.argmax(tab[fit].mean(0)))]
        return h.mean()

    avail = [k for k in pool if (OUT / f"probs_f_{k}.npz").exists()]
    C = {}
    for k in avail:
        d = np.load(OUT / f"probs_f_{k}.npz"); P = dict(zip(d["names_train"].tolist(), d["p_train"].astype(np.float32)))
        C[k] = np.stack([P[n] for n in names])
    single = {k: cv(C[k]) for k in avail}
    print(f"=== {meth}: " + ", ".join(f"{k} {s:.3f}" for k, s in sorted(single.items(), key=lambda x: -x[1])), flush=True)
    sel = [max(single, key=single.get)]; cur = single[sel[0]]; S = C[sel[0]].copy()
    for _ in range(max_add):
        best = max(avail, key=lambda k: cv((S + C[k]) / (len(sel) + 1)))
        new = cv((S + C[best]) / (len(sel) + 1))
        if new <= cur + min_gain:
            break
        sel.append(best); S += C[best]; cur = new
        print(f"   + {best:14s} -> CV {cur:.4f}", flush=True)
    print(f"   selected for {meth}: {sel}  (model-only CV {cur:.4f})", flush=True)
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boxdir", default="box2", help="folder under output/ with the box-search results")
    ap.add_argument("--out", default="sub_final.pt", help="submission file name under output/")
    a = ap.parse_args()
    tr = LocalisationDataset(TASK / "release", "train")
    gt = {m["image_name"]: tr[i]["mask128"] for i, m in enumerate(tr.meta)}
    acc = json.load(open(OUT / "global_acc.json"))
    for meth, pool in POOL.items():
        H.MODELS[meth] = select(meth, pool, tr, gt, acc)            # repeats in the list act as weights
    H.run_combine(types.SimpleNamespace(boxdir=a.boxdir, out=a.out))


if __name__ == "__main__":
    main()
