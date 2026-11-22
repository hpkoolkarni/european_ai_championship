"""A. Baseline: classify each image as 'holes' (watermark almost everywhere) or 'patch' (small watermarked patch)
from the whole-image bit accuracy, then predict all ones (holes) or all zeros (patch).
The threshold per method is fitted on train; 5-fold cross-validation reports an honest estimate.
Run from the task folder:
    .venv/bin/python output/a_baseline.py              writes output/sub_a_baseline.pt and output/global_acc.json
    API_KEY=... .venv/bin/python output/a_baseline.py --submit     also uploads the file"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK))
from wmloc.data import LocalisationDataset                       # noqa: E402
from wmloc.utils import bit_accuracy, miou_image, to_tensor       # noqa: E402
from wmloc.watermarks import load_watermark                       # noqa: E402
import submission_template as st                                  # noqa: E402

METHODS = ("pixelseal", "trustmark", "stegastamp")
OUT = TASK / "output"
ONES, ZEROS = np.ones((128, 128), np.uint8), np.zeros((128, 128), np.uint8)


def global_accuracy(ds, device):
    """Bit accuracy of the whole-image decode against the known message, per image name."""
    acc = {}
    for meth in METHODS:
        wm = load_watermark(meth, models_dir=TASK / "models", device=device)
        for i, m in enumerate(ds.meta):
            if m["method"] == meth:
                it = ds[i]
                bits = wm.decode(to_tensor(it["image"]))[0].cpu().numpy()
                acc[it["image_name"]] = bit_accuracy(bits, it["message"])
        print(f"  decoded {meth}", flush=True)
    return acc


def best_threshold(acc, s_ones, s_zeros):
    """Threshold t maximising mean score of: all ones if acc >= t else all zeros."""
    cands = np.unique(np.concatenate([[0.0, 1.01], (np.sort(acc)[1:] + np.sort(acc)[:-1]) / 2]))
    scores = [np.mean(np.where(acc >= t, s_ones, s_zeros)) for t in cands]
    return float(cands[int(np.argmax(scores))])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--submit", action="store_true"); a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, test = LocalisationDataset(TASK / "release", "train"), LocalisationDataset(TASK / "release", "test")
    print("decoding train"); acc_tr = global_accuracy(train, device)
    print("decoding test"); acc_te = global_accuracy(test, device)
    json.dump({"train": acc_tr, "test": acc_te}, open(OUT / "global_acc.json", "w"))

    thr, cv_scores = {}, []
    rng = np.random.default_rng(0)
    for meth in METHODS:
        idx = [i for i, m in enumerate(train.meta) if m["method"] == meth]
        gts = [train[i]["mask128"] for i in idx]
        acc = np.array([acc_tr[train.meta[i]["image_name"]] for i in idx])
        s1 = np.array([miou_image(ONES, g) for g in gts]); s0 = np.array([miou_image(ZEROS, g) for g in gts])
        thr[meth] = best_threshold(acc, s1, s0)
        fold = rng.permutation(len(idx)) % 5                     # 5-fold CV estimate
        for k in range(5):
            t = best_threshold(acc[fold != k], s1[fold != k], s0[fold != k])
            cv_scores += list(np.where(acc[fold == k] >= t, s1[fold == k], s0[fold == k]))
        fit = np.mean(np.where(acc >= thr[meth], s1, s0))
        print(f"{meth:10s} threshold={thr[meth]:.3f}  train score={fit:.3f}  oracle={np.mean(np.maximum(s1, s0)):.3f}")
    print(f"5-fold CV score (all methods): {np.mean(cv_scores):.4f}")

    names, masks = [], []
    for m in test.meta:
        names.append(m["image_name"])
        masks.append(ONES if acc_te[m["image_name"]] >= thr[m["method"]] else ZEROS)
    path = OUT / "sub_a_baseline.pt"
    torch.save({"image_names": names, "masks": torch.from_numpy(np.stack(masks)).bool()}, path)
    print(f"wrote {path}: {len(names)} masks, predicted holes share={np.mean([k.mean() for k in masks]):.2f}")
    if a.submit:
        st.BASE_URL, st.API_KEY = "http://34.147.229.55", os.environ["API_KEY"]
        st.submit(str(path))


if __name__ == "__main__":
    main()
