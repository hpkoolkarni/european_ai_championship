"""E3. Predict with the fine-tuned WAM segmenter and write a submission.
Thresholds are chosen on the 510 real train images; 5-fold cross-validation gives the honest score estimate.
Two decision rules are compared by CV and the better one is used for the test set:
  plain : watermarked where p >= t[method]
  gated : the baseline's holes/patch decision (output/global_acc.json, from a_baseline.py) picks one of two
          thresholds per method, t[method, type]
Writes output/probs_wam.npz (probabilities, reused later) and output/sub_e_wam.pt; --submit uploads the file.
Run:  .venv/bin/python output/e_predict.py            (add --submit with API_KEY exported to upload)"""
import argparse
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
from e_wam_finetune import load_model, predict                       # noqa: E402
from a_baseline import best_threshold                                # noqa: E402
import submission_template as st                                     # noqa: E402

METHODS = ("pixelseal", "trustmark", "stegastamp")
GRID = np.round(np.linspace(0.05, 0.95, 19), 2)
ONES, ZEROS = np.ones((128, 128), np.uint8), np.zeros((128, 128), np.uint8)


def fit(groups, table, rows):
    """Best threshold index per group, using only the given rows of the (images x thresholds) score table."""
    return {g: int(np.argmax(table[[i for i in rows if groups[i] == g]].mean(0)))
            for g in set(groups[i] for i in rows)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--submit", action="store_true"); a = ap.parse_args()
    model = load_model(OUT / "ckpt" / "wam_ft.pt")
    tr, te = LocalisationDataset(TASK / "release", "train"), LocalisationDataset(TASK / "release", "test")
    tr_items = [tr[i] for i in range(len(tr))]
    p_tr = predict(model, [it["image"] for it in tr_items])
    p_te = np.concatenate([predict(model, [te[i]["image"] for i in range(j, min(j + 64, len(te)))])
                           for j in range(0, len(te), 64)])
    np.savez(OUT / "probs_wam.npz", names_train=[it["image_name"] for it in tr_items], p_train=p_tr.astype(np.float16),
             names_test=[m["image_name"] for m in te.meta], p_test=p_te.astype(np.float16))

    gts = [it["mask128"] for it in tr_items]
    table = np.array([[miou_image((p >= t).astype(np.uint8), g) for t in GRID] for p, g in zip(p_tr, gts)])
    meth_tr = [it["method"] for it in tr_items]; meth_te = [m["method"] for m in te.meta]
    rules = {"plain": (meth_tr, meth_te)}

    acc_path = OUT / "global_acc.json"
    if acc_path.exists():                                         # holes/patch type from the baseline's decoder check
        acc = json.load(open(acc_path))
        thr_a = {}
        for meth in METHODS:
            idx = [i for i in range(len(tr_items)) if meth_tr[i] == meth]
            ac = np.array([acc["train"][tr_items[i]["image_name"]] for i in idx])
            s1 = np.array([miou_image(ONES, gts[i]) for i in idx]); s0 = np.array([miou_image(ZEROS, gts[i]) for i in idx])
            thr_a[meth] = best_threshold(ac, s1, s0)
        typ = lambda name, meth, split: "holes" if acc[split][name] >= thr_a[meth] else "patch"   # noqa: E731
        rules["gated"] = ([(meth_tr[i], typ(tr_items[i]["image_name"], meth_tr[i], "train")) for i in range(len(tr_items))],
                          [(m["method"], typ(m["image_name"], m["method"], "test")) for m in te.meta])
    else:
        print("output/global_acc.json not found (run a_baseline.py first): only the plain rule is tried")

    fold = np.random.default_rng(0).permutation(len(tr_items)) % 5
    cv = {}
    for name, (g_tr, _) in rules.items():
        held = []
        for k in range(5):
            t = fit(g_tr, table, [i for i in range(len(fold)) if fold[i] != k])
            held += [table[i, t.get(g_tr[i], 9)] for i in range(len(fold)) if fold[i] == k]
        cv[name] = float(np.mean(held))
        full = fit(g_tr, table, range(len(fold)))
        print(f"rule {name:5s}: 5-fold CV score={cv[name]:.4f}  thresholds={ {str(g): float(GRID[i]) for g, i in sorted(full.items(), key=str)} }")
    rule = max(cv, key=cv.get)
    g_tr, g_te = rules[rule]
    t = fit(g_tr, table, range(len(g_tr)))
    masks = np.stack([(p >= GRID[t[g]]).astype(np.uint8) for p, g in zip(p_te, g_te)])
    path = OUT / "sub_e_wam.pt"
    torch.save({"image_names": [m["image_name"] for m in te.meta], "masks": torch.from_numpy(masks).bool()}, path)
    print(f"using rule '{rule}' (CV {cv[rule]:.4f}); wrote {path}: {len(masks)} masks, mean watermarked share={masks.mean():.3f}")
    if a.submit:
        st.BASE_URL, st.API_KEY = "http://34.147.229.55", os.environ["API_KEY"]
        st.submit(str(path))


if __name__ == "__main__":
    main()
