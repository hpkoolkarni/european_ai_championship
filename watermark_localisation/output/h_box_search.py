"""H. Known-message box search for patch images (uses the message given in the metadata of every image).
Check on the real train patches: the official TrustMark decoder reads the known message at ~100 % bit accuracy from the
exact watermarked rectangle, still at ~97 % when its edges are up to 10 % off, and at chance (~51 %, never > 90 %) from
a same-size box anywhere else. StegaStamp works the same way within ~2-5 %. So a box that decodes the known message
IS the watermarked rectangle.
  search (GPU):  for every image the holes-or-patch decision calls "patch" (as in g_combine.py), take up to 4 region
                 proposals from the averaged model probabilities (connected regions at 3 thresholds), search each
                 proposal's box (coarse: every edge moved by -20..+20 % of the box size, 625 boxes; then edge-by-edge
                 refinement in 16 / 8 / 4 / 2 px steps), scoring each box by how strongly the decoder reads the known
                 message. Writes output/box/<method>_<split>_<shard>.npz (best box, bit accuracy and margin per proposal).
  --combine (CPU): for TrustMark and StegaStamp, choose by 5-fold CV on the real train images the acceptance level
                 (bit accuracy) and the rule (accepted boxes replace the model's mask, or are added to it); accepted
                 boxes are used only where the decoder confirms them. Writes output/sub_h_box.pt. Nothing is uploaded.
Run:  .venv/bin/python output/h_box_search.py --method trustmark --split both
      .venv/bin/python output/h_box_search.py --combine"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

TASK = Path(__file__).resolve().parents[1]
OUT = TASK / "output"
sys.path.insert(0, str(TASK)); sys.path.insert(0, str(OUT))
from wmloc.data import LocalisationDataset                          # noqa: E402
from wmloc.utils import downsample_mask, miou_image                 # noqa: E402
from a_baseline import best_threshold                                # noqa: E402

METHODS = ("pixelseal", "trustmark", "stegastamp")
MODELS = {"pixelseal": ["ps_v4_real", "ps_spec_real"], "trustmark": ["tm_spec_real"], "stegastamp": ["ss_v3_real"]}  # = submission v3
RES = {"trustmark": (245, "bilinear"), "stegastamp": (400, "bicubic")}    # as the decoder wrappers resize
MAX_ASPECT = {"trustmark": 2.2, "stegastamp": 16}                  # brief: TrustMark patches have aspect <= 2
GRID = np.concatenate([[0.0], np.round(np.linspace(0.05, 0.95, 19), 2), [1.01]])
ONES, ZEROS = np.ones((128, 128), np.uint8), np.zeros((128, 128), np.uint8)
S = 512


def averaged_probs(models):
    P = {}
    for k in models:
        d = np.load(OUT / f"probs_f_{k}.npz")
        for n, p in list(zip(d["names_train"].tolist(), d["p_train"])) + list(zip(d["names_test"].tolist(), d["p_test"])):
            P[n] = P.get(n, 0) + p.astype(np.float32) / len(models)
    return P


def holes_decision(tr, te, meth):
    """Holes-or-patch decision for every train and test image of a method, exactly as in g_combine.py."""
    acc = json.load(open(OUT / "global_acc.json"))
    gt = {m["image_name"]: tr[i]["mask128"] for i, m in enumerate(tr.meta) if m["method"] == meth}
    names = list(gt)
    ac = np.array([acc["train"][n] for n in names])
    t = best_threshold(ac, np.array([miou_image(ONES, gt[n]) for n in names]), np.array([miou_image(ZEROS, gt[n]) for n in names]))
    out = {n: acc["train"][n] >= t for n in names}
    out.update({m["image_name"]: acc["test"][m["image_name"]] >= t for m in te.meta if m["method"] == meth})
    return out


def proposals(p, k=4, levels=(0.15, 0.3, 0.5)):
    """Up to k boxes (y0, y1, x0, x1 in 512 px) around connected high-probability regions, strongest first."""
    cand = []
    for t in levels:
        lab, n = ndimage.label(p >= t, structure=np.ones((3, 3)))
        for j in range(1, n + 1):
            ys, xs = np.where(lab == j)
            if len(ys) >= 4:
                cand.append((float(p[ys, xs].sum()), (ys.min() * 4, (ys.max() + 1) * 4, xs.min() * 4, (xs.max() + 1) * 4)))
    cand.sort(key=lambda c: -c[0]); out = []
    for _, b in cand:
        if all(box_iou(b, o) < 0.8 for o in out):
            out.append(b)
        if len(out) == k:
            break
    return out


def box_iou(a, b):
    iy = max(0, min(a[1], b[1]) - max(a[0], b[0])); ix = max(0, min(a[3], b[3]) - max(a[2], b[2]))
    inter = iy * ix; ua = (a[1] - a[0]) * (a[3] - a[2]) + (b[1] - b[0]) * (b[3] - b[2]) - inter
    return inter / max(ua, 1)


class Scorer:
    """Bit accuracy and margin (mean logit x sign of the known bit) of the decoder on many boxes of one image."""

    def __init__(self, method, device):
        from wmloc.watermarks import load_watermark
        self.method, self.dev = method, device
        self.wm = load_watermark(method, models_dir=TASK / "models", device=device)
        self.res, self.mode = RES[method]

    def __call__(self, img_t, boxes, msg):
        """img_t: 1 x 3 x 512 x 512 float on the device; boxes: list of (y0, y1, x0, x1); msg: +-1 tensor."""
        acc, mar = [], []
        for i in range(0, len(boxes), 128):
            crops = [F.interpolate(img_t[:, :, y0:y1, x0:x1], size=(self.res,) * 2, mode=self.mode, align_corners=False,
                                   antialias=True) for y0, y1, x0, x1 in boxes[i:i + 128]]
            x = torch.cat(crops).clamp(0, 1)
            with torch.no_grad():
                z = self.wm.logits(x).to(self.dev).float()[:, :len(msg)]
            acc.append(((z > 0).float() * 2 - 1 == msg).float().mean(1).cpu()); mar.append((z * msg).mean(1).cpu())
        return torch.cat(acc).numpy(), torch.cat(mar).numpy()


def valid(b, method):
    y0, y1, x0, x1 = b; h, w = y1 - y0, x1 - x0
    return 0 <= y0 and y1 <= S and 0 <= x0 and x1 <= S and h >= 12 and w >= 12 and max(h, w) / min(h, w) <= MAX_ASPECT[method]


def search(score, img_t, msg, box, method, coarse=(-0.2, -0.1, 0.0, 0.1, 0.2)):
    y0, y1, x0, x1 = box; h, w = y1 - y0, x1 - x0
    cands = {(int(round(y0 + a * h)), int(round(y1 + b * h)), int(round(x0 + c * w)), int(round(x1 + d * w)))
             for a in coarse for b in coarse for c in coarse for d in coarse}
    cands = [b for b in cands if valid(b, method)]
    if not cands:
        return None
    acc, mar = score(img_t, cands, msg); j = int(np.argmax(mar)); best, bm, ba = cands[j], mar[j], acc[j]
    for step in (16, 8, 4, 2):
        improved = True
        while improved:
            improved = False
            nb = [tuple(v + s * (e == k) for e, v in enumerate(best)) for k in range(4) for s in (-step, step)]
            nb = [b for b in nb if valid(b, method)]
            if not nb:
                break
            acc, mar = score(img_t, nb, msg); j = int(np.argmax(mar))
            if mar[j] > bm:
                best, bm, ba, improved = nb[j], mar[j], acc[j], True
    return best, float(ba), float(bm)


def run_search(a):
    dev = "cuda"
    tr, te = LocalisationDataset(TASK / "release", "train"), LocalisationDataset(TASK / "release", "test")
    hol = holes_decision(tr, te, a.method); P = averaged_probs(MODELS[a.method]); score = Scorer(a.method, dev)
    (OUT / a.boxdir).mkdir(exist_ok=True)
    extra = [averaged_probs([m]) for m in (a.prop_models or [])]   # each model's own map: other regions light up
    for split, ds in (("train", tr), ("test", te)):
        if a.split not in (split, "both"):
            continue
        idx = [i for i, m in enumerate(ds.meta) if m["method"] == a.method and not hol[m["image_name"]]][a.shard::a.nshards]
        names, boxes, accs, mars, t0 = [], [], [], [], time.time()
        for c, i in enumerate(idx):
            it = ds[i]; n = it["image_name"]
            img_t = torch.from_numpy(it["image"]).permute(2, 0, 1)[None].float().div(255).to(dev)
            msg = torch.from_numpy(it["message"].astype(np.float32) * 2 - 1).to(dev)
            props = proposals(P[n])
            for Q in extra:                                       # more candidates, also from weak responses
                for b in proposals(Q[n], k=a.kprop, levels=(0.08, 0.15, 0.3)):
                    if len(props) < a.maxprop and all(box_iou(b, o) < 0.8 for o in props):
                        props.append(b)
            K = max(4, a.maxprop if extra else 4)
            res = [search(score, img_t, msg, b, a.method) for b in props]
            res = [r for r in res if r is not None] + [((0, 0, 0, 0), 0.0, -1e9)] * K
            names.append(n); boxes.append([r[0] for r in res[:K]]); accs.append([r[1] for r in res[:K]]); mars.append([r[2] for r in res[:K]])
            if (c + 1) % 10 == 0:
                print(f"{a.method} {split} shard {a.shard}: {c + 1}/{len(idx)}  best bit-acc so far median "
                      f"{np.median([max(x) for x in accs]):.3f}  {time.time() - t0:.0f}s", flush=True)
        np.savez(OUT / a.boxdir / f"{a.method}_{split}_{a.shard}.npz", names=names, boxes=np.array(boxes, np.int32),
                 accs=np.array(accs, np.float32), margins=np.array(mars, np.float32))
        print(f"wrote output/{a.boxdir}/{a.method}_{split}_{a.shard}.npz ({len(names)} images, {time.time() - t0:.0f}s)", flush=True)


def box_masks(boxes, accs, level):
    """128-grid mask of the union of the accepted, non-overlapping boxes (None if no box reaches the level)."""
    keep = []
    for b, ac in sorted(zip(boxes, accs), key=lambda t: -t[1]):
        if ac >= level and all(box_iou(tuple(b), tuple(k)) < 0.3 for k in keep):
            keep.append(b)
    if not keep:
        return None
    m = np.zeros((S, S), np.uint8)
    for y0, y1, x0, x1 in keep:
        m[y0:y1, x0:x1] = 1
    return downsample_mask(m)


def run_combine(a):
    tr, te = LocalisationDataset(TASK / "release", "train"), LocalisationDataset(TASK / "release", "test")
    gt = {m["image_name"]: tr[i]["mask128"] for i, m in enumerate(tr.meta)}
    final, total = {}, []
    for meth in METHODS:
        hol = holes_decision(tr, te, meth); P = averaged_probs(MODELS[meth])
        names = [m["image_name"] for m in tr.meta if m["method"] == meth]; holes = np.array([hol[n] for n in names])
        BX = {}
        for f in glob.glob(str(OUT / a.boxdir / f"{meth}_*_*.npz")):
            d = np.load(f)
            BX.update({n: (b, ac) for n, b, ac in zip(d["names"].tolist(), d["boxes"], d["accs"])})
        rules = [("model", None)] + ([(f"{mode} >= {lv}", (mode, lv)) for mode in ("replace", "add")
                                      for lv in (0.75, 0.8, 0.85, 0.9, 0.95)] if BX else [])

        def mask(n, g, rule, h):
            m = (P[n] >= g).astype(np.uint8)
            if rule is None or h or n not in BX:
                return m
            bm = box_masks(*BX[n], rule[1])
            return m if bm is None else (bm if rule[0] == "replace" else np.maximum(bm, m))

        fold = np.random.default_rng(0).permutation(len(names)) % 5; cv = {}
        for rname, rule in rules:
            tab = np.array([[miou_image(mask(n, g, rule, h), gt[n]) for g in GRID] for n, h in zip(names, holes)])
            held = np.zeros(len(names))
            for k in range(5):
                for b in (True, False):
                    fit = (fold != k) & (holes == b); ev = (fold == k) & (holes == b)
                    if ev.any():
                        held[ev] = tab[ev, int(np.argmax(tab[fit].mean(0)))]
            thr = {b: GRID[int(np.argmax(tab[holes == b].mean(0)))] for b in (True, False)}
            cv[rname] = (held.mean(), thr, rule, held)
        # the rule itself is also chosen by CV: per fold, the rule best on the other folds
        rnames = list(cv); held_r = np.zeros(len(names))
        for k in range(5):
            best_r = max(rnames, key=lambda r: cv[r][3][fold != k].mean()); held_r[fold == k] = cv[best_r][3][fold == k]
        best = max(rnames, key=lambda r: cv[r][0])
        for r in sorted(rnames, key=lambda r: -cv[r][0])[:4]:
            print(f"{meth:10s} {r:16s} CV={cv[r][0]:.4f}" + ("   <== used" if r == best else ""))
        print(f"{meth:10s} honest CV with the rule also chosen per fold: {held_r.mean():.4f}   (model only: {cv['model'][0]:.4f})")
        total.append(held_r); final[meth] = (cv[best][1], cv[best][2], hol, P, BX)
        for m in te.meta:
            if m["method"] == meth:
                n = m["image_name"]; thr, rule = final[meth][0], final[meth][1]
                final[n] = mask(n, thr[hol[n]], rule, hol[n])
    print(f"overall honest CV: {np.concatenate(total).mean():.4f}")
    names = [m["image_name"] for m in te.meta]
    torch.save({"image_names": names, "masks": torch.from_numpy(np.stack([final[n] for n in names])).bool()}, OUT / a.out)
    print(f"wrote output/{a.out} ({len(names)} masks)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=("trustmark", "stegastamp"))
    ap.add_argument("--split", default="both", choices=("train", "test", "both"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--combine", action="store_true")
    ap.add_argument("--models", nargs="+", help="override the averaged models for --method (and for --combine)")
    ap.add_argument("--prop-models", nargs="+", help="extra models whose own maps give more candidate boxes")
    ap.add_argument("--kprop", type=int, default=3, help="candidates per extra model")
    ap.add_argument("--maxprop", type=int, default=12, help="max candidates per image")
    ap.add_argument("--boxdir", default="box", help="folder under output/ for the search results")
    ap.add_argument("--out", default="sub_h_box.pt", help="--combine: submission file name under output/")
    a = ap.parse_args()
    if a.models and a.method:
        MODELS[a.method] = a.models
    run_combine(a) if a.combine else run_search(a)


if __name__ == "__main__":
    main()
