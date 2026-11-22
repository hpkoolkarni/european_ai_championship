"""E1. Synthetic training data for the localisation segmenter, made with the task's own embedders.
Each sample: a COCO photo (random square crop, resized to 512 x 512), one of the three methods, a random message,
and one of four recipes that mirror the task brief:
  holes      (40 %) whole photo watermarked, then original pixels pasted back inside 1-3 COCO objects
  outpaint   (10 %) a large rectangle watermarked at its own size, the rest of the photo clean
  patch      (25 %) a small rectangle watermarked at its own size (in place, or carried over from another photo)
  transplant (25 %) a rectangle of another photo watermarked at its own size; only an object inside it is pasted
Labels follow the official rule: 512 x 512 mask -> 128 x 128 grid with wmloc.utils.downsample_mask.
Writes output/synth/shard_<k>.npz with images (N,512,512,3 uint8), masks (N,128,128 uint8), method, recipe.
Run one process per GPU (run_e_synth.sbatch does this):  .venv/bin/python output/e_synth.py --nshards 4 --n 1500"""
import argparse
import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK))
from wmloc.utils import downsample_mask                           # noqa: E402
from wmloc.watermarks import load_watermark                       # noqa: E402

OUT = TASK / "output"
COCO = OUT / "data" / "coco"
METHODS = ("pixelseal", "trustmark", "stegastamp")
ASPECT = {"pixelseal": 16, "trustmark": 2, "stegastamp": 16}      # max aspect ratio of the watermarked rectangle (brief)
RECIPES = ("holes", "outpaint", "patch", "transplant")
RECIPE_P = (0.40, 0.10, 0.25, 0.25)
S = 512


def load_coco():
    d = json.loads((COCO / "instances_val2017.json").read_text())
    objs = {}
    for a in d["annotations"]:
        if a["iscrowd"] == 0 and isinstance(a["segmentation"], list):
            objs.setdefault(a["image_id"], []).append(a["segmentation"])
    return sorted(d["images"], key=lambda im: im["id"]), objs


_TAR = {}


def open_photo(file_name):
    """COCO photo from output/data/coco/val2017.tar (one file instead of 5,000: saves inodes on scratch),
    or from the unpacked val2017/ folder if the tar does not exist."""
    tar_path = COCO / "val2017.tar"
    if tar_path.exists():
        if "tf" not in _TAR:
            _TAR["tf"] = tarfile.open(tar_path)
            _TAR["idx"] = {Path(m.name).name: m for m in _TAR["tf"].getmembers() if m.isfile()}
        return Image.open(io.BytesIO(_TAR["tf"].extractfile(_TAR["idx"][file_name]).read()))
    return Image.open(COCO / "val2017" / file_name)


def load_photo(info, objs, rng):
    """Random square crop resized to 512; returns the photo and its object masks (512 x 512 bool) in the same frame."""
    im = open_photo(info["file_name"]).convert("RGB")
    w, h = im.size; s = min(w, h)
    ox, oy = int(rng.integers(0, w - s + 1)), int(rng.integers(0, h - s + 1))
    photo = np.asarray(im.crop((ox, oy, ox + s, oy + s)).resize((S, S), Image.Resampling.BICUBIC))
    masks = []
    for seg in objs.get(info["id"], []):
        canvas = Image.new("L", (S, S), 0); draw = ImageDraw.Draw(canvas)
        for poly in seg:
            pts = [((x - ox) * S / s, (y - oy) * S / s) for x, y in zip(poly[0::2], poly[1::2])]
            if len(pts) >= 3:
                draw.polygon(pts, fill=1)
        m = np.asarray(canvas, bool)
        if m.any():
            masks.append(m)
    return photo, masks


def rect(rng, area_frac, max_aspect):
    """Random rectangle (y0, y1, x0, x1) inside 512 x 512 with about the given area share and aspect <= max_aspect."""
    a = area_frac * S * S
    r = float(np.exp(rng.uniform(-np.log(max_aspect), np.log(max_aspect))))          # width / height
    h = int(np.clip(np.sqrt(a / r), 8, S)); w = int(np.clip(a / h, 8, S))
    return fit_aspect(*place(rng, h, w), max_aspect)


def place(rng, h, w):
    y0, x0 = int(rng.integers(0, S - h + 1)), int(rng.integers(0, S - w + 1))
    return y0, y0 + h, x0, x0 + w


def fit_aspect(y0, y1, x0, x1, lim):
    """Widen the short side (centred, kept inside the image) until the aspect ratio is <= lim."""
    h, w = y1 - y0, x1 - x0
    if w > lim * h:
        h2 = int(np.ceil(w / lim)); c = (y0 + y1) // 2; y0 = max(0, min(S - h2, c - h2 // 2)); y1 = y0 + h2
    elif h > lim * w:
        w2 = int(np.ceil(h / lim)); c = (x0 + x1) // 2; x0 = max(0, min(S - w2, c - w2 // 2)); x1 = x0 + w2
    return y0, y1, x0, x1


def box_label(r):
    lab = np.zeros((S, S), np.uint8); lab[r[0]:r[1], r[2]:r[3]] = 1
    return lab


def embed_box(wm, src, r, msg):
    """Watermark the rectangle r of src at the rectangle's own size."""
    y0, y1, x0, x1 = r
    return wm.embed(np.ascontiguousarray(src[y0:y1, x0:x1]), msg)


def make_holes(wm, method, photo, masks, other, other_masks, msg, rng):
    marked = wm.embed(photo, msg)
    cand = [m for m in masks if 0.003 <= m.mean() <= 0.3]
    hole = np.zeros((S, S), bool)
    if cand:
        for k in rng.choice(len(cand), size=min(len(cand), int(rng.integers(1, 4))), replace=False):
            hole |= cand[k]
    else:
        y0, y1, x0, x1 = rect(rng, rng.uniform(0.01, 0.15), 4); hole[y0:y1, x0:x1] = True
    if rng.random() < 0.5:                                        # edits often reach a little past the object
        hole = ndimage.binary_dilation(hole, iterations=int(rng.integers(1, 9)))
    return np.where(hole[..., None], photo, marked), (~hole).astype(np.uint8)


def make_outpaint(wm, method, photo, masks, other, other_masks, msg, rng):
    r = rect(rng, rng.uniform(0.4, 0.9), min(ASPECT[method], 4))
    img = photo.copy(); img[r[0]:r[1], r[2]:r[3]] = embed_box(wm, photo, r, msg)
    return img, box_label(r)


def make_patch(wm, method, photo, masks, other, other_masks, msg, rng):
    area = float(np.exp(rng.uniform(np.log(0.002), np.log(0.25))))
    r = rect(rng, area, ASPECT[method])
    src = photo if rng.random() < 0.5 else other                  # in place, or a rectangle from another photo
    img = photo.copy(); img[r[0]:r[1], r[2]:r[3]] = embed_box(wm, src, r, msg)
    return img, box_label(r)


def make_transplant(wm, method, photo, masks, other, other_masks, msg, rng):
    cand = [m for m in other_masks if 0.002 <= m.mean() <= 0.25]
    if not cand:
        return make_patch(wm, method, photo, masks, other, other_masks, msg, rng)
    o = cand[int(rng.integers(len(cand)))]
    ys, xs = np.where(o)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    my, mx = rng.uniform(0, 0.3) * (y1 - y0), rng.uniform(0, 0.3) * (x1 - x0)    # embedding box = object box + margin
    r = fit_aspect(max(0, int(y0 - my)), min(S, int(y1 + my)), max(0, int(x0 - mx)), min(S, int(x1 + mx)), ASPECT[method])
    marked = other.copy(); marked[r[0]:r[1], r[2]:r[3]] = embed_box(wm, other, r, msg)
    keep = o & box_label(r).astype(bool)
    return np.where(keep[..., None], marked, photo), keep.astype(np.uint8)


MAKERS = (make_holes, make_outpaint, make_patch, make_transplant)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_PROCID", 0)))
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="synth", help="folder under output/")
    ap.add_argument("--npy", action="store_true",
                    help="write 5 plain .npy files per shard (memory-mappable, for multi-GPU training) instead of one .npz")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed * 1000 + a.shard)
    infos, objs = load_coco()
    infos = infos[a.shard::a.nshards]                             # each shard uses its own photos
    wms = {m: load_watermark(m, models_dir=TASK / "models", device="cuda") for m in METHODS}

    images = np.zeros((a.n, S, S, 3), np.uint8); labels = np.zeros((a.n, 128, 128), np.uint8)
    meth = np.zeros(a.n, np.uint8); rec = np.zeros(a.n, np.uint8)
    msgs = np.zeros((a.n, 256), np.uint8)                         # embedded message, padded to 256 bits
    k, fails, t0 = 0, 0, time.time()
    while k < a.n:
        mi = k % 3; wm = wms[METHODS[mi]]
        ri = int(rng.choice(4, p=RECIPE_P))
        msg = rng.integers(0, 2, wm.nbits).tolist()
        try:
            photo, pm = load_photo(infos[int(rng.integers(len(infos)))], objs, rng)
            other, om = load_photo(infos[int(rng.integers(len(infos)))], objs, rng)
            img, lab = MAKERS[ri](wm, METHODS[mi], photo, pm, other, om, msg, rng)
        except Exception as e:                                    # e.g. an embedder refusing a tiny crop
            fails += 1
            print(f"skip ({RECIPES[ri]}, {METHODS[mi]}): {e}", flush=True)
            if fails > 200:
                raise
            continue
        lab128 = downsample_mask(lab)
        if lab128.min() == lab128.max():                          # watermark vanished on the grid: real data never has this
            continue
        images[k], labels[k], meth[k], rec[k] = img, lab128, mi, ri
        msgs[k, :len(msg)] = msg
        k += 1
        if k % 100 == 0:
            print(f"shard {a.shard}: {k}/{a.n}  {time.time() - t0:.0f}s", flush=True)
    (OUT / a.outdir).mkdir(parents=True, exist_ok=True)
    if a.npy:
        path = OUT / a.outdir / f"shard_{a.shard}_*.npy"
        for key, arr in (("images", images), ("masks", labels), ("method", meth), ("recipe", rec), ("msgs", msgs)):
            np.save(OUT / a.outdir / f"shard_{a.shard}_{key}.npy", arr)
    else:
        path = OUT / a.outdir / f"shard_{a.shard}.npz"
        np.savez(path, images=images, masks=labels, method=meth, recipe=rec, msgs=msgs)
    print(f"wrote {path}  watermarked share mean={labels.mean():.3f}  recipes={np.bincount(rec, minlength=4).tolist()}  skipped={fails}")


if __name__ == "__main__":
    main()
