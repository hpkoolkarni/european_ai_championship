"""E3. Synthetic training data v3: mask shapes and photos matched to the real training data.
Measured on the 510 real train images against synthetic v1 (output/synth): v1 "holes" followed COCO object outlines,
patches were half the real size, too elongated and never in several pieces, transplanted objects left strong seams,
10 % "outpaint" images (40-90 % watermarked) have no real counterpart, and the COCO photos were much sharper than the
real images. v3:
  holes (50 %)  whole photo watermarked, then 1-5 random blobs (7 % plain rectangles, 35 % placed on the border) restored to the clean original;
                the shapes ignore the image content; total clean share ~ lognormal, median 7 %; pieces may touch the border
  patch (50 %)  each piece watermarked at its rectangle's own size; 40 % kept as the rectangle, 36 % cut to an irregular
                blob inside it, 24 % as 2-3 separate pieces; total share ~ lognormal, median 4.5 %; elongation per method
                (measured median: PixelSeal / StegaStamp 2.1, TrustMark 1.45, capped by the brief's limits 16 / 2 / 16); 30 % of
                rectangles pushed against the image border; 85 % of pieces from the same photo (no seam), 15 % from another
  photos        COCO val2017 square crops whose sharpness (mean |Laplacian|) matches the real train images: a target is
                drawn from the real images, the closest of 4 random crops is kept, then mildly blurred if still too sharp
Labels follow the official rule (wmloc.utils.downsample_mask). Writes output/<outdir>/shard_<k>_{images,masks,method,
recipe,msgs}.npy (recipe 0 holes, 1 patch rectangle, 2 patch irregular, 3 patch multi-piece; msgs padded to 256 bits) and
prints the real and the generated shape and sharpness statistics side by side, so the match can be checked.
  GPU, one process per GPU (run_f3.sbatch):  .venv/bin/python output/e_synth3.py --nshards 4 --n 6000 --outdir synth3
  with --residual it also writes shard_<k>_resid.npy: the exact watermark residual (image - clean cover, int8), the
  dense training target of the residual head (f_msg_train.py --w-resid)
  CPU check of shapes and photos, no watermarking, nothing written:  .venv/bin/python output/e_synth3.py --dry --n 600"""
import argparse
import io
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
from wmloc.data import LocalisationDataset                       # noqa: E402
from wmloc.utils import downsample_mask                           # noqa: E402

OUT = TASK / "output"
COCO_TAR = OUT / "data" / "coco" / "val2017.tar"
METHODS = ("pixelseal", "trustmark", "stegastamp")
NBITS = {"pixelseal": 256, "trustmark": 100, "stegastamp": 100}
MAX_ASPECT = {"pixelseal": 16, "trustmark": 2, "stegastamp": 16}  # brief: limits of the watermarked rectangle
ELONG_SIGMA = {"pixelseal": 1.0, "trustmark": 0.6, "stegastamp": 1.0}    # tuned so the measured elongation matches
RECIPES = ("holes", "patch_rect", "patch_blob", "patch_multi")
S = 512


def sharpness(img):
    g = img.astype(np.float32) @ np.array([.299, .587, .114], np.float32)
    return float(np.abs(ndimage.laplace(g)).mean())


class Photos:
    """Random 512 x 512 square crops of the COCO photos in val2017.tar (each shard uses its own photos)."""

    def __init__(self, shard, nshards):
        self.tf = tarfile.open(COCO_TAR)
        mem = sorted((m for m in self.tf.getmembers() if m.isfile() and m.name.endswith(".jpg")), key=lambda m: m.name)
        self.members = mem[shard::nshards]

    def crop(self, rng):
        m = self.members[int(rng.integers(len(self.members)))]
        im = Image.open(io.BytesIO(self.tf.extractfile(m).read())).convert("RGB")
        w, h = im.size; s = min(w, h)
        ox, oy = int(rng.integers(0, w - s + 1)), int(rng.integers(0, h - s + 1))
        return np.asarray(im.crop((ox, oy, ox + s, oy + s)).resize((S, S), Image.Resampling.BICUBIC))

    def matched(self, rng, target, tries=4):
        """The crop closest in sharpness to target (out of `tries`), blurred a little if still more than 15 % sharper."""
        cands = [self.crop(rng) for _ in range(tries)]
        sh = [sharpness(c) for c in cands]
        k = int(np.argmin([abs(np.log(x / target)) for x in sh]))
        photo, cur = cands[k], sh[k]
        for sigma in (0.4, 0.6, 0.8, 1.0, 1.3):
            if cur <= 1.15 * target:
                break
            photo = np.clip(ndimage.gaussian_filter(cands[k].astype(np.float32), (sigma, sigma, 0)), 0, 255)
            photo = photo.round().astype(np.uint8); cur = sharpness(photo)
        return photo


def lognormal(rng, median, sigma, lo, hi):
    return float(np.clip(median * np.exp(sigma * rng.standard_normal()), lo, hi))


def elongation(rng, sigma, cap):
    return float(min(np.exp(abs(sigma * rng.standard_normal())), cap))


def wobble(rng, t):
    """Radius 1 with random low-frequency bumps; normalised so the shape keeps the area of the plain ellipse."""
    r = np.ones_like(t)
    for k in range(2, 6):
        r += rng.uniform(0, 0.22) / np.sqrt(k - 1) * np.cos(k * t + rng.uniform(0, 2 * np.pi))
    r = np.clip(r, 0.35, None)
    return r / np.sqrt(np.mean(r ** 2))


def blob(rng, cy, cx, area, elong, angle):
    """Polygon of a random smooth blob with about the given area (pixels), elongation and rotation."""
    t = np.linspace(0, 2 * np.pi, 64, endpoint=False); r = wobble(rng, t)
    a = np.sqrt(area * elong / np.pi); b = a / elong
    x, y = a * r * np.cos(t), b * r * np.sin(t); ca, sa = np.cos(angle), np.sin(angle)
    return [(cx + ca * u - sa * v, cy + sa * u + ca * v) for u, v in zip(x, y)]


STRENGTH = {"pixelseal": (0.5, 1.0), "trustmark": (1.0, 1.0), "stegastamp": (0.7, 1.0)}   # --strength ranges


class Weaker:
    """Embedder at reduced strength: cover + alpha * (watermarked - cover). The decoder check found the real PixelSeal
    and StegaStamp watermarks 10-15 % weaker and more variable than ours (match: strength ~0.6-0.7 and ~0.85-0.9);
    TrustMark matched as it is."""

    def __init__(self, wm, alpha):
        self.wm, self.alpha = wm, alpha

    def embed(self, x, msg):
        x = np.ascontiguousarray(x); w = self.wm.embed(x, msg).astype(np.float32)
        return np.clip(x + self.alpha * (w - x), 0, 255).round().astype(np.uint8)


def embed(wm, x, msg):
    return np.array(x, copy=True) if wm is None else wm.embed(np.ascontiguousarray(x), msg)   # wm None: --dry


def make_holes(wm, method, photo, get_other, msg, rng):
    marked = embed(wm, photo, msg)
    k = int(rng.choice([1, 2, 3, 4, 5], p=[0.38, 0.27, 0.18, 0.11, 0.06]))
    parts = rng.dirichlet(np.ones(k)) * lognormal(rng, 0.07, 1.4, 0.004, 0.35)
    canvas = Image.new("L", (S, S), 0); d = ImageDraw.Draw(canvas)
    for p in parts:
        area, e = max(p, 0.0015) * S * S, elongation(rng, 0.8, 5)
        cy, cx = rng.uniform(0, S, 2)
        if rng.random() < 0.35:                                   # real restored areas often touch the border
            if rng.random() < 0.5:
                cy = rng.choice([0.0, float(S)])
            else:
                cx = rng.choice([0.0, float(S)])
        if rng.random() < 0.07:
            h = np.sqrt(area / e); w = area / h
            if rng.random() < 0.5:
                h, w = w, h
            d.rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], fill=1)
        else:
            d.polygon(blob(rng, cy, cx, area, e, rng.uniform(0, np.pi)), fill=1)
    hole = np.asarray(canvas, bool)
    return np.where(hole[..., None], photo, marked), (~hole).astype(np.uint8), 0, photo


def patch_rect(rng, method, share):
    """Rectangle (y0, y1, x0, x1) inside the image with about the given area share and the method's elongation."""
    e = elongation(rng, ELONG_SIGMA[method], MAX_ASPECT[method]); area = share * S * S
    h = np.sqrt(area / e); w = area / h
    if rng.random() < 0.5:
        h, w = w, h
    h, w = int(np.clip(round(h), 8, S)), int(np.clip(round(w), 8, S))
    y0, x0 = int(rng.integers(0, S - h + 1)), int(rng.integers(0, S - w + 1))
    if rng.random() < 0.3:                                        # real patches touch the image border fairly often
        if rng.random() < 0.5:
            y0 = 0 if rng.random() < 0.5 else S - h
        else:
            x0 = 0 if rng.random() < 0.5 else S - w
    return y0, y0 + h, x0, x0 + w


def patch_piece(wm, method, img, src, msg, rng, share, irregular, clean):
    """Watermark one rectangle of src at its own size; paste the whole rectangle, or only a blob inside it, into img.
    clean gets the same pixels WITHOUT the watermark (the cover of the final image, for the residual target)."""
    y0, y1, x0, x1 = patch_rect(rng, method, share); h, w = y1 - y0, x1 - x0
    marked = embed(wm, src[y0:y1, x0:x1], msg)
    sub = np.ones((h, w), bool)
    if irregular:
        t = np.linspace(0, 2 * np.pi, 64, endpoint=False); r = wobble(rng, t)
        g = rng.uniform(0.9, 1.2)                                 # sometimes larger than the box: some edges cut straight
        pts = [(w / 2 + g * w / 2 * ri * np.cos(ti), h / 2 + g * h / 2 * ri * np.sin(ti)) for ri, ti in zip(r, t)]
        canvas = Image.new("L", (w, h), 0); ImageDraw.Draw(canvas).polygon(pts, fill=1)
        b = np.asarray(canvas, bool)
        if b.sum() >= 0.3 * h * w:
            sub = b
    region = img[y0:y1, x0:x1]; region[sub] = marked[sub]
    cregion = clean[y0:y1, x0:x1]; cregion[sub] = src[y0:y1, x0:x1][sub]
    keep = np.zeros((S, S), bool); keep[y0:y1, x0:x1] = sub
    return keep


def make_patch(wm, method, photo, get_other, msg, rng):
    kind = int(rng.choice(3, p=[0.40, 0.36, 0.24]))              # rectangle, irregular, multi-piece (tuned to the real shares)
    total = lognormal(rng, 0.045, 1.35, 0.003, 0.35)
    if kind < 2:
        shares, irregular = [total], [kind == 1]
    else:
        n = int(rng.choice([2, 3], p=[0.7, 0.3]))
        shares = list(rng.dirichlet(np.ones(n)) * total); irregular = [rng.random() < 0.5 for _ in shares]
    img = photo.copy(); clean = photo.copy(); lab = np.zeros((S, S), bool)
    for sh, irr in zip(shares, irregular):
        src = photo if rng.random() < 0.85 else get_other()
        lab |= patch_piece(wm, method, img, src, msg, rng, max(sh, 0.002), irr, clean)
    return img, lab.astype(np.uint8), 1 + kind, clean


def shape_stats(m128, holes):
    """Pieces, and for the largest piece: share of its bounding box, elongation, touches the border.
    The piece is the watermarked area for patch images and the clean area for holes images."""
    piece = (1 - m128) if holes else m128
    lab, n = ndimage.label(piece, structure=np.ones((3, 3)))
    if n == 0:
        return n, np.nan, np.nan, np.nan
    sizes = ndimage.sum(piece, lab, range(1, n + 1)); big = int(np.argmax(sizes)) + 1
    ys, xs = np.where(lab == big); h, w = np.ptp(ys) + 1, np.ptp(xs) + 1
    return n, sizes[big - 1] / (h * w), max(h, w) / min(h, w), float(min(ys.min(), xs.min()) == 0 or max(ys.max(), xs.max()) == 127)


def report(tag, masks, meth, holes, sharp):
    for mi, m in enumerate(METHODS):
        for hl in (True, False):
            sel = np.where((meth == mi) & (holes == hl))[0]
            if len(sel) == 0:
                continue
            st = np.array([shape_stats(masks[i], hl) for i in sel], float); share = masks[sel].mean((1, 2))
            print(f"{tag:6s} {m:10s} {'holes' if hl else 'patch'} n={len(sel):5d}  watermarked share {np.median(share):.3f} "
                  f"[{np.percentile(share, 25):.3f}-{np.percentile(share, 75):.3f}]  pieces>1 {np.mean(st[:, 0] > 1):.2f}  "
                  f"near-rect {np.nanmean(st[:, 1] > 0.9):.2f}  elongation {np.nanmedian(st[:, 2]):.2f}  "
                  f"touches border {np.nanmean(st[:, 3]):.2f}  sharpness {np.median(sharp[sel]):.1f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_PROCID", 0)))
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--outdir", default="synth3", help="folder under output/")
    ap.add_argument("--dry", action="store_true", help="no watermarking, nothing written: only check shapes and photos")
    ap.add_argument("--residual", action="store_true",
                    help="also save the watermark residual (image - clean cover, int8) as shard_<k>_resid.npy")
    ap.add_argument("--strength", action="store_true",
                    help="random watermark strength per image (ranges in STRENGTH), to match the real decoder margins")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed * 1000 + a.shard)

    ds = LocalisationDataset(TASK / "release", "train")
    real = [ds[i] for i in range(len(ds))]
    targets = np.array([sharpness(it["image"]) for it in real])
    if a.shard == 0:
        report("real", np.stack([it["mask128"] for it in real]), np.array([METHODS.index(it["method"]) for it in real]),
               np.array([bool(it["full_watermark"]) for it in real]), targets)
    photos = Photos(a.shard, a.nshards)
    if a.dry:
        wms = {m: None for m in METHODS}
    else:
        from wmloc.watermarks import load_watermark
        wms = {m: load_watermark(m, models_dir=TASK / "models", device="cuda") for m in METHODS}

    images = np.zeros((a.n, S, S, 3), np.uint8); labels = np.zeros((a.n, 128, 128), np.uint8)
    meth = np.zeros(a.n, np.uint8); rec = np.zeros(a.n, np.uint8); msgs = np.zeros((a.n, 256), np.uint8)
    sharp = np.zeros(a.n, np.float32)
    resid = np.zeros((a.n, S, S, 3), np.int8) if a.residual else None
    k, fails, t0 = 0, 0, time.time()
    while k < a.n:
        mi = k % 3; method = METHODS[mi]
        msg = rng.integers(0, 2, NBITS[method]).tolist()
        target = float(rng.choice(targets))
        make = make_holes if rng.random() < 0.5 else make_patch
        try:
            photo = photos.matched(rng, target)
            wm = wms[method]
            if a.strength and wm is not None:
                wm = Weaker(wm, float(rng.uniform(*STRENGTH[method])))
            img, lab, ri, clean = make(wm, method, photo, lambda: photos.matched(rng, target), msg, rng)
        except Exception as e:                                    # e.g. an embedder refusing a tiny crop
            fails += 1
            print(f"skip ({make.__name__}, {method}): {e}", flush=True)
            if fails > 200:
                raise
            continue
        lab128 = downsample_mask(lab)
        if lab128.min() == lab128.max():                          # watermark vanished on the grid: real data never has this
            continue
        images[k], labels[k], meth[k], rec[k], sharp[k] = img, lab128, mi, ri, sharpness(img)
        msgs[k, :len(msg)] = msg
        if resid is not None:
            resid[k] = np.clip(img.astype(np.int16) - clean.astype(np.int16), -127, 127)
        k += 1
        if k % 500 == 0:
            print(f"shard {a.shard}: {k}/{a.n}  {time.time() - t0:.0f}s", flush=True)
    report(f"gen{a.shard}", labels, meth, rec == 0, sharp)
    print(f"shard {a.shard}: recipes {dict(zip(RECIPES, np.bincount(rec, minlength=4).tolist()))}  skipped={fails}  "
          f"{time.time() - t0:.0f}s", flush=True)
    if a.dry:
        return
    (OUT / a.outdir).mkdir(parents=True, exist_ok=True)
    for key, arr in [("images", images), ("masks", labels), ("method", meth), ("recipe", rec), ("msgs", msgs)] + (
            [("resid", resid)] if resid is not None else []):
        np.save(OUT / a.outdir / f"shard_{a.shard}_{key}.npy", arr)
    print(f"wrote output/{a.outdir}/shard_{a.shard}_*.npy", flush=True)


if __name__ == "__main__":
    main()
