"""F2. Fine-tune a trained specialist on the REAL train images, honestly, with 5 folds.
The specialists score 0.62-0.87 on held-out synthetic images but 0.49-0.66 on the real ones: some difference between
our synthetic data and the organisers' data remains. The most direct fix is to show the model real images too.
For each fold k = 0..4 (the same folds g_combine.py uses for its thresholds, per method):
  1. start again from the synthetic-trained checkpoint output/ckpt/f_<init>.pt (its backbone and methods are reused)
  2. train --steps steps; every batch is --real-share real images from the other 4 folds (plus, with --pseudo, test
     images whose patch box the official decoder confirmed with the known message), the rest synthetic images
     (--data), so the model cannot simply memorise the ~136 real images it sees
  3. every --eval-every steps, predict the held-out fold (never trained on) and all test images of these methods
Options (all off by default, so the default run is exactly the earlier one):
  --pseudo      test-image pseudo-labels from the box search (output/box/<method>_test_*.npz): a box the decoder reads
                the known message from (bit accuracy >= --pseudo-level) is the watermarked rectangle, so that image's
                mask is known. These come from the real TEST distribution and from an independent "view" (the decoder
                + message), so training on them does not just repeat the model's own mistakes (co-training)
  --llrd D      layer-wise learning-rate decay: the backbone is split into 4 depth groups in order; the deepest keeps
                --lr, each earlier group gets D times less (early layers are the pretrained pattern detectors)
  --swa         also evaluate the average of the weights saved at every evaluation from step --eval-every on
                (stochastic weight averaging: flatter minimum, usually generalises better with little data)
Afterwards, the number of steps (or "swa") with the best held-out score over all 5 folds is chosen (one choice for all
folds), and output/probs_f_<name>.npz gets, at that choice:
  p_train  each real train image predicted by the fold model that never saw it (out-of-fold, so g_combine's scores and
           thresholds stay honest)
  p_test   the average of the 5 fold models
Fold models are not kept (only their predictions), to save disk. Nothing is uploaded.
Run:  .venv/bin/python output/f_real_ft.py --init ss_v3 --name ss_v3_real --data synth4"""
import argparse
import copy
import glob
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

TASK = Path(__file__).resolve().parents[1]
OUT = TASK / "output"
sys.path.insert(0, str(TASK)); sys.path.insert(0, str(OUT))
from wmloc.data import LocalisationDataset                          # noqa: E402
from wmloc.utils import miou_image                                   # noqa: E402
from f_seg_train import METHODS, Seg, freeze_pretrained_bn, loss_fn, onehot, predict   # noqa: E402


def pseudo_labels(methods, level):
    """Test images whose box the decoder confirmed: (image, 128-grid mask, method id) from output/box/<m>_test_*.npz."""
    from h_box_search import box_masks
    te = LocalisationDataset(TASK / "release", "test", methods=methods); pos = {m["image_name"]: i for i, m in enumerate(te.meta)}
    imgs, masks, mids = [], [], []
    for m in methods:
        for f in sorted(glob.glob(str(OUT / "box" / f"{m}_test_*.npz"))):
            d = np.load(f)
            for n, b, ac in zip(d["names"].tolist(), d["boxes"], d["accs"]):
                bm = box_masks(b, ac, level) if n in pos else None
                if bm is not None:
                    imgs.append(te[pos[n]]["image"]); masks.append(bm); mids.append(METHODS.index(m))
    return imgs, masks, np.array(mids, int)


def param_groups(model, lr, llrd):
    enc = [p for p in model.enc.parameters() if p.requires_grad]
    head = list(model.proj.parameters()) + list(model.out.parameters())
    if llrd >= 1.0:
        return [{"params": enc, "lr": lr}, {"params": head, "lr": lr * 10}]
    chunks = np.array_split(np.arange(len(enc)), 4)                 # registration order ~ depth order
    return [{"params": [enc[i] for i in c], "lr": lr * llrd ** (3 - g)} for g, c in enumerate(chunks)] + \
        [{"params": head, "lr": lr * 10}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="trained specialist: output/ckpt/f_<init>.pt")
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", default="synth4", help="synthetic data folder under output/ (.npy shards) mixed in")
    ap.add_argument("--steps", type=int, default=1500, help="per fold")
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--real-share", type=float, default=0.5, help="share of each batch from real (and pseudo) images")
    ap.add_argument("--lr", type=float, default=2e-5, help="backbone learning rate; the head uses 10x")
    ap.add_argument("--llrd", type=float, default=1.0, help="layer-wise lr decay per backbone depth group (1 = off)")
    ap.add_argument("--swa", action="store_true", help="also score the average of the evaluated weights")
    ap.add_argument("--pseudo", action="store_true", help="add decoder-confirmed test boxes as extra real images")
    ap.add_argument("--pseudo-level", type=float, default=0.85, help="bit accuracy needed to accept a box")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--minutes", type=float, default=55, help="expected total; only a warning after fold 0 if exceeded")
    a = ap.parse_args()
    torch.manual_seed(0)
    ck0 = torch.load(OUT / "ckpt" / f"f_{a.init}.pt", map_location="cpu")
    backbone, methods = ck0["backbone"], ck0["methods"]; keep = [METHODS.index(m) for m in methods]

    tr = LocalisationDataset(TASK / "release", "train", methods=methods)
    real = [tr[i] for i in range(len(tr))]
    r_img = np.stack([it["image"] for it in real]); r_gt = np.stack([it["mask128"] for it in real])
    r_mid = np.array([METHODS.index(it["method"]) for it in real]); r_full = np.array([bool(it["full_watermark"]) for it in real])
    fold = np.zeros(len(real), int)
    for m in keep:                                                  # same folds as g_combine.py (per method, rng 0)
        idx = np.where(r_mid == m)[0]; fold[idx] = np.random.default_rng(0).permutation(len(idx)) % 5
    te = LocalisationDataset(TASK / "release", "test", methods=methods)
    t_img = [te[i]["image"] for i in range(len(te))]; t_mid = np.array([METHODS.index(m["method"]) for m in te.meta])
    p_img, p_gt, p_mid = pseudo_labels(methods, a.pseudo_level) if a.pseudo else ([], [], np.zeros(0, int))

    X, Y, Mid = [], [], []
    for f in sorted(glob.glob(str(OUT / a.data / "shard_*_images.npy"))):
        me = np.load(f.replace("_images", "_method")); sel = np.where(np.isin(me, keep))[0]
        X.append(np.load(f, mmap_mode="r")[sel]); Y.append(np.load(f.replace("_images", "_masks"))[sel]); Mid.append(me[sel])
    Xs, Ys, Ms = torch.from_numpy(np.concatenate(X)), torch.from_numpy(np.concatenate(Y)), np.concatenate(Mid)
    print(f"[{a.name}] init f_{a.init} ({backbone}, {methods}); real {len(real)} images in 5 folds; pseudo-labelled test "
          f"{len(p_img)}; synthetic {len(Xs)} from {a.data}; test {len(te)}; real share {a.real_share}, llrd {a.llrd}, "
          f"swa {a.swa}", flush=True)

    t_start = time.time(); n_eval = a.steps // a.eval_every; n_col = n_eval + 1 + int(a.swa)   # last column = swa
    oof = np.zeros((n_col, len(real), 128, 128), np.float16)        # held-out predictions at every eval point
    tst = np.zeros((n_col, len(te), 128, 128), np.float32)          # sum of the fold models' test predictions
    steps = a.steps
    for k in range(5):
        model = Seg(backbone); model.load_state_dict(ck0["model"]); model.cuda().train(); freeze_pretrained_bn(model, backbone)
        opt = torch.optim.AdamW(param_groups(model, a.lr, a.llrd), weight_decay=0.01)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
        tr_i, ho_i = np.where(fold != k)[0], np.where(fold == k)[0]
        Xr = torch.from_numpy(np.concatenate([r_img[tr_i]] + ([np.stack(p_img)] if len(p_img) else [])))
        Yr = torch.from_numpy(np.concatenate([r_gt[tr_i]] + ([np.stack(p_gt)] if len(p_gt) else [])))
        Mr = np.concatenate([r_mid[tr_i], p_mid])
        swa, n_swa = None, 0

        def evaluate(j, step, m=None):
            m = m or model
            P = predict(m, r_img[ho_i], r_mid[ho_i]); oof[j, ho_i] = P
            tst[j] += predict(m, t_img, t_mid)
            s = np.array([miou_image((p >= 0.5).astype(np.uint8), g) for p, g in zip(P, r_gt[ho_i])])
            h = r_full[ho_i]
            print(f"[{a.name}] fold {k} {step}: held-out real={s.mean():.4f}  holes={s[h].mean():.3f}  "
                  f"patch={s[~h].mean():.3f}  {time.time() - t_start:.0f}s", flush=True)
            model.train(); freeze_pretrained_bn(model, backbone)

        evaluate(0, "step 0")
        n_real = max(1, int(round(a.bs * a.real_share)))
        for step in range(1, steps + 1):
            ir, js = torch.randint(len(Xr), (n_real,)), torch.randint(len(Xs), (a.bs - n_real,))
            x = torch.cat([Xr[ir], Xs[js]]).permute(0, 3, 1, 2).contiguous().cuda(non_blocking=True)
            y = torch.cat([Yr[ir], Ys[js]]).float().cuda(non_blocking=True)
            mid = np.concatenate([Mr[ir.numpy()], Ms[js.numpy()]])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = model(x, onehot(mid).cuda())
            loss = loss_fn(z.float(), y)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            if step % a.eval_every == 0:
                evaluate(step // a.eval_every, f"step {step}")
                if a.swa:                                           # running average of the evaluated weights
                    sd = {kk: v.detach().float().clone() for kk, v in model.state_dict().items()}
                    swa = sd if swa is None else {kk: swa[kk] + sd[kk] for kk in swa}; n_swa += 1
        if a.swa and swa is not None:
            avg = copy.deepcopy(model); ref = model.state_dict()
            avg.load_state_dict({kk: (swa[kk] / n_swa).to(ref[kk].dtype) if ref[kk].is_floating_point() else ref[kk] for kk in ref})
            evaluate(n_col - 1, "swa", avg); del avg
        if k == 0:                                                  # time check after fold 0: keep 5 folds inside --minutes
            per_fold = (time.time() - t_start) / 60
            if 5 * per_fold > a.minutes:
                print(f"[{a.name}] WARNING: fold 0 took {per_fold:.1f} min; 5 folds exceed --minutes {a.minutes}", flush=True)
        del model, opt

    s_all = np.array([[miou_image((oof[j, i] >= 0.5).astype(np.uint8), r_gt[i]) for i in range(len(real))]
                      for j in range(n_col)])
    label = lambda j: "swa" if (a.swa and j == n_col - 1) else f"{j * a.eval_every:5d} steps"   # noqa: E731
    for j in range(n_col):
        print(f"[{a.name}] out-of-fold real score after {label(j)}: {s_all[j].mean():.4f}", flush=True)
    jb = int(np.argmax(s_all.mean(1)))
    np.savez(OUT / f"probs_f_{a.name}.npz", names_train=[it["image_name"] for it in real], p_train=oof[jb],
             names_test=[m["image_name"] for m in te.meta], p_test=(tst[jb] / 5).astype(np.float16))
    print(f"[{a.name}] BEST out-of-fold real score={s_all[jb].mean():.4f} after {label(jb)} "
          f"(start: {s_all[0].mean():.4f}); wrote probs_f_{a.name}.npz", flush=True)


if __name__ == "__main__":
    main()
