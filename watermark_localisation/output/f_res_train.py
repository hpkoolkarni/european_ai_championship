"""F4. Message-aware specialist + RESIDUAL head (dense supervision). Builds on f_msg_train.py (unchanged).
Why: a binary mask is very little supervision per image (a blob is a few numbers), but our generator knows the clean
cover, so the exact watermark residual w = image - cover is known at every pixel (output/<data>/shard_*_resid.npy,
from e_synth3.py --residual). A head that predicts w is a learned "watermark remover": the best such predictor is the
posterior mean E[w | image] (Tweedie / denoising), so the network must model what the watermark looks like LOCALLY,
with a target at all 512 x 512 x 3 values instead of a few blob parameters - the most useful signal for small and faint
pieces. The predicted strength |w_hat|, pooled to the official 128 grid, is one more input of the mask head.
Model = f_msg_train.MsgSeg (backbone, frozen decoder prior, per-cell bit head, message match c / c0 / energy e) plus
  residual head   F (128 x 128 x 128) -> 3x3 conv -> 1x1 conv to 48 channels -> pixel shuffle x4 -> 3 x 512 x 512 = w_hat/4
  mask head       now reads [F, c, c0, e, r] with r = 4x4 average of |w_hat| (the residual strength per cell)
  loss            mask BCE + soft mIoU, + --w-bits * bit BCE (watermarked cells), + --w-resid * L1(w_hat, w/4) on
                  synthetic images only (real images have no cover, so no residual target)
Phases as in f_msg_train.py: --phase synth (train on synthetic data, best real checkpoint, probs) and --phase real
(5-fold real-image fine-tune, out-of-fold train + 5-fold-average test predictions, step count chosen once).
Nothing is uploaded.
Run:  .venv/bin/python output/f_res_train.py --phase synth --backbone pixelseal --methods pixelseal --name psr_v5 --data synth5
      .venv/bin/python output/f_res_train.py --phase real --init psr_v5 --name psr_v5_real --data synth5"""
import argparse
import glob
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TASK = Path(__file__).resolve().parents[1]
OUT = TASK / "output"
sys.path.insert(0, str(TASK)); sys.path.insert(0, str(OUT))
from f_seg_train import ENCODERS, METHODS, freeze_pretrained_bn, loss_fn, onehot   # noqa: E402
from f_msg_train import NBITS, MsgSeg, bit_loss, load_real, score, signs           # noqa: E402


class ResSeg(MsgSeg):
    def __init__(self, backbone, method, width=128, use_prior=True):
        super().__init__(backbone, method, width, use_prior)
        self.res = nn.Sequential(nn.Conv2d(width, 64, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, 48, 1),
                                 nn.PixelShuffle(4))
        block = lambda i: [nn.Conv2d(i, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(inplace=True)]  # noqa: E731
        self.head = nn.Sequential(*block(width + 4), nn.Conv2d(width, width, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(width, 1, 1))

    def forward(self, x_uint8, oh, s):
        """Returns mask logits B x 128 x 128, bit logits B x nbits x 128 x 128, residual B x 3 x 512 x 512 (w / 4)."""
        x01 = x_uint8.float() / 255; t = 0
        for f, p in zip(self.enc(x01, oh), self.proj):
            f = p(f)
            t = t + (F.adaptive_avg_pool2d(f, 128) if f.shape[-1] > 128 else F.interpolate(f, size=(128, 128), mode="bilinear", align_corners=False))
        feat = self.trunk(t)
        z = self.bits(feat)
        if self.prior:
            with torch.no_grad():
                z0 = self.prior[0].features(x01.float())["per_position_logits"][:, :self.nbits].float()
            z0 = F.interpolate(z0, size=(128, 128), mode="bilinear", align_corners=False) / 10
            z = z + self.a * z0
        else:
            z0 = torch.zeros_like(z)
        w = self.res(feat)
        r = F.avg_pool2d(w.abs().mean(1, keepdim=True), 4)
        sb = s[:, :, None, None]
        c, c0, e = (z * sb).mean(1, keepdim=True), (z0 * sb).mean(1, keepdim=True), (z ** 2).mean(1, keepdim=True)
        return self.head(torch.cat([feat, c, c0, e, r], 1))[:, 0], z, w


@torch.no_grad()
def predict(model, images, mids, sgn, bs=12):
    model.eval(); out = []
    for i in range(0, len(images), bs):
        x = torch.from_numpy(np.stack(images[i:i + bs])).permute(0, 3, 1, 2).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z, _, _ = model(x, onehot(mids[i:i + bs]).cuda(), torch.from_numpy(sgn[i:i + bs]).cuda())
        out.append(torch.sigmoid(z.float()).cpu().numpy())
    model.train(); freeze_pretrained_bn(model, model.backbone)
    return np.concatenate(out)


def load_synth(data, keep, nbits):
    X, Y, Mid, Ms, R = [], [], [], [], []
    for f in sorted(glob.glob(str(OUT / data / "shard_*_images.npy"))):
        me = np.load(f.replace("_images", "_method")); sel = np.where(np.isin(me, keep))[0]
        X.append(np.load(f, mmap_mode="r")[sel]); Y.append(np.load(f.replace("_images", "_masks"))[sel])
        Mid.append(me[sel]); Ms.append(np.load(f.replace("_images", "_msgs"))[sel][:, :nbits])
        R.append(np.load(f.replace("_images", "_resid"), mmap_mode="r")[sel])
    return np.concatenate(X), np.concatenate(Y), np.concatenate(Mid), np.concatenate(Ms).astype(np.float32), np.concatenate(R)


def make_opt(model, lr, steps, warm):
    head = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
    opt = torch.optim.AdamW([{"params": model.enc.parameters(), "lr": lr}, {"params": head, "lr": lr * 10}], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    return opt, sched


def train_step(model, opt, sched, x, y, mid, msg01, resid, has_res, a):
    """resid: B x 512 x 512 x 3 int8 (zeros where unknown), has_res: B bool (synthetic images)."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        z, zb, w = model(x, onehot(mid).cuda(), torch.from_numpy(signs(msg01, a.no_msg)).cuda())
    loss = loss_fn(z.float(), y) + a.w_bits * bit_loss(zb.float(), torch.from_numpy(msg01).cuda(), y)
    hr = torch.from_numpy(has_res).cuda()
    if hr.any():
        tgt = torch.from_numpy(resid).cuda().permute(0, 3, 1, 2).float() / 4
        loss = loss + a.w_resid * (w.float() - tgt).abs()[hr].mean()
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    return loss.item()


def phase_synth(a):
    method = a.methods[0]; keep = [METHODS.index(method)]; nb = NBITS[method]
    X, Y, Mid, Ms, R = load_synth(a.data, keep, nb)
    perm = np.random.default_rng(0).permutation(len(X)); held, tr = perm[:300], perm[300:]
    real, test = load_real("train", a.methods, nb), load_real("test", a.methods, nb)
    print(f"[{a.name}] {method} on {a.backbone} + residual head; synthetic {len(tr)} + {len(held)} held out; real "
          f"{len(real['names'])}; messages {'OFF (control)' if a.no_msg else 'on'}; w_resid {a.w_resid}", flush=True)
    model = ResSeg(a.backbone, method).to_device("cuda").train(); freeze_pretrained_bn(model, a.backbone)
    opt, sched = make_opt(model, a.lr, a.steps, 300); best, t0, run = -1.0, time.time(), None

    def evaluate(step):
        nonlocal best
        s_syn = score(predict(model, X[held], Mid[held], signs(Ms[held], a.no_msg)), Y[held]).mean()
        s = score(predict(model, real["images"], real["mid"], signs(real["msg"], a.no_msg)), real["gt"]); h = real["full"]
        mark = ""
        if s.mean() > best:
            best = s.mean(); mark = "  (best, saved)"
            torch.save({"model": model.state_dict(), "backbone": a.backbone, "methods": a.methods, "step": step,
                        "no_msg": a.no_msg, "res_model": True}, OUT / "ckpt" / f"f_{a.name}.pt")
        print(f"[{a.name}] step {step}: held-out synthetic={s_syn:.4f}  real-train={s.mean():.4f}  holes={s[h].mean():.3f}  "
              f"patch={s[~h].mean():.3f}{mark}", flush=True)

    evaluate(0)
    for step in range(1, a.steps + 1):
        idx = np.sort(tr[np.random.randint(len(tr), size=a.bs)])
        x = torch.from_numpy(X[idx]).permute(0, 3, 1, 2).contiguous().cuda(non_blocking=True)
        loss = train_step(model, opt, sched, x, torch.from_numpy(Y[idx]).float().cuda(), Mid[idx], Ms[idx],
                          np.ascontiguousarray(R[idx]), np.ones(len(idx), bool), a)
        run = loss if run is None else 0.98 * run + 0.02 * loss
        if step % 100 == 0:
            print(f"[{a.name}] step {step}  loss={run:.4f}  {time.time() - t0:.0f}s", flush=True)
        done = (time.time() - t0) / 60 > a.minutes
        if step % a.eval_every == 0 or step == a.steps or done:
            evaluate(step)
        if done:
            print(f"[{a.name}] time limit reached at step {step}", flush=True); break
    print(f"[{a.name}] BEST real-train score={best:.4f}", flush=True)
    ck = torch.load(OUT / "ckpt" / f"f_{a.name}.pt", map_location="cpu"); model.load_state_dict(ck["model"])
    np.savez(OUT / f"probs_f_{a.name}.npz", names_train=real["names"],
             p_train=predict(model, real["images"], real["mid"], signs(real["msg"], a.no_msg)).astype(np.float16),
             names_test=test["names"], p_test=predict(model, test["images"], test["mid"], signs(test["msg"], a.no_msg)).astype(np.float16))
    print(f"[{a.name}] wrote probs_f_{a.name}.npz (best checkpoint from step {ck['step']})", flush=True)


def phase_real(a):
    ck0 = torch.load(OUT / "ckpt" / f"f_{a.init}.pt", map_location="cpu")
    backbone, methods = ck0["backbone"], ck0["methods"]; a.no_msg = ck0["no_msg"]; method = methods[0]; nb = NBITS[method]
    keep = [METHODS.index(method)]
    real, test = load_real("train", methods, nb), load_real("test", methods, nb)
    fold = np.random.default_rng(0).permutation(len(real["names"])) % 5      # same folds as g_combine.py
    Xs, Ys, Ms_, Mss, Rs = load_synth(a.data, keep, nb)
    R_img = np.stack(real["images"]); n_eval = a.steps // a.eval_every; zero_res = np.zeros((1, 512, 512, 3), np.int8)
    oof = np.zeros((n_eval + 1, len(R_img), 128, 128), np.float16); tst = np.zeros((n_eval + 1, len(test["names"]), 128, 128), np.float32)
    print(f"[{a.name}] init f_{a.init} ({backbone}, {method}, messages {'OFF' if a.no_msg else 'on'}); real {len(R_img)} in "
          f"5 folds; synthetic {len(Xs)}", flush=True)
    t_start = time.time()
    for k in range(5):
        model = ResSeg(backbone, method); model.load_state_dict(ck0["model"]); model.to_device("cuda").train()
        freeze_pretrained_bn(model, backbone); opt, sched = make_opt(model, a.lr, a.steps, 100)
        tr_i, ho_i = np.where(fold != k)[0], np.where(fold == k)[0]

        def evaluate(j, step):
            P = predict(model, R_img[ho_i], real["mid"][ho_i], signs(real["msg"][ho_i], a.no_msg)); oof[j, ho_i] = P
            tst[j] += predict(model, test["images"], test["mid"], signs(test["msg"], a.no_msg))
            s = score(P, real["gt"][ho_i]); h = real["full"][ho_i]
            print(f"[{a.name}] fold {k} step {step}: held-out real={s.mean():.4f}  holes={s[h].mean():.3f}  patch={s[~h].mean():.3f}  "
                  f"{time.time() - t_start:.0f}s", flush=True)

        evaluate(0, 0); half = a.bs // 2
        for step in range(1, a.steps + 1):
            ir, js = tr_i[np.random.randint(len(tr_i), size=half)], np.sort(np.random.randint(len(Xs), size=a.bs - half))
            x = torch.from_numpy(np.concatenate([R_img[ir], Xs[js]])).permute(0, 3, 1, 2).contiguous().cuda(non_blocking=True)
            y = torch.from_numpy(np.concatenate([real["gt"][ir], Ys[js]])).float().cuda()
            res = np.concatenate([np.repeat(zero_res, half, 0), np.ascontiguousarray(Rs[js])])
            train_step(model, opt, sched, x, y, np.concatenate([real["mid"][ir], Ms_[js]]),
                       np.concatenate([real["msg"][ir], Mss[js]]), res, np.r_[np.zeros(half, bool), np.ones(a.bs - half, bool)], a)
            if step % a.eval_every == 0:
                evaluate(step // a.eval_every, step)
        del model, opt
    s_all = np.array([score(oof[j], real["gt"]) for j in range(n_eval + 1)])
    for j in range(n_eval + 1):
        print(f"[{a.name}] out-of-fold real score after {j * a.eval_every:5d} steps: {s_all[j].mean():.4f}", flush=True)
    jb = int(np.argmax(s_all.mean(1)))
    np.savez(OUT / f"probs_f_{a.name}.npz", names_train=real["names"], p_train=oof[jb], names_test=test["names"],
             p_test=(tst[jb] / 5).astype(np.float16))
    print(f"[{a.name}] BEST out-of-fold real score={s_all[jb].mean():.4f} after {jb * a.eval_every} steps "
          f"(start: {s_all[0].mean():.4f}); wrote probs_f_{a.name}.npz", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("synth", "real"), required=True)
    ap.add_argument("--backbone", choices=sorted(ENCODERS))
    ap.add_argument("--methods", nargs=1, choices=METHODS)
    ap.add_argument("--init", help="phase real: output/ckpt/f_<init>.pt from phase synth")
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", default="synth5", help="synthetic data with *_resid.npy (e_synth3.py --residual)")
    ap.add_argument("--no-msg", action="store_true", help="control: messages replaced by zeros")
    ap.add_argument("--w-bits", type=float, default=1.0)
    ap.add_argument("--w-resid", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=None, help="default 10000 (synth) / 750 per fold (real)")
    ap.add_argument("--minutes", type=float, default=45)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=None, help="default 1e-4 (synth) / 2e-5 (real); head 10x")
    ap.add_argument("--seed", type=int, default=0, help="random seed for init, batches and augmentation (folds stay fixed)")
    ap.add_argument("--eval-every", type=int, default=None, help="default 1000 (synth) / 250 (real)")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    synth = a.phase == "synth"
    a.steps = a.steps or (10000 if synth else 750); a.lr = a.lr or (1e-4 if synth else 2e-5)
    a.eval_every = a.eval_every or (1000 if synth else 250)
    phase_synth(a) if synth else phase_real(a)


if __name__ == "__main__":
    main()
