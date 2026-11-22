"""F3. Message-aware specialist: uses the KNOWN message of every image (given in the metadata).
Why (detection theory): our models so far only see the image, so at best they detect watermark "energy". A detector
that knows the message can add the bit evidence coherently, c = mean_b z_b * s_b (matched filter), which is far more
sensitive for weak signals (small pieces, faint PixelSeal) - exactly where we lose most points. Measured with the
pretrained decoders' own per-position bit outputs: on PixelSeal holes images the coherent map separates watermarked
from clean cells better than the energy map (AUC 0.648 vs 0.562), but those maps are only 8 x 8 and fail on patches
(pattern at the wrong scale). So the model must learn a fine, local bit decoder - WAM's per-pixel message head - on top
of a backbone that already knows the watermark.
Model (per method), everything on the official 128 x 128 grid:
  backbone      the method's own decoder backbone (PixelSeal ConvNeXt / TrustMark ResNet-50) or the steganalysis
                ResNet-34, all stages projected to 128 channels and summed (as f_seg_train.py), then 2 conv blocks -> F
  prior         the frozen official decoder at its own resolution: per-position bit logits z0 (8 x 8, upsampled), i.e.
                the decoder's own evidence; none for StegaStamp (its ONNX decoder has no per-position output)
  bit head      z(p) = a * z0(p) + conv1x1(F)(p): N bit logits per cell, starting AS the decoder (a = 1, conv = 0)
                and learning a finer, local decoder; trained with the known bits on watermarked cells only (WAM)
  match         c(p) = mean_b z_b(p) s_b and c0(p) = mean_b z0_b(p) s_b (s_b = +-1 known bits), and the energy
                e(p) = mean_b z_b(p)^2: the matched-filter statistic and its message-blind counterpart
  mask head     2 conv blocks on [F, c, c0, e] -> 1 logit per cell (the message acts like SAM's prompt)
  loss          BCE + (1 - soft mIoU) on the mask (as before) + --w-bits * BCE of the bits on watermarked cells
  --no-msg      control run: the messages are replaced by zeros, so c and c0 carry no information; everything else
                is identical. Comparing the two tells whether the MESSAGE helps, not just the bigger head.
Phases:
  --phase synth  train on synthetic data (output/<data>, with messages) like f_seg_train.py; scores held-out synthetic
                 and real train images every --eval-every steps; best real checkpoint -> output/ckpt/f_<name>.pt;
                 probs -> output/probs_f_<name>.npz (same format as before, so g_combine works)
  --phase real   5-fold fine-tune on the real train images from --init (as f_real_ft.py: half real / half synthetic
                 per batch, out-of-fold train predictions, 5-fold-average test predictions, step count chosen once)
Nothing is uploaded.
Run:  .venv/bin/python output/f_msg_train.py --phase synth --backbone pixelseal --methods pixelseal --name psm_v4
      .venv/bin/python output/f_msg_train.py --phase real --init psm_v4 --name psm_v4_real"""
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
from wmloc.data import LocalisationDataset                          # noqa: E402
from wmloc.utils import miou_image                                   # noqa: E402
from wmloc.watermarks import load_watermark                          # noqa: E402
from f_seg_train import ENCODERS, METHODS, freeze_pretrained_bn, loss_fn, onehot   # noqa: E402

NBITS = {"pixelseal": 256, "trustmark": 100, "stegastamp": 100}


class MsgSeg(nn.Module):
    def __init__(self, backbone, method, width=128, use_prior=True):
        super().__init__()
        self.backbone, self.method, self.nbits = backbone, method, NBITS[method]
        self.enc = ENCODERS[backbone]()
        with torch.no_grad():
            chans = [f.shape[1] for f in self.enc(torch.zeros(1, 3, 512, 512), torch.zeros(1, 3))]
        self.proj = nn.ModuleList([nn.Conv2d(c, width, 1) for c in chans])
        block = lambda i: [nn.Conv2d(i, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(inplace=True)]  # noqa: E731
        self.trunk = nn.Sequential(*block(width), *block(width))
        self.bits = nn.Conv2d(width, self.nbits, 1)
        nn.init.zeros_(self.bits.weight); nn.init.zeros_(self.bits.bias)
        self.a = nn.Parameter(torch.ones(1))
        self.head = nn.Sequential(*block(width + 3), nn.Conv2d(width, width, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(width, 1, 1))
        self.prior = None
        if use_prior and method in ("pixelseal", "trustmark"):
            self.prior = [load_watermark(method, models_dir=TASK / "models", device="cpu")]   # list: not a submodule
            for p in (self.prior[0].det if method == "pixelseal" else self.prior[0].net).parameters():
                p.requires_grad_(False)

    def to_device(self, dev):
        self.to(dev)
        if self.prior:
            w = self.prior[0]; w.device = dev
            (w.det if self.method == "pixelseal" else w.net).to(dev).eval()
        return self

    def forward(self, x_uint8, oh, s):
        """x: B x 3 x 512 x 512 uint8, oh: B x 3 method one-hot, s: B x nbits known bits as +-1 (zeros = unknown).
        Returns mask logits B x 128 x 128 and bit logits B x nbits x 128 x 128."""
        x01 = x_uint8.float() / 255; t = 0
        for f, p in zip(self.enc(x01, oh), self.proj):
            f = p(f)
            t = t + (F.adaptive_avg_pool2d(f, 128) if f.shape[-1] > 128 else F.interpolate(f, size=(128, 128), mode="bilinear", align_corners=False))
        feat = self.trunk(t)
        z = self.bits(feat)
        if self.prior:
            with torch.no_grad():
                z0 = self.prior[0].features(x01.float())["per_position_logits"][:, :self.nbits].float()
            z0 = F.interpolate(z0, size=(128, 128), mode="bilinear", align_corners=False) / 10   # logits are ~10
            z = z + self.a * z0
        else:
            z0 = torch.zeros_like(z)
        sb = s[:, :, None, None]
        c, c0, e = (z * sb).mean(1, keepdim=True), (z0 * sb).mean(1, keepdim=True), (z ** 2).mean(1, keepdim=True)
        return self.head(torch.cat([feat, c, c0, e], 1))[:, 0], z


def bit_loss(z, msg01, y):
    """BCE of the per-cell bit logits against the known bits, on watermarked cells only."""
    tgt = msg01[:, :, None, None].expand_as(z)
    l = F.binary_cross_entropy_with_logits(z, tgt, reduction="none").mean(1)
    return (l * y).sum() / y.sum().clamp_min(1.0)


@torch.no_grad()
def predict(model, images, mids, signs, bs=12):
    model.eval(); out = []
    for i in range(0, len(images), bs):
        x = torch.from_numpy(np.stack(images[i:i + bs])).permute(0, 3, 1, 2).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z, _ = model(x, onehot(mids[i:i + bs]).cuda(), torch.from_numpy(signs[i:i + bs]).cuda())
        out.append(torch.sigmoid(z.float()).cpu().numpy())
    model.train(); freeze_pretrained_bn(model, model.backbone)
    return np.concatenate(out)


def load_synth(data, keep, nbits):
    X, Y, Mid, Ms = [], [], [], []
    for f in sorted(glob.glob(str(OUT / data / "shard_*_images.npy"))):
        me = np.load(f.replace("_images", "_method")); sel = np.where(np.isin(me, keep))[0]
        X.append(np.load(f, mmap_mode="r")[sel]); Y.append(np.load(f.replace("_images", "_masks"))[sel])
        Mid.append(me[sel]); Ms.append(np.load(f.replace("_images", "_msgs"))[sel][:, :nbits])
    return np.concatenate(X), np.concatenate(Y), np.concatenate(Mid), np.concatenate(Ms).astype(np.float32)


def load_real(split, methods, nbits):
    ds = LocalisationDataset(TASK / "release", split, methods=methods); items = [ds[i] for i in range(len(ds))]
    out = {"names": [it["image_name"] for it in items], "images": [it["image"] for it in items],
           "mid": np.array([METHODS.index(it["method"]) for it in items]),
           "msg": np.stack([it["message"][:nbits] for it in items]).astype(np.float32)}
    if split == "train":
        out["gt"] = np.stack([it["mask128"] for it in items]); out["full"] = np.array([bool(it["full_watermark"]) for it in items])
    return out


def signs(msg01, no_msg):
    return np.zeros_like(msg01) if no_msg else (msg01 * 2 - 1).astype(np.float32)


def make_opt(model, lr, steps, warm):
    head = list(model.proj.parameters()) + list(model.trunk.parameters()) + list(model.bits.parameters()) + \
        list(model.head.parameters()) + [model.a]
    opt = torch.optim.AdamW([{"params": model.enc.parameters(), "lr": lr}, {"params": head, "lr": lr * 10}], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    return opt, sched


def train_step(model, opt, sched, x, y, mid, msg01, no_msg, w_bits):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        z, zb = model(x, onehot(mid).cuda(), torch.from_numpy(signs(msg01, no_msg)).cuda())
    m = torch.from_numpy(msg01).cuda()
    loss = loss_fn(z.float(), y) + w_bits * bit_loss(zb.float(), m, y)
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    return loss.item()


def score(P, gts):
    return np.array([miou_image((p >= 0.5).astype(np.uint8), g) for p, g in zip(P, gts)])


def phase_synth(a):
    method = a.methods[0]; keep = [METHODS.index(method)]; nb = NBITS[method]
    X, Y, Mid, Ms = load_synth(a.data, keep, nb)
    perm = np.random.default_rng(0).permutation(len(X)); held, tr = perm[:300], perm[300:]
    real, test = load_real("train", a.methods, nb), load_real("test", a.methods, nb)
    print(f"[{a.name}] {method} on {a.backbone}; synthetic {len(tr)} + {len(held)} held out; real {len(real['names'])}; "
          f"messages {'OFF (control)' if a.no_msg else 'on'}", flush=True)
    model = MsgSeg(a.backbone, method).to_device("cuda").train(); freeze_pretrained_bn(model, a.backbone)
    opt, sched = make_opt(model, a.lr, a.steps, 300); best, t0, run = -1.0, time.time(), None

    def evaluate(step):
        nonlocal best
        s_syn = score(predict(model, X[held], Mid[held], signs(Ms[held], a.no_msg)), Y[held]).mean()
        s = score(predict(model, real["images"], real["mid"], signs(real["msg"], a.no_msg)), real["gt"]); h = real["full"]
        mark = ""
        if s.mean() > best:
            best = s.mean(); mark = "  (best, saved)"
            torch.save({"model": model.state_dict(), "backbone": a.backbone, "methods": a.methods, "step": step,
                        "no_msg": a.no_msg, "msg_model": True}, OUT / "ckpt" / f"f_{a.name}.pt")
        print(f"[{a.name}] step {step}: held-out synthetic={s_syn:.4f}  real-train={s.mean():.4f}  holes={s[h].mean():.3f}  "
              f"patch={s[~h].mean():.3f}  a={model.a.item():.2f}{mark}", flush=True)

    evaluate(0)
    for step in range(1, a.steps + 1):
        idx = tr[np.random.randint(len(tr), size=a.bs)]
        x = torch.from_numpy(X[idx]).permute(0, 3, 1, 2).contiguous().cuda(non_blocking=True)
        loss = train_step(model, opt, sched, x, torch.from_numpy(Y[idx]).float().cuda(), Mid[idx], Ms[idx], a.no_msg, a.w_bits)
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
    backbone, methods, no_msg = ck0["backbone"], ck0["methods"], ck0["no_msg"]; method = methods[0]; nb = NBITS[method]
    keep = [METHODS.index(method)]
    real, test = load_real("train", methods, nb), load_real("test", methods, nb)
    fold = np.random.default_rng(0).permutation(len(real["names"])) % 5      # same folds as g_combine.py
    Xs, Ys, Ms_, Mss = load_synth(a.data, keep, nb)
    R_img = np.stack(real["images"]); n_eval = a.steps // a.eval_every
    oof = np.zeros((n_eval + 1, len(R_img), 128, 128), np.float16); tst = np.zeros((n_eval + 1, len(test["names"]), 128, 128), np.float32)
    print(f"[{a.name}] init f_{a.init} ({backbone}, {method}, messages {'OFF' if no_msg else 'on'}); real {len(R_img)} in 5 folds; "
          f"synthetic {len(Xs)}", flush=True)
    t_start = time.time()
    for k in range(5):
        model = MsgSeg(backbone, method); model.load_state_dict(ck0["model"]); model.to_device("cuda").train()
        freeze_pretrained_bn(model, backbone); opt, sched = make_opt(model, a.lr, a.steps, 100)
        tr_i, ho_i = np.where(fold != k)[0], np.where(fold == k)[0]

        def evaluate(j, step):
            P = predict(model, R_img[ho_i], real["mid"][ho_i], signs(real["msg"][ho_i], no_msg)); oof[j, ho_i] = P
            tst[j] += predict(model, test["images"], test["mid"], signs(test["msg"], no_msg))
            s = score(P, real["gt"][ho_i]); h = real["full"][ho_i]
            print(f"[{a.name}] fold {k} step {step}: held-out real={s.mean():.4f}  holes={s[h].mean():.3f}  patch={s[~h].mean():.3f}  "
                  f"{time.time() - t_start:.0f}s", flush=True)

        evaluate(0, 0); half = a.bs // 2
        for step in range(1, a.steps + 1):
            ir, js = tr_i[np.random.randint(len(tr_i), size=half)], np.random.randint(len(Xs), size=a.bs - half)
            x = torch.from_numpy(np.concatenate([R_img[ir], Xs[js]])).permute(0, 3, 1, 2).contiguous().cuda(non_blocking=True)
            y = torch.from_numpy(np.concatenate([real["gt"][ir], Ys[js]])).float().cuda()
            train_step(model, opt, sched, x, y, np.concatenate([real["mid"][ir], Ms_[js]]),
                       np.concatenate([real["msg"][ir], Mss[js]]), no_msg, a.w_bits)
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
    ap.add_argument("--data", default="synth4")
    ap.add_argument("--no-msg", action="store_true", help="control: messages replaced by zeros")
    ap.add_argument("--w-bits", type=float, default=1.0)
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
