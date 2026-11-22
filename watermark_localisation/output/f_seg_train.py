"""F. Backup segmenters, trained on the same synthetic data as the WAM run (output/synth/shard_*.npz).
  --backbone stego      ImageNet ResNet-34 with the steganalysis changes: first conv stride 1 (keeps full resolution),
                        extra input planes = 3 high-pass channels (fine detail, where watermarks live) + 3 method planes
  --backbone pixelseal  PixelSeal's own detector backbone (ConvNeXt-V2), already tuned to the PixelSeal pattern
  --backbone trustmark  TrustMark's own decoder backbone (ResNet-50), already tuned to the TrustMark pattern
All share one head: every encoder stage is resized to the 128 x 128 grid, projected to 128 channels and summed, then
two 3x3 conv blocks give one logit per official cell. Loss = BCE + (1 - soft mIoU over both classes), as for WAM.
--methods restricts training and scoring to some watermarks (specialists).
Every --eval-every steps it scores (a) 300 held-out synthetic images: did it learn? and (b) the real train images of
its methods: does it carry over? The best real-train checkpoint goes to output/ckpt/f_<name>.pt; at the end its
probabilities for the real train and test images of its methods go to output/probs_f_<name>.npz (used by g_combine.py).
Run:  .venv/bin/python output/f_seg_train.py --backbone stego --name stego_all"""
import argparse
import glob
import math
import sys
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

TASK = Path(__file__).resolve().parents[1]
OUT = TASK / "output"
sys.path.insert(0, str(TASK))
from wmloc.data import LocalisationDataset                         # noqa: E402
from wmloc.utils import miou_image                                  # noqa: E402
from wmloc.watermarks import load_watermark                         # noqa: E402

METHODS = ("pixelseal", "trustmark", "stegastamp")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class StegoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = timm.create_model("resnet34", pretrained=True, features_only=True, in_chans=9)
        self.net.conv1.stride = (1, 1)                              # no downsampling in the first layer
        lap = torch.tensor([[0., -1., 0.], [-1., 4., -1.], [0., -1., 0.]])
        self.register_buffer("lap", lap.view(1, 1, 3, 3).repeat(3, 1, 1, 1))

    def forward(self, x01, onehot):
        rgb = (x01 - MEAN.to(x01.device)) / STD.to(x01.device)
        high = F.conv2d(x01, self.lap, padding=1, groups=3) * 20    # watermark residuals are only ~1/255
        m = onehot[:, :, None, None].expand(-1, -1, *x01.shape[-2:])
        return self.net(torch.cat([rgb, high, m], 1))


class PixelSealEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.det = load_watermark("pixelseal", models_dir=TASK / "models", device="cpu").det

    def forward(self, x01, onehot):
        x, feats = self.det.preprocess(x01), []
        for down, stage in zip(self.det.convnext.downsample_layers, self.det.convnext.stages):
            x = stage(down(x)); feats.append(x)
        return feats


class TrustMarkEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = load_watermark("trustmark", models_dir=TASK / "models", device="cpu").net

    def forward(self, x01, onehot):
        n = self.net; x = n.maxpool(n.relu(n.bn1(n.conv1(x01 * 2 - 1)))); feats = []
        for k in ("layer1", "layer2", "layer3", "layer4"):
            x = getattr(n, k)(x); feats.append(x)
        return feats


ENCODERS = {"stego": StegoEncoder, "pixelseal": PixelSealEncoder, "trustmark": TrustMarkEncoder}


class Seg(nn.Module):
    def __init__(self, backbone, width=128):
        super().__init__()
        self.enc = ENCODERS[backbone]()
        with torch.no_grad():
            chans = [f.shape[1] for f in self.enc(torch.zeros(1, 3, 512, 512), torch.zeros(1, 3))]
        self.proj = nn.ModuleList([nn.Conv2d(c, width, 1) for c in chans])
        block = lambda: [nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(inplace=True)]  # noqa: E731
        self.out = nn.Sequential(*block(), *block(), nn.Conv2d(width, 1, 1))

    def forward(self, x_uint8, onehot):
        """B x 3 x 512 x 512 uint8, B x 3 method one-hot -> B x 128 x 128 logits on the official grid."""
        s = 0
        for f, p in zip(self.enc(x_uint8.float() / 255, onehot), self.proj):
            f = p(f)
            f = F.adaptive_avg_pool2d(f, 128) if f.shape[-1] > 128 else F.interpolate(f, size=(128, 128), mode="bilinear", align_corners=False)
            s = s + f
        return self.out(s)[:, 0]


def freeze_pretrained_bn(model, backbone):
    """Keep the batch-norm statistics of the watermark decoders as trained (TrustMark is a ResNet with BN)."""
    if backbone == "trustmark":
        for m in model.enc.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


def loss_fn(z, y):
    bce = F.binary_cross_entropy_with_logits(z, y)
    p = torch.sigmoid(z); q, n = 1 - p, 1 - y; d = (1, 2)
    iou1 = (p * y).sum(d) / (p + y - p * y).sum(d).clamp_min(1e-6)
    iou0 = (q * n).sum(d) / (q + n - q * n).sum(d).clamp_min(1e-6)
    return bce + 1 - 0.5 * (iou0 + iou1).mean()


def onehot(method_ids):
    return F.one_hot(torch.as_tensor(method_ids, dtype=torch.long), 3).float()


@torch.no_grad()
def predict(model, images, method_ids, bs=16):
    """N x 512 x 512 x 3 uint8 (array or list) -> N x 128 x 128 probabilities."""
    model.eval(); out = []
    for i in range(0, len(images), bs):
        x = torch.from_numpy(np.stack(images[i:i + bs])).permute(0, 3, 1, 2).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = model(x, onehot(method_ids[i:i + bs]).cuda())
        out.append(torch.sigmoid(z.float()).cpu().numpy())
    model.train()
    return np.concatenate(out)


def score(P, gts):
    return np.array([miou_image((p >= 0.5).astype(np.uint8), g) for p, g in zip(P, gts)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=sorted(ENCODERS), required=True)
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--minutes", type=float, default=80)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-4, help="backbone learning rate; the new head uses 10x")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--data", default="synth", help="synthetic data folder under output/ (.npz or .npy shards)")
    a = ap.parse_args()
    torch.manual_seed(0)
    keep = [METHODS.index(m) for m in a.methods]

    X, Y, Mid = [], [], []
    npy = sorted(glob.glob(str(OUT / a.data / "shard_*_images.npy")))
    for f in npy:                                                 # e_synth3 format: plain .npy files, read only this method
        me = np.load(f.replace("_images", "_method")); sel = np.where(np.isin(me, keep))[0]
        X.append(np.load(f, mmap_mode="r")[sel]); Y.append(np.load(f.replace("_images", "_masks"))[sel]); Mid.append(me[sel])
    for f in [] if npy else sorted(glob.glob(str(OUT / a.data / "shard_*.npz"))):
        d = np.load(f); sel = np.isin(d["method"], keep)
        X.append(d["images"][sel]); Y.append(d["masks"][sel]); Mid.append(d["method"][sel])
    X, Y, Mid = np.concatenate(X), np.concatenate(Y), np.concatenate(Mid)
    perm = np.random.default_rng(0).permutation(len(X)); held, tr_idx = perm[:300], perm[300:]
    Xt, Yt, Mt = torch.from_numpy(X[tr_idx]), torch.from_numpy(Y[tr_idx]), Mid[tr_idx]
    print(f"[{a.name}] synthetic: {len(tr_idx)} train + {len(held)} held out, methods {a.methods}", flush=True)

    ds = LocalisationDataset(TASK / "release", "train", methods=a.methods)
    real = [ds[i] for i in range(len(ds))]
    r_img, r_gt = [it["image"] for it in real], [it["mask128"] for it in real]
    r_mid = np.array([METHODS.index(it["method"]) for it in real]); r_full = np.array([bool(it["full_watermark"]) for it in real])

    model = Seg(a.backbone).cuda().train(); freeze_pretrained_bn(model, a.backbone)
    opt = torch.optim.AdamW([{"params": model.enc.parameters(), "lr": a.lr},
                             {"params": list(model.proj.parameters()) + list(model.out.parameters()), "lr": a.lr * 10}],
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 300) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))

    def evaluate(step):
        nonlocal best
        s_syn = score(predict(model, X[held], Mid[held]), Y[held]).mean()
        P = predict(model, r_img, r_mid); s = score(P, r_gt)
        parts = "  ".join(f"{METHODS[m][:5]}-{'holes' if f else 'patch'}={s[(r_mid == m) & (r_full == f)].mean():.3f}"
                          for m in keep for f in (True, False))
        mark = ""
        if s.mean() > best:
            best = s.mean(); mark = "  (best, saved)"
            torch.save({"model": model.state_dict(), "backbone": a.backbone, "methods": a.methods, "step": step},
                       OUT / "ckpt" / f"f_{a.name}.pt")
        model.train(); freeze_pretrained_bn(model, a.backbone)
        print(f"[{a.name}] step {step}: held-out synthetic={s_syn:.4f}  real-train={s.mean():.4f}  {parts}{mark}", flush=True)

    best = -1.0
    evaluate(0)
    t0, run = time.time(), None
    for step in range(1, a.steps + 1):
        idx = torch.randint(len(Xt), (a.bs,))
        x = Xt[idx].permute(0, 3, 1, 2).contiguous().cuda(non_blocking=True)
        y = Yt[idx].float().cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = model(x, onehot(Mt[idx.numpy()]).cuda())
        loss = loss_fn(z.float(), y)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
        if step % 100 == 0:
            print(f"[{a.name}] step {step}  loss={run:.4f}  {time.time() - t0:.0f}s", flush=True)
        done = (time.time() - t0) / 60 > a.minutes
        if step % a.eval_every == 0 or step == a.steps or done:
            evaluate(step)
        if done:
            print(f"[{a.name}] time limit reached at step {step}"); break
    print(f"[{a.name}] BEST real-train score={best:.4f}", flush=True)

    ck = torch.load(OUT / "ckpt" / f"f_{a.name}.pt", map_location="cpu"); model.load_state_dict(ck["model"])
    te = LocalisationDataset(TASK / "release", "test", methods=a.methods)
    t_mid = np.array([METHODS.index(m["method"]) for m in te.meta])
    p_te = np.concatenate([predict(model, [te[i]["image"] for i in range(j, min(j + 64, len(te)))], t_mid[j:j + 64])
                           for j in range(0, len(te), 64)])
    np.savez(OUT / f"probs_f_{a.name}.npz", names_train=[it["image_name"] for it in real],
             p_train=predict(model, r_img, r_mid).astype(np.float16),
             names_test=[m["image_name"] for m in te.meta], p_test=p_te.astype(np.float16))
    print(f"[{a.name}] wrote probs_f_{a.name}.npz (best checkpoint from step {ck['step']})")


if __name__ == "__main__":
    main()
