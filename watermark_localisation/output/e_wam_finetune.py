"""E2. Fine-tune the Watermark Anything (WAM, MIT weights) extractor to segment PixelSeal / TrustMark / StegaStamp.
Keeps WAM's ViT-B encoder and pixel decoder; output channel 0 (WAM's "watermarked or not" map) is the mask logit,
WAM's 32 message channels are ignored. Runs at 512 x 512 (absolute position grid stretched from 16x16 to 32x32) so
small patches keep their fine pattern. Logits are averaged over 4 x 4 pixel cells onto the official 128 x 128 grid.
Trains only on output/synth/shard_*.npz; every --eval-every steps it scores the 510 real train images with the
official mIoU (threshold 0.5) and keeps the best checkpoint in output/ckpt/wam_ft.pt.
Run:  .venv/bin/python output/e_wam_finetune.py --steps 4000 --minutes 75"""
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
sys.path.insert(0, str(TASK)); sys.path.insert(0, str(OUT / "third_party" / "watermark-anything"))
from wmloc.data import LocalisationDataset                         # noqa: E402
from wmloc.utils import miou_image                                  # noqa: E402
from watermark_anything.modules.vit import ImageEncoderViT          # noqa: E402
from watermark_anything.modules.pixel_decoder import PixelDecoder   # noqa: E402

METHODS = ("pixelseal", "trustmark", "stegastamp")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class WamSeg(nn.Module):
    """WAM extractor (configs/extractor.yaml 'sam_base') with channel 0 used as the watermark mask logit."""

    def __init__(self, size=512):
        super().__init__()
        self.size = size
        self.image_encoder = ImageEncoderViT(img_size=256, patch_size=16, embed_dim=768, out_chans=768, depth=12,
                                             num_heads=12, mlp_ratio=4, qkv_bias=True, use_rel_pos=True,
                                             window_size=8, global_attn_indexes=(2, 5, 8, 11))
        self.pixel_decoder = PixelDecoder(embed_dim=768, nbits=32, upscale_stages=[4, 2, 2],
                                          upscale_type="bilinear", sigmoid_output=False)
        if size != 256:                                            # placeholder of the right shape, filled by load_*
            self.image_encoder.pos_embed = nn.Parameter(torch.zeros(1, size // 16, size // 16, 768))

    def load_wam(self, path):
        sd = torch.load(path, map_location="cpu")
        sd = sd.get("model", sd)
        det = {k[len("detector."):]: v for k, v in sd.items() if k.startswith("detector.")}
        pe = det.pop("image_encoder.pos_embed")                    # 1 x 16 x 16 x 768, trained at 256 px
        missing, unexpected = self.load_state_dict(det, strict=False)
        assert missing == ["image_encoder.pos_embed"] and not unexpected, (missing, unexpected)
        g = self.size // 16
        pe = F.interpolate(pe.permute(0, 3, 1, 2), size=(g, g), mode="bicubic", align_corners=False).permute(0, 2, 3, 1)
        self.image_encoder.pos_embed.data.copy_(pe)               # relative positions are resized inside WAM's attention

    def forward(self, x_uint8):
        """B x 3 x 512 x 512 uint8 -> B x 128 x 128 logits on the official grid."""
        x = (x_uint8.float() / 255 - MEAN.to(x_uint8.device)) / STD.to(x_uint8.device)
        if x.shape[-1] != self.size:
            x = F.interpolate(x, size=(self.size,) * 2, mode="bilinear", align_corners=False, antialias=True)
        z = self.pixel_decoder(self.image_encoder(x))[:, 0:1]
        return F.adaptive_avg_pool2d(z, 128)[:, 0]


def load_model(path, device="cuda"):
    ck = torch.load(path, map_location="cpu")
    m = WamSeg(ck["size"]); m.load_state_dict(ck["model"])
    return m.to(device).eval()


def loss_fn(z, y):
    """BCE + (1 - soft mIoU over both classes), the per-image metric of the task made differentiable."""
    bce = F.binary_cross_entropy_with_logits(z, y)
    p = torch.sigmoid(z); q, n = 1 - p, 1 - y; d = (1, 2)
    iou1 = (p * y).sum(d) / (p + y - p * y).sum(d).clamp_min(1e-6)
    iou0 = (q * n).sum(d) / (q + n - q * n).sum(d).clamp_min(1e-6)
    return bce + 1 - 0.5 * (iou0 + iou1).mean()


@torch.no_grad()
def predict(model, images, bs=16):
    """List/array of H x W x 3 uint8 images (512 x 512) -> N x 128 x 128 probabilities."""
    out = []
    for i in range(0, len(images), bs):
        x = torch.from_numpy(np.stack(images[i:i + bs])).permute(0, 3, 1, 2).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = model(x)
        out.append(torch.sigmoid(z.float()).cpu().numpy())
    return np.concatenate(out)


def load_real_train():
    ds = LocalisationDataset(TASK / "release", "train")
    items = [ds[i] for i in range(len(ds))]
    return {"images": [it["image"] for it in items], "gts": [it["mask128"] for it in items],
            "method": [it["method"] for it in items], "full": [bool(it["full_watermark"]) for it in items]}


def evaluate(model, real):
    model.eval()
    P = predict(model, real["images"])
    model.train()
    s = np.array([miou_image((p >= 0.5).astype(np.uint8), g) for p, g in zip(P, real["gts"])])
    parts = []
    for meth in METHODS:
        for full, name in ((True, "holes"), (False, "patch")):
            sel = [i for i in range(len(s)) if real["method"][i] == meth and real["full"][i] == full]
            parts.append(f"{meth[:5]}-{name}={s[sel].mean():.3f}")
    return float(s.mean()), "  ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--minutes", type=float, default=75)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=500)
    a = ap.parse_args()
    torch.manual_seed(0)

    files = sorted(glob.glob(str(OUT / "synth" / "shard_*.npz")))
    X, Y = [], []
    for f in files:
        d = np.load(f); X.append(d["images"]); Y.append(d["masks"])
    X = torch.from_numpy(np.concatenate(X)); Y = torch.from_numpy(np.concatenate(Y))
    print(f"synthetic images: {len(X)} from {len(files)} shards", flush=True)
    real = load_real_train()

    model = WamSeg(a.size); model.load_wam(OUT / "third_party" / "wam_mit.pth"); model.cuda().train()
    opt = torch.optim.AdamW([{"params": model.image_encoder.parameters(), "lr": a.lr * 0.3},
                             {"params": model.pixel_decoder.parameters(), "lr": a.lr}], weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))

    best, info = evaluate(model, real)
    print(f"step 0 (WAM as downloaded): real-train score={best:.4f}  {info}", flush=True)
    best = -1.0
    t0, run = time.time(), 0.0
    for step in range(1, a.steps + 1):
        idx = torch.randint(len(X), (a.bs,))
        x = X[idx].permute(0, 3, 1, 2).contiguous().cuda(non_blocking=True)
        y = Y[idx].float().cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = model(x)
        loss = loss_fn(z.float(), y)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 50 == 0:
            print(f"step {step}  loss={run:.4f}  {time.time() - t0:.0f}s", flush=True)
        done = (time.time() - t0) / 60 > a.minutes
        if step % a.eval_every == 0 or step == a.steps or done:
            score, info = evaluate(model, real)
            mark = ""
            if score > best:
                best = score; mark = "  (best, saved)"
                torch.save({"model": model.state_dict(), "size": a.size, "step": step, "score": score},
                           OUT / "ckpt" / "wam_ft.pt")
            print(f"step {step}: real-train score={score:.4f}  {info}{mark}", flush=True)
        if done:
            print(f"time limit reached at step {step}")
            break
    print(f"BEST real-train score={best:.4f}")


if __name__ == "__main__":
    main()
