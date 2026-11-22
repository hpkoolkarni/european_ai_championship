"""E2b. Improved WAM fine-tune, following WAM's own recipe (github.com/facebookresearch/watermark-anything: train.py,
losses/detperceptual.py and the published pretraining log, where detection only starts improving after ~165k images):
  - more data: output/synth2/shard_*_{images,masks,method,msgs}.npy (48,000 images, memory-mapped, shared by the GPUs)
  - all 4 GPUs of the node (DistributedDataParallel via torchrun), global batch 64 -> ~1M images in 15k steps
  - WAM's native 256 px input (no stretched position grid); the 512 px images are resized on the GPU
  - WAM's detection loss: per-pixel BCE on the full-resolution map (128-grid label upsampled); the soft-IoU term of the
    competition metric is only added for the last part of training (--iou-from)
  - WAM's second task, message decoding: a small head reads the encoder features averaged over the watermarked area and
    predicts that image's bits (one head per method: 256 / 100 / 100 bits), weight --w-bits. It forces the encoder to
    pick up the watermark signal itself, which is what WAM's training relies on (its decode weight is 6-10x detection)
  - lower learning rate with a long warm-up (WAM: 5e-5 to 8e-5, 10 warm-up epochs)
Rank 0 scores 300 held-out synthetic and the 510 real train images every --eval-every steps (threshold 0.5) and keeps
the best real-train checkpoint in output/ckpt/wam_ft.pt (same format as before, so e_predict.py works unchanged).
Run:  python -m torch.distributed.run --standalone --nproc_per_node=4 output/e_wam_train2.py"""
import argparse
import glob
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

TASK = Path(__file__).resolve().parents[1]
OUT = TASK / "output"
sys.path.insert(0, str(TASK)); sys.path.insert(0, str(OUT))
from wmloc.utils import miou_image                                  # noqa: E402
from e_wam_finetune import WamSeg, MEAN, STD, predict, load_real_train  # noqa: E402

METHODS = ("pixelseal", "trustmark", "stegastamp")
NBITS = (256, 100, 100)


class WamAux(nn.Module):
    """WAM extractor + message heads. Returns the full-resolution mask logits, per-image bit logits and the
    watermarked area (in encoder tokens) used for the message pooling."""

    def __init__(self, size):
        super().__init__()
        self.seg = WamSeg(size)
        self.heads = nn.ModuleList([nn.Linear(768, n) for n in NBITS])

    def forward(self, x_uint8, method, mask128):
        s = self.seg
        x = (x_uint8.float() / 255 - MEAN.to(x_uint8.device)) / STD.to(x_uint8.device)
        x = F.interpolate(x, size=(s.size,) * 2, mode="bilinear", align_corners=False, antialias=True)
        feat = s.image_encoder(x)                                   # B x 768 x size/16 x size/16
        z = s.pixel_decoder(feat)[:, 0]                             # B x size x size
        w = F.adaptive_avg_pool2d(mask128[:, None].float(), feat.shape[-1])        # watermarked share per token
        pooled = (feat * w).sum((2, 3)) / w.sum((2, 3)).clamp_min(1e-3)
        bits = [self.heads[m](pooled[i]) for i, m in enumerate(method.tolist())]
        return z, bits, w.sum((1, 2, 3))


def soft_miou_loss(z128, y):
    p = torch.sigmoid(z128); q, n = 1 - p, 1 - y; d = (1, 2)
    iou1 = (p * y).sum(d) / (p + y - p * y).sum(d).clamp_min(1e-6)
    iou0 = (q * n).sum(d) / (q + n - q * n).sum(d).clamp_min(1e-6)
    return 1 - 0.5 * (iou0 + iou1).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--bs", type=int, default=16, help="per GPU")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=1500)
    ap.add_argument("--w-bits", type=float, default=1.0)
    ap.add_argument("--iou-from", type=float, default=0.6, help="add the soft-mIoU term after this share of steps")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--data", default="synth2", help="synthetic data folder under output/ (.npy shards with msgs)")
    a = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world, local = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local); dev = torch.device("cuda", local)
    torch.manual_seed(rank)
    log = (lambda *s: print(*s, flush=True)) if rank == 0 else (lambda *s: None)

    files = sorted(glob.glob(str(OUT / a.data / "shard_*_images.npy")))
    IM = [np.load(f, mmap_mode="r") for f in files]
    MK = [np.load(f.replace("_images", "_masks")) for f in files]
    ME = [np.load(f.replace("_images", "_method")) for f in files]
    MS = [np.load(f.replace("_images", "_msgs")) for f in files]
    index = [(s, i) for s in range(len(files)) for i in range(len(IM[s]))]
    perm = np.random.default_rng(0).permutation(len(index))
    held = [index[k] for k in perm[:300]]; train_ids = [index[k] for k in perm[300:]]
    log(f"synthetic: {len(train_ids)} train + {len(held)} held out from {len(files)} shards; world size {world}")
    rng = np.random.default_rng(1000 + rank)

    def batch(ids):
        x = torch.from_numpy(np.stack([IM[s][i] for s, i in ids])).permute(0, 3, 1, 2).contiguous()
        y = torch.from_numpy(np.stack([MK[s][i] for s, i in ids]))
        me = torch.tensor([int(ME[s][i]) for s, i in ids])
        ms = torch.from_numpy(np.stack([MS[s][i] for s, i in ids]))
        return x.to(dev, non_blocking=True), y.to(dev).float(), me, ms.to(dev).float()

    model = WamAux(a.size); model.seg.load_wam(OUT / "third_party" / "wam_mit.pth"); model.to(dev).train()
    ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local], find_unused_parameters=True)
    opt = torch.optim.AdamW([{"params": model.seg.image_encoder.parameters(), "lr": a.lr},
                             {"params": model.seg.pixel_decoder.parameters(), "lr": a.lr},
                             {"params": model.heads.parameters(), "lr": a.lr * 10}], weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / a.warmup) * max(0.02, 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps))))

    real = load_real_train() if rank == 0 else None
    best = -1.0

    def evaluate(step):
        nonlocal best
        seg = model.seg; seg.eval()
        P = predict(seg, [IM[s][i] for s, i in held])
        s_syn = np.mean([miou_image((p >= 0.5).astype(np.uint8), MK[s][i]) for p, (s, i) in zip(P, held)])
        P = predict(seg, real["images"])
        sc = np.array([miou_image((p >= 0.5).astype(np.uint8), g) for p, g in zip(P, real["gts"])])
        parts = []
        for meth in METHODS:
            for full, name in ((True, "holes"), (False, "patch")):
                sel = [i for i in range(len(sc)) if real["method"][i] == meth and real["full"][i] == full]
                parts.append(f"{meth[:5]}-{name}={sc[sel].mean():.3f}")
        mark = ""
        if sc.mean() > best:
            best = sc.mean(); mark = "  (best, saved)"
            torch.save({"model": seg.state_dict(), "size": a.size, "step": step, "score": float(best)}, OUT / "ckpt" / "wam_ft.pt")
        seg.train()
        log(f"step {step}: held-out synthetic={s_syn:.4f}  real-train={sc.mean():.4f}  {'  '.join(parts)}{mark}")

    if rank == 0:
        evaluate(0)
    dist.barrier()
    t0 = time.time(); run = {}
    for step in range(1, a.steps + 1):
        ids = [train_ids[k] for k in rng.integers(len(train_ids), size=a.bs)]
        x, y, me, ms = batch(ids)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z, bits, area = ddp(x, me, y)
        z = z.float()
        y_full = F.interpolate(y[:, None], size=z.shape[-2:], mode="nearest")[:, 0]
        det = F.binary_cross_entropy_with_logits(z, y_full)
        bl, accs = [], []
        for i, m in enumerate(me.tolist()):
            if area[i] > 0.5:                                       # at least half a token carries the watermark
                tgt = ms[i, :NBITS[m]]
                bl.append(F.binary_cross_entropy_with_logits(bits[i].float(), tgt))
                accs.append(((bits[i].float() > 0).float() == tgt).float().mean())
        bit_loss = torch.stack(bl).mean() if bl else z.sum() * 0
        loss = det + a.w_bits * bit_loss
        if step > a.iou_from * a.steps:
            loss = loss + soft_miou_loss(F.adaptive_avg_pool2d(z[:, None], 128)[:, 0], y)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        for k, v in (("det", det.item()), ("bits", bit_loss.item()), ("bitacc", torch.stack(accs).mean().item() if accs else 0.5)):
            run[k] = v if k not in run else 0.98 * run[k] + 0.02 * v
        if step % 100 == 0:
            log(f"step {step}  detect={run['det']:.4f}  bits={run['bits']:.4f}  bit-acc={run['bitacc']:.3f}  {time.time() - t0:.0f}s")
        stop = torch.tensor([float((time.time() - t0) / 60 > a.minutes)], device=dev)
        dist.all_reduce(stop, op=dist.ReduceOp.MAX)                # all GPUs stop together
        if step % a.eval_every == 0 or step == a.steps or stop.item():
            if rank == 0:
                evaluate(step)
            dist.barrier()
        if stop.item():
            log(f"time limit reached at step {step}"); break
    log(f"BEST real-train score={best:.4f}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
