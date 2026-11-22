"""J. Exhaustive window scan for TrustMark patch images: decode a grid of windows (9 sizes, stride 20 % of the size), refine
the best non-overlapping hits with h_box_search.search, and append them to the box2 candidates -> output/box3/.
Run (one node, 4 GPUs; copy output/box2/stegastamp_*.npz into output/box3 first):
  srun -n4 --gres=gpu:4 bash -c 'export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID; .venv/bin/python output/j_tm_scan.py'
then: .venv/bin/python output/i_final_combine.py --boxdir box3 --out sub_final.pt"""
import os, sys, time, glob, numpy as np, torch
from pathlib import Path
T = Path(__file__).resolve().parents[1]; OUT = T / 'output'
sys.path.insert(0, str(T)); sys.path.insert(0, str(OUT))
import h_box_search as H
from wmloc.data import LocalisationDataset
meth = 'trustmark'; shard, nsh = int(os.environ['SLURM_PROCID']), int(os.environ['SLURM_NTASKS'])
tr, te = LocalisationDataset(T/'release', 'train'), LocalisationDataset(T/'release', 'test')
hol = H.holes_decision(tr, te, meth); sc = H.Scorer(meth, 'cuda')
BX = {}
for f in glob.glob(str(OUT/'box2'/f'{meth}_*_*.npz')):
    d = np.load(f); BX.update({n: (b, a, m) for n, b, a, m in zip(d['names'].tolist(), d['boxes'], d['accs'], d['margins'])})
W = []
for s in (40, 52, 68, 88, 114, 148, 192, 250, 324):
    st = max(8, int(0.2 * s)); pos = sorted(set(list(range(0, 512 - s + 1, st)) + [512 - s]))
    W += [(y, y + s, x, x + s) for y in pos for x in pos]
items = [('train', tr, i) for i, m in enumerate(tr.meta) if m['method'] == meth and not hol[m['image_name']]] + \
        [('test', te, i) for i, m in enumerate(te.meta) if m['method'] == meth and not hol[m['image_name']]]
items = items[shard::nsh]; out = {'train': [], 'test': []}; t0 = time.time(); ADD = 4
for c, (split, ds, i) in enumerate(items):
    it = ds[i]; n = it['image_name']
    img = torch.from_numpy(it['image']).permute(2, 0, 1)[None].float().div(255).cuda(); msg = torch.from_numpy(it['message'].astype(np.float32) * 2 - 1).cuda()
    acc, mar = sc(img, W, msg); order = np.argsort(-mar); hits = []
    for j in order:
        if acc[j] < 0.7 or len(hits) == ADD: break
        if all(H.box_iou(W[j], h) < 0.3 for h in hits): hits.append(W[j])
    res = [H.search(sc, img, msg, b, meth) for b in hits]; res = [r for r in res if r is not None]
    res += [((0, 0, 0, 0), 0.0, -1e9)] * (ADD - len(res))
    b0, a0, m0 = BX.get(n, (np.zeros((0, 4), np.int32), np.zeros(0, np.float32), np.zeros(0, np.float32)))
    out[split].append((n, np.concatenate([np.asarray(b0, np.int32).reshape(-1, 4), np.array([r[0] for r in res], np.int32)]),
                       np.concatenate([a0, np.array([r[1] for r in res], np.float32)]), np.concatenate([m0, np.array([r[2] for r in res], np.float32)])))
    if (c + 1) % 10 == 0: print(f"shard {shard}: {c+1}/{len(items)}  {time.time()-t0:.0f}s  last best scan acc {acc.max():.3f}", flush=True)
for split, L in out.items():
    if not L: continue
    K = max(len(x[1]) for x in L)
    pad = lambda v, fill: np.concatenate([v, np.full((K - len(v),) + v.shape[1:], fill, v.dtype)])
    np.savez(OUT/'box3'/f'{meth}_{split}_s{shard}.npz', names=[x[0] for x in L], boxes=np.stack([pad(x[1], 0) for x in L]),
             accs=np.stack([pad(x[2], 0.0) for x in L]), margins=np.stack([pad(x[3], -1e9) for x in L]))
print(f"shard {shard} done: {len(items)} images, {time.time()-t0:.0f}s", flush=True)
