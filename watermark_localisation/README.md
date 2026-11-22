# Watermark localisation (images)

Every test image carries one of three invisible watermarks (PixelSeal, TrustMark, StegaStamp), either almost everywhere with clean "holes" restored, or only in one or more small patches. The task is to predict the watermarked cells on the official 128 × 128 grid; the score is the per-image mIoU. The message embedded in each image is given in the metadata.

## Layout

| Path | Contents |
|---|---|
| `STARTER_KIT.txt`, `download_models.py`, `submission_template.py`, `wmloc/`, `requirements.txt` | Organisers' starter kit: model wrappers (`embed`, `logits`, `decode`, `features`), data loading, the official mask downsampling and mIoU. |
| `output/` | Our experiments (scripts, Slurm jobs). Every script writes its artefacts (checkpoints, probabilities, submissions, logs) next to itself in `output/`. |
| `output/global_acc.json` | Whole-image bit accuracy per image, from `a_baseline.py` (input of the holes-or-patch decision). |

## Pipeline (in the order it was built)

| Step | Script | Idea |
|---|---|---|
| A | `a_baseline.py` | Classify each image as "holes" or "patch" from the whole-image bit accuracy (threshold per method, 5-fold CV), predict all ones or all zeros. |
| E0 | `setup_wam_coco.sh`, `pack_coco.sh` | Download Watermark Anything (WAM, MIT weights) and COCO val2017; pack the photos into one tar (inode quota). |
| E1 | `e_synth.py` | Synthetic training data with the task's own embedders: COCO photos, four recipes (holes, outpaint, patch, transplant), official label downsampling. |
| E2 | `e_wam_finetune.py`, `e_wam_train2.py` | Fine-tune WAM's extractor as a segmenter; v2 follows WAM's own recipe (DDP on 4 GPUs, 48k images, message-decoding auxiliary head). |
| E3 | `e_synth3.py` | Synthetic data v3, with mask shapes, sizes, elongation and photo sharpness matched to the 510 real train images; optional exact watermark residuals. |
| E3 | `e_predict.py` | Predict with the WAM segmenter; plain vs gated thresholds chosen by CV. |
| F | `f_seg_train.py` | Specialist segmenters on the methods' own decoder backbones (PixelSeal ConvNeXt, TrustMark ResNet-50) or a steganalysis ResNet-34. |
| F2 | `f_real_ft.py` | Honest 5-fold fine-tune on the real train images (mixed with synthetic), out-of-fold predictions; optional pseudo-labels, layer-wise LR decay, SWA. |
| F3 | `f_msg_train.py` | Message-aware specialist: uses the known message as a matched filter (per-cell bit head + message match → mask head). |
| F4 | `f_res_train.py` | F3 plus a residual head trained on the exact watermark residual (dense supervision). |
| G | `g_combine.py` | Per method, choose the best candidate (baseline or any model's probabilities, gated thresholds) by 5-fold CV on the real train images; write the submission. |
| H | `h_box_search.py` | Known-message box search for patch images: a box from which the official decoder reads the known message is the watermarked rectangle (TrustMark, StegaStamp). `--prop-models` takes candidate boxes from several models' maps. |
| I | `i_final_combine.py` | Final submission: per method, greedy ensemble selection (Caruana) over the real-fine-tuned models by 5-fold CV, then the box search on top. |

The `run_*.sbatch` files are the Slurm jobs that ran these steps (one node, 4 GPUs, one process per GPU); their header comments say what each job ran and why.

## Results

Honest 5-fold CV mIoU on the 510 real train images (out-of-fold predictions, thresholds chosen inside the folds). The public leaderboard tracked it closely (CV 0.646-0.655 scored 0.67).

| Step | PixelSeal | TrustMark | StegaStamp | Overall |
|---|---|---|---|---|
| Baseline: all ones / all zeros by the holes-or-patch decision (A) | 0.449 | 0.451 | 0.459 | 0.453 |
| WAM fine-tunes (E2) | baseline level | baseline level | baseline level | 0.449 |
| Specialists on COCO synthetic data (F) | 0.533 | 0.560 | 0.571 | 0.555 |
| + synthetic data matched to the real images (E3, synth3/synth4) | 0.547 | 0.560 | 0.671 | 0.593 |
| + averaging models | 0.557 | 0.573 | 0.674 | 0.601 |
| + 5-fold fine-tune on the real images (F2) | 0.592 | 0.597 | 0.750 | 0.646 |
| + known-message box search (H) | 0.592 | 0.609 | 0.763 | 0.655 |
| + message-aware / residual TrustMark models, box candidates from 5 models | 0.592 | 0.642 | 0.763 | 0.666 |
| + retuned fine-tune (lr 4e-5, 75 % real, 1,250 steps for StegaStamp), ensemble selection (I) | **0.600** | **0.643** | **0.791** | **0.678** |

Final ensembles (`i_final_combine.py`): PixelSeal `ps_v4_r2 + ps_spec_real + psr_v5_real`; TrustMark `tmr_v5_r2 + tmm_v4_real + tm_spec_ft2 + tss_v4_real` with the box search; StegaStamp `ss_v3_r3` with the box search.

## What we learned

- **Real images beat everything else.** The fine-tune on the 510 real images gave +0.04 to +0.10 per method; the specialists' score on real images plateaus early on synthetic data alone.
- **The covers differ.** A simple classifier on image statistics separates the competition images from our COCO covers with AUC 0.91-0.94: the competition images are much cleaner in their fine noise layer (more perfectly flat pixels, more peaked residuals), while COCO photos carry camera and JPEG grain. Light non-local-means denoising of the covers brings them closer (AUC 0.86); passing them through the Stable Diffusion VAE does not (0.95). Not retrained on denoised covers for lack of time.
- **The known message is most useful through the official decoders.** TrustMark reads the known message at ~1.00 bit accuracy from the exact patch box and still ~0.97 with the box edges 10 % off, and never above 0.9 at a wrong place: a GLRT over the box. As model input (F3) the message did not help PixelSeal (exact control run: 0.529 vs 0.529) but the message-aware TrustMark model was the best TrustMark member.
- **Failure analysis.** Holes images fail when the clean pieces are small and many (geometry, not signal strength); PixelSeal patches fail where the watermark itself is barely decodable even at the exact box; small TrustMark patches were decodable but not proposed, which the multi-model box candidates fixed.
- **Did not help (measured):** WAM (memorises 24k images, 256 px input loses the signal), one shared model for all three methods (0.43), the re-embedding matched filter (AUC ~0.5), decoder heat maps and gradient attribution, mask clean-up rules, per-image expected-IoU thresholds (+0.002), a combined holes-or-patch decision (+0.000), layer-wise LR decay, SWA and test pseudo-labels in the fine-tune.

## How to run

From this folder, after the starter-kit setup (`STARTER_KIT.txt`):

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python download_models.py
.venv/bin/python -c "from wmloc.data import unzip; unzip('train.zip', 'release'); unzip('test.zip', 'release')"
bash output/setup_wam_coco.sh            # login node: WAM code and weights, COCO photos
.venv/bin/python output/a_baseline.py    # writes output/global_acc.json
sbatch output/run_f3.sbatch              # example job; submit jobs from this folder
.venv/bin/python output/g_combine.py     # combined submission: output/sub_g_best.pt
.venv/bin/python output/i_final_combine.py --boxdir box2 --out sub_final.pt   # final submission (after the F2-F4 and H jobs)
```

Uploading (`--submit` on `a_baseline.py`, `e_predict.py`, `g_combine.py`; or `submission_template.submit(path)`) needs `API_KEY` in the environment.
