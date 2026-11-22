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
| H | `h_box_search.py` | Known-message box search for patch images: a box from which the official decoder reads the known message is the watermarked rectangle (TrustMark, StegaStamp). |

The `run_*.sbatch` files are the Slurm jobs that ran these steps (one node, 4 GPUs, one process per GPU); their header comments say what each job ran and why.

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
```

Uploading (`--submit` on `a_baseline.py`, `e_predict.py`, `g_combine.py`) needs `API_KEY` in the environment.
