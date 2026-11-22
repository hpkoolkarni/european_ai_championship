#!/bin/bash
# F as one batch job on ONE node: four backup models side by side, one per A100 (organisers' one-node rule).
#   GPU 0  stego_all  steganalysis-style ResNet-34, all three watermarks
#   GPU 1  ps_spec    PixelSeal specialist on PixelSeal's own detector backbone
#   GPU 2  tm_spec    TrustMark specialist on TrustMark's own decoder backbone
#   GPU 3  stego_ss   steganalysis-style ResNet-34, StegaStamp only (its decoder is ONNX, so no own backbone)
# Each writes output/logs/f_<name>.log, output/ckpt/f_<name>.pt and output/probs_f_<name>.npz. Nothing is uploaded.
#SBATCH --job-name=tt_wm_backups
#SBATCH --account=training2666
#SBATCH --reservation=cispahack
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=01:45:00
#SBATCH --output=output/logs/%x_%j.log
cd "$SLURM_SUBMIT_DIR"   # submit from watermark_localisation/: sbatch output/run_f_parallel.sbatch
srun --cpu-bind=none bash -c '
export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID}
case ${SLURM_LOCALID} in
  0) N=stego_all; A="--backbone stego";;
  1) N=ps_spec;   A="--backbone pixelseal --methods pixelseal";;
  2) N=tm_spec;   A="--backbone trustmark --methods trustmark";;
  3) N=stego_ss;  A="--backbone stego --methods stegastamp";;
esac
.venv/bin/python output/f_seg_train.py ${A} --name ${N} --minutes 80 > output/logs/f_${N}.log 2>&1'
