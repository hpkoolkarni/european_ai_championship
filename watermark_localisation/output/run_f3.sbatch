#!/bin/bash
# Synthetic data v3 (shapes and photos matched to the real train data) + specialists retrained on it, as ONE job on
# ONE node (team rule: at most one job, running or queued, per team):
#   1. 24,000 images with e_synth3.py, one process per GPU (~20-25 min)   -> output/synth3/shard_<k>_*.npy
#   2. four models side by side, one per GPU (<= 45 min of training each)
#      GPU 0  ps_v3   PixelSeal specialist on PixelSeal's own detector backbone
#      GPU 1  tm_v3   TrustMark specialist on TrustMark's own decoder backbone
#      GPU 2  ss_v3   StegaStamp specialist, steganalysis-style ResNet-34
#      GPU 3  pss_v3  PixelSeal on the steganalysis ResNet-34 (second opinion for PixelSeal)
# New names, so the first backup models (f_ps_spec, f_tm_spec, f_stego_ss) stay untouched. Nothing is uploaded.
#SBATCH --job-name=tt_wm_v3
#SBATCH --account=training2666
#SBATCH --reservation=cispahack
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=01:45:00
#SBATCH --output=output/logs/%x_%j.log
cd "$SLURM_SUBMIT_DIR"   # submit from watermark_localisation/: sbatch output/run_f3.sbatch
if [ ! -f output/synth3/shard_3_msgs.npy ]; then
  srun --cpu-bind=none bash -c 'CUDA_VISIBLE_DEVICES=$SLURM_LOCALID .venv/bin/python output/e_synth3.py --nshards 4 --n 6000 --outdir synth3' || exit 1
fi
srun --cpu-bind=none bash -c '
export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID}
case ${SLURM_LOCALID} in
  0) N=ps_v3;  A="--backbone pixelseal --methods pixelseal";;
  1) N=tm_v3;  A="--backbone trustmark --methods trustmark";;
  2) N=ss_v3;  A="--backbone stego --methods stegastamp";;
  3) N=pss_v3; A="--backbone stego --methods pixelseal";;
esac
.venv/bin/python output/f_seg_train.py ${A} --name ${N} --data synth3 --minutes 45 > output/logs/f_${N}.log 2>&1'
