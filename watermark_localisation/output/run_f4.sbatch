#!/bin/bash
# Specialists retrained on synth4 (shapes, sharpness AND watermark strength matched to the real train data), ONE job on
# ONE node (team rule: at most one job, running or queued, per team). synth4 already exists (made by run_e3.sbatch).
#   GPU 0  ps_v4   PixelSeal specialist on PixelSeal's own detector backbone
#   GPU 1  tm_v4   TrustMark specialist on TrustMark's own decoder backbone
#   GPU 2  ss_v4   StegaStamp specialist, steganalysis-style ResNet-34
#   GPU 3  tss_v4  TrustMark on the steganalysis ResNet-34 (full resolution: may find small pieces better)
# New names, so all earlier models stay untouched. <= 45 min of training each. Nothing is uploaded.
#SBATCH --job-name=tt_wm_v4
#SBATCH --account=training2666
#SBATCH --reservation=cispahack
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=01:10:00
#SBATCH --output=output/logs/%x_%j.log
cd "$SLURM_SUBMIT_DIR"   # submit from watermark_localisation/: sbatch output/run_f4.sbatch
[ -f output/synth4/shard_3_msgs.npy ] || { echo "output/synth4 missing"; exit 1; }
srun --cpu-bind=none bash -c '
export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID}
case ${SLURM_LOCALID} in
  0) N=ps_v4;  A="--backbone pixelseal --methods pixelseal";;
  1) N=tm_v4;  A="--backbone trustmark --methods trustmark";;
  2) N=ss_v4;  A="--backbone stego --methods stegastamp";;
  3) N=tss_v4; A="--backbone stego --methods trustmark";;
esac
.venv/bin/python output/f_seg_train.py ${A} --name ${N} --data synth4 --minutes 45 > output/logs/f_${N}.log 2>&1'
