#!/bin/bash
# WAM v2 fine-tune on matched synthetic data with random watermark strength, ONE job on ONE node:
#   1. 24,000 images with e_synth3.py --strength (~15 min)      -> output/synth4/shard_<k>_*.npy
#      (same shapes and photos as synth3, plus PixelSeal / StegaStamp embedded at a random 50-100 % / 70-100 % strength,
#       matching the decoder margins measured on the real train images)
#   2. WAM fine-tune on all 4 GPUs (<= 60 min)                 -> output/ckpt/wam_ft.pt
#   3. thresholds by cross-validation + test prediction (~3 min) -> output/probs_wam.npz, output/sub_e_wam.pt
# Submit only when the team has no other job running or queued (the waiter does this). Nothing is uploaded.
#SBATCH --job-name=tt_wm_wam3
#SBATCH --account=training2666
#SBATCH --reservation=cispahack
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=01:40:00
#SBATCH --output=output/logs/%x_%j.log
cd "$SLURM_SUBMIT_DIR"   # submit from watermark_localisation/: sbatch output/run_e3.sbatch
if [ ! -f output/synth4/shard_3_msgs.npy ]; then
  srun --cpu-bind=none bash -c 'CUDA_VISIBLE_DEVICES=$SLURM_LOCALID .venv/bin/python output/e_synth3.py --nshards 4 --n 6000 --seed 4 --strength --outdir synth4' || exit 1
fi
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 output/e_wam_train2.py --data synth4 --steps 15000 --minutes 60 || exit 1
.venv/bin/python output/e_predict.py
