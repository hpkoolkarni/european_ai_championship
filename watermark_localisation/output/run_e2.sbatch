#!/bin/bash
# Improved WAM pipeline as ONE job on ONE node (team rule: at most one job, running or queued, per team):
#   1. 48,000 new synthetic images with messages (4 GPUs, ~20 min)   -> output/synth2/shard_<k>_*.npy
#   2. WAM fine-tune on all 4 GPUs (<= 60 min)                       -> output/ckpt/wam_ft.pt
#   3. thresholds by cross-validation + test prediction (~3 min)     -> output/probs_wam.npz, output/sub_e_wam.pt
# Nothing is uploaded. Submit only when the team has no other job running or queued.
#SBATCH --job-name=tt_wm_wam2
#SBATCH --account=training2666
#SBATCH --reservation=cispahack
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=01:50:00
#SBATCH --output=output/logs/%x_%j.log
cd "$SLURM_SUBMIT_DIR"   # submit from watermark_localisation/: sbatch output/run_e2.sbatch
if [ ! -f output/synth2/shard_3_msgs.npy ]; then
  srun --cpu-bind=none bash -c 'CUDA_VISIBLE_DEVICES=$SLURM_LOCALID .venv/bin/python output/e_synth.py --nshards 4 --n 12000 --seed 1 --outdir synth2 --npy' || exit 1
fi
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 output/e_wam_train2.py --steps 15000 --minutes 60 || exit 1
.venv/bin/python output/e_predict.py
