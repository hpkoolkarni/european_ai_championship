#!/bin/bash
# E1 as a batch job: 4 processes, one per A100, each writes output/synth/shard_<k>.npz (1,500 images each, 6,000 total).
#SBATCH --job-name=tt_wm_synth
#SBATCH --account=training2666
#SBATCH --reservation=cispahack
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=output/logs/%x_%j.log
cd "$SLURM_SUBMIT_DIR"   # submit from watermark_localisation/: sbatch output/run_e_synth.sbatch
srun --cpu-bind=none bash -c 'CUDA_VISIBLE_DEVICES=$SLURM_LOCALID .venv/bin/python output/e_synth.py --nshards 4 --n 1500'
