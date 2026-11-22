#!/bin/bash
# E2 + E3 as a batch job on one A100: fine-tune WAM (stops after 75 minutes at the latest), then predict and write
# output/sub_e_wam.pt. Nothing is uploaded.
#SBATCH --job-name=tt_wm_wam
#SBATCH --account=training2666
#SBATCH --reservation=cispahack
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:45:00
#SBATCH --output=output/logs/%x_%j.log
cd "$SLURM_SUBMIT_DIR"   # submit from watermark_localisation/: sbatch output/run_e_train.sbatch
.venv/bin/python output/e_wam_finetune.py --steps 4000 --minutes 75 && .venv/bin/python output/e_predict.py
