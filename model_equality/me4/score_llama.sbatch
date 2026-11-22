#!/bin/bash
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:45:00
#SBATCH --output=score_llama_%j.log
# inference only: log-likelihood of the saved scored answers under local Qwen2.5 checkpoints (no API, no training)
R="$SLURM_SUBMIT_DIR"   # submit from model_equality/me4/: sbatch score_llama.sbatch
export HF_HOME=$R/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd $R
nvidia-smi --query-gpu=name,memory.total --format=csv
srun venv/bin/python -u score.py unsloth/Llama-3.2-1B-Instruct unsloth/Llama-3.2-3B-Instruct
