#!/bin/bash
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=00:40:00
#SBATCH --output=toolfit_%j.log
# inference only: base-model probability of each tool choice, families A (Qwen2.5-3B) and B (Qwen2.5-1.5B)
R="$SLURM_SUBMIT_DIR"   # submit from model_equality/reference_models/: sbatch toolfit.sbatch
export HF_HOME=$R/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd $R
CUDA_VISIBLE_DEVICES=0 venv/bin/python -u tools_fit.py Qwen/Qwen2.5-3B-Instruct ep-01,ep-02,ep-04,ep-12,ep-14,ep-17,ep-18,ep-23,ep-24,ep-25,ep-33 &
CUDA_VISIBLE_DEVICES=1 venv/bin/python -u tools_fit.py Qwen/Qwen2.5-1.5B-Instruct ep-03,ep-06,ep-07,ep-08,ep-20,ep-21,ep-22,ep-26,ep-28,ep-30,ep-32 &
wait
echo ALLDONE
