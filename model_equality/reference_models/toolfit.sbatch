#!/bin/bash
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=24
#SBATCH --time=00:40:00
#SBATCH --output=toolfit_%j.log
# inference only: base-model probability of each tool choice under each family's exact base
#   A Qwen2.5-3B-Instruct, B Qwen2.5-1.5B-Instruct, C Llama-3.2-1B-Instruct
# options (environment): FP8=1 / FP8=t simulated fp8 copies, MINREP=40 OUTSUF=-new only newly collected answers
R="$SLURM_SUBMIT_DIR"   # submit from model_equality/reference_models/: sbatch toolfit.sbatch
export HF_HOME=$R/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd $R
CUDA_VISIBLE_DEVICES=0 venv/bin/python -u tools_fit.py Qwen/Qwen2.5-3B-Instruct ep-01,ep-02,ep-04,ep-12,ep-14,ep-17,ep-18,ep-23,ep-24,ep-25,ep-33 &
CUDA_VISIBLE_DEVICES=1 venv/bin/python -u tools_fit.py Qwen/Qwen2.5-1.5B-Instruct ep-03,ep-06,ep-07,ep-08,ep-20,ep-21,ep-22,ep-26,ep-28,ep-30,ep-32 &
CUDA_VISIBLE_DEVICES=2 venv/bin/python -u tools_fit.py unsloth/Llama-3.2-1B-Instruct ep-05,ep-09,ep-10,ep-11,ep-13,ep-15,ep-16,ep-19,ep-27,ep-29,ep-31 &
wait
echo ALLDONE
