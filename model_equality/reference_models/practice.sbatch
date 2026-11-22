#!/bin/bash
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=24
#SBATCH --time=00:30:00
#SBATCH --output=practice_%j.log
# practice check: same pipeline on ref-01..09 with the practice base family (SmolLM2), one GPU per size
R="$SLURM_SUBMIT_DIR"   # submit from model_equality/reference_models/ after: bash get_smol.sh
export HF_HOME=$R/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TARGET=ref
cd $R
E=ref-01,ref-02,ref-03,ref-04,ref-05,ref-06,ref-07,ref-08,ref-09
i=0
for m in HuggingFaceTB/SmolLM2-135M-Instruct HuggingFaceTB/SmolLM2-360M-Instruct HuggingFaceTB/SmolLM2-1.7B-Instruct; do
  ( CUDA_VISIBLE_DEVICES=$i venv/bin/python -u score.py $m && CUDA_VISIBLE_DEVICES=$i venv/bin/python -u tools_fit.py $m $E ) &
  i=$((i+1))
done
wait
echo ALLDONE
