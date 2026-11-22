#!/bin/bash
# Known-message box search (h_box_search.py), inference only, ONE job on ONE node (team rule: one job per team):
#   GPU 0      TrustMark, all patch images (train + test)
#   GPU 1-3    StegaStamp, patch images split in 3 shards (its ONNX decoder is slower)
# then, on the node's CPUs, --combine: acceptance level and rule chosen by 5-fold CV on the real train images,
# writes output/sub_h_box.pt. Proposals come from the submission-v3 models. Nothing is uploaded.
#SBATCH --job-name=tt_wm_box
#SBATCH --account=training2666
#SBATCH --reservation=cispahack
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=output/logs/%x_%j.log
cd "$SLURM_SUBMIT_DIR"   # submit from watermark_localisation/: sbatch output/run_h.sbatch
srun --cpu-bind=none bash -c '
export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID}
if [ ${SLURM_LOCALID} -eq 0 ]; then A="--method trustmark"; N=tm; else A="--method stegastamp --shard $((SLURM_LOCALID - 1)) --nshards 3"; N=ss$((SLURM_LOCALID - 1)); fi
.venv/bin/python output/h_box_search.py ${A} --split both > output/logs/h_box_${N}.log 2>&1'
.venv/bin/python output/h_box_search.py --combine
