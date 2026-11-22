#!/bin/bash
# Re-run of what the inode-quota crash of job 15904938 cost, ONE job on ONE node (team rule: one job per team):
#   GPU 0-2  wider StegaStamp box search on the TEST patch images, split in 3 shards (train part was saved already)
#   GPU 3    PixelSeal residual model: the 5-fold real-image fine-tune (its synthetic checkpoint f_psr_v5.pt survived)
# Nothing is uploaded.
#SBATCH --job-name=tt_wm_g8
#SBATCH --account=training2666
#SBATCH --reservation=cispahack
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=output/logs/%x_%j.log
cd "$SLURM_SUBMIT_DIR"   # submit from watermark_localisation/: sbatch output/run_g8.sbatch
srun --cpu-bind=none bash -c '
export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID}
if [ ${SLURM_LOCALID} -lt 3 ]; then
  .venv/bin/python output/h_box_search.py --method stegastamp --split test --shard ${SLURM_LOCALID} --nshards 3 --boxdir box2 \
    --prop-models ss_v3 ss_v4 stego_ss stego_all ss_v3_real > output/logs/h_box2_ss_test${SLURM_LOCALID}.log 2>&1
else
  .venv/bin/python output/f_res_train.py --phase real --init psr_v5 --name psr_v5_real --data synth5 > output/logs/f_psr_v5_real.log 2>&1
fi'
