#!/bin/bash
# Tuning round 2 of the real-image fine-tune (the retuning lr 4e-5 / 75 % real gave StegaStamp 0.749 -> 0.762 and it was
# still rising at the last step; PixelSeal 0.580 -> 0.585). ONE job, ONE node.
#   GPU 0  ss_v3  -> ss_v3_r3     StegaStamp: retuned AND twice the steps (1500 per fold)
#   GPU 1  tmm_v4 -> tmm_v4_r2    TrustMark message-aware model, retuned lr 4e-5
#   GPU 2  tmr_v5 -> tmr_v5_r2    TrustMark residual model, retuned lr 4e-5
#   GPU 3  ps_spec-> ps_spec_r2   PixelSeal second model, retuned lr 4e-5 / 75 % real
# Step counts are still chosen on out-of-fold scores. Nothing is uploaded.
#SBATCH --job-name=tt_wm_g10
#SBATCH --account=training2666
#SBATCH --reservation=cispahack
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=01:10:00
#SBATCH --output=output/logs/%x_%j.log
cd "$SLURM_SUBMIT_DIR"   # submit from watermark_localisation/: sbatch output/run_g10.sbatch
srun --cpu-bind=none bash -c '
export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID}
P=".venv/bin/python output"
case ${SLURM_LOCALID} in
  0) $P/f_real_ft.py --init ss_v3 --name ss_v3_r3 --data synth4 --steps 1500 --lr 4e-5 --real-share 0.75 > output/logs/f_ss_v3_r3.log 2>&1;;
  1) $P/f_msg_train.py --phase real --init tmm_v4 --name tmm_v4_r2 --data synth4 --lr 4e-5 > output/logs/f_tmm_v4_r2.log 2>&1;;
  2) $P/f_res_train.py --phase real --init tmr_v5 --name tmr_v5_r2 --data synth5 --lr 4e-5 > output/logs/f_tmr_v5_r2.log 2>&1;;
  3) $P/f_real_ft.py --init ps_spec --name ps_spec_r2 --data synth4 --steps 750 --lr 4e-5 --real-share 0.75 > output/logs/f_ps_spec_r2.log 2>&1;;
esac'
