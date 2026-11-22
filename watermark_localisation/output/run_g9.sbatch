#!/bin/bash
# "Do what worked, tuned": the real-image fine-tune (5 honest folds) - the biggest proven lever - ONE job, ONE node.
#   GPU 0  ss_v3  -> ss_v3_r2     StegaStamp, retuned: lr 4e-5 (was 2e-5), 75 % real per batch (was 50 %)
#   GPU 1  ps_v4  -> ps_v4_r2     PixelSeal, the same retuning
#   GPU 2  tss_v4 -> tss_v4_real  TrustMark full-resolution model, fine-tuned for the first time (new ensemble member)
#   GPU 3  psm_v4 -> psm_v4_real  PixelSeal message-aware model, fine-tuned for the first time (new ensemble member)
# The step count (0/250/500/750) is still chosen once on out-of-fold scores, so a too-aggressive setting falls back.
# Each writes output/logs/f_<name>.log and output/probs_f_<name>.npz. Nothing is uploaded.
#SBATCH --job-name=tt_wm_g9
#SBATCH --account=training2666
#SBATCH --reservation=cispahack
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=output/logs/%x_%j.log
cd "$SLURM_SUBMIT_DIR"   # submit from watermark_localisation/: sbatch output/run_g9.sbatch
srun --cpu-bind=none bash -c '
export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID}
P=".venv/bin/python output"
TUNE="--data synth4 --steps 750 --lr 4e-5 --real-share 0.75"
case ${SLURM_LOCALID} in
  0) $P/f_real_ft.py --init ss_v3  --name ss_v3_r2    ${TUNE} > output/logs/f_ss_v3_r2.log 2>&1;;
  1) $P/f_real_ft.py --init ps_v4  --name ps_v4_r2    ${TUNE} > output/logs/f_ps_v4_r2.log 2>&1;;
  2) $P/f_real_ft.py --init tss_v4 --name tss_v4_real --data synth4 --steps 750 > output/logs/f_tss_v4_real.log 2>&1;;
  3) $P/f_msg_train.py --phase real --init psm_v4 --name psm_v4_real --data synth4 > output/logs/f_psm_v4_real.log 2>&1;;
esac'
