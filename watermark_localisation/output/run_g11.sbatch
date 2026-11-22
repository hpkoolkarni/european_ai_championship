#!/bin/bash
# Seed ensemble (v8): the best fine-tune setups of g9/g10 rerun with new seeds (same folds, same inits; new batches,
# augmentation and head init), to average with the seed-0 models in i_final_combine.py. ONE node, 4 GPUs.
#   GPU 0  ss_v3  -> ss_v3_s1    StegaStamp, lr 4e-5, 75 % real, 1250 steps (the g10 best)
#   GPU 1  ps_v4  -> ps_v4_s1    PixelSeal, lr 4e-5, 75 % real, 500 steps (g9 best at 250)
#   GPU 2  tmr_v5 -> tmr_v5_s1   TrustMark residual model, lr 4e-5, 500 steps; then ps_v4 -> ps_v4_s2 (seed 2)
#   GPU 3  tmm_v4 -> tmm_v4_s1   TrustMark message-aware model, lr 4e-5, 500 steps; then psr_v5 -> psr_v5_r2 (retuned)
# (In the actual run the two follow-up models were started as extra job steps on the GPUs that had finished.)
# Step counts are chosen on out-of-fold scores. Result: PixelSeal 0.600 -> 0.603 (ps_v4_s2 + ps_spec_real + psr_v5_r2),
# TrustMark and StegaStamp unchanged; overall CV 0.6789, public 0.6788. Nothing is uploaded.
#SBATCH --job-name=tt_wm_g11
#SBATCH --account=training2666
#SBATCH --reservation=cispahack
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=00:50:00
#SBATCH --output=output/logs/%x_%j.log
cd "$SLURM_SUBMIT_DIR"   # submit from watermark_localisation/: sbatch output/run_g11.sbatch
srun --cpu-bind=none bash -c '
export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID}
P=".venv/bin/python output"
case ${SLURM_LOCALID} in
  0) $P/f_real_ft.py --init ss_v3 --name ss_v3_s1 --data synth4 --steps 1250 --lr 4e-5 --real-share 0.75 --seed 1 > output/logs/f_ss_v3_s1.log 2>&1;;
  1) $P/f_real_ft.py --init ps_v4 --name ps_v4_s1 --data synth4 --steps 500 --lr 4e-5 --real-share 0.75 --seed 1 > output/logs/f_ps_v4_s1.log 2>&1;;
  2) $P/f_res_train.py --phase real --init tmr_v5 --name tmr_v5_s1 --data synth5 --steps 500 --lr 4e-5 --seed 1 > output/logs/f_tmr_v5_s1.log 2>&1
     $P/f_real_ft.py --init ps_v4 --name ps_v4_s2 --data synth4 --steps 500 --lr 4e-5 --real-share 0.75 --seed 2 > output/logs/f_ps_v4_s2.log 2>&1;;
  3) $P/f_msg_train.py --phase real --init tmm_v4 --name tmm_v4_s1 --data synth4 --steps 500 --lr 4e-5 --seed 1 > output/logs/f_tmm_v4_s1.log 2>&1
     $P/f_res_train.py --phase real --init psr_v5 --name psr_v5_r2 --data synth5 --steps 500 --lr 4e-5 > output/logs/f_psr_v5_r2.log 2>&1;;
esac'
