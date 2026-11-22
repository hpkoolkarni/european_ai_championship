#!/bin/bash
# Message-aware specialists (f_msg_train.py), ONE job on ONE node (team rule: one job per team). Per GPU: phase synth
# on synth4 (<= 45 min), then the 5-fold real-image fine-tune from that checkpoint (750 steps per fold, ~40 min).
#   GPU 0  psm_v4       PixelSeal, own decoder backbone + frozen decoder prior, messages ON
#   GPU 1  tmm_v4       TrustMark, own decoder backbone + frozen decoder prior, messages ON
#   GPU 2  ssm_v4       StegaStamp, steganalysis ResNet-34 (no decoder prior), messages ON
#   GPU 3  psm_v4_ctrl  PixelSeal, identical model but messages OFF: the control that isolates the message's effect
# Compare: psm_v4 vs psm_v4_ctrl (message effect), and *_real vs ps_v4_real / tm_spec_real / ss_v3_real (vs the
# current best). Each writes output/logs/f_<name>.log, output/ckpt/f_<name>.pt, output/probs_f_<name>(_real).npz.
# Nothing is uploaded.
#SBATCH --job-name=tt_wm_msg
#SBATCH --account=training2666
#SBATCH --reservation=cispahack
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=output/logs/%x_%j.log
cd "$SLURM_SUBMIT_DIR"   # submit from watermark_localisation/: sbatch output/run_m.sbatch
srun --cpu-bind=none bash -c '
export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID}
case ${SLURM_LOCALID} in
  0) N=psm_v4;      A="--backbone pixelseal --methods pixelseal";;
  1) N=tmm_v4;      A="--backbone trustmark --methods trustmark";;
  2) N=ssm_v4;      A="--backbone stego --methods stegastamp";;
  3) N=psm_v4_ctrl; A="--backbone pixelseal --methods pixelseal --no-msg";;
esac
.venv/bin/python output/f_msg_train.py --phase synth ${A} --name ${N} --data synth4 --minutes 45 > output/logs/f_${N}.log 2>&1 &&
.venv/bin/python output/f_msg_train.py --phase real --init ${N} --name ${N}_real --data synth4 > output/logs/f_${N}_real.log 2>&1'
