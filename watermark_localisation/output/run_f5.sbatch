#!/bin/bash
# Fine-tune the best specialists on the REAL train images (5 honest folds, half real / half synthetic per batch), ONE
# job on ONE node (team rule: at most one job, running or queued, per team). One specialist per GPU:
#   GPU 0  ps_spec_real  from f_ps_spec (best PixelSeal)
#   GPU 1  tm_spec_real  from f_tm_spec (best TrustMark)
#   GPU 2  ss_v3_real    from f_ss_v3   (best StegaStamp)
#   GPU 3  ps_v3_real    from f_ps_v3   (second PixelSeal; averaging the two PixelSeal models helped before)
# The --init names are updated to the best models once run_f4 (synth4) has finished. Each writes
# output/logs/f_<name>.log and output/probs_f_<name>.npz (out-of-fold train + 5-fold-average test predictions). Nothing
# is uploaded.
#SBATCH --job-name=tt_wm_real
#SBATCH --account=training2666
#SBATCH --reservation=cispahack
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=01:20:00
#SBATCH --output=output/logs/%x_%j.log
cd "$SLURM_SUBMIT_DIR"   # submit from watermark_localisation/: sbatch output/run_f5.sbatch
srun --cpu-bind=none bash -c '
export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID}
case ${SLURM_LOCALID} in
  0) I=ps_v4;;
  1) I=tm_spec;;
  2) I=ss_v3;;
  3) I=ps_spec;;
esac
.venv/bin/python output/f_real_ft.py --init ${I} --name ${I}_real --data synth4 --steps 750 > output/logs/f_${I}_real.log 2>&1'
