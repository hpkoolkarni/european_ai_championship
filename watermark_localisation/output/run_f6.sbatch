#!/bin/bash
# Two things side by side on ONE node (team rule: one job per team):
#   GPU 0-2  improved real-image fine-tune (f_real_ft.py): layer-wise learning-rate decay 0.75 + weight averaging (SWA),
#            and for TrustMark / StegaStamp also the decoder-confirmed test boxes as pseudo-labels (--pseudo).
#            Same starting models as run_f5, so the result compares directly with ps_v4_real / tm_spec_real / ss_v3_real.
#   GPU 3    synth5: 24,000 new synthetic images (shapes, sharpness, strength matched, as synth4) that also store the
#            exact watermark residual (--residual), the dense training target for f_res_train.py (next job)
# Nothing is uploaded.
#SBATCH --job-name=tt_wm_ft2
#SBATCH --account=training2666
#SBATCH --reservation=cispahack
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=01:15:00
#SBATCH --output=output/logs/%x_%j.log
cd "$SLURM_SUBMIT_DIR"   # submit from watermark_localisation/: sbatch output/run_f6.sbatch
srun --cpu-bind=none bash -c '
export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID}
FT="--data synth4 --steps 750 --llrd 0.75 --swa"
case ${SLURM_LOCALID} in
  0) .venv/bin/python output/f_real_ft.py --init ss_v3   --name ss_v3_ft2 ${FT} --pseudo > output/logs/f_ss_v3_ft2.log 2>&1;;
  1) .venv/bin/python output/f_real_ft.py --init tm_spec --name tm_spec_ft2 ${FT} --pseudo > output/logs/f_tm_spec_ft2.log 2>&1;;
  2) .venv/bin/python output/f_real_ft.py --init ps_v4   --name ps_v4_ft2 ${FT} > output/logs/f_ps_v4_ft2.log 2>&1;;
  3) for k in 0 1 2 3; do
       SLURM_PROCID=$k .venv/bin/python output/e_synth3.py --shard $k --nshards 4 --n 6000 --seed 5 --strength --residual --outdir synth5 || exit 1
     done > output/logs/synth5.log 2>&1;;
esac'
