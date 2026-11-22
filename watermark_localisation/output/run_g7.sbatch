#!/bin/bash
# Job 2, ONE node (team rule: one job per team). Needs output/synth5 (made by run_f6.sbatch).
#   GPU 0  wider box search, TrustMark: candidate boxes from EVERY TrustMark model's own map (also weak responses),
#          because the failure analysis showed the worst TrustMark patches are small but decodable at the exact box
#          (bit accuracy 0.96) - our best model just does not propose them -> output/box2/trustmark_*
#          then the real-image fine-tune of the message-aware TrustMark model tmm_v4 (it beat tm_v4 0.536 vs 0.493)
#   GPU 1  residual-head PixelSeal model on synth5: synthetic training, then the 5-fold real-image fine-tune
#   GPU 2  residual-head TrustMark model on synth5: the same
#   GPU 3  wider box search, StegaStamp (as GPU 0) -> output/box2/stegastamp_*, then the residual-head StegaStamp model
# Each writes output/logs/<name>.log and output/probs_f_<name>.npz (box search: output/box2/*.npz). Nothing is uploaded.
#SBATCH --job-name=tt_wm_g7
#SBATCH --account=training2666
#SBATCH --reservation=cispahack
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=output/logs/%x_%j.log
cd "$SLURM_SUBMIT_DIR"   # submit from watermark_localisation/: sbatch output/run_g7.sbatch
[ -f output/synth5/shard_3_resid.npy ] || { echo "output/synth5 missing"; exit 1; }
srun --cpu-bind=none bash -c '
export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID}
P=".venv/bin/python output"
RES="--data synth5 --minutes 45"
case ${SLURM_LOCALID} in
  0) $P/h_box_search.py --method trustmark --split both --boxdir box2 --prop-models tm_spec tss_v4 tm_v3 tm_v4 tm_spec_real > output/logs/h_box2_tm.log 2>&1
     $P/f_msg_train.py --phase real --init tmm_v4 --name tmm_v4_real --data synth4 > output/logs/f_tmm_v4_real.log 2>&1;;
  1) $P/f_res_train.py --phase synth --backbone pixelseal --methods pixelseal --name psr_v5 ${RES} > output/logs/f_psr_v5.log 2>&1 &&
     $P/f_res_train.py --phase real --init psr_v5 --name psr_v5_real --data synth5 > output/logs/f_psr_v5_real.log 2>&1;;
  2) $P/f_res_train.py --phase synth --backbone trustmark --methods trustmark --name tmr_v5 ${RES} > output/logs/f_tmr_v5.log 2>&1 &&
     $P/f_res_train.py --phase real --init tmr_v5 --name tmr_v5_real --data synth5 > output/logs/f_tmr_v5_real.log 2>&1;;
  3) $P/h_box_search.py --method stegastamp --split both --boxdir box2 --prop-models ss_v3 ss_v4 stego_ss stego_all ss_v3_real > output/logs/h_box2_ss.log 2>&1
     $P/f_res_train.py --phase synth --backbone stego --methods stegastamp --name ssr_v5 --data synth5 --minutes 40 > output/logs/f_ssr_v5.log 2>&1 &&
     $P/f_res_train.py --phase real --init ssr_v5 --name ssr_v5_real --data synth5 > output/logs/f_ssr_v5_real.log 2>&1;;
esac'
