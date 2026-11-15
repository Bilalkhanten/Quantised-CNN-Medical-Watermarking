#!/usr/bin/env bash
# Full experiment protocol of Section 3.9.  Edit the paths, then:  bash run_all.sh  (GPU strongly recommended; each training run is ~100+ epochs)
# Resumable: finished steps (existing output files) are skipped.
set -euo pipefail
MIAS=${MIAS:?set MIAS=/path/to/mias}            # mdb*.pgm + Info.txt
PRETRAIN=${PRETRAIN:-""}                        # e.g. "/data/cbis /data/chestxray14"  (Stage 1-2 pre-training sets; empty = MIAS train only)
CROSS=${CROSS:-""}                              # e.g. "CBIS=/data/cbis CXR14=/data/cxr14 BraTS=/data/brats LIDC=/data/lidc"  (held-out test images, never used in training)
DENOISER=${DENOISER:-""}                        # optional TorchScript DnCNN (sigma=15) -> relabels the Table 10 denoising row
SEEDS=${SEEDS:-"0 1 2 3 4"}
TR="python train.py --mias $MIAS ${PRETRAIN:+--pretrain $PRETRAIN}"
EV="python evaluate.py --mias $MIAS ${DENOISER:+--denoiser $DENOISER}"
step() { local out=$1; shift; if [ -e "$out" ]; then echo "skip $out"; else echo ">> $*"; "$@"; fi; }

# 1. main model, 5 seeds (seed changes init, batch order, payload keys, attack draws; the patient split is fixed by --split_seed 0)
for s in $SEEDS; do
  step runs/full_s$s/final_int8sim.pt $TR --out runs/full_s$s --seed $s
  step runs/full_s$s/eval.json       $EV --run runs/full_s$s --seed $((100+s)) ${CROSS:+--cross $CROSS} --out runs/full_s$s/eval.json
  step runs/full_s$s/eval_fp32.json  $EV --run runs/full_s$s --which fp32 --seed $((100+s)) --out runs/full_s$s/eval_fp32.json
done
# 2. ablations (seed 0).  (i)-(v),(viii),(ix) are compared against the FP32 model; (xi),(xii) against the final model
for ab in no_attack:noattack no_bch:nobch no_stn:nostn no_roi:noroi spatial_encoder:spatial global_pool:globalpool no_rk:nork no_d:nod; do
  flag=${ab%%:*}; name=${ab##*:}
  step runs/abl_$name/stage3.pt  $TR --out runs/abl_$name --seed 0 --$flag
  step runs/abl_$name/eval.json  $EV --run runs/abl_$name --which fp32 --out runs/abl_$name/eval.json
done
step runs/abl_ptq/final_int8sim.pt        $TR --out runs/abl_ptq --seed 0 --ptq
step runs/abl_ptq/eval.json               $EV --run runs/abl_ptq --out runs/abl_ptq/eval.json
step runs/abl_nofinetune/final_int8sim.pt $TR --out runs/abl_nofinetune --seed 0 --no_finetune
step runs/abl_nofinetune/eval.json        $EV --run runs/abl_nofinetune --out runs/abl_nofinetune/eval.json
# (vii) embedding-strength sweep on the FP32 model: alpha x 0.7 and x 1.3 around the calibrated 40 dB point
step runs/alpha07.json $EV --run runs/full_s0 --which fp32 --alpha_scale 0.7 --out runs/alpha07.json
step runs/alpha13.json $EV --run runs/full_s0 --which fp32 --alpha_scale 1.3 --out runs/alpha13.json
# 3. baselines (same data, payload, attacks, 40 dB).  ReDMark / MBRS: wrap the authors' code with embed()/extract() and call baselines.run_eval()
step runs/base_dwtsvd.json python baselines.py dwtsvd --mias $MIAS --out runs/base_dwtsvd.json
step runs/base_hidden.json python baselines.py hidden --mias $MIAS --epochs 60 --out runs/base_hidden.json
# 4. diagnostic impact (Table 11 companion): ROC-AUC of a classifier on original vs watermarked test images, DeLong test
step runs/full_s0/diag.json python diag_classifier.py --run runs/full_s0 --mias $MIAS
# 5. INT8 export (run cost measurement ON THE DEVICE: see README)
step runs/full_s0/onnx/enc_int8.onnx python export_onnx.py --run runs/full_s0 --calib $MIAS --alpha "$(python -c "import json;print(json.load(open('runs/full_s0/eval.json'))['alpha'])")"
# 6. statistics, then the manuscript tables
python stats.py summary runs/full_s*/eval.json | tee runs/summary.txt
python stats.py compare runs/full_s0/eval.json runs/base_hidden.json | tee runs/compare_hidden.txt
ABL="noattack=runs/abl_noattack/eval.json nobch=runs/abl_nobch/eval.json nostn=runs/abl_nostn/eval.json noroi=runs/abl_noroi/eval.json spatial=runs/abl_spatial/eval.json globalpool=runs/abl_globalpool/eval.json nork=runs/abl_nork/eval.json nod=runs/abl_nod/eval.json ptq=runs/abl_ptq/eval.json nofinetune=runs/abl_nofinetune/eval.json alpha07=runs/alpha07.json alpha13=runs/alpha13.json"
BASE="dwtsvd=runs/base_dwtsvd.json hidden=runs/base_hidden.json"; [ -e runs/base_redmark.json ] && BASE="$BASE redmark=runs/base_redmark.json"; [ -e runs/base_mbrs.json ] && BASE="$BASE mbrs=runs/base_mbrs.json"
python fill_docx.py --docx "${DOCX:?set DOCX=manuscript.docx}" --out manuscript_measured.docx --full runs/full_s*/eval.json --fp32 runs/full_s*/eval_fp32.json \
   --baseline $BASE --abl $ABL ${COST:+--cost $COST}
