#!/bin/bash
# Seven-corpus DER evaluation of an assembled experiment directory (single checkpoint, no averaging).
#   bash eval_tf.sh <exp_dir>        exp_dir = {<name>.toml, checkpoints/epoch_0001/pytorch_model.bin, val_metric_summary.lst}
# Inference protocol = DiariZen (Han et al., ICASSP 2025 / TASLP 2026): 16 s segments, VBx clustering
# [Landini et al., CSL 2022] with WeSpeaker ResNet34-LM embeddings, one speaker-count range (1-20)
# shared by all corpora, DER by dscore with collar 0 (and 0.25 for reference).  Macro = mean of 7 sets.
# Environment overrides:
#   SETS="AMI AliMeeting"   restrict the corpora            SPK_MIN / SPK_MAX   speaker range (default 1 / 20)
#   DSCORE       path to https://github.com/nryant/dscore          (default $ROOT/dscore)
#   EMB          WeSpeaker embedding model (pyannote 3.1 format)  (default $ROOT/pretrained/pyannote/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin;
#                keep "pyannote/" in the path, pyannote chooses its loader from the path string)
#   DIARIZEN_HUB DiariZen hub directory with plda/               (default $ROOT/pretrained/diarizen-wavlm-base-s80-md)
#   DATA         directory with <corpus>/eval/{wav.scp,rttm}     (default $ROOT/data, see data/README.md)
set -eu
ulimit -n 2048
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY=${PY:-python}
DSCORE=${DSCORE:-$ROOT/dscore}
EMB=${EMB:-$ROOT/pretrained/pyannote/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin}
DIARIZEN_HUB=${DIARIZEN_HUB:-$ROOT/pretrained/diarizen-wavlm-base-s80-md}
data_dir=${DATA:-$ROOT/data}; dtype=test
cd "$ROOT"
EXP=${1:?usage: bash eval_tf.sh <exp_dir>}
for f in "$DSCORE/score.py" "$EMB" "$DIARIZEN_HUB/plda" "$EXP/checkpoints/epoch_0001/pytorch_model.bin"; do
    [ -e "$f" ] || { echo "[eval_tf] missing: $f  (see README 'Installation' / conf/run.toml)" >&2; exit 1; }
done
config=$(ls "$EXP"/*.toml | head -n 1)
SUMMARY="$EXP/val_metric_summary.lst"
seg_duration=16; seg_step=0.1; ahc_thr=0.6; mcs=13; Fa=0.07; Fb=0.8; lda_dim=128; max_iters=20
MIN=${SPK_MIN:-1}; MAX=${SPK_MAX:-20}
SYS=$EXP/infer_VBx_seg16_tf/avg_ckpt1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
RES=$EXP/tf_der.txt; : > "$RES"
for dset in ${SETS:-AMI AISHELL4 AliMeeting RAMC VoxConverse MSDWild NOTSOFAR}; do
    OUT=$SYS/${dtype}/${dset}
    echo "==== infer: $dset (speakers ${MIN}-${MAX}) ===="
    $PY infer_avg.py -C "$config" -i "${data_dir}/${dset}/eval/wav.scp" -o "$OUT" \
        --embedding_model "$EMB" --diarizen_hub "$DIARIZEN_HUB" \
        --avg_ckpt_num 1 --val_metric Loss --val_mode best --val_metric_summary "$SUMMARY" \
        --seg_duration $seg_duration --segmentation_step $seg_step --clustering_method VBxClustering \
        --ahc_threshold $ahc_thr --min_cluster_size $mcs --Fa $Fa --Fb $Fb --lda_dim $lda_dim \
        --max_iters $max_iters --min_speakers $MIN --max_speakers $MAX --batch_size 8 || { echo "[$dset] FAILED"; continue; }
    for collar in 0 0.25; do
        $PY "$DSCORE/score.py" -r "${data_dir}/${dset}/eval/rttm" -s "$OUT"/*.rttm --collar $collar \
            > "$OUT/result_collar${collar}" 2>&1 || true
        der=$(grep -iE "OVERALL" "$OUT/result_collar${collar}" | tail -1 | awk '{print $4}')
        echo "[$dset collar$collar] DER=$der" | tee -a "$RES"
    done
done
echo "=== DONE ==="
echo "macro(collar 0): $(grep 'collar0]' $RES | awk -F= '{s+=$2;n++} END{printf "%.2f (n=%d)", s/n, n}')"
echo "macro(collar 0.25): $(grep 'collar0.25]' $RES | awk -F= '{s+=$2;n++} END{printf "%.2f (n=%d)", s/n, n}')"
