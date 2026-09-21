#!/bin/bash
# Usage:  bash run_stage.sh <0|1|2|3> [--method obdllm|svdllm|fwsvd] [--ratio RHO] [--conf conf/run.toml]
#
#   0  extract the dense fine-tuned WavLM-Large teacher from the DiariZen EEND checkpoint
#   1  training-free compression with ONE factorization rule at ONE ratio, then the seven-corpus evaluation
#   2  recovery fine-tuning of that model, averaging of the Phase-B epoch checkpoints, evaluation
#   3  Params / MACs and GPU / CPU speed-up under the protocol of Han et al. (TASLP 2026)
#
# All settings live in the config file; --method / --ratio override [compress] for this call.  All outputs
# go to <exp>/stage<N>/; each stage reads only the outputs of the previous ones.
set -eu
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); cd "$ROOT"
stage=${1:?usage: bash run_stage.sh <0|1|2|3> [--method M] [--ratio R] [--conf FILE]}; shift
CONF=conf/run.toml; METHOD=""; RATIO=""
while [ $# -gt 0 ]; do case $1 in
  --method) METHOD=$2; shift 2 ;;  --ratio) RATIO=$2; shift 2 ;;  --conf) CONF=$2; shift 2 ;;
  *) echo "unknown option $1" >&2; exit 1 ;; esac; done
PY=${PY:-python}
$PY - <<'CHECK' || exit 1
import importlib, sys
def ok(m):
    try: importlib.import_module(m); return True
    except Exception: return False
missing = [m for m in ("toml", "diarizen", "pyannote.audio") if not ok(m)]
if missing:
    sys.exit(f"[run_stage] not installed in this Python ({sys.executable}): {', '.join(missing)}\n"
             "            follow README.md 'Installation' (this repository's requirements.txt, then DiariZen)")
CHECK
eval "$($PY scripts/_config.py "$CONF")"
METHOD=${METHOD:-$COMPRESS_METHOD}; RATIO=${RATIO:-$COMPRESS_RATIO}; TAG=$METHOD-r$RATIO
export CUDA_VISIBLE_DEVICES=$ENV_CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

E=$PATHS_EXP
TEACHER=$E/stage0/wavlm-large-ft.bin
LOG=$E/logs; mkdir -p "$LOG"
lowrank_bin() { echo "$E/stage1/$1/wavlm-large-ft-$1.bin"; }     # tag -> compressed encoder

# eval_tf.sh reads its assets and protocol from the environment
export DSCORE=$PATHS_DSCORE EMB=$PATHS_EMBEDDING DIARIZEN_HUB=$PATHS_DIARIZEN_HUB \
       DATA=$PATHS_DATA SETS=$EVAL_SETS SPK_MIN=$EVAL_SPK_MIN SPK_MAX=$EVAL_SPK_MAX
evaluate() { bash eval_tf.sh "$1" 2>&1 | tee "$LOG/eval_$(basename "$1").log" | grep -E "^\[|^macro"; }

case $stage in
0)
  echo "### stage 0: teacher extraction -> $TEACHER"
  $PY scripts/extract_ft_wavlm.py --diar_ckpt "$PATHS_TEACHER_DIAR" --config_src "$PATHS_WAVLM_CONFIG" --out "$TEACHER"
  ;;

1)
  echo "### stage 1: training-free compression  method=$METHOD  rho=$RATIO  -> $E/stage1/$TAG"
  $PY scripts/build_lowrank.py --method "$METHOD" --ratio "$RATIO" --tag "$TAG" \
      --teacher_wavlm "$TEACHER" --teacher_diar "$PATHS_TEACHER_DIAR" \
      --train_scp "$PATHS_DATA/_pooled/train/wav.scp" --train_rttm "$PATHS_DATA/_pooled/train/rttm" \
      --train_uem "$PATHS_DATA/_pooled/train/all.uem" \
      --win_per_corpus "$COMPRESS_WIN_PER_CORPUS" --chunk_per_corpus "$COMPRESS_CHUNK_PER_CORPUS" \
      --damp "$COMPRESS_DAMP" --round_to "$COMPRESS_ROUND_TO" --seed "$COMPRESS_SEED" \
      --stats_cache "$E/stage1/cache/stats.pt" --spectra_cache "$E/stage1/cache/spectra_$METHOD.pt" \
      --out_bin "$(lowrank_bin $TAG)" --exp_dir "$E/stage1/$TAG" 2>&1 | tee "$LOG/build_$TAG.log"
  evaluate "$E/stage1/$TAG"
  ;;

2)
  tag=$TAG
  echo "### stage 2: recovery of $tag"
  $PY scripts/recover.py --compressed "$(lowrank_bin $tag)" --exp_dir "$E/stage2/${tag}_recover" \
      --ft_diar "$PATHS_TEACHER_DIAR" --teacher "$TEACHER" --train_dir "$PATHS_DATA/_pooled/recover" --dev_dir "$PATHS_DATA/_pooled/dev" \
      --lora_rank "$RECOVERY_LORA_RANK" --lr "$RECOVERY_LR" --head_lr "$RECOVERY_HEAD_LR" \
      --epochs_a "$RECOVERY_EPOCHS_A" --epochs_b "$RECOVERY_EPOCHS_B" --batch "$RECOVERY_BATCH" \
      --distill_lambda "$RECOVERY_DISTILL_LAMBDA" --distill_S "$RECOVERY_DISTILL_S" \
      --dev_subset "$RECOVERY_DEV_SUBSET" --eval_every "$RECOVERY_EVAL_EVERY" 2>&1 | tee "$LOG/recover_$tag.log"
  $PY scripts/average_epochs.py --recover_exp "$E/stage2/${tag}_recover" --out_exp "$E/stage2/${tag}_recover_avg5" \
      --wavlm_bin "$(lowrank_bin $tag)" --ckpts "$RECOVERY_AVG_CKPTS"
  evaluate "$E/stage2/${tag}_recover_avg5"
  ;;

3)
  echo "### stage 3: efficiency"
  mkdir -p "$E/stage3"
  M=("dense=$TEACHER" "pruned80=$BENCH_PRUNED" "$TAG=$(lowrank_bin $TAG)")
  $PY scripts/bench_macs.py --models "${M[@]}" --out "$E/stage3/macs.json" | tee "$LOG/macs.log"
  $PY scripts/bench_speed.py --device cuda --batches "$BENCH_GPU_BATCHES" --iters "$BENCH_GPU_ITERS" \
      --models "${M[@]}" --out "$E/stage3/speed_gpu.json" | tee "$LOG/speed_gpu.log"
  OMP_NUM_THREADS=$BENCH_CPU_THREADS $PY scripts/bench_speed.py --device cpu --batches "$BENCH_CPU_BATCHES" \
      --iters "$BENCH_CPU_ITERS" --models "${M[@]}" --out "$E/stage3/speed_cpu.json" | tee "$LOG/speed_cpu.log"
  ;;

*) echo "unknown stage '$stage' (0-3)" >&2; exit 1 ;;
esac
echo "### stage $stage done"
