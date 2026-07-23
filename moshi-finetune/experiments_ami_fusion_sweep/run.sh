#!/usr/bin/env bash
# Compare EPAD controllers and fusion weights on the same checkpoint, ten AMI
# samples, and fixed random seed. Runs sequentially on one GPU.
#
# Usage:
#   bash experiments_ami_fusion_sweep/run.sh
#   GPU_ID=0 SEED=42 bash experiments_ami_fusion_sweep/run.sh
set -uo pipefail

ROOT=/home2/s20235100/Conversational-AI/personaplex_MBG/moshi-finetune
EXPDIR="$ROOT/experiments_ami_fusion_sweep_2e-5"
PY=/home2/s20235100/miniconda3/envs/plex/bin/python
GPU="${GPU_ID:-0}"
SEED="${SEED:-42}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
CONFIG="$ROOT/experiments_ami/configs/ami_vap_base.yaml"
CKPT="$ROOT/output/hyades/output/ami_vap_2e-5/checkpoints/checkpoint_000500"
TEST_JSONL="$ROOT/experiments_ami/ami_test_10.jsonl"
OUT_ROOT="$ROOT/result/ami_fusion_sweep_2e-5_$RUN_TAG"
LOGDIR="$EXPDIR/logs/$RUN_TAG"

cd "$ROOT"
mkdir -p "$OUT_ROOT" "$LOGDIR"
export PYTHONUNBUFFERED=1

if [ ! -f "$CKPT/consolidated/lora.safetensors" ]; then
    echo "Checkpoint not found: $CKPT" >&2
    exit 2
fi
if [ ! -f "$CONFIG" ]; then
    echo "Config not found: $CONFIG" >&2
    exit 2
fi
if [ ! -f "$TEST_JSONL" ]; then
    echo "10-sample JSONL not found: $TEST_JSONL" >&2
    exit 2
fi
if ! CUDA_VISIBLE_DEVICES="$GPU" "$PY" -c \
    'import torch; raise SystemExit(0 if torch.cuda.is_available() and torch.cuda.device_count() == 1 else 1)'
then
    echo "GPU_ID=$GPU is not visible to CUDA." >&2
    exit 3
fi

run_setting() {
    name="$1"
    shift
    out_dir="$OUT_ROOT/$name"
    mkdir -p "$out_dir"
    echo "[$(date +%T)] [GPU$GPU] START $name"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$ROOT/test_inference2.py" \
        --config "$CONFIG" \
        --ckpt-dir "$CKPT" \
        --test-jsonl "$TEST_JSONL" \
        --output-dir "$out_dir" \
        --seed "$SEED" \
        "$@" > "$LOGDIR/$name.log" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "[$(date +%T)] [GPU$GPU] FAILED $name rc=$rc -- see $LOGDIR/$name.log" >&2
        return "$rc"
    fi
    echo "[$(date +%T)] [GPU$GPU] DONE  $name -> $out_dir"
}

failed=0

# Baselines: implicit backbone only and the previous explicit hard-gate policies.
# run_setting implicit_only --epad-control none || failed=$((failed + 1))
# run_setting legacy_inject --epad-control legacy || failed=$((failed + 1))
# Main ablation: change only the explicit BC evidence weight.
run_setting fusion_bc005 --epad-control fusion --fusion-bc-weight 0.05 || failed=$((failed + 1))
run_setting fusion_bc010 --epad-control fusion --fusion-bc-weight 0.10 || failed=$((failed + 1))
run_setting fusion_bc025 --epad-control fusion --fusion-bc-weight 0.25 || failed=$((failed + 1))
run_setting fusion_bc050 --epad-control fusion --fusion-bc-weight 0.50 || failed=$((failed + 1))
run_setting fusion_bc100 --epad-control fusion --fusion-bc-weight 1.00 || failed=$((failed + 1))

# Auxiliary-evidence ablation around the moderate BC=0.25 operating point.
run_setting fusion_bc025_vap025 \
    --epad-control fusion --fusion-bc-weight 0.25 --fusion-vap-weight 0.25 \
    || failed=$((failed + 1))
run_setting fusion_bc025_vad025 \
    --epad-control fusion --fusion-bc-weight 0.25 --fusion-vad-weight 0.25 \
    || failed=$((failed + 1))
run_setting fusion_bc025_vap025_vad025 \
    --epad-control fusion --fusion-bc-weight 0.25 \
    --fusion-vap-weight 0.25 --fusion-vad-weight 0.25 \
    || failed=$((failed + 1))

"$PY" "$EXPDIR/summarize.py" "$OUT_ROOT" || failed=$((failed + 1))

echo "[$(date +%T)] COMPLETE (failed=$failed)"
echo "Results: $OUT_ROOT"
echo "Summary: $OUT_ROOT/summary.csv"
exit "$failed"
