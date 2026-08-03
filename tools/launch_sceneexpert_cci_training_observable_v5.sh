#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/afs/task3_2/visitor36/projects/SceneRepair_v3"
RUN_NAME="furniture_clean512_observable_v5"
RUN_DIR="$ROOT/runs/$RUN_NAME"
PID_FILE="$ROOT/runs/$RUN_NAME.pid"

cd "$ROOT"
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "training is already running as PID $(cat "$PID_FILE")" >&2
  exit 1
fi
if [[ -e "$RUN_DIR/metrics.jsonl" ]]; then
  echo "refusing to append to existing run: $RUN_DIR" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=4
export PYTHONPATH="$ROOT/.python_packages${PYTHONPATH:+:$PYTHONPATH}"

nohup python3 -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m tools.train_furniture \
  --manifest data/training/clean512_v1/clean512_manifest_v1.json \
  --vocab data/training/clean512_v1/furniture_vocab_clean512_v1.json \
  --functional-rules configs/functional_partner_rules_hssd_v1.json \
  --compatibility-rules configs/functional_partner_category_compatibility_v1.json \
  --output-dir "$RUN_DIR" \
  --label-mode observable_v3 \
  --epochs 100 \
  --batch-size 64 \
  --train-samples-per-scene 16 \
  --val-samples-per-scene 4 \
  --workers 8 \
  > "$ROOT/runs/$RUN_NAME.stdout.log" \
  2> "$ROOT/runs/$RUN_NAME.stderr.log" &

pid=$!
echo "$pid" > "$PID_FILE"
echo "started $RUN_NAME as PID $pid"
