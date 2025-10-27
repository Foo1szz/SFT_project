#!/bin/bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

MODEL_CONFIG=${MODEL_CONFIG:-$PROJECT_ROOT/configs/model.yaml}
TRAIN_CONFIG=${TRAIN_CONFIG:-$PROJECT_ROOT/configs/train_lora.yaml}
DATA_CONFIGS=${DATA_CONFIGS:-"$PROJECT_ROOT/configs/data_math.yaml $PROJECT_ROOT/configs/data_gsm8k.yaml"}
EVAL_CONFIG=${EVAL_CONFIG:-$PROJECT_ROOT/configs/eval.yaml}
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_ROOT/outputs/checkpoints/lora_run}
LOG_FILE=${LOG_FILE:-$OUTPUT_DIR/train.log}

mkdir -p "$OUTPUT_DIR"

cd "$PROJECT_ROOT"

python "$PROJECT_ROOT/scripts/train/train.py" \
  --config "$MODEL_CONFIG" \
  --train-config "$TRAIN_CONFIG" \
  --data-configs $DATA_CONFIGS \
  --eval-config "$EVAL_CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --use-lora \
  --log-file "$LOG_FILE"
