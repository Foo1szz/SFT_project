#!/bin/bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

MODEL_CONFIG=${MODEL_CONFIG:-$PROJECT_ROOT/configs/model.yaml}
EVAL_CONFIG=${EVAL_CONFIG:-$PROJECT_ROOT/configs/eval.yaml}
DATA_CONFIGS=${DATA_CONFIGS:-"$PROJECT_ROOT/configs/data_math.yaml $PROJECT_ROOT/configs/data_gsm8k.yaml"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:?Environment variable CHECKPOINT_PATH must point to a saved checkpoint}
LOG_FILE=${LOG_FILE:-$PROJECT_ROOT/logs/eval.log}

cd "$PROJECT_ROOT"

python "$PROJECT_ROOT/scripts/evaluate/run_eval.py" \
  --config "$MODEL_CONFIG" \
  --eval-config "$EVAL_CONFIG" \
  --data-configs $DATA_CONFIGS \
  --checkpoint-path "$CHECKPOINT_PATH" \
  --log-file "$LOG_FILE"
