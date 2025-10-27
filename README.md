# SFT Project

This project implements a supervised fine-tuning (SFT) pipeline that supports full-parameter and LoRA-based training for large language models on the MATH and GSM8K datasets. It follows the plan documented in `plan.md` and provides end-to-end workflows for data preparation, training, checkpoint management, and evaluation.

## Quick Start

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare data**
   - Place raw datasets under `data/raw/`.
   - Use `src/data/preprocess.py` (to be authored per dataset policy) or custom scripts to convert data into the standardized JSONL format expected by the training pipeline.

3. **Run training**
   ```bash
   python scripts/train/train.py \
     --config configs/model.yaml \
     --train-config configs/train_lora.yaml \
     --data-configs configs/data_math.yaml configs/data_gsm8k.yaml \
     --eval-config configs/eval.yaml \
     --output-dir outputs/checkpoints/example_run \
     --use-lora true
   ```

4. **Evaluate checkpoints**
   ```bash
   python scripts/evaluate/run_eval.py \
     --config configs/model.yaml \
     --eval-config configs/eval.yaml \
     --data-configs configs/data_math.yaml configs/data_gsm8k.yaml \
     --checkpoint-path outputs/checkpoints/example_run/checkpoint-best
   ```

## Repository Layout

- `configs/`: YAML configuration files for models, training, data, and evaluation.
- `data/`: Raw and processed dataset storage plus prompt templates.
- `docs/`: Experiment reports and evaluation summaries.
- `logs/`: Combined training and evaluation logging output.
- `outputs/`: Saved checkpoints and evaluation artifacts.
- `scripts/`: Command-line entry points for training and evaluation.
- `src/`: Reusable Python modules for data processing, model handling, training, and evaluation.

Refer to `plan.md` for a detailed roadmap and architectural rationale.

## Attention Heatmaps (MATH-500)

Generate attention heatmaps for 5 random queries per difficulty level (1–5) from MATH-500 and save six layers per query (first two, middle two, last two):

```bash
python scripts/evaluate/attention_heatmaps.py \
  --model-path /mnt/sharedata/ssd_large/common/LLMs/deepseek-math-7b-rl \
  --data-file /mnt/sharedata/ssd_large/common/datasets/MATH-500/test.jsonl \
  --output-dir outputs/attention_maps \
  --per-level 5 \
  --max-plot-tokens 128
```

Outputs are saved under `outputs/attention_maps/level_{L}/sample_{k}/layer_{ii}.png` with a `meta.json` containing the query and token list. If your model backend does not support returning attentions (e.g., certain FlashAttention-only builds), disable such optimizations or load a variant that supports `output_attentions=True`.
