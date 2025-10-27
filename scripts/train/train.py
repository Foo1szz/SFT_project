#!/usr/bin/env python
"""
Training entry point for the SFT project.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import List

from transformers import set_seed

from src.data.dataset import SFTDataModule
from src.models.base_model import (
    load_model,
    load_tokenizer,
    maybe_merge_lora_weights,
    save_model_artifacts,
)
from src.training.accelerator import build_training_arguments
from src.training.trainer import SFTTrainer
from src.utils.config import load_yaml_file
from src.utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT training script")
    parser.add_argument("--config", required=True, help="Path to model YAML config.")
    parser.add_argument(
        "--train-config", required=True, help="Path to training YAML config."
    )
    parser.add_argument(
        "--data-configs",
        nargs="+",
        required=True,
        help="List of dataset YAML configuration files.",
    )
    parser.add_argument(
        "--eval-config",
        required=False,
        help="Optional evaluation config path (for metadata only).",
    )
    parser.add_argument("--output-dir", required=False, help="Overrides output dir.")
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Enable LoRA fine-tuning (expects `lora` block in training config).",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Resume training from an existing checkpoint.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Optional subset of datasets to include (by dataset_name).",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional log file path.",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Logging verbosity level.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbosity, args.log_file)

    model_config = load_yaml_file(args.config)
    training_config = load_yaml_file(args.train_config)

    data_configs: List[dict] = [load_yaml_file(path) for path in args.data_configs]
    if args.datasets:
        whitelist = set(args.datasets)
        data_configs = [
            cfg for cfg in data_configs if cfg.get("dataset_name") in whitelist
        ]
        if not data_configs:
            raise ValueError("No dataset configs left after filtering by --datasets.")

    output_dir = args.output_dir or training_config.get("output_dir")
    if not output_dir:
        raise ValueError("Output directory must be specified via CLI or training config.")
    os.makedirs(output_dir, exist_ok=True)

    set_seed(training_config.get("seed", 42))

    tokenizer = load_tokenizer(model_config)

    model = load_model(model_config, training_config, use_lora=args.use_lora)
    model.resize_token_embeddings(len(tokenizer))

    data_module = SFTDataModule(data_configs, tokenizer, seed=training_config.get("seed", 42))
    datasets = data_module.build_datasets()

    training_args = build_training_arguments(training_config, output_dir)

    trainer = SFTTrainer(model=model, tokenizer=tokenizer, training_args=training_args, datasets=datasets)

    train_metrics = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    LOGGER.info("Training metrics: %s", train_metrics)

    if datasets.get("validation") is not None:
        eval_metrics = trainer.evaluate()
        LOGGER.info("Validation metrics: %s", eval_metrics)

    trainer.save_model()

    lora_config = training_config.get("lora") or {}
    if args.use_lora and lora_config.get("merge_weights"):
        merged_model = maybe_merge_lora_weights(model, merge=True)
        merged_output = os.path.join(output_dir, "merged")
        os.makedirs(merged_output, exist_ok=True)
        save_model_artifacts(merged_model, tokenizer, merged_output)
    elif lora_config.get("save_merged_weights"):
        LOGGER.warning(
            "`save_merged_weights` requested but `--use-lora` is disabled; skipping."
        )

    # Always persist tokenizer alongside trainer.save_model()
    save_model_artifacts(model, tokenizer, output_dir)


if __name__ == "__main__":
    main()
