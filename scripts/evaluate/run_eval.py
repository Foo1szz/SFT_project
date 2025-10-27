#!/usr/bin/env python
"""
Inference and evaluation entry point for saved checkpoints.
"""

from __future__ import annotations

import argparse
import json
import logging

from src.evaluation.evaluator import SFTCheckpointEvaluator
from src.utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an SFT checkpoint.")
    parser.add_argument("--config", required=True, help="Model config YAML path.")
    parser.add_argument("--eval-config", required=True, help="Evaluation config YAML path.")
    parser.add_argument(
        "--data-configs",
        nargs="+",
        required=True,
        help="Dataset config YAML paths.",
    )
    parser.add_argument(
        "--checkpoint-path",
        required=True,
        help="Checkpoint directory produced during training.",
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
    evaluator = SFTCheckpointEvaluator(
        model_config_path=args.config,
        data_config_paths=args.data_configs,
        eval_config_path=args.eval_config,
    )
    metrics = evaluator.evaluate(args.checkpoint_path)
    LOGGER.info("Evaluation metrics:\n%s", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
