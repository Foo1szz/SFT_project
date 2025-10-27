"""
High-level training orchestration for SFT fine-tuning.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
)

from .callbacks import IntervalCheckpointCallback, SaveBestModelCallback

LOGGER = logging.getLogger(__name__)


class SFTTrainer:
    """
    Thin wrapper on top of the Hugging Face Trainer with project defaults.
    """

    def __init__(
        self,
        model,
        tokenizer,
        training_args,
        datasets,
    ) -> None:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        callbacks = [
            SaveBestModelCallback(training_args.output_dir),
        ]
        if training_args.save_strategy == "steps" and training_args.save_steps:
            callbacks.append(IntervalCheckpointCallback(training_args.save_steps))

        self.trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=datasets.get("train"),
            eval_dataset=datasets.get("validation"),
            data_collator=data_collator,
            callbacks=callbacks,
        )

    def train(self, resume_from_checkpoint: Optional[str] = None) -> Dict[str, float]:
        LOGGER.info("Starting training. Resume checkpoint: %s", resume_from_checkpoint)
        result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        self.trainer.save_state()
        return result.metrics

    def evaluate(self) -> Dict[str, float]:
        LOGGER.info("Running final evaluation.")
        return self.trainer.evaluate()

    def save_model(self) -> None:
        self.trainer.save_model()
