"""
Custom callbacks used during training.
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import Optional

from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

LOGGER = logging.getLogger(__name__)


class SaveBestModelCallback(TrainerCallback):
    """
    Saves the best checkpoint based on evaluation loss.
    """

    def __init__(self, output_dir: str) -> None:
        self.best_metric: Optional[float] = None
        self.output_dir = output_dir
        self._is_best_step = False

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        metrics = kwargs.get("metrics") or {}
        loss = metrics.get("eval_loss")
        self._is_best_step = False
        if loss is None:
            return control

        if self.best_metric is None or loss < self.best_metric:
            self.best_metric = loss
            LOGGER.info("New best eval loss %.4f, saving checkpoint.", loss)
            control.should_save = True
            self._is_best_step = True
        return control

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        if not self._is_best_step:
            return control
        checkpoint_dir = kwargs.get("checkpoint_dir")
        if not checkpoint_dir:
            return control
        destination = os.path.join(self.output_dir, "checkpoint-best")
        LOGGER.info("Copying best checkpoint to %s", destination)
        os.makedirs(destination, exist_ok=True)
        for item in os.listdir(checkpoint_dir):
            src_path = os.path.join(checkpoint_dir, item)
            dst_path = os.path.join(destination, item)
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            else:
                shutil.copy2(src_path, dst_path)
        return control


class IntervalCheckpointCallback(TrainerCallback):
    """
    Forces checkpoint saving on a fixed interval, even when evaluation is disabled.
    """

    def __init__(self, save_steps: int) -> None:
        self.save_steps = save_steps

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        if (
            self.save_steps
            and state.global_step > 0
            and state.global_step % self.save_steps == 0
        ):
            control.should_save = True
        return control
