"""
Wrapper utilities for constructing Hugging Face TrainingArguments with optional
FSDP or DeepSpeed configurations.
"""

from __future__ import annotations

from typing import Any, Dict

from transformers import TrainingArguments


def build_training_arguments(
    training_config: Dict[str, Any],
    output_dir: str,
) -> TrainingArguments:
    args_dict: Dict[str, Any] = {
        "output_dir": output_dir,
        "num_train_epochs": training_config.get("num_train_epochs", 1.0),
        "learning_rate": training_config.get("learning_rate", 5e-5),
        "lr_scheduler_type": training_config.get("lr_scheduler_type", "linear"),
        "warmup_steps": training_config.get("warmup_steps"),
        "warmup_ratio": training_config.get("warmup_ratio"),
        "weight_decay": training_config.get("weight_decay", 0.0),
        "per_device_train_batch_size": training_config.get("per_device_train_batch_size", 1),
        "per_device_eval_batch_size": training_config.get("per_device_eval_batch_size", 1),
        "gradient_accumulation_steps": training_config.get("gradient_accumulation_steps", 1),
        "max_grad_norm": training_config.get("max_grad_norm", 1.0),
        "evaluation_strategy": training_config.get("eval_strategy", "no"),
        "eval_steps": training_config.get("eval_steps"),
        "logging_steps": training_config.get("logging_steps", 100),
        "save_strategy": training_config.get("save_strategy", "steps"),
        "save_steps": training_config.get("save_steps"),
        "save_total_limit": training_config.get("save_total_limit"),
        "bf16": training_config.get("bf16", False),
        "fp16": training_config.get("fp16", False),
        "seed": training_config.get("seed", 42),
        "report_to": training_config.get("report_to", ["none"]),
        "gradient_checkpointing": training_config.get("gradient_checkpointing", False),
    }

    fsdp_config = training_config.get("fsdp_config")
    if fsdp_config:
        args_dict.update(
            {
                "fsdp": fsdp_config.get("fsdp"),
                "fsdp_config": {
                    key: value
                    for key, value in fsdp_config.items()
                    if key != "fsdp"
                },
            }
        )

    deepspeed_config = training_config.get("deepspeed_config")
    if deepspeed_config:
        args_dict["deepspeed"] = deepspeed_config

    # Remove keys with None to avoid HF validation issues.
    clean_args = {k: v for k, v in args_dict.items() if v is not None}
    return TrainingArguments(**clean_args)
