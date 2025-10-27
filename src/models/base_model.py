"""
Model loading utilities with optional LoRA injection.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

LOGGER = logging.getLogger(__name__)


def _parse_dtype(name: Optional[str]):
    if name is None:
        return None
    normalized = str(name).lower()
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype {name}")
    return mapping[normalized]


def load_tokenizer(model_config: Dict[str, Any]) -> PreTrainedTokenizerBase:
    tokenizer_name = model_config.get("tokenizer_name") or model_config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=model_config.get("trust_remote_code", False),
    )
    pad_token = model_config.get("pad_token")
    if tokenizer.pad_token is None:
        if pad_token:
            tokenizer.add_special_tokens({"pad_token": pad_token})
        else:
            tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    use_lora: bool,
) -> PreTrainedModel:
    dtype = _parse_dtype(model_config.get("dtype"))
    kwargs = model_config.get("model_kwargs", {})
    LOGGER.info("Loading base model %s", model_config["model_name"])
    model = AutoModelForCausalLM.from_pretrained(
        model_config["model_name"],
        trust_remote_code=model_config.get("trust_remote_code", False),
        torch_dtype=dtype,
        use_flash_attention_2=model_config.get("use_flash_attention_2", False),
        **kwargs,
    )

    lora_cfg = training_config.get("lora")
    if use_lora:
        if not lora_cfg:
            raise ValueError("LoRA requested but no `lora` section found in training config.")
        LOGGER.info("Injecting LoRA adapters with config: %s", lora_cfg)
        lora_config = LoraConfig(
            r=lora_cfg.get("r", 8),
            lora_alpha=lora_cfg.get("alpha", 16),
            lora_dropout=lora_cfg.get("dropout", 0.05),
            bias=lora_cfg.get("bias", "none"),
            task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
            target_modules=lora_cfg.get("target_modules"),
        )
        if lora_cfg.get("int8_training"):
            model = prepare_model_for_int8_training(model)
        model = get_peft_model(model, lora_config)
        if lora_cfg.get("use_gradient_checkpointing"):
            model.gradient_checkpointing_enable()
    else:
        if training_config.get("gradient_checkpointing"):
            model.gradient_checkpointing_enable()

    return model


def maybe_merge_lora_weights(model: PreTrainedModel, merge: bool) -> PreTrainedModel:
    if merge and hasattr(model, "merge_and_unload"):
        LOGGER.info("Merging LoRA weights into the base model.")
        model = model.merge_and_unload()
    return model


def save_model_artifacts(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: str,
) -> None:
    LOGGER.info("Saving model to %s", output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
