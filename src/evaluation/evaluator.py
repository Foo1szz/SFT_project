"""
End-to-end checkpoint evaluation for MATH and GSM8K datasets.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from ..data.dataset import (
    DatasetConfig,
    build_prompt,
    dataset_config_from_dict,
    normalize_jsonl,
)
from ..models.base_model import load_tokenizer
from ..utils.config import load_yaml_file
from .gsm8k_eval import evaluate_gsm8k
from .math_eval import evaluate_math

LOGGER = logging.getLogger(__name__)


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class SFTCheckpointEvaluator:
    """
    Loads a fine-tuned checkpoint and evaluates it on target datasets.
    """

    def __init__(
        self,
        model_config_path: str,
        data_config_paths: Iterable[str],
        eval_config_path: str,
    ) -> None:
        self.model_config = load_yaml_file(model_config_path)
        self.eval_config = load_yaml_file(eval_config_path)
        self.tokenizer = load_tokenizer(self.model_config)
        self.device = _select_device()
        self.data_configs = {
            cfg.name: cfg
            for cfg in self._load_data_configs(data_config_paths)
        }
        LOGGER.info("Evaluator initialized on device %s", self.device)

    def _load_data_configs(
        self,
        paths: Iterable[str],
    ) -> List[DatasetConfig]:
        return [
            dataset_config_from_dict(load_yaml_file(path))
            for path in paths
        ]

    def _load_model(self, checkpoint_path: str) -> torch.nn.Module:
        checkpoint_dir = Path(checkpoint_path)
        LOGGER.info("Loading checkpoint from %s", checkpoint_dir)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir,
            trust_remote_code=self.model_config.get("trust_remote_code", False),
        )
        adapter_config = checkpoint_dir / "adapter_config.json"
        if adapter_config.exists():
            base_model_name = self.model_config["model_name"]
            LOGGER.info("Detected LoRA adapters. Loading base model: %s", base_model_name)
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                trust_remote_code=self.model_config.get("trust_remote_code", False),
            )
            model = PeftModel.from_pretrained(base_model, checkpoint_dir)
        model.to(self.device)
        model.eval()
        return model

    def _prepare_samples(self, dataset_name: str) -> List[Dict[str, str]]:
        cfg = self.data_configs[dataset_name]
        data_file = cfg.test_file or cfg.validation_file
        if not data_file:
            raise ValueError(f"Dataset {dataset_name} does not provide test or validation file.")
        records = normalize_jsonl(data_file)
        samples: List[Dict[str, str]] = []
        for record in records:
            rendered = build_prompt(record, cfg)
            samples.append(
                {
                    "prompt": rendered["prompt"],
                    "reference": rendered["response"],
                }
            )
        return samples

    def _generate(
        self,
        model,
        prompts: List[str],
    ) -> List[str]:
        batch_size = self.eval_config.get("batch_size", 1)
        max_new_tokens = self.eval_config.get("max_new_tokens", 256)
        temperature = self.eval_config.get("temperature", 0.0)
        top_p = self.eval_config.get("top_p", 1.0)
        top_k = self.eval_config.get("top_k", 50)

        results: List[str] = []
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "do_sample": temperature > 0.0,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            with torch.no_grad():
                outputs = model.generate(**inputs, **generation_kwargs)
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for prompt, full_output in zip(batch_prompts, decoded):
                if full_output.startswith(prompt):
                    generated = full_output[len(prompt) :].strip()
                else:
                    generated = full_output.strip()
                results.append(generated)
        return results

    def evaluate(self, checkpoint_path: str) -> Dict[str, Dict[str, float]]:
        model = self._load_model(checkpoint_path)
        metrics: Dict[str, Dict[str, float]] = {}
        for dataset_name in self.eval_config.get("datasets", []):
            if dataset_name not in self.data_configs:
                LOGGER.warning("Dataset %s not present in data configs; skipping.", dataset_name)
                continue
            samples = self._prepare_samples(dataset_name)
            prompts = [sample["prompt"] for sample in samples]
            predictions = self._generate(model, prompts)
            pairs = list(zip(predictions, [s["reference"] for s in samples]))
            if dataset_name.lower() == "math":
                result = evaluate_math(pairs)
            elif dataset_name.lower() == "gsm8k":
                result = evaluate_gsm8k(pairs)
            else:
                raise ValueError(f"Unsupported evaluation dataset: {dataset_name}")
            metrics[dataset_name] = result
            if self.eval_config.get("write_predictions", True):
                self._write_predictions(dataset_name, samples, predictions)
        return metrics

    def _write_predictions(
        self,
        dataset_name: str,
        samples: List[Dict[str, str]],
        predictions: List[str],
    ) -> None:
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        output_dir = Path("outputs") / "eval" / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{timestamp}.json"
        payload = [
            {
                "prompt": sample["prompt"],
                "reference": sample["reference"],
                "prediction": pred,
            }
            for sample, pred in zip(samples, predictions)
        ]
        LOGGER.info("Writing predictions for %s to %s", dataset_name, path)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
