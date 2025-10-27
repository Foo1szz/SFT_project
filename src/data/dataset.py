"""
Dataset preparation utilities for the SFT project.
"""

from __future__ import annotations

import json
import logging
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from datasets import DatasetDict, concatenate_datasets, load_dataset
from jinja2 import Template
from transformers import PreTrainedTokenizerBase

LOGGER = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    name: str
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    test_file: Optional[str] = None
    prompt_template: str = "{{instruction}}\n{{input}}\nResponse:"
    answer_template: str = "{{output}}"
    input_fields: Optional[List[str]] = None
    output_field: str = "output"
    max_source_length: int = 2048
    max_target_length: int = 512
    shuffle: bool = True
    num_proc: Optional[int] = None


def _to_config(config: Dict[str, Any]) -> DatasetConfig:
    data = {**config}
    dataset_name = data.pop("dataset_name", None) or data.pop("name", None)
    if not dataset_name:
        raise ValueError("Dataset configuration must include `dataset_name`.")
    return DatasetConfig(name=dataset_name, **data)


def _render_template(template: str, example: Dict[str, Any]) -> str:
    return Template(template).render(**example)


def dataset_config_from_dict(config: Dict[str, Any]) -> DatasetConfig:
    """
    Public adapter for converting dictionaries into DatasetConfig instances.
    """
    return _to_config(config)


def build_prompt(example: Dict[str, Any], cfg: DatasetConfig) -> Dict[str, Any]:
    prompt_fields: Dict[str, Any] = {}
    if cfg.input_fields:
        for field in cfg.input_fields:
            prompt_fields[field] = example.get(field, "")
    else:
        prompt_fields = example
    rendered_prompt = _render_template(cfg.prompt_template, prompt_fields)
    answer_context = dict(example)
    answer_context.update(prompt_fields)
    rendered_answer = _render_template(cfg.answer_template, answer_context)
    return {
        **example,
        "prompt": rendered_prompt.strip(),
        "response": rendered_answer.strip(),
    }


class SFTDataModule:
    """
    Loads fine-tuning datasets and prepares tokenized inputs.
    """

    def __init__(
        self,
        data_configs: Iterable[Dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        seed: int = 42,
    ) -> None:
        self.dataset_configs = [_to_config(cfg) for cfg in data_configs]
        self.tokenizer = tokenizer
        self.seed = seed

    def build_datasets(self) -> DatasetDict:
        datasets: Dict[str, List] = {}
        for cfg in self.dataset_configs:
            LOGGER.info("Loading dataset %s", cfg.name)
            ds = self._load_single_dataset(cfg)
            for split in ds.keys():
                datasets.setdefault(split, []).append(ds[split])

        combined = {
            split: concatenate_datasets(parts) if len(parts) > 1 else parts[0]
            for split, parts in datasets.items()
        }
        dataset_dict = DatasetDict(combined)
        if "train" in dataset_dict:
            dataset_dict["train"] = dataset_dict["train"].shuffle(seed=self.seed)
        return dataset_dict

    def _load_single_dataset(self, cfg: DatasetConfig) -> DatasetDict:
        data_files: Dict[str, str] = {}
        if cfg.train_file:
            data_files["train"] = cfg.train_file
        if cfg.validation_file:
            data_files["validation"] = cfg.validation_file
        if cfg.test_file:
            data_files["test"] = cfg.test_file

        if not data_files:
            raise ValueError(f"No data files specified for dataset {cfg.name}.")

        for split, path in data_files.items():
            if not pathlib.Path(path).exists():
                LOGGER.warning(
                    "File %s for dataset %s (%s) does not exist yet.",
                    path,
                    cfg.name,
                    split,
                )

        dataset = load_dataset("json", data_files=data_files)
        dataset = dataset.map(
            lambda ex: build_prompt(ex, cfg),
            desc=f"render prompts for {cfg.name}",
            num_proc=cfg.num_proc,
        )

        reference_split = "train" if "train" in dataset else next(iter(dataset.keys()))
        remove_columns = [
            col
            for col in dataset[reference_split].column_names
            if col not in {"prompt", "response"}
        ]

        tokenized = dataset.map(
            lambda ex: self._tokenize_example(ex, cfg),
            batched=False,
            remove_columns=remove_columns,
            desc=f"tokenize {cfg.name}",
            num_proc=cfg.num_proc,
        )
        return tokenized

    def _tokenize_example(
        self,
        example: Dict[str, Any],
        cfg: DatasetConfig,
    ) -> Dict[str, Any]:
        prompt = example["prompt"]
        response = example["response"]

        prompt_enc = self.tokenizer(
            prompt,
            truncation=True,
            max_length=cfg.max_source_length,
            padding=False,
            add_special_tokens=True,
        )
        response_enc = self.tokenizer(
            response,
            truncation=True,
            max_length=cfg.max_target_length,
            padding=False,
            add_special_tokens=True,
        )

        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is not None and (
            not response_enc["input_ids"]
            or response_enc["input_ids"][-1] != eos_token_id
        ):
            response_enc["input_ids"].append(eos_token_id)
            response_enc["attention_mask"].append(1)

        input_ids = prompt_enc["input_ids"] + response_enc["input_ids"]
        attention_mask = prompt_enc["attention_mask"] + response_enc["attention_mask"]
        labels = [-100] * len(prompt_enc["input_ids"]) + response_enc["input_ids"]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt": prompt,
            "response": response,
        }


def normalize_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Utility helper used during preprocessing to transform JSONL into dictionaries.
    """
    output: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            output.append(json.loads(line))
    return output
