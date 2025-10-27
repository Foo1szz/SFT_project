"""
Utilities for loading and merging YAML configuration files.
"""

from __future__ import annotations

import copy
import pathlib
from typing import Any, Dict, Iterable, Optional

import yaml


def _ensure_path(path: Any) -> pathlib.Path:
    if isinstance(path, pathlib.Path):
        return path
    if isinstance(path, str):
        return pathlib.Path(path)
    raise TypeError(f"Unsupported path type: {type(path)!r}")


def load_yaml_file(path: Any) -> Dict[str, Any]:
    """
    Load a YAML file into a Python dictionary.
    """
    file_path = _ensure_path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration at {file_path} must be a mapping.")
    return data


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries, giving precedence to override.
    """
    merged: Dict[str, Any] = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_and_merge_configs(
    primary: Any,
    overrides: Optional[Iterable[Any]] = None,
) -> Dict[str, Any]:
    """
    Load a primary YAML config and merge optional overrides.
    """
    config = load_yaml_file(primary)
    for override in overrides or []:
        config = merge_dicts(config, load_yaml_file(override))
    return config
