"""
Evaluation helpers for the GSM8K dataset.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, Tuple

ANSWER_PATTERN = re.compile(r"[Aa]nswer\s*[:\-]\s*(.*)")


def normalize_numeric_answer(text: str) -> str:
    if text is None:
        return ""
    normalized = text.strip()
    match = ANSWER_PATTERN.search(normalized)
    if match:
        normalized = match.group(1)
    normalized = normalized.replace(",", "")
    normalized = normalized.split("\n")[0]
    normalized = normalized.strip(" .")
    return normalized


def evaluate_gsm8k(examples: Iterable[Tuple[str, str]]) -> Dict[str, float]:
    total = 0
    correct = 0
    for prediction, reference in examples:
        total += 1
        pred_norm = normalize_numeric_answer(prediction)
        ref_norm = normalize_numeric_answer(reference)
        if pred_norm == ref_norm:
            correct += 1
    accuracy = correct / total if total else 0.0
    return {"accuracy": accuracy, "total": total, "correct": correct}
