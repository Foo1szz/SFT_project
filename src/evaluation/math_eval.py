"""
Evaluation helpers for the MATH dataset.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple


BOXED_PATTERN = re.compile(r"\\boxed\{([^}]*)\}")
LATEX_CLEANUP = re.compile(r"\\\\")
WHITESPACE = re.compile(r"\s+")


def extract_math_answer(text: str) -> str:
    """
    Extract the final answer from a MATH-style solution.
    """
    if not text:
        return ""
    text = text.strip()
    match = BOXED_PATTERN.search(text)
    if match:
        candidate = match.group(1)
    else:
        candidate = text.splitlines()[-1]
    candidate = LATEX_CLEANUP.sub("", candidate)
    candidate = WHITESPACE.sub(" ", candidate)
    return candidate.strip(" .")


def evaluate_math(examples: Iterable[Tuple[str, str]]) -> Dict[str, float]:
    """
    Compute accuracy over (prediction, reference) pairs.
    """
    total = 0
    correct = 0
    detailed: List[Dict[str, str]] = []
    for prediction, reference in examples:
        total += 1
        pred_norm = extract_math_answer(prediction)
        ref_norm = extract_math_answer(reference)
        if pred_norm == ref_norm:
            correct += 1
        detailed.append(
            {"prediction": pred_norm, "reference": ref_norm, "raw_prediction": prediction}
        )
    accuracy = correct / total if total else 0.0
    return {"accuracy": accuracy, "total": total, "correct": correct, "samples": detailed}
