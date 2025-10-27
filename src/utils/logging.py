"""
Shared logging helpers for CLI entry points and libraries.
"""

from __future__ import annotations

import logging
import os
from typing import Optional


def setup_logging(verbosity: int = 1, log_file: Optional[str] = None) -> None:
    """
    Configure logging for the project.
    """
    level = logging.WARNING
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO

    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
