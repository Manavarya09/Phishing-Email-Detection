"""Utility helpers for logging, reproducibility, and simple operations."""

import logging
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create or fetch a configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


def set_seed(seed: int) -> None:
    """Set seeds for reproducibility across numpy, random, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dir(path: Path) -> None:
    """Create directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


LOGGER = get_logger("phishing_xai")
