"""Text cleaning and dataset splitting utilities."""

import re
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import config
from src.utils import LOGGER


def clean_text(text: str) -> str:
    """
    Basic text cleaning that preserves URLs and key tokens.

    - Lowercase
    - Remove extra whitespace
    - Strip minimal punctuation while keeping URL characters
    """
    text = text.lower()
    # Remove punctuation except URL-friendly characters
    text = re.sub(r"[^\w\s:/?&.=+-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_dataset(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Split dataset into train/validation/test sets with stratification.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    df = df.copy()
    df["clean_text"] = df["text"].astype(str).apply(clean_text)
    X = df["clean_text"]
    y = df["label"]

    # First split off validation+test
    temp_size = config.val_size + config.test_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=temp_size,
        stratify=y,
        random_state=config.random_seed,
    )

    # Split temp into validation and test equally
    val_ratio = config.val_size / temp_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=1 - val_ratio,
        stratify=y_temp,
        random_state=config.random_seed,
    )

    LOGGER.info("Split dataset: train=%d, val=%d, test=%d", len(X_train), len(X_val), len(X_test))
    return X_train, X_val, X_test, y_train, y_val, y_test
