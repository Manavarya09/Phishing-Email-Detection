"""Data loading utilities."""

from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils import LOGGER


def load_dataset(path: str | Path) -> pd.DataFrame:
    """
    Load phishing email dataset from CSV.

    Args:
        path: Path to CSV file containing at least 'text' and 'label' columns.

    Returns:
        DataFrame with loaded data.

    Raises:
        FileNotFoundError: If the CSV is missing.
        ValueError: If the CSV is empty or missing required columns.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Please place phishing_emails.csv in data/.")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Loaded dataset is empty.")

    required_cols = {"text", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {required_cols}")

    LOGGER.info("Loaded dataset with %d rows from %s", len(df), csv_path)
    return df
