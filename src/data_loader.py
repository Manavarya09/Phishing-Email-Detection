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

    # Check for required columns - support both 'text' and 'email_text'
    if 'label' not in df.columns:
        raise ValueError("Dataset must contain 'label' column")
    
    # Rename email_text to text if needed for consistency
    if 'email_text' in df.columns and 'text' not in df.columns:
        df = df.rename(columns={'email_text': 'text'})
    elif 'text' not in df.columns:
        raise ValueError("Dataset must contain either 'text' or 'email_text' column")

    LOGGER.info("Loaded dataset with %d rows from %s", len(df), csv_path)
    return df
