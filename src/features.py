"""Feature engineering: TF-IDF and lightweight metadata features."""

import re
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from src.config import config


def _extract_metadata_features(texts: Iterable[str]) -> np.ndarray:
    """Compute simple metadata features for each text."""
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    special_pattern = re.compile(r"[!@#$%^&*(),.?\":{}|<>]")
    keyword_pattern = re.compile("|".join([re.escape(k) for k in config.suspicious_keywords]), re.IGNORECASE)

    features = []
    for t in texts:
        text = t or ""
        urls = len(url_pattern.findall(text))
        digits = sum(c.isdigit() for c in text)
        special_chars = len(special_pattern.findall(text))
        words = text.split()
        word_lengths = [len(w) for w in words] if words else [0]
        avg_word_len = float(np.mean(word_lengths))
        suspicious_flag = 1 if keyword_pattern.search(text) else 0
        features.append(
            [
                urls,
                digits,
                special_chars,
                len(text),
                avg_word_len,
                suspicious_flag,
            ]
        )
    return np.array(features, dtype=float)


def fit_transform_tfidf(train_texts: Iterable[str]) -> Tuple[csr_matrix, TfidfVectorizer]:
    """Fit TF-IDF vectorizer and transform training texts."""
    vectorizer = TfidfVectorizer(
        max_features=config.max_features,
        ngram_range=config.ngram_range,
        min_df=config.min_df,
        max_df=config.max_df,
    )
    X_tfidf = vectorizer.fit_transform(train_texts)
    return X_tfidf, vectorizer


def transform_tfidf(texts: Iterable[str], vectorizer: TfidfVectorizer) -> csr_matrix:
    """Transform texts using a fitted TF-IDF vectorizer."""
    return vectorizer.transform(texts)


def fit_transform_metadata(train_texts: Iterable[str]) -> Tuple[np.ndarray, StandardScaler]:
    """Fit scaler on metadata features and transform training data."""
    raw_meta = _extract_metadata_features(train_texts)
    scaler = StandardScaler()
    X_meta = scaler.fit_transform(raw_meta)
    return X_meta, scaler


def transform_metadata(texts: Iterable[str], scaler: StandardScaler) -> np.ndarray:
    """Transform texts into scaled metadata features."""
    raw_meta = _extract_metadata_features(texts)
    return scaler.transform(raw_meta)


def combine_features(tfidf_matrix: csr_matrix, meta_features: np.ndarray) -> csr_matrix:
    """Horizontally stack sparse TF-IDF features with dense metadata features."""
    meta_sparse = csr_matrix(meta_features)
    return hstack([tfidf_matrix, meta_sparse])
