"""Classic machine learning models for phishing detection."""

from pathlib import Path
from typing import Any, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from src.config import config
from src.evaluation import compute_classification_metrics
from src.utils import LOGGER


def _log_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> dict:
    metrics = compute_classification_metrics(y_true, y_pred)
    LOGGER.info("%s validation metrics: %s", model_name, metrics)
    return metrics


def train_logistic_regression(
    X_train, y_train, X_val, y_val
) -> Tuple[LogisticRegression, dict]:
    """Train Logistic Regression classifier."""
    model = LogisticRegression(max_iter=200, C=config.logistic_reg_c, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    metrics = _log_metrics(y_val, y_pred, "Logistic Regression")
    return model, metrics


def train_svm(
    X_train, y_train, X_val, y_val
) -> Tuple[LinearSVC, dict]:
    """Train linear SVM classifier."""
    model = LinearSVC(C=config.svm_c, max_iter=config.svm_max_iter)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    metrics = _log_metrics(y_val, y_pred, "Linear SVM")
    return model, metrics


def train_random_forest(
    X_train, y_train, X_val, y_val
) -> Tuple[RandomForestClassifier, dict]:
    """Train Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=config.rf_estimators,
        max_depth=config.rf_max_depth,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    metrics = _log_metrics(y_val, y_pred, "Random Forest")
    return model, metrics


def save_model(model: Any, path: Path) -> None:
    """Persist model to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    LOGGER.info("Saved model to %s", path)
