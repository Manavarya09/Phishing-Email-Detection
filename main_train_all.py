"""
Orchestrate the full training pipeline:
- load data
- preprocess and split
- build features
- train classic ML models
- train BERT model
- evaluate and persist artifacts
"""

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

from src import preprocessing
from src.bert_model import EmailDataset, evaluate as evaluate_bert, train_bert_model
from src.classic_models import (
    save_model,
    train_logistic_regression,
    train_random_forest,
    train_svm,
)
from src.config import config
from src.data_loader import load_dataset
from src.evaluation import compute_classification_metrics
from src.features import (
    combine_features,
    fit_transform_metadata,
    fit_transform_tfidf,
    transform_metadata,
    transform_tfidf,
)
from src.utils import LOGGER, set_seed


def train_classic_models(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train classic models, select best by F1, and evaluate on test."""
    models = []

    logreg_model, logreg_metrics = train_logistic_regression(X_train, y_train, X_val, y_val)
    models.append(("logreg", logreg_model, logreg_metrics))

    svm_model, svm_metrics = train_svm(X_train, y_train, X_val, y_val)
    models.append(("svm", svm_model, svm_metrics))

    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_val, y_val)
    models.append(("random_forest", rf_model, rf_metrics))

    best_name, best_model, best_metrics = max(models, key=lambda m: m[2].get("f1", 0.0))
    LOGGER.info("Best classic model: %s with F1=%.3f", best_name, best_metrics.get("f1", 0.0))

    # Save artifacts
    save_model(svm_model, config.classic_tfidf_svm_path)
    save_model(rf_model, config.classic_tfidf_rf_path)
    joblib.dump(best_model, config.classic_best_model_path)

    # Evaluate best on test
    if hasattr(best_model, "predict_proba"):
        test_proba = best_model.predict_proba(X_test)[:, 1]
    else:
        decision = best_model.decision_function(X_test)
        test_proba = 1 / (1 + np.exp(-decision))
    test_pred = best_model.predict(X_test)
    test_metrics = compute_classification_metrics(y_test, test_pred, test_proba)
    LOGGER.info("Classic best model test metrics: %s", test_metrics)
    return best_model, test_metrics


def main():
    set_seed(config.random_seed)
    df = load_dataset(config.data_path)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessing.split_dataset(df)

    # Feature engineering
    X_tfidf_train, vectorizer = fit_transform_tfidf(X_train)
    X_tfidf_val = transform_tfidf(X_val, vectorizer)
    X_tfidf_test = transform_tfidf(X_test, vectorizer)
    joblib.dump(vectorizer, config.tfidf_vectorizer_path)

    X_meta_train, scaler = fit_transform_metadata(X_train)
    X_meta_val = transform_metadata(X_val, scaler)
    X_meta_test = transform_metadata(X_test, scaler)
    joblib.dump(scaler, config.metadata_scaler_path)

    X_train_comb = combine_features(X_tfidf_train, X_meta_train)
    X_val_comb = combine_features(X_tfidf_val, X_meta_val)
    X_test_comb = combine_features(X_tfidf_test, X_meta_test)

    LOGGER.info("Training classic models...")
    _, classic_test_metrics = train_classic_models(X_train_comb, y_train, X_val_comb, y_val, X_test_comb, y_test)

    LOGGER.info("Training BERT model...")
    bert_model, tokenizer = train_bert_model(X_train, y_train, X_val, y_val)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = EmailDataset(X_test, y_test, tokenizer, config.bert_max_length)
    test_loader = DataLoader(test_dataset, batch_size=config.bert_batch_size, shuffle=False)
    bert_test_metrics = evaluate_bert(bert_model, test_loader, device)
    LOGGER.info("BERT test metrics: %s", bert_test_metrics)

    print("\n=== Test Metrics Summary ===")
    print("Classic (best):", classic_test_metrics)
    print("BERT:", bert_test_metrics)


if __name__ == "__main__":
    main()
