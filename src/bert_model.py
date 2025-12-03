"""Fine-tuning DistilBERT for phishing detection."""

from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.config import config
from src.evaluation import compute_classification_metrics
from src.utils import LOGGER, set_seed


class EmailDataset(Dataset):
    """Torch dataset for tokenized emails."""

    def __init__(self, texts: Iterable[str], labels: Iterable[int], tokenizer, max_length: int):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        item = {key: tensor[idx] for key, tensor in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_bert_model(
    train_texts: Iterable[str],
    train_labels: Iterable[int],
    val_texts: Iterable[str],
    val_labels: Iterable[int],
):
    """Fine-tune DistilBERT and save the model + tokenizer."""
    set_seed(config.random_seed)
    device = _get_device()
    LOGGER.info("Using device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(config.pretrained_model_name, num_labels=2)
    model.to(device)

    train_dataset = EmailDataset(train_texts, train_labels, tokenizer, config.bert_max_length)
    val_dataset = EmailDataset(val_texts, val_labels, tokenizer, config.bert_max_length)

    train_loader = DataLoader(train_dataset, batch_size=config.bert_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.bert_batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.bert_learning_rate, weight_decay=config.bert_weight_decay)
    total_steps = config.bert_epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in range(config.bert_epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.bert_epochs} - training"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        avg_loss = total_loss / max(1, len(train_loader))

        val_metrics = evaluate(model, val_loader, device)
        LOGGER.info("Epoch %d - train_loss=%.4f - val_metrics=%s", epoch + 1, avg_loss, val_metrics)

    save_dir = Path(config.bert_model_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    LOGGER.info("Saved fine-tuned BERT model to %s", save_dir)
    return model, tokenizer


def evaluate(model, data_loader: DataLoader, device: torch.device) -> dict:
    """Evaluate a BERT model on a dataloader."""
    model.eval()
    all_preds, all_labels, all_probas = [], [], []
    with torch.no_grad():
        for batch in data_loader:
            labels = batch["labels"].numpy()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            all_probas.extend(probs[:, 1].tolist())
    return compute_classification_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probas))


def load_finetuned_model():
    """Load fine-tuned model and tokenizer from disk."""
    model_dir = Path(config.bert_model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"BERT model directory not found at {model_dir}. Train the model first.")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = _get_device()
    model.to(device)
    model.eval()
    return model, tokenizer, device


def predict_bert(texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Predict labels and probabilities using the saved BERT model."""
    model, tokenizer, device = load_finetuned_model()
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=config.bert_max_length,
        return_tensors="pt",
    )
    with torch.no_grad():
        batch = {k: v.to(device) for k, v in encodings.items()}
        outputs = model(**batch)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
    return preds, probs[:, 1]
