"""Checkpoint save/load helpers for the BERT study project."""

from __future__ import annotations

from pathlib import Path

import torch

from .classification import BertForSequenceClassification
from .model import BertConfig
from .tokenizer import WordPieceTokenizer


def save_sequence_classification_checkpoint(
    path: Path,
    *,
    model: BertForSequenceClassification,
    tokenizer: WordPieceTokenizer,
    config: BertConfig,
    labels: list[str],
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "tokenizer": tokenizer.to_dict(),
        "config": config.__dict__.copy(),
        "labels": labels,
    }
    torch.save(payload, path)


def load_sequence_classification_checkpoint(
    path: Path,
) -> tuple[BertForSequenceClassification, WordPieceTokenizer, BertConfig, list[str]]:
    payload = torch.load(path, weights_only=False)
    config = BertConfig(**payload["config"])
    labels = [str(label) for label in payload["labels"]]
    tokenizer = WordPieceTokenizer.from_dict(payload["tokenizer"])
    model = BertForSequenceClassification(config, num_labels=len(labels))
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, tokenizer, config, labels
