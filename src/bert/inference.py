"""Inference helpers for sentence-level classification."""

from __future__ import annotations

import torch

from .batching import encode_text_batch
from .classification import BertForSequenceClassification
from .tokenizer import WordPieceTokenizer


def predict_sequence_class(
    model: BertForSequenceClassification,
    tokenizer: WordPieceTokenizer,
    text: str,
    *,
    labels: list[str],
) -> tuple[str, dict[str, float]]:
    model.eval()
    batch = encode_text_batch(tokenizer, [text])

    with torch.no_grad():
        logits, _ = model(
            batch.token_ids,
            batch.token_type_ids,
            attention_mask=batch.attention_mask,
        )
        probabilities = torch.softmax(logits[0], dim=-1)

    scores = {
        label: float(probability)
        for label, probability in zip(labels, probabilities.tolist(), strict=True)
    }
    prediction = labels[int(probabilities.argmax().item())]
    return prediction, scores
