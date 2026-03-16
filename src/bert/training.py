"""Training and evaluation helpers for sequence classification."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .batching import encode_text_batch
from .classification import BertForSequenceClassification
from .data import ClassificationExample
from .model import BertConfig
from .tokenizer import WordPieceTokenizer


@dataclass(frozen=True)
class SequenceClassificationTrainingConfig:
    vocab_size: int = 128
    hidden_size: int = 64
    max_position_embeddings: int = 128
    num_hidden_layers: int = 2
    num_heads: int = 4
    intermediate_size: int = 256
    batch_size: int = 8
    learning_rate: float = 3e-4
    num_epochs: int = 10
    seed: int = 0


@dataclass(frozen=True)
class SequenceClassificationArtifacts:
    model: BertForSequenceClassification
    tokenizer: WordPieceTokenizer
    config: BertConfig
    labels: list[str]
    train_losses: list[float]
    validation_accuracy: float


def _build_batches(
    examples: list[ClassificationExample],
    *,
    batch_size: int,
) -> list[list[ClassificationExample]]:
    return [examples[index : index + batch_size] for index in range(0, len(examples), batch_size)]


def evaluate_sequence_classifier(
    model: BertForSequenceClassification,
    tokenizer: WordPieceTokenizer,
    examples: list[ClassificationExample],
    *,
    label_to_id: dict[str, int],
    batch_size: int,
) -> float:
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_examples in _build_batches(examples, batch_size=batch_size):
            batch = encode_text_batch(tokenizer, [example.text for example in batch_examples])
            labels = torch.tensor(
                [label_to_id[example.label] for example in batch_examples],
                dtype=torch.long,
            )
            logits, _ = model(
                batch.token_ids,
                batch.token_type_ids,
                attention_mask=batch.attention_mask,
            )
            predictions = logits.argmax(dim=-1)
            correct += int((predictions == labels).sum().item())
            total += labels.numel()

    return correct / total


def train_sequence_classifier(
    train_examples: list[ClassificationExample],
    validation_examples: list[ClassificationExample],
    *,
    label_to_id: dict[str, int],
    labels: list[str],
    training_config: SequenceClassificationTrainingConfig,
) -> SequenceClassificationArtifacts:
    torch.manual_seed(training_config.seed)

    tokenizer = WordPieceTokenizer.fit(
        [example.text for example in train_examples],
        vocab_size=training_config.vocab_size,
        min_frequency=1,
    )
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=training_config.hidden_size,
        max_position_embeddings=training_config.max_position_embeddings,
        num_hidden_layers=training_config.num_hidden_layers,
        num_heads=training_config.num_heads,
        intermediate_size=training_config.intermediate_size,
        hidden_dropout_prob=0.0,
    )
    model = BertForSequenceClassification(config, num_labels=len(labels))
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate)

    train_losses: list[float] = []
    for _ in range(training_config.num_epochs):
        model.train()
        for batch_examples in _build_batches(train_examples, batch_size=training_config.batch_size):
            batch = encode_text_batch(tokenizer, [example.text for example in batch_examples])
            labels_tensor = torch.tensor(
                [label_to_id[example.label] for example in batch_examples],
                dtype=torch.long,
            )
            _, loss = model(
                batch.token_ids,
                batch.token_type_ids,
                attention_mask=batch.attention_mask,
                labels=labels_tensor,
            )
            assert loss is not None
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

    validation_accuracy = evaluate_sequence_classifier(
        model,
        tokenizer,
        validation_examples,
        label_to_id=label_to_id,
        batch_size=training_config.batch_size,
    )
    return SequenceClassificationArtifacts(
        model=model,
        tokenizer=tokenizer,
        config=config,
        labels=labels,
        train_losses=train_losses,
        validation_accuracy=validation_accuracy,
    )
