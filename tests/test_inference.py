from pathlib import Path

from bert.checkpoint import (
    load_sequence_classification_checkpoint,
    save_sequence_classification_checkpoint,
)
from bert.classification import BertForSequenceClassification
from bert.inference import predict_sequence_class
from bert.model import BertConfig
from bert.tokenizer import WordPieceTokenizer


def test_predict_sequence_class_returns_label_and_scores() -> None:
    tokenizer = WordPieceTokenizer(
        vocab=[
            "[PAD]",
            "[CLS]",
            "[SEP]",
            "[MASK]",
            "[UNK]",
            "i",
            "feel",
            "great",
        ]
    )
    model = BertForSequenceClassification(
        BertConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=8,
            max_position_embeddings=8,
            num_hidden_layers=1,
            num_heads=2,
            intermediate_size=16,
            hidden_dropout_prob=0.0,
        ),
        num_labels=2,
    )
    labels = ["joy", "sadness"]

    prediction, scores = predict_sequence_class(
        model,
        tokenizer,
        "I feel great",
        labels=labels,
    )

    assert prediction in labels
    assert set(scores) == set(labels)
    assert abs(sum(scores.values()) - 1.0) < 1e-6


def test_sequence_classification_checkpoint_supports_inference(tmp_path: Path) -> None:
    tokenizer = WordPieceTokenizer(
        vocab=[
            "[PAD]",
            "[CLS]",
            "[SEP]",
            "[MASK]",
            "[UNK]",
            "i",
            "feel",
            "bad",
        ]
    )
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=8,
        max_position_embeddings=8,
        num_hidden_layers=1,
        num_heads=2,
        intermediate_size=16,
        hidden_dropout_prob=0.0,
    )
    model = BertForSequenceClassification(config, num_labels=2)
    checkpoint_path = tmp_path / "classifier.pt"
    labels = ["joy", "sadness"]

    save_sequence_classification_checkpoint(
        checkpoint_path,
        model=model,
        tokenizer=tokenizer,
        config=config,
        labels=labels,
    )
    restored_model, restored_tokenizer, _, restored_labels = (
        load_sequence_classification_checkpoint(checkpoint_path)
    )

    prediction, scores = predict_sequence_class(
        restored_model,
        restored_tokenizer,
        "I feel bad",
        labels=restored_labels,
    )

    assert prediction in restored_labels
    assert set(scores) == set(restored_labels)
