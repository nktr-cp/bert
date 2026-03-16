import json
from pathlib import Path

from bert.checkpoint import (
    load_sequence_classification_checkpoint,
    save_sequence_classification_checkpoint,
)
from bert.data import build_label_vocabulary, load_classification_examples, split_examples
from bert.training import SequenceClassificationTrainingConfig, train_sequence_classifier


def test_load_classification_examples_reads_jsonl(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps({"text": "I feel great", "label": "joy"}),
                json.dumps({"text": "I feel awful", "label": "sadness"}),
            ]
        ),
        encoding="utf-8",
    )

    examples = load_classification_examples(dataset_path)

    assert [(example.text, example.label) for example in examples] == [
        ("I feel great", "joy"),
        ("I feel awful", "sadness"),
    ]


def test_train_sequence_classifier_returns_artifacts_and_checkpoint_round_trip(
    tmp_path: Path,
) -> None:
    examples = [
        {"text": "I feel happy", "label": "joy"},
        {"text": "This is wonderful", "label": "joy"},
        {"text": "I feel sad", "label": "sadness"},
        {"text": "This is terrible", "label": "sadness"},
    ]
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        "\n".join(json.dumps(example) for example in examples),
        encoding="utf-8",
    )

    loaded_examples = load_classification_examples(dataset_path)
    train_examples, validation_examples = split_examples(
        loaded_examples,
        validation_fraction=0.5,
        seed=0,
    )
    label_to_id, labels = build_label_vocabulary(loaded_examples)

    artifacts = train_sequence_classifier(
        train_examples,
        validation_examples,
        label_to_id=label_to_id,
        labels=labels,
        training_config=SequenceClassificationTrainingConfig(
            vocab_size=32,
            hidden_size=16,
            max_position_embeddings=16,
            num_hidden_layers=1,
            num_heads=2,
            intermediate_size=32,
            batch_size=2,
            learning_rate=1e-3,
            num_epochs=1,
            seed=0,
        ),
    )

    assert artifacts.labels == labels
    assert artifacts.train_losses
    assert 0.0 <= artifacts.validation_accuracy <= 1.0

    checkpoint_path = tmp_path / "classifier.pt"
    save_sequence_classification_checkpoint(
        checkpoint_path,
        model=artifacts.model,
        tokenizer=artifacts.tokenizer,
        config=artifacts.config,
        labels=artifacts.labels,
    )
    restored_model, restored_tokenizer, restored_config, restored_labels = (
        load_sequence_classification_checkpoint(checkpoint_path)
    )

    assert restored_labels == artifacts.labels
    assert restored_tokenizer.vocab == artifacts.tokenizer.vocab
    assert restored_config.hidden_size == artifacts.config.hidden_size
    assert restored_model.classifier.out_features == len(labels)
