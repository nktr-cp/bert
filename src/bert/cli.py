"""Project entrypoint for manual experiments."""

from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path

from .checkpoint import (
    load_sequence_classification_checkpoint,
    save_sequence_classification_checkpoint,
)
from .data import build_label_vocabulary, load_classification_examples, split_examples
from .tokenizer import WordPieceTokenizer
from .training import (
    SequenceClassificationTrainingConfig,
    evaluate_sequence_classifier,
    train_sequence_classifier,
)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="bert",
        description="BERT implementation workspace.",
    )
    subparsers = parser.add_subparsers(dest="command")

    tokenize_parser = subparsers.add_parser(
        "tokenize",
        help="Train a tiny WordPiece tokenizer and inspect an encoded example.",
    )
    tokenize_parser.add_argument("text", help="Input text to tokenize.")
    tokenize_parser.add_argument(
        "--corpus",
        type=Path,
        help="Optional newline-delimited corpus used to fit the tokenizer.",
    )
    tokenize_parser.add_argument("--vocab-size", type=int, default=64)

    train_parser = subparsers.add_parser(
        "train-classifier",
        help="Train a minimal BERT sequence classifier from a JSONL dataset.",
    )
    train_parser.add_argument("dataset", type=Path)
    train_parser.add_argument("--checkpoint-out", type=Path)
    train_parser.add_argument("--validation-fraction", type=float, default=0.2)
    train_parser.add_argument("--vocab-size", type=int, default=128)
    train_parser.add_argument("--hidden-size", type=int, default=64)
    train_parser.add_argument("--max-position-embeddings", type=int, default=128)
    train_parser.add_argument("--num-hidden-layers", type=int, default=2)
    train_parser.add_argument("--num-heads", type=int, default=4)
    train_parser.add_argument("--intermediate-size", type=int, default=256)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--num-epochs", type=int, default=10)
    train_parser.add_argument("--seed", type=int, default=0)

    evaluate_parser = subparsers.add_parser(
        "evaluate-classifier",
        help="Evaluate a saved sequence classification checkpoint on a JSONL dataset.",
    )
    evaluate_parser.add_argument("checkpoint", type=Path)
    evaluate_parser.add_argument("dataset", type=Path)
    evaluate_parser.add_argument("--batch-size", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "tokenize":
        corpus = [args.text]
        if args.corpus is not None:
            corpus = args.corpus.read_text(encoding="utf-8").splitlines()
        tokenizer = WordPieceTokenizer.fit(corpus, vocab_size=args.vocab_size)
        encoding = tokenizer.encode(args.text)
        print(f"tokens={encoding.tokens}")
        print(f"token_ids={encoding.token_ids}")
        print(f"token_type_ids={encoding.token_type_ids}")
        return

    if args.command == "train-classifier":
        examples = load_classification_examples(args.dataset)
        train_examples, validation_examples = split_examples(
            examples,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
        label_to_id, labels = build_label_vocabulary(examples)
        training_config = SequenceClassificationTrainingConfig(
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            max_position_embeddings=args.max_position_embeddings,
            num_hidden_layers=args.num_hidden_layers,
            num_heads=args.num_heads,
            intermediate_size=args.intermediate_size,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            seed=args.seed,
        )
        artifacts = train_sequence_classifier(
            train_examples,
            validation_examples,
            label_to_id=label_to_id,
            labels=labels,
            training_config=training_config,
        )
        print(f"final_train_loss={artifacts.train_losses[-1]:.4f}")
        print(f"validation_accuracy={artifacts.validation_accuracy:.4f}")
        print(f"labels={artifacts.labels}")
        if args.checkpoint_out is not None:
            save_sequence_classification_checkpoint(
                args.checkpoint_out,
                model=artifacts.model,
                tokenizer=artifacts.tokenizer,
                config=artifacts.config,
                labels=artifacts.labels,
            )
            print(f"checkpoint={args.checkpoint_out}")
        return

    if args.command == "evaluate-classifier":
        model, tokenizer, _, labels = load_sequence_classification_checkpoint(args.checkpoint)
        examples = load_classification_examples(args.dataset)
        label_to_id = {label: index for index, label in enumerate(labels)}
        accuracy = evaluate_sequence_classifier(
            model,
            tokenizer,
            examples,
            label_to_id=label_to_id,
            batch_size=args.batch_size,
        )
        print(f"accuracy={accuracy:.4f}")
