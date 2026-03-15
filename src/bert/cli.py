"""Project entrypoint for manual experiments."""

from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path

from .tokenizer import WordPieceTokenizer


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
