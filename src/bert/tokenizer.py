"""Tokenizer primitives for the BERT study project."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Self

from .basic_tokenizer import BasicTokenizer
from .wordpiece import (
    build_character_vocab,
    compute_pair_scores,
    greedy_segment,
    initialize_wordpieces,
    merge_pair_pieces,
    merge_pair_token,
)


@dataclass(frozen=True)
class BertSpecialTokens:
    pad: str = "[PAD]"
    cls: str = "[CLS]"
    sep: str = "[SEP]"
    mask: str = "[MASK]"
    unk: str = "[UNK]"

    def as_list(self) -> list[str]:
        return [self.pad, self.cls, self.sep, self.mask, self.unk]


@dataclass(frozen=True)
class BertEncoding:
    tokens: list[str]
    token_ids: list[int]
    token_type_ids: list[int]


class WordPieceTokenizer:
    """A small WordPiece tokenizer with BERT-style special tokens."""

    def __init__(
        self,
        vocab: list[str],
        *,
        special_tokens: BertSpecialTokens | None = None,
        lowercase: bool = True,
        unknown_token: str | None = None,
    ) -> None:
        self.special_tokens = special_tokens or BertSpecialTokens()
        self.lowercase = lowercase
        self.basic_tokenizer = BasicTokenizer(lowercase=lowercase)
        self.vocab = vocab
        self.token_to_id = {token: idx for idx, token in enumerate(vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(vocab)}
        self.unknown_token = unknown_token or self.special_tokens.unk

        missing = [
            token for token in self.special_tokens.as_list() if token not in self.token_to_id
        ]
        if missing:
            msg = f"vocab is missing required special tokens: {missing}"
            raise ValueError(msg)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.special_tokens.pad]

    @property
    def cls_id(self) -> int:
        return self.token_to_id[self.special_tokens.cls]

    @property
    def sep_id(self) -> int:
        return self.token_to_id[self.special_tokens.sep]

    @property
    def mask_id(self) -> int:
        return self.token_to_id[self.special_tokens.mask]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.unknown_token]

    @classmethod
    def fit(
        cls,
        texts: list[str],
        *,
        vocab_size: int,
        min_frequency: int = 2,
        lowercase: bool = True,
        special_tokens: BertSpecialTokens | None = None,
    ) -> Self:
        if vocab_size < 5:
            msg = "vocab_size must leave room for the required special tokens"
            raise ValueError(msg)

        resolved_special_tokens = special_tokens or BertSpecialTokens()
        basic_tokenizer = BasicTokenizer(lowercase=lowercase)
        words = [token for text in texts for token in basic_tokenizer.tokenize(text)]
        if not words:
            msg = "texts must contain at least one tokenizable word"
            raise ValueError(msg)

        word_counts = Counter(words)
        vocab = resolved_special_tokens.as_list()

        # Seed the vocabulary with characters so greedy segmentation has a base alphabet.
        char_tokens = build_character_vocab(word_counts)
        for token in char_tokens:
            if token not in vocab:
                vocab.append(token)

        tokenized_words = initialize_wordpieces(word_counts)

        while len(vocab) < vocab_size:
            pair_scores = compute_pair_scores(tokenized_words, word_counts, min_frequency)
            if not pair_scores:
                break

            best_pair = max(pair_scores, key=pair_scores.get)
            merged_token = merge_pair_token(best_pair)
            if merged_token in vocab:
                break

            vocab.append(merged_token)
            tokenized_words = {
                word: merge_pair_pieces(pieces, best_pair)
                for word, pieces in tokenized_words.items()
            }

        return cls(
            vocab=vocab,
            special_tokens=resolved_special_tokens,
            lowercase=lowercase,
        )

    def tokenize(self, text: str) -> list[str]:
        wordpieces: list[str] = []
        for token in self.basic_tokenizer.tokenize(text):
            wordpieces.extend(greedy_segment(token, self.token_to_id, self.unknown_token))
        return wordpieces

    def encode(self, text: str) -> BertEncoding:
        tokens = [self.special_tokens.cls, *self.tokenize(text), self.special_tokens.sep]
        token_ids = [self.token_to_id.get(token, self.unk_id) for token in tokens]
        token_type_ids = [0] * len(tokens)
        return BertEncoding(tokens=tokens, token_ids=token_ids, token_type_ids=token_type_ids)

    def encode_pair(self, text_a: str, text_b: str) -> BertEncoding:
        tokens_a = self.tokenize(text_a)
        tokens_b = self.tokenize(text_b)
        tokens = [
            self.special_tokens.cls,
            *tokens_a,
            self.special_tokens.sep,
            *tokens_b,
            self.special_tokens.sep,
        ]
        token_ids = [self.token_to_id.get(token, self.unk_id) for token in tokens]
        token_type_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
        return BertEncoding(tokens=tokens, token_ids=token_ids, token_type_ids=token_type_ids)

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        tokens = [self.id_to_token[token_id] for token_id in token_ids]
        if skip_special_tokens:
            specials = set(self.special_tokens.as_list())
            tokens = [token for token in tokens if token not in specials]

        words: list[str] = []
        for token in tokens:
            if token.startswith("##") and words:
                words[-1] = words[-1] + token.removeprefix("##")
            else:
                words.append(token)
        return " ".join(words)

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "wordpiece",
            "vocab": self.vocab,
            "lowercase": self.lowercase,
            "special_tokens": {
                "pad": self.special_tokens.pad,
                "cls": self.special_tokens.cls,
                "sep": self.special_tokens.sep,
                "mask": self.special_tokens.mask,
                "unk": self.special_tokens.unk,
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Self:
        special_tokens_payload = payload["special_tokens"]
        if not isinstance(special_tokens_payload, dict):
            msg = "special_tokens must be a dictionary"
            raise TypeError(msg)

        vocab = payload["vocab"]
        if not isinstance(vocab, list):
            msg = "vocab must be a list"
            raise TypeError(msg)

        return cls(
            vocab=[str(token) for token in vocab],
            lowercase=bool(payload.get("lowercase", True)),
            special_tokens=BertSpecialTokens(
                pad=str(special_tokens_payload["pad"]),
                cls=str(special_tokens_payload["cls"]),
                sep=str(special_tokens_payload["sep"]),
                mask=str(special_tokens_payload["mask"]),
                unk=str(special_tokens_payload["unk"]),
            ),
        )
