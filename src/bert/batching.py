"""Batch collation helpers for BERT-style inputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .tokenizer import BertEncoding, WordPieceTokenizer


@dataclass(frozen=True)
class BertBatch:
    token_ids: Tensor
    attention_mask: Tensor
    token_type_ids: Tensor


def collate_encodings(
    encodings: list[BertEncoding],
    *,
    pad_id: int,
) -> BertBatch:
    if not encodings:
        msg = "encodings must not be empty"
        raise ValueError(msg)

    max_length = max(len(encoding.token_ids) for encoding in encodings)
    batch_size = len(encodings)

    token_ids = torch.full((batch_size, max_length), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
    token_type_ids = torch.zeros((batch_size, max_length), dtype=torch.long)

    for row, encoding in enumerate(encodings):
        sequence_length = len(encoding.token_ids)
        if sequence_length != len(encoding.token_type_ids):
            msg = "token_ids and token_type_ids must have the same length"
            raise ValueError(msg)

        token_ids[row, :sequence_length] = torch.tensor(encoding.token_ids, dtype=torch.long)
        # Pads stay at 0 so attention can ignore them later.
        attention_mask[row, :sequence_length] = 1
        token_type_ids[row, :sequence_length] = torch.tensor(
            encoding.token_type_ids,
            dtype=torch.long,
        )

    return BertBatch(
        token_ids=token_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )


def encode_text_batch(
    tokenizer: WordPieceTokenizer,
    texts: list[str],
) -> BertBatch:
    encodings = [tokenizer.encode(text) for text in texts]
    return collate_encodings(encodings, pad_id=tokenizer.pad_id)


def encode_text_pair_batch(
    tokenizer: WordPieceTokenizer,
    text_pairs: list[tuple[str, str]],
) -> BertBatch:
    encodings = [tokenizer.encode_pair(text_a, text_b) for text_a, text_b in text_pairs]
    return collate_encodings(encodings, pad_id=tokenizer.pad_id)
