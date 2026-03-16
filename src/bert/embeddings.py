"""Embedding stack for BERT inputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class BertEmbeddingConfig:
    vocab_size: int
    hidden_size: int
    max_position_embeddings: int
    type_vocab_size: int = 2
    layer_norm_eps: float = 1e-12
    hidden_dropout_prob: float = 0.1


class BertEmbeddings(nn.Module):
    """Sum token, position, and token type embeddings like the original BERT input stack."""

    def __init__(self, config: BertEmbeddingConfig) -> None:
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size,
            config.hidden_size,
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        token_ids: Tensor,
        token_type_ids: Tensor,
    ) -> Tensor:
        if token_ids.ndim != 2:
            msg = "token_ids must have shape (batch_size, sequence_length)"
            raise ValueError(msg)
        if token_type_ids.shape != token_ids.shape:
            msg = "token_type_ids must have the same shape as token_ids"
            raise ValueError(msg)

        batch_size, sequence_length = token_ids.shape
        if sequence_length > self.config.max_position_embeddings:
            msg = (
                "sequence_length must be <= "
                f"max_position_embeddings ({self.config.max_position_embeddings})"
            )
            raise ValueError(msg)

        position_ids = torch.arange(sequence_length, device=token_ids.device).unsqueeze(0)
        position_ids = position_ids.expand(batch_size, sequence_length)

        word_embeddings = self.word_embeddings(token_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        return self.dropout(embeddings)
