"""BERT backbone composed from embeddings and encoder blocks."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from .blocks import BertBlockConfig, TransformerEncoderBlock
from .embeddings import BertEmbeddingConfig, BertEmbeddings


@dataclass(frozen=True)
class BertConfig:
    vocab_size: int
    hidden_size: int
    max_position_embeddings: int
    num_hidden_layers: int
    num_heads: int
    intermediate_size: int
    type_vocab_size: int = 2
    hidden_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12

    def embedding_config(self) -> BertEmbeddingConfig:
        return BertEmbeddingConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            layer_norm_eps=self.layer_norm_eps,
            hidden_dropout_prob=self.hidden_dropout_prob,
        )

    def block_config(self) -> BertBlockConfig:
        return BertBlockConfig(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_heads=self.num_heads,
            hidden_dropout_prob=self.hidden_dropout_prob,
            layer_norm_eps=self.layer_norm_eps,
        )


class BertModel(nn.Module):
    """A minimal BERT backbone returning sequence and pooled representations."""

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config.embedding_config())
        block_config = config.block_config()
        self.layers = nn.ModuleList(
            TransformerEncoderBlock(block_config) for _ in range(config.num_hidden_layers)
        )
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = nn.Tanh()

    def forward(
        self,
        token_ids: Tensor,
        token_type_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        x = self.embeddings(token_ids, token_type_ids)
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)

        cls_state = x[:, 0, :]
        pooled = self.pooler_activation(self.pooler(cls_state))
        return x, pooled
