"""Transformer block components for the BERT study project."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from .attention import MultiHeadSelfAttention


@dataclass(frozen=True)
class BertBlockConfig:
    hidden_size: int
    intermediate_size: int
    num_heads: int
    hidden_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12


class FeedForward(nn.Module):
    """A position-wise feed-forward network matching the BERT encoder layout."""

    def __init__(self, config: BertBlockConfig) -> None:
        super().__init__()
        self.in_proj = nn.Linear(config.hidden_size, config.intermediate_size)
        self.activation = nn.GELU()
        self.out_proj = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_proj(x)
        x = self.activation(x)
        x = self.out_proj(x)
        return self.dropout(x)


class TransformerEncoderBlock(nn.Module):
    """A pre-norm encoder block with self-attention and feed-forward sublayers."""

    def __init__(self, config: BertBlockConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = MultiHeadSelfAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
        )
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.feed_forward_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = FeedForward(config)

    def forward(
        self,
        x: Tensor,
        *,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        attention_out = self.attention(
            self.attention_norm(x),
            attention_mask=attention_mask,
        )
        x = x + self.attention_dropout(attention_out)
        x = x + self.feed_forward(self.feed_forward_norm(x))
        return x
