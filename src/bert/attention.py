"""Attention primitives for the BERT study project."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def make_attention_mask(attention_mask: Tensor) -> Tensor:
    """Expand a padding mask so each query position shares the same visible keys."""
    if attention_mask.ndim != 2:
        msg = "attention_mask must have shape (batch_size, sequence_length)"
        raise ValueError(msg)
    return attention_mask.unsqueeze(1).to(dtype=torch.bool)


def _scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    scale: float,
    attention_mask: Tensor | None = None,
) -> dict[str, Tensor]:
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    masked_scores = scores
    if attention_mask is not None:
        masked_scores = scores.masked_fill(~attention_mask, float("-inf"))

    attention_weights = torch.softmax(masked_scores, dim=-1)
    output = torch.matmul(attention_weights, v)

    return {
        "scores": scores,
        "masked_scores": masked_scores,
        "attention_weights": attention_weights,
        "output": output,
    }


class SingleHeadSelfAttention(nn.Module):
    """A single bidirectional self-attention head with padding-mask support."""

    def __init__(self, *, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        x: Tensor,
        *,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        return self._compute_attention(x, attention_mask=attention_mask)["output"]

    def _compute_attention(
        self,
        x: Tensor,
        *,
        attention_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if x.ndim != 3:
            msg = "x must have shape (batch_size, sequence_length, hidden_size)"
            raise ValueError(msg)
        if x.size(-1) != self.hidden_size:
            msg = f"last dimension must be hidden_size ({self.hidden_size})"
            raise ValueError(msg)

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        expanded_mask: Tensor | None = None
        if attention_mask is not None:
            expanded_mask = make_attention_mask(attention_mask)
        attention = _scaled_dot_product_attention(
            q,
            k,
            v,
            scale=1.0 / math.sqrt(self.hidden_size),
            attention_mask=expanded_mask,
        )

        result = {
            "q": q,
            "k": k,
            "v": v,
            "scores": attention["scores"],
            "masked_scores": attention["masked_scores"],
            "attention_weights": attention["attention_weights"],
            "output": attention["output"],
        }
        if expanded_mask is not None:
            result["attention_mask"] = expanded_mask
        return result

    def inspect(
        self,
        x: Tensor,
        *,
        attention_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        return self._compute_attention(x, attention_mask=attention_mask)


class MultiHeadSelfAttention(nn.Module):
    """A minimal multi-head bidirectional self-attention module."""

    def __init__(self, *, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            msg = "hidden_size must be divisible by num_heads"
            raise ValueError(msg)

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def _split_heads(self, x: Tensor) -> Tensor:
        batch_size, sequence_length, _ = x.shape
        x = x.view(batch_size, sequence_length, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        batch_size, _, sequence_length, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, sequence_length, self.hidden_size)

    def forward(
        self,
        x: Tensor,
        *,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        return self._compute_attention(x, attention_mask=attention_mask)["output"]

    def _compute_attention(
        self,
        x: Tensor,
        *,
        attention_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if x.ndim != 3:
            msg = "x must have shape (batch_size, sequence_length, hidden_size)"
            raise ValueError(msg)
        if x.size(-1) != self.hidden_size:
            msg = f"last dimension must be hidden_size ({self.hidden_size})"
            raise ValueError(msg)

        q = self._split_heads(self.query(x))
        k = self._split_heads(self.key(x))
        v = self._split_heads(self.value(x))

        expanded_mask: Tensor | None = None
        if attention_mask is not None:
            expanded_mask = make_attention_mask(attention_mask).unsqueeze(1)

        attention = _scaled_dot_product_attention(
            q,
            k,
            v,
            scale=1.0 / math.sqrt(self.head_dim),
            attention_mask=expanded_mask,
        )
        merged = self._merge_heads(attention["output"])
        output = self.proj(merged)

        result = {
            "q": q,
            "k": k,
            "v": v,
            "scores": attention["scores"],
            "masked_scores": attention["masked_scores"],
            "attention_weights": attention["attention_weights"],
            "head_outputs": attention["output"],
            "merged": merged,
            "output": output,
        }
        if expanded_mask is not None:
            result["attention_mask"] = expanded_mask
        return result

    def inspect(
        self,
        x: Tensor,
        *,
        attention_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        return self._compute_attention(x, attention_mask=attention_mask)
