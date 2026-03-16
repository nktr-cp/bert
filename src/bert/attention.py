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

        scale = 1.0 / math.sqrt(self.hidden_size)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        expanded_mask: Tensor | None = None
        masked_scores = scores
        if attention_mask is not None:
            expanded_mask = make_attention_mask(attention_mask)
            masked_scores = scores.masked_fill(~expanded_mask, float("-inf"))

        attention_weights = torch.softmax(masked_scores, dim=-1)
        output = torch.matmul(attention_weights, v)

        result = {
            "q": q,
            "k": k,
            "v": v,
            "scores": scores,
            "masked_scores": masked_scores,
            "attention_weights": attention_weights,
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
