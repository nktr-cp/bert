"""Masked language modeling utilities and prediction head."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .model import BertConfig, BertModel

IGNORE_INDEX = -100


@dataclass(frozen=True)
class MaskedLanguageModelingBatch:
    token_ids: Tensor
    labels: Tensor
    prediction_mask: Tensor


def create_mlm_inputs(
    token_ids: Tensor,
    *,
    mask_token_id: int,
    special_token_ids: set[int],
    vocab_size: int,
    mlm_probability: float = 0.15,
    mask_replace_prob: float = 0.8,
    random_replace_prob: float = 0.1,
) -> MaskedLanguageModelingBatch:
    if token_ids.ndim != 2:
        msg = "token_ids must have shape (batch_size, sequence_length)"
        raise ValueError(msg)

    if not 0.0 <= mlm_probability <= 1.0:
        msg = "mlm_probability must be between 0 and 1"
        raise ValueError(msg)

    if mask_replace_prob + random_replace_prob > 1.0:
        msg = "mask_replace_prob + random_replace_prob must be <= 1"
        raise ValueError(msg)

    candidate_mask = torch.ones_like(token_ids, dtype=torch.bool)
    for token_id in special_token_ids:
        candidate_mask &= token_ids != token_id

    random_draws = torch.rand_like(token_ids, dtype=torch.float)
    prediction_mask = candidate_mask & (random_draws < mlm_probability)

    labels = token_ids.clone()
    labels[~prediction_mask] = IGNORE_INDEX

    masked_token_ids = token_ids.clone()

    replace_draws = torch.rand_like(token_ids, dtype=torch.float)
    replace_with_mask = prediction_mask & (replace_draws < mask_replace_prob)
    replace_with_random = prediction_mask & (
        (replace_draws >= mask_replace_prob)
        & (replace_draws < mask_replace_prob + random_replace_prob)
    )

    masked_token_ids[replace_with_mask] = mask_token_id

    random_tokens = torch.randint(
        low=0,
        high=vocab_size,
        size=token_ids.shape,
        device=token_ids.device,
    )
    masked_token_ids[replace_with_random] = random_tokens[replace_with_random]

    return MaskedLanguageModelingBatch(
        token_ids=masked_token_ids,
        labels=labels,
        prediction_mask=prediction_mask,
    )


class BertOnlyMLMHead(nn.Module):
    """Prediction head used to map encoder states back into token logits."""

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states: Tensor, embedding_weight: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return F.linear(hidden_states, embedding_weight, self.bias)


class BertForMaskedLM(nn.Module):
    """BERT backbone plus a masked language modeling head."""

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.bert = BertModel(config)
        self.mlm_head = BertOnlyMLMHead(config)

    def forward(
        self,
        token_ids: Tensor,
        token_type_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        sequence_output, _ = self.bert(
            token_ids,
            token_type_ids,
            attention_mask=attention_mask,
        )
        logits = self.mlm_head(
            sequence_output,
            self.bert.embeddings.word_embeddings.weight,
        )

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=IGNORE_INDEX,
            )

        return logits, loss
