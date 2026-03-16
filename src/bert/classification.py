"""Sequence classification heads built on top of the BERT backbone."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from .model import BertConfig, BertModel


class BertForSequenceClassification(nn.Module):
    """BERT backbone plus a sentence-level classification head."""

    def __init__(self, config: BertConfig, *, num_labels: int) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(
        self,
        token_ids: Tensor,
        token_type_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        _, pooled_output = self.bert(
            token_ids,
            token_type_ids,
            attention_mask=attention_mask,
        )
        logits = self.classifier(self.dropout(pooled_output))

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return logits, loss
