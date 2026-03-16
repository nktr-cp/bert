import torch

from bert.classification import BertForSequenceClassification
from bert.model import BertConfig


def test_bert_for_sequence_classification_returns_logits_and_loss() -> None:
    model = BertForSequenceClassification(
        BertConfig(
            vocab_size=16,
            hidden_size=8,
            max_position_embeddings=8,
            num_hidden_layers=1,
            num_heads=2,
            intermediate_size=16,
            hidden_dropout_prob=0.0,
        ),
        num_labels=4,
    )
    token_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    token_type_ids = torch.tensor([[0, 0, 0], [0, 1, 1]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.long)
    labels = torch.tensor([2, 1], dtype=torch.long)

    logits, loss = model(
        token_ids,
        token_type_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    assert logits.shape == (2, 4)
    assert loss is not None
    assert loss.ndim == 0


def test_bert_for_sequence_classification_accepts_missing_labels() -> None:
    model = BertForSequenceClassification(
        BertConfig(
            vocab_size=16,
            hidden_size=8,
            max_position_embeddings=8,
            num_hidden_layers=1,
            num_heads=2,
            intermediate_size=16,
            hidden_dropout_prob=0.0,
        ),
        num_labels=3,
    )
    token_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    token_type_ids = torch.tensor([[0, 0, 0]], dtype=torch.long)

    logits, loss = model(token_ids, token_type_ids)

    assert logits.shape == (1, 3)
    assert loss is None


def test_bert_for_sequence_classification_uses_pooled_cls_representation() -> None:
    model = BertForSequenceClassification(
        BertConfig(
            vocab_size=16,
            hidden_size=8,
            max_position_embeddings=8,
            num_hidden_layers=1,
            num_heads=2,
            intermediate_size=16,
            hidden_dropout_prob=0.0,
        ),
        num_labels=2,
    )
    token_ids = torch.tensor([[1, 2, 3], [4, 2, 3]], dtype=torch.long)
    token_type_ids = torch.tensor([[0, 0, 0], [0, 0, 0]], dtype=torch.long)

    logits, _ = model(token_ids, token_type_ids)

    assert not torch.allclose(logits[0], logits[1])
