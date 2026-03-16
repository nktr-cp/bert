import torch

from bert.mlm import IGNORE_INDEX, BertForMaskedLM, create_mlm_inputs
from bert.model import BertConfig


def test_create_mlm_inputs_masks_only_non_special_tokens() -> None:
    torch.manual_seed(0)
    token_ids = torch.tensor([[1, 5, 6, 2, 0]], dtype=torch.long)

    batch = create_mlm_inputs(
        token_ids,
        mask_token_id=3,
        special_token_ids={0, 1, 2, 3},
        vocab_size=10,
        mlm_probability=1.0,
        mask_replace_prob=1.0,
        random_replace_prob=0.0,
    )

    assert batch.token_ids.tolist() == [[1, 3, 3, 2, 0]]
    assert batch.labels.tolist() == [[IGNORE_INDEX, 5, 6, IGNORE_INDEX, IGNORE_INDEX]]
    assert batch.prediction_mask.tolist() == [[False, True, True, False, False]]


def test_create_mlm_inputs_can_leave_masked_token_unchanged() -> None:
    torch.manual_seed(0)
    token_ids = torch.tensor([[1, 5, 6, 2]], dtype=torch.long)

    batch = create_mlm_inputs(
        token_ids,
        mask_token_id=3,
        special_token_ids={1, 2, 3},
        vocab_size=10,
        mlm_probability=1.0,
        mask_replace_prob=0.0,
        random_replace_prob=0.0,
    )

    assert batch.token_ids.tolist() == [[1, 5, 6, 2]]
    assert batch.labels.tolist() == [[IGNORE_INDEX, 5, 6, IGNORE_INDEX]]


def test_bert_for_masked_lm_returns_logits_and_loss() -> None:
    model = BertForMaskedLM(
        BertConfig(
            vocab_size=16,
            hidden_size=8,
            max_position_embeddings=8,
            num_hidden_layers=1,
            num_heads=2,
            intermediate_size=16,
            hidden_dropout_prob=0.0,
        )
    )
    token_ids = torch.tensor([[1, 3, 6, 2]], dtype=torch.long)
    token_type_ids = torch.tensor([[0, 0, 0, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)
    labels = torch.tensor([[IGNORE_INDEX, 5, IGNORE_INDEX, IGNORE_INDEX]], dtype=torch.long)

    logits, loss = model(
        token_ids,
        token_type_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    assert logits.shape == (1, 4, 16)
    assert loss is not None
    assert loss.ndim == 0
