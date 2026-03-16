import torch

from bert.model import BertConfig, BertModel


def test_bert_model_returns_sequence_and_pooled_outputs() -> None:
    model = BertModel(
        BertConfig(
            vocab_size=32,
            hidden_size=8,
            max_position_embeddings=16,
            num_hidden_layers=2,
            num_heads=2,
            intermediate_size=16,
            hidden_dropout_prob=0.0,
        )
    )
    token_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7]], dtype=torch.long)
    token_type_ids = torch.tensor([[0, 0, 0, 0], [0, 0, 1, 1]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.long)

    sequence_output, pooled_output = model(
        token_ids,
        token_type_ids,
        attention_mask=attention_mask,
    )

    assert sequence_output.shape == (2, 4, 8)
    assert pooled_output.shape == (2, 8)


def test_bert_model_accepts_missing_attention_mask() -> None:
    model = BertModel(
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
    token_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    token_type_ids = torch.tensor([[0, 0, 0]], dtype=torch.long)

    sequence_output, pooled_output = model(token_ids, token_type_ids)

    assert sequence_output.shape == (1, 3, 8)
    assert pooled_output.shape == (1, 8)


def test_bert_model_pooled_output_depends_on_cls_position() -> None:
    model = BertModel(
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
    token_ids = torch.tensor([[1, 2, 3], [4, 2, 3]], dtype=torch.long)
    token_type_ids = torch.tensor([[0, 0, 0], [0, 0, 0]], dtype=torch.long)

    _, pooled_output = model(token_ids, token_type_ids)

    assert not torch.allclose(pooled_output[0], pooled_output[1])
