import pytest
import torch

from bert.embeddings import BertEmbeddingConfig, BertEmbeddings


def test_bert_embeddings_returns_expected_shape() -> None:
    embeddings = BertEmbeddings(
        BertEmbeddingConfig(
            vocab_size=32,
            hidden_size=16,
            max_position_embeddings=12,
            hidden_dropout_prob=0.0,
        )
    )

    token_ids = torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long)
    token_type_ids = torch.tensor([[0, 0, 0], [0, 1, 0]], dtype=torch.long)

    output = embeddings(token_ids, token_type_ids)

    assert output.shape == (2, 3, 16)


def test_bert_embeddings_uses_position_and_token_type_information() -> None:
    embeddings = BertEmbeddings(
        BertEmbeddingConfig(
            vocab_size=16,
            hidden_size=8,
            max_position_embeddings=8,
            hidden_dropout_prob=0.0,
        )
    )

    repeated_tokens = torch.tensor([[3, 3]], dtype=torch.long)
    token_type_ids = torch.tensor([[0, 1]], dtype=torch.long)

    output = embeddings(repeated_tokens, token_type_ids)

    assert not torch.allclose(output[:, 0, :], output[:, 1, :])


def test_bert_embeddings_rejects_shape_mismatch() -> None:
    embeddings = BertEmbeddings(
        BertEmbeddingConfig(
            vocab_size=16,
            hidden_size=8,
            max_position_embeddings=8,
            hidden_dropout_prob=0.0,
        )
    )

    token_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    token_type_ids = torch.tensor([[0, 0]], dtype=torch.long)

    with pytest.raises(ValueError, match="token_type_ids must have the same shape"):
        embeddings(token_ids, token_type_ids)


def test_bert_embeddings_rejects_sequences_longer_than_position_limit() -> None:
    embeddings = BertEmbeddings(
        BertEmbeddingConfig(
            vocab_size=16,
            hidden_size=8,
            max_position_embeddings=2,
            hidden_dropout_prob=0.0,
        )
    )

    token_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    token_type_ids = torch.tensor([[0, 0, 0]], dtype=torch.long)

    with pytest.raises(ValueError, match="sequence_length must be <="):
        embeddings(token_ids, token_type_ids)
