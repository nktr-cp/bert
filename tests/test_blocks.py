import torch

from bert.blocks import BertBlockConfig, FeedForward, TransformerEncoderBlock


def test_feed_forward_returns_expected_shape() -> None:
    feed_forward = FeedForward(
        BertBlockConfig(
            hidden_size=8,
            intermediate_size=16,
            num_heads=2,
            hidden_dropout_prob=0.0,
        )
    )
    x = torch.randn(2, 3, 8)

    output = feed_forward(x)

    assert output.shape == (2, 3, 8)


def test_transformer_encoder_block_returns_expected_shape() -> None:
    block = TransformerEncoderBlock(
        BertBlockConfig(
            hidden_size=8,
            intermediate_size=16,
            num_heads=2,
            hidden_dropout_prob=0.0,
        )
    )
    x = torch.randn(2, 4, 8)
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.long)

    output = block(x, attention_mask=attention_mask)

    assert output.shape == (2, 4, 8)


def test_transformer_encoder_block_propagates_attention_mask_to_attention_module() -> None:
    block = TransformerEncoderBlock(
        BertBlockConfig(
            hidden_size=8,
            intermediate_size=16,
            num_heads=2,
            hidden_dropout_prob=0.0,
        )
    )
    x = torch.randn(1, 3, 8)
    attention_mask = torch.tensor([[1, 1, 0]], dtype=torch.long)

    inspected = block.attention.inspect(
        block.attention_norm(x),
        attention_mask=attention_mask,
    )

    masked_column = inspected["attention_weights"][0, :, :, 2]
    assert torch.allclose(masked_column, torch.zeros_like(masked_column))
