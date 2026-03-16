import pytest
import torch

from bert.attention import MultiHeadSelfAttention, SingleHeadSelfAttention, make_attention_mask


def test_make_attention_mask_expands_padding_mask_for_key_positions() -> None:
    attention_mask = torch.tensor([[1, 1, 0]], dtype=torch.long)

    expanded = make_attention_mask(attention_mask)

    assert expanded.shape == (1, 1, 3)
    assert expanded.tolist() == [[[True, True, False]]]


def test_single_head_self_attention_returns_expected_shape() -> None:
    attention = SingleHeadSelfAttention(hidden_size=4)
    x = torch.randn(2, 3, 4)

    output = attention(x)

    assert output.shape == (2, 3, 4)


def test_single_head_self_attention_padding_mask_zeros_out_masked_key_weights() -> None:
    attention = SingleHeadSelfAttention(hidden_size=4)
    x = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        ]
    )
    attention_mask = torch.tensor([[1, 1, 0]], dtype=torch.long)

    inspected = attention.inspect(x, attention_mask=attention_mask)

    masked_column = inspected["attention_weights"][0, :, 2]
    assert torch.allclose(masked_column, torch.zeros_like(masked_column))


def test_single_head_self_attention_rejects_wrong_hidden_size() -> None:
    attention = SingleHeadSelfAttention(hidden_size=4)
    x = torch.randn(1, 3, 5)

    try:
        attention(x)
    except ValueError as error:
        assert "hidden_size (4)" in str(error)
    else:
        raise AssertionError("expected ValueError for hidden-size mismatch")


def test_multi_head_self_attention_returns_expected_shape() -> None:
    attention = MultiHeadSelfAttention(hidden_size=8, num_heads=2)
    x = torch.randn(2, 3, 8)

    output = attention(x)

    assert output.shape == (2, 3, 8)


def test_multi_head_self_attention_inspect_exposes_split_head_shapes() -> None:
    attention = MultiHeadSelfAttention(hidden_size=8, num_heads=2)
    x = torch.randn(1, 4, 8)

    inspected = attention.inspect(x)

    assert inspected["q"].shape == (1, 2, 4, 4)
    assert inspected["head_outputs"].shape == (1, 2, 4, 4)
    assert inspected["merged"].shape == (1, 4, 8)


def test_multi_head_self_attention_padding_mask_zeros_out_masked_key_weights() -> None:
    attention = MultiHeadSelfAttention(hidden_size=8, num_heads=2)
    x = torch.randn(1, 3, 8)
    attention_mask = torch.tensor([[1, 1, 0]], dtype=torch.long)

    inspected = attention.inspect(x, attention_mask=attention_mask)

    masked_column = inspected["attention_weights"][0, :, :, 2]
    assert torch.allclose(masked_column, torch.zeros_like(masked_column))


def test_multi_head_self_attention_rejects_non_divisible_head_config() -> None:
    with pytest.raises(ValueError, match="divisible by num_heads"):
        MultiHeadSelfAttention(hidden_size=10, num_heads=3)
