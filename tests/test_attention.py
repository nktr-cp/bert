import torch

from bert.attention import SingleHeadSelfAttention, make_attention_mask


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
