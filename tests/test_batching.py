import pytest

from bert.batching import BertBatch, collate_encodings, encode_text_batch, encode_text_pair_batch
from bert.tokenizer import BertEncoding, WordPieceTokenizer


def test_collate_encodings_pads_to_max_length_and_marks_attention() -> None:
    batch = collate_encodings(
        [
            BertEncoding(
                tokens=["[CLS]", "hello", "[SEP]"],
                token_ids=[1, 11, 2],
                token_type_ids=[0, 0, 0],
            ),
            BertEncoding(
                tokens=["[CLS]", "hello", "world", "[SEP]"],
                token_ids=[1, 11, 12, 2],
                token_type_ids=[0, 0, 0, 0],
            ),
        ],
        pad_id=0,
    )

    assert isinstance(batch, BertBatch)
    assert batch.token_ids.tolist() == [[1, 11, 2, 0], [1, 11, 12, 2]]
    assert batch.attention_mask.tolist() == [[1, 1, 1, 0], [1, 1, 1, 1]]
    assert batch.token_type_ids.tolist() == [[0, 0, 0, 0], [0, 0, 0, 0]]


def test_collate_encodings_preserves_token_type_ids_for_sentence_pairs() -> None:
    batch = collate_encodings(
        [
            BertEncoding(
                tokens=["[CLS]", "happy", "[SEP]", "sad", "[SEP]"],
                token_ids=[1, 21, 2, 22, 2],
                token_type_ids=[0, 0, 0, 1, 1],
            ),
            BertEncoding(
                tokens=["[CLS]", "joy", "[SEP]", "anger", "now", "[SEP]"],
                token_ids=[1, 31, 2, 32, 33, 2],
                token_type_ids=[0, 0, 0, 1, 1, 1],
            ),
        ],
        pad_id=0,
    )

    assert batch.token_type_ids.tolist() == [[0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 1]]
    assert batch.attention_mask.tolist() == [[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]]


def test_encode_text_batch_uses_tokenizer_pad_id() -> None:
    tokenizer = WordPieceTokenizer(
        vocab=[
            "[PAD]",
            "[CLS]",
            "[SEP]",
            "[MASK]",
            "[UNK]",
            "i",
            "love",
            "bert",
        ],
    )

    batch = encode_text_batch(tokenizer, ["I love BERT", "BERT"])

    assert batch.token_ids.shape == (2, 5)
    assert batch.token_ids[1, -1].item() == tokenizer.pad_id
    assert batch.attention_mask.tolist() == [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]


def test_encode_text_pair_batch_collates_sentence_pairs() -> None:
    tokenizer = WordPieceTokenizer(
        vocab=[
            "[PAD]",
            "[CLS]",
            "[SEP]",
            "[MASK]",
            "[UNK]",
            "happy",
            "very",
            "sad",
        ],
    )

    batch = encode_text_pair_batch(
        tokenizer,
        [("happy", "sad"), ("very happy", "sad")],
    )

    assert batch.token_ids.shape == (2, 6)
    assert batch.attention_mask.tolist() == [[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]]
    assert batch.token_type_ids.tolist() == [[0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 1, 1]]


def test_collate_encodings_rejects_empty_batches() -> None:
    with pytest.raises(ValueError, match="encodings must not be empty"):
        collate_encodings([], pad_id=0)
