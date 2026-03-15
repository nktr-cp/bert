from bert.tokenizer import BasicTokenizer, WordPieceTokenizer


def test_basic_tokenizer_splits_words_and_punctuation() -> None:
    tokenizer = BasicTokenizer()

    assert tokenizer.tokenize("Hello, BERT!") == ["hello", ",", "bert", "!"]


def test_wordpiece_encode_wraps_sequence_with_special_tokens() -> None:
    tokenizer = WordPieceTokenizer.fit(
        ["I love BERT tokenizers", "BERT loves special tokens"],
        vocab_size=32,
        min_frequency=1,
    )

    encoding = tokenizer.encode("I love BERT")

    assert encoding.tokens[0] == "[CLS]"
    assert encoding.tokens[-1] == "[SEP]"
    assert len(encoding.tokens) == len(encoding.token_ids) == len(encoding.token_type_ids)
    assert set(encoding.token_type_ids) == {0}


def test_wordpiece_fit_keeps_required_special_tokens_and_vocab_budget() -> None:
    tokenizer = WordPieceTokenizer.fit(
        ["playing played player", "playing playful player"],
        vocab_size=18,
        min_frequency=1,
    )

    assert tokenizer.vocab[:5] == ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]
    assert tokenizer.vocab_size <= 18


def test_wordpiece_fit_learns_merge_that_avoids_character_by_character_tokenization() -> None:
    tokenizer = WordPieceTokenizer.fit(
        ["playing playing played player", "playing player"],
        vocab_size=24,
        min_frequency=1,
    )

    tokens = tokenizer.tokenize("playing")

    assert tokens != ["[UNK]"]
    assert len(tokens) < len("playing")
    assert any(len(token.removeprefix("##")) > 1 for token in tokens)


def test_wordpiece_encode_pair_assigns_token_type_ids() -> None:
    tokenizer = WordPieceTokenizer.fit(
        ["happy model", "sad model"],
        vocab_size=24,
        min_frequency=1,
    )

    encoding = tokenizer.encode_pair("happy", "sad")
    first_sep_index = encoding.tokens.index("[SEP]")

    assert encoding.tokens == ["[CLS]", "happy", "[SEP]", "sad", "[SEP]"]
    assert encoding.token_type_ids[: first_sep_index + 1] == [0, 0, 0]
    assert encoding.token_type_ids[first_sep_index + 1 :] == [1, 1]


def test_wordpiece_decode_joins_continuation_pieces() -> None:
    tokenizer = WordPieceTokenizer(
        vocab=["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]", "play", "##ing"],
    )

    decoded = tokenizer.decode([tokenizer.cls_id, 5, 6, tokenizer.sep_id])

    assert decoded == "playing"


def test_wordpiece_round_trips_through_serialization() -> None:
    tokenizer = WordPieceTokenizer.fit(
        ["emotion analysis with bert", "bert likes compact unit tests"],
        vocab_size=30,
        min_frequency=1,
    )

    restored = WordPieceTokenizer.from_dict(tokenizer.to_dict())

    assert restored.vocab == tokenizer.vocab
    assert restored.tokenize("emotion analysis") == tokenizer.tokenize("emotion analysis")
