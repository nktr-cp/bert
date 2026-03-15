"""Core WordPiece training and segmentation helpers."""

from __future__ import annotations

from collections import Counter


def build_character_vocab(words: Counter[str]) -> list[str]:
    return sorted(
        {char if index == 0 else f"##{char}" for word in words for index, char in enumerate(word)}
    )


def initialize_wordpieces(words: Counter[str]) -> dict[str, list[str]]:
    return {
        word: [char if index == 0 else f"##{char}" for index, char in enumerate(word)]
        for word in words
    }


def compute_pair_scores(
    tokenized_words: dict[str, list[str]],
    word_counts: Counter[str],
    min_frequency: int,
) -> dict[tuple[str, str], float]:
    pair_counts: Counter[tuple[str, str]] = Counter()
    piece_counts: Counter[str] = Counter()

    for word, pieces in tokenized_words.items():
        frequency = word_counts[word]
        for piece in pieces:
            piece_counts[piece] += frequency
        for left, right in zip(pieces, pieces[1:], strict=False):
            pair_counts[(left, right)] += frequency

    scores: dict[tuple[str, str], float] = {}
    for pair, pair_frequency in pair_counts.items():
        if pair_frequency < min_frequency:
            continue
        left, right = pair
        # WordPiece chooses merges by normalized association rather than raw pair count.
        scores[pair] = pair_frequency / (piece_counts[left] * piece_counts[right])
    return scores


def merge_pair_token(pair: tuple[str, str]) -> str:
    left, right = pair
    return left + right.removeprefix("##")


def merge_pair_pieces(pieces: list[str], pair: tuple[str, str]) -> list[str]:
    merged: list[str] = []
    index = 0
    merged_token = merge_pair_token(pair)

    while index < len(pieces):
        if index < len(pieces) - 1 and (pieces[index], pieces[index + 1]) == pair:
            merged.append(merged_token)
            index += 2
            continue
        merged.append(pieces[index])
        index += 1
    return merged


def greedy_segment(word: str, token_to_id: dict[str, int], unknown_token: str) -> list[str]:
    if not word:
        return []

    pieces: list[str] = []
    start = 0
    while start < len(word):
        end = len(word)
        current_piece: str | None = None

        # Greedy longest-match-first is the core WordPiece decoding rule.
        while start < end:
            candidate = word[start:end]
            if start > 0:
                candidate = f"##{candidate}"
            if candidate in token_to_id:
                current_piece = candidate
                break
            end -= 1

        if current_piece is None:
            return [unknown_token]

        pieces.append(current_piece)
        start = end

    return pieces
