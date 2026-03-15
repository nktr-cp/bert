"""Basic tokenization before WordPiece segmentation."""

from __future__ import annotations

import re


class BasicTokenizer:
    """A compact pre-tokenizer before WordPiece segmentation."""

    _token_pattern = re.compile(r"[A-Za-z0-9]+|[^\w\s]", re.UNICODE)

    def __init__(self, *, lowercase: bool = True) -> None:
        self.lowercase = lowercase

    def tokenize(self, text: str) -> list[str]:
        normalized = text.strip()
        if self.lowercase:
            normalized = normalized.lower()
        return self._token_pattern.findall(normalized)
