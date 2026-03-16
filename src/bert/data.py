"""Dataset helpers for sentence-level classification experiments."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ClassificationExample:
    text: str
    label: str


def load_classification_examples(path: Path) -> list[ClassificationExample]:
    examples: list[ClassificationExample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        text = payload.get("text")
        label = payload.get("label")
        if not isinstance(text, str) or not isinstance(label, str):
            msg = "each JSONL row must contain string fields 'text' and 'label'"
            raise ValueError(msg)
        examples.append(ClassificationExample(text=text, label=label))

    if not examples:
        msg = f"no classification examples found in {path}"
        raise ValueError(msg)
    return examples


def build_label_vocabulary(
    examples: list[ClassificationExample],
) -> tuple[dict[str, int], list[str]]:
    labels = sorted({example.label for example in examples})
    label_to_id = {label: index for index, label in enumerate(labels)}
    return label_to_id, labels


def split_examples(
    examples: list[ClassificationExample],
    *,
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[list[ClassificationExample], list[ClassificationExample]]:
    if not 0.0 < validation_fraction < 1.0:
        msg = "validation_fraction must be between 0 and 1"
        raise ValueError(msg)

    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    split_index = max(1, int(len(shuffled) * (1.0 - validation_fraction)))
    split_index = min(split_index, len(shuffled) - 1)
    return shuffled[:split_index], shuffled[split_index:]
