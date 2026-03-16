# BERT

Small, learning-oriented BERT implementation in Python 3.13 with `uv`.

## Setup

```bash
uv sync --dev
```

## Current scope

- uv-based project scaffold
- package entrypoint
- CI for lint, format, and CLI smoke test
- WordPiece tokenizer with BERT special tokens
- batch collation for padding, attention masks, and token type ids
- BERT embedding stack with token, position, and token type embeddings
- single-head bidirectional self-attention with padding-mask support
- multi-head bidirectional self-attention with output projection
- encoder feed-forward, residual connections, and LayerNorm
- BERT backbone with stacked encoder blocks and pooled `[CLS]` output
- masked language modeling corruption and MLM prediction head
- sequence classification head for emotion analysis
- sequence-classification training, evaluation, and checkpointing
