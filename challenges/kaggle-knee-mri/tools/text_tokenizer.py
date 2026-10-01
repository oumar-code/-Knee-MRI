#!/usr/bin/env python3
"""Utilities for building a simple text vocabulary and tokenizing reports.

This lightweight tokenizer is designed to work with the Knee MRI challenge
without needing a heavyweight NLP dependency stack. It fits a vocabulary on
all report texts and converts each report into integer token IDs.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List, Sequence


class SimpleTokenizer:
    """Whitespace/punctuation-aware tokenizer for clinical reports."""

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    CLS_TOKEN = "<cls>"
    SEP_TOKEN = "<sep>"

    def __init__(self, vocab_size: int = 5000, min_freq: int = 1):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = []

    @staticmethod
    def normalize_text(text: str) -> str:
        if text is None:
            return ""
        text = str(text).lower()
        text = text.replace("/", " ")
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def fit(self, texts: Iterable[str]) -> "SimpleTokenizer":
        counts = Counter()
        for text in texts:
            for token in self.normalize_text(text).split():
                if token:
                    counts[token] += 1

        specials = [self.PAD_TOKEN, self.UNK_TOKEN, self.CLS_TOKEN, self.SEP_TOKEN]
        ordered = [w for w, c in counts.most_common() if c >= self.min_freq]
        vocab = specials + ordered[: max(0, self.vocab_size - len(specials))]

        self.word2idx = {token: idx for idx, token in enumerate(vocab)}
        self.idx2word = vocab
        return self

    def encode(self, text: str, max_len: int | None = None) -> List[int]:
        normalized = self.normalize_text(text)
        tokens = [self.CLS_TOKEN] + normalized.split() if normalized else [self.CLS_TOKEN]
        ids = []
        for token in tokens:
            ids.append(self.word2idx.get(token, self.word2idx.get(self.UNK_TOKEN, 1)))
        if max_len is not None:
            if len(ids) < max_len:
                ids = ids + [self.word2idx.get(self.PAD_TOKEN, 0)] * (max_len - len(ids))
            else:
                ids = ids[:max_len]
        return ids

    def encode_many(self, texts: Sequence[str], max_len: int | None = None) -> List[List[int]]:
        return [self.encode(text, max_len=max_len) for text in texts]

    def __len__(self) -> int:
        return len(self.idx2word)
