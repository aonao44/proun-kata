"""Tests for the CTC beam search decoder with phonotactic constraints."""
from __future__ import annotations

import torch

from asr.ctc_decode import (
    BeamSearchConfig,
    beam_search_ctc,
    build_token_catalog,
    load_phone_language_model,
    _cluster_penalty,
)


class _FakeTokenizer:
    vocab = ["[PAD]", "p", "b"]
    vocab_size = len(vocab)

    def convert_ids_to_tokens(self, idx: int) -> str:
        return self.vocab[idx]


def test_beam_search_returns_expected_sequence() -> None:
    tokenizer = _FakeTokenizer()
    catalog = build_token_catalog(tokenizer)
    lm = load_phone_language_model(None, {"p", "b"})
    config = BeamSearchConfig(beam_size=4)

    logits = torch.tensor(
        [
            [5.0, 8.0, 1.0],
            [5.0, 2.0, 7.0],
        ],
        dtype=torch.float32,
    )

    result = beam_search_ctc(
        logits,
        blank_id=0,
        catalog=catalog,
        lm=lm,
        config=config,
    )

    assert result.tokens == (1, 2)
    assert result.canonical == ("p", "b")
    assert len(result.alignment) == logits.shape[0]


def test_language_model_fallback_when_missing_file() -> None:
    tokenizer = _FakeTokenizer()
    catalog = build_token_catalog(tokenizer)
    lm = load_phone_language_model("does/not/exist.json", {"p", "b"})

    logits = torch.tensor([[5.0, 7.0, 1.0]], dtype=torch.float32)
    result = beam_search_ctc(
        logits,
        blank_id=0,
        catalog=catalog,
        lm=lm,
        config=BeamSearchConfig(beam_size=2),
    )

    assert result.canonical[0] in {"p", "b"}


def test_cluster_penalty_counts_long_run() -> None:
    sequence = ("s", "t", "ɹ", "s")
    assert _cluster_penalty(sequence) == 2
