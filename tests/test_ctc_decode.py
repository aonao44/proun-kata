"""Tests for the CTC beam search decoder with heuristic scoring."""
from __future__ import annotations

import torch

from asr.ctc_decode import (
    BeamSearchConfig,
    beam_search_ctc,
    build_token_catalog,
    load_phone_language_model,
)


class _FakeTokenizer:
    def __init__(self, vocab: list[str]):
        self.vocab = vocab
        self.vocab_size = len(vocab)

    def convert_ids_to_tokens(self, idx: int) -> str:
        return self.vocab[idx]


def test_beam_search_returns_expected_sequence() -> None:
    tokenizer = _FakeTokenizer(["[PAD]", "p", "b"])
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
    tokenizer = _FakeTokenizer(["[PAD]", "p", "b"])
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


def test_th_bonus_prefers_th_over_nearby_f() -> None:
    tokenizer = _FakeTokenizer(["[PAD]", "θ", "f"])
    catalog = build_token_catalog(tokenizer)
    lm = load_phone_language_model(None, {"θ", "f"})
    config = BeamSearchConfig(beam_size=3, th_bonus=0.5, th_margin=0.3)

    logits = torch.tensor(
        [
            [0.0, 2.0, 2.1],
            [2.5, 0.5, 0.1],
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

    assert result.canonical == ("θ",)


def test_vowel_bonus_encourages_vowel_after_consonant_run() -> None:
    tokenizer = _FakeTokenizer(["[PAD]", "s", "ə"])
    catalog = build_token_catalog(tokenizer)
    lm = load_phone_language_model(None, {"s", "ə"})
    config = BeamSearchConfig(
        beam_size=3,
        vowel_bonus=0.6,
        repeat_penalty=0.0,
        final_cons_penalty=0.0,
        th_bonus=0.0,
    )

    logits = torch.tensor(
        [
            [0.0, 3.0, 1.0],
            [0.0, 2.2, 2.0],
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

    assert result.canonical == ("s", "ə")


def test_kg_onset_bonus_prefers_g_over_k_with_front_vowel() -> None:
    tokenizer = _FakeTokenizer(["[PAD]", "K", "G", "IH"])
    catalog = build_token_catalog(tokenizer)
    lm = load_phone_language_model(None, {"k", "ɡ", "ɪ"})
    config = BeamSearchConfig(
        beam_size=4,
        vowel_bonus=0.0,
        repeat_penalty=0.0,
        final_cons_penalty=0.0,
        th_bonus=0.0,
        use_voicing=False,
    )

    logits = torch.tensor(
        [
            [0.0, 5.0, 4.9, 4.8],
            [0.0, 0.5, 0.1, 5.5],
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

    assert result.canonical[0] == "ɡ"


def test_flap_bonus_prefers_flap_between_vowels() -> None:
    tokenizer = _FakeTokenizer(["[PAD]", "AA", "DX", "D", "IH"])
    catalog = build_token_catalog(tokenizer)
    lm = load_phone_language_model(None, {"ɑ", "ɾ", "d", "ɪ"})
    config = BeamSearchConfig(
        beam_size=4,
        vowel_bonus=0.0,
        repeat_penalty=0.0,
        final_cons_penalty=0.0,
        th_bonus=0.0,
        use_voicing=False,
    )

    logits = torch.tensor(
        [
            [0.0, 6.0, 0.5, 0.4, 0.3],
            [0.0, 0.3, 4.6, 4.5, 4.0],
            [0.0, 0.2, 0.1, 0.1, 6.0],
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

    assert "ɾ" in result.canonical
