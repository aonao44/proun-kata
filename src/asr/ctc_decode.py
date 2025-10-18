"""CTC beam search decoding with lightweight English heuristics."""
from __future__ import annotations

import json
import logging
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch

from core.normalize import ENGLISH_VOWELS, normalize_symbol

LOGGER = logging.getLogger(__name__)

NEG_INF = -1e9
KG_ONSET_BONUS = 0.18
KG_MARGIN = 0.22
FRONT_VOWELS = {"ɪ", "i", "e", "ɛ"}
FLAP_BONUS = 0.20
DD_REPEAT_PENALTY = 0.12
TH_INITIAL_PENALTY = 0.10
DELTA_KEY = "delta_sum"
FRAME_MS = 20.0
FLAP_MAX_MS = 80.0


def _log_add(a: float, b: float) -> float:
    if a <= NEG_INF:
        return b
    if b <= NEG_INF:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


@dataclass(frozen=True)
class TokenCatalog:
    """Precomputed lookup tables for tokenizer ids and canonical phones."""

    id_to_token: List[str]
    id_to_canonical: Dict[int, Tuple[str, ...]]
    canonical_to_id: Dict[str, int]

    def canonical_symbol(self, token_id: int) -> str | None:
        values = self.id_to_canonical.get(token_id)
        if not values:
            return None
        return values[0]


def build_token_catalog(tokenizer) -> TokenCatalog:  # type: ignore[no-untyped-def]
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None:
        vocab_size = len(tokenizer)
    id_to_token: List[str] = []
    id_to_canonical: Dict[int, Tuple[str, ...]] = {}
    canonical_to_id: Dict[str, int] = {}

    for idx in range(vocab_size):
        token = tokenizer.convert_ids_to_tokens(idx)
        if token is None:
            token = ""
        id_to_token.append(str(token))
        canonical = tuple(normalize_symbol(str(token), log_unknown=False))
        id_to_canonical[idx] = canonical
        for symbol in canonical:
            canonical_to_id.setdefault(symbol, idx)
    return TokenCatalog(id_to_token=id_to_token, id_to_canonical=id_to_canonical, canonical_to_id=canonical_to_id)


class PhoneLanguageModel:
    """Thin wrapper around a bigram log-probability table."""

    def __init__(self, transitions: Mapping[str, Mapping[str, float]], default_log_prob: float = -5.0) -> None:
        self._transitions: Dict[str, Dict[str, float]] = {
            str(prev): {str(curr): float(score) for curr, score in mapping.items()}
            for prev, mapping in transitions.items()
        }
        self._default_log_prob = float(default_log_prob)

    def transition_log_prob(self, prev_symbol: str | None, next_symbol: str) -> float:
        previous = prev_symbol or "<s>"
        table = self._transitions.get(previous)
        if table is None:
            return self._default_log_prob
        return table.get(next_symbol, self._default_log_prob)


def load_phone_language_model(
    path: str | Path | None,
    known_symbols: Iterable[str],
    *,
    default_log_prob: float = -5.0,
) -> PhoneLanguageModel:
    target_path = Path(path) if path else None
    if target_path and target_path.exists():
        try:
            data = json.loads(target_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return PhoneLanguageModel(data, default_log_prob=default_log_prob)
        except json.JSONDecodeError:
            pass
    return PhoneLanguageModel(_build_uniform_transitions(known_symbols), default_log_prob=default_log_prob)


def _build_uniform_transitions(symbols: Iterable[str]) -> Dict[str, Dict[str, float]]:
    phones = sorted({symbol for symbol in symbols if symbol})
    if not phones:
        return {"<s>": {}}
    log_prob = -math.log(len(phones))
    transitions: Dict[str, Dict[str, float]] = {"<s>": {symbol: log_prob for symbol in phones}}
    for symbol in phones:
        transitions[symbol] = {next_symbol: log_prob for next_symbol in phones}
    return transitions


@dataclass(frozen=True)
class BeamSearchConfig:
    """Scoring parameters for the miniature beam search decoder."""

    beam_size: int = 12
    vowel_bonus: float = 0.15
    repeat_penalty: float = 0.2
    final_cons_penalty: float = 0.3
    th_bonus: float = 0.0
    th_margin: float = 0.2
    use_voicing: bool = False
    voicing_bonus: float = 0.2
    voicing_threshold: float = 0.4
    length_norm: bool = True
    lm_weight: float = 0.0
    insertion_penalty: float = 0.0
    candidate_top_k: int = 0


@dataclass
class BeamEntry:
    log_p_blank: float = NEG_INF
    log_p_non_blank: float = NEG_INF
    lm_score: float = 0.0
    canonical: Tuple[str, ...] = ()
    durations: Tuple[int, ...] = ()


@dataclass(frozen=True)
class BeamSearchResult:
    tokens: Tuple[int, ...]
    canonical: Tuple[str, ...]
    alignment: List[int]
    score: float


@dataclass(frozen=True)
class Candidate:
    token_id: int
    canonical_symbol: str | None
    log_prob: float
    heuristics: Tuple[str, ...] = ()


VOICING_FLIP = {
    "p": "b",
    "t": "d",
    "k": "ɡ",
    "f": "v",
    "s": "z",
    "θ": "ð",
}

VOICED_PHONES = {
    "b",
    "d",
    "ɡ",
    "v",
    "z",
    "ʒ",
    "ð",
    "m",
    "n",
    "ŋ",
    "l",
    "ɹ",
    "w",
    "j",
    "ɾ",
    "ʔ",
} | {
    "i",
    "iː",
    "ɪ",
    "e",
    "eɪ",
    "ɛ",
    "æ",
    "ɑ",
    "ɑː",
    "ɚ",
    "ə",
    "o",
    "oʊ",
    "ɔ",
    "ʊ",
    "u",
    "uː",
    "aɪ",
    "aʊ",
    "ɔɪ",
    "ʌ",
}

TH_COMPETITORS = {
    "θ": ("f", "s"),
    "ð": ("d", "z"),
}

SILENCE_SYMBOLS = {"SIL", "SP", "NSN"}
FINAL_CONSONANTS = {"p", "b", "t", "d", "k", "ɡ"}


def beam_search_ctc(
    logits: torch.Tensor,
    *,
    blank_id: int,
    catalog: TokenCatalog,
    lm: PhoneLanguageModel,
    config: BeamSearchConfig,
) -> BeamSearchResult:
    if logits.ndim != 2:
        raise ValueError("logits must be a 2-D tensor (time, vocab)")

    log_probs = torch.log_softmax(logits, dim=-1)
    log_probs_np = log_probs.detach().cpu().numpy()
    time_steps, vocab_size = log_probs_np.shape

    beam: Dict[Tuple[int, ...], BeamEntry] = {
        (): BeamEntry(log_p_blank=0.0, log_p_non_blank=NEG_INF, lm_score=0.0, canonical=(), durations=())
    }
    stats: Counter[str] = Counter()
    accum: Dict[str, float] = {DELTA_KEY: 0.0, "delta_count": 0.0}
    voicing_stats: Counter[str] = Counter()

    top_k = config.candidate_top_k
    if top_k <= 0:
        top_k = max(min(vocab_size, config.beam_size * 4), 32)

    for t in range(time_steps):
        row = log_probs_np[t]
        next_beam: Dict[Tuple[int, ...], BeamEntry] = {}

        top_indices = np.argpartition(row, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(row[top_indices])[::-1]]
        if top_indices.size >= 2:
            accum[DELTA_KEY] += float(row[top_indices[0]] - row[top_indices[1]])
            accum["delta_count"] += 1.0

        for prefix, entry in beam.items():
            total_prob = _log_add(entry.log_p_blank, entry.log_p_non_blank)

            blank_log = row[blank_id]
            blank_entry = next_beam.setdefault(
                prefix,
                BeamEntry(
                    log_p_blank=NEG_INF,
                    log_p_non_blank=NEG_INF,
                    lm_score=entry.lm_score,
                    canonical=entry.canonical,
                    durations=entry.durations,
                ),
            )
            blank_entry.log_p_blank = _log_add(blank_entry.log_p_blank, total_prob + blank_log)

            last_token_id = prefix[-1] if prefix else None
            previous_symbol = entry.canonical[-1] if entry.canonical else None

            for token_id in top_indices:
                if token_id == blank_id:
                    continue
                candidates = _candidate_symbols(
                    token_id,
                    row,
                    entry.canonical,
                    entry.durations,
                    previous_symbol,
                    catalog,
                    config,
                    voicing_stats,
                )
                for candidate in candidates:
                    if candidate.log_prob <= NEG_INF:
                        continue
                    _register_heuristics(stats, candidate.heuristics)
                    if candidate.token_id == last_token_id:
                        if prefix in next_beam:
                            same_entry = next_beam[prefix]
                        else:
                            same_entry = BeamEntry(
                                log_p_blank=NEG_INF,
                                log_p_non_blank=NEG_INF,
                                lm_score=entry.lm_score,
                                canonical=entry.canonical,
                                durations=entry.durations,
                            )
                            next_beam[prefix] = same_entry
                        if same_entry.durations:
                            updated = list(same_entry.durations)
                            updated[-1] += 1
                            if same_entry.canonical and same_entry.canonical[-1] == "ɾ":
                                threshold_frames = int(FLAP_MAX_MS / FRAME_MS)
                                if updated[-1] == threshold_frames:
                                    same_entry.log_p_non_blank -= FLAP_BONUS
                            same_entry.durations = tuple(updated)
                        same_entry.log_p_non_blank = _log_add(
                            same_entry.log_p_non_blank,
                            entry.log_p_non_blank + candidate.log_prob,
                        )
                        continue

                    new_prefix = prefix + (candidate.token_id,)
                    new_canonical = entry.canonical
                    new_lm_score = entry.lm_score
                    new_durations = entry.durations
                    if candidate.canonical_symbol:
                        new_canonical = entry.canonical + (candidate.canonical_symbol,)
                        new_durations = entry.durations + (1,)
                        if config.lm_weight:
                            prev_for_lm = previous_symbol or "<s>"
                            new_lm_score += lm.transition_log_prob(prev_for_lm, candidate.canonical_symbol)
                    if new_prefix in next_beam:
                        new_entry = next_beam[new_prefix]
                    else:
                        new_entry = BeamEntry(
                            log_p_blank=NEG_INF,
                            log_p_non_blank=NEG_INF,
                            lm_score=new_lm_score,
                            canonical=new_canonical,
                            durations=new_durations,
                        )
                        next_beam[new_prefix] = new_entry
                    new_entry.log_p_non_blank = _log_add(
                        new_entry.log_p_non_blank,
                        total_prob + candidate.log_prob,
                    )

        beam = _prune(next_beam, config)

    best_prefix, best_entry, best_score = _best_entry(beam, config)
    alignment = _ctc_viterbi_alignment(log_probs_np, list(best_prefix), blank_id)

    if best_entry.canonical and best_entry.canonical[-1] in FINAL_CONSONANTS:
        stats["final_penalty"] += 1

    if LOGGER.isEnabledFor(logging.DEBUG):
        summary = (
            "bonus: vowel=%d repeat=%d final=%d th=%d flap=%d kg_onset=%d voicing=%d"
            % (
                stats.get("vowel_bonus", 0),
                stats.get("repeat_penalty", 0),
                stats.get("final_penalty", 0),
                stats.get("th_bonus", 0),
                stats.get("flap_bonus", 0),
                stats.get("kg_onset", 0),
                stats.get("voicing_flip", 0),
            )
        )
        delta_avg = 0.0
        if accum["delta_count"]:
            delta_avg = accum[DELTA_KEY] / accum["delta_count"]
        if voicing_stats:
            detail = ", ".join(f"{k}={v}" for k, v in sorted(voicing_stats.items()))
        else:
            detail = "none"
        LOGGER.debug(
            "dec_beam=%d %s delta=%.3f voicing=%s",
            config.beam_size,
            summary,
            delta_avg,
            detail,
        )

    return BeamSearchResult(
        tokens=best_prefix,
        canonical=best_entry.canonical,
        alignment=alignment,
        score=best_score,
    )


def _entry_score(entry: BeamEntry, config: BeamSearchConfig) -> float:
    ctc_score = _log_add(entry.log_p_blank, entry.log_p_non_blank)
    length = len(entry.canonical)
    total = ctc_score + (config.lm_weight * entry.lm_score) - (config.insertion_penalty * length)
    if length > 0 and entry.canonical[-1] in FINAL_CONSONANTS:
        total -= config.final_cons_penalty
    if config.length_norm and length > 0:
        total /= length
    return total


def _prune(beam: MutableMapping[Tuple[int, ...], BeamEntry], config: BeamSearchConfig) -> Dict[Tuple[int, ...], BeamEntry]:
    scored: List[Tuple[float, Tuple[int, ...], BeamEntry]] = [
        (_entry_score(entry, config), prefix, entry)
        for prefix, entry in beam.items()
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    top = scored[: max(1, config.beam_size)]
    return {prefix: entry for _, prefix, entry in top}


def _best_entry(
    beam: Mapping[Tuple[int, ...], BeamEntry],
    config: BeamSearchConfig,
) -> Tuple[Tuple[int, ...], BeamEntry, float]:
    best_prefix: Tuple[int, ...] | None = None
    best_entry: BeamEntry | None = None
    best_score = -math.inf
    for prefix, entry in beam.items():
        score = _entry_score(entry, config)
        if best_prefix is None or score > best_score:
            best_prefix = prefix
            best_entry = entry
            best_score = score
    assert best_prefix is not None and best_entry is not None
    return best_prefix, best_entry, best_score


def _candidate_symbols(
    token_id: int,
    row: Sequence[float],
    history: Tuple[str, ...],
    durations: Tuple[int, ...],
    previous_symbol: str | None,
    catalog: TokenCatalog,
    config: BeamSearchConfig,
    voicing_stats: Counter[str],
) -> List[Candidate]:
    base_log = row[token_id]
    if base_log <= NEG_INF:
        return []
    canonical = catalog.id_to_canonical.get(token_id, ())
    canonical_symbol = canonical[0] if canonical else None
    adjusted_log = base_log
    heuristics: List[str] = []
    prospective_durations = durations + (1,) if canonical_symbol else durations

    if canonical_symbol in TH_COMPETITORS:
        competitor_logs = [
            row[catalog.canonical_to_id[alt]]
            for alt in TH_COMPETITORS[canonical_symbol]
            if alt in catalog.canonical_to_id
        ]
        if competitor_logs and config.th_bonus > 0.0:
            best_alt = max(competitor_logs)
            if best_alt - base_log <= config.th_margin:
                adjusted_log += config.th_bonus
                heuristics.append("th_bonus")
        if (not history) or history[-1] in SILENCE_SYMBOLS:
            adjusted_log -= TH_INITIAL_PENALTY

    if canonical_symbol and canonical_symbol in ENGLISH_VOWELS:
        if _needs_vowel_bonus(history):
            adjusted_log += config.vowel_bonus
            heuristics.append("vowel_bonus")

    if (
        canonical_symbol
        and canonical_symbol == previous_symbol
        and canonical_symbol not in ENGLISH_VOWELS
        and canonical_symbol not in SILENCE_SYMBOLS
    ):
        adjusted_log -= config.repeat_penalty
        heuristics.append("repeat_penalty")

    if canonical_symbol == "ɡ" and _eligible_for_kg_onset(history):
        k_id = catalog.canonical_to_id.get("k")
        if k_id is not None:
            diff = abs(row[k_id] - base_log)
        else:
            diff = None
        front_scores = [
            row[catalog.canonical_to_id[v]]
            for v in FRONT_VOWELS
            if v in catalog.canonical_to_id
        ]
        if (
            diff is not None
            and diff <= KG_MARGIN
            and front_scores
            and max(front_scores) > base_log - 0.2
        ):
            adjusted_log += KG_ONSET_BONUS
            heuristics.append("kg_onset")

    if canonical_symbol == "ɾ" and _eligible_for_flap(history, prospective_durations):
        front_scores = [
            row[catalog.canonical_to_id[v]]
            for v in FRONT_VOWELS
            if v in catalog.canonical_to_id
        ]
        if front_scores and max(front_scores) > base_log - 0.25:
            adjusted_log += FLAP_BONUS
            heuristics.append("flap_bonus")

    if canonical_symbol == "d" and previous_symbol == "d":
        adjusted_log -= DD_REPEAT_PENALTY
        heuristics.append("flap_penalty")

    candidates: List[Candidate] = [Candidate(token_id, canonical_symbol, adjusted_log, tuple(heuristics))]

    if (
        config.use_voicing
        and canonical_symbol in VOICING_FLIP
        and previous_symbol in VOICED_PHONES
    ):
        prob = math.exp(base_log)
        if prob < config.voicing_threshold:
            voiced_symbol = VOICING_FLIP[canonical_symbol]
            voiced_id = catalog.canonical_to_id.get(voiced_symbol)
            if voiced_id is not None:
                voiced_canonical = catalog.id_to_canonical.get(voiced_id, ())
                voiced_symbol_actual = voiced_canonical[0] if voiced_canonical else voiced_symbol
                voiced_log = row[voiced_id] + config.voicing_bonus
                voicing_stats[f"{canonical_symbol}->{voiced_symbol_actual}"] += 1
                candidates.append(
                    Candidate(
                        token_id=voiced_id,
                        canonical_symbol=voiced_symbol_actual,
                        log_prob=voiced_log,
                        heuristics=("voicing_flip",),
                    )
                )

    return candidates


def _needs_vowel_bonus(history: Tuple[str, ...]) -> bool:
    seen = 0
    for symbol in reversed(history):
        if symbol in SILENCE_SYMBOLS:
            continue
        seen += 1
        if symbol in ENGLISH_VOWELS:
            return False
        if seen >= 3:
            return True
    return seen > 0


def _eligible_for_kg_onset(history: Tuple[str, ...]) -> bool:
    if not history:
        return True
    last = history[-1]
    if last in SILENCE_SYMBOLS:
        return True
    if last in ENGLISH_VOWELS or last == "s":
        return False
    return True


def _eligible_for_flap(history: Tuple[str, ...], durations: Tuple[int, ...]) -> bool:
    if len(history) < 1:
        return False
    prev = history[-1]
    if prev in SILENCE_SYMBOLS or prev not in ENGLISH_VOWELS:
        return False
    if len(history) >= 2 and history[-2] in SILENCE_SYMBOLS:
        return False
    if len(durations) < 1:
        return False
    last_duration_ms = durations[-1] * FRAME_MS
    if last_duration_ms >= FLAP_MAX_MS:
        return False
    return True


def _register_heuristics(stats: Counter[str], heuristics: Tuple[str, ...]) -> None:
    for name in heuristics:
        stats[name] += 1


def _ctc_viterbi_alignment(
    log_probs: np.ndarray,
    tokens: List[int],
    blank_id: int,
) -> List[int]:
    time_steps, _ = log_probs.shape
    if not tokens:
        return [blank_id] * time_steps

    extended: List[int] = [blank_id]
    for token in tokens:
        extended.append(token)
        extended.append(blank_id)
    state_count = len(extended)

    dp = np.full((time_steps, state_count), NEG_INF, dtype=float)
    back = np.zeros((time_steps, state_count), dtype=np.int32)

    dp[0, 0] = log_probs[0, blank_id]
    if state_count > 1:
        dp[0, 1] = log_probs[0, extended[1]]

    for t in range(1, time_steps):
        for s in range(state_count):
            log_prob = log_probs[t, extended[s]]
            best_prev = dp[t - 1, s]
            best_state = s
            if s > 0 and dp[t - 1, s - 1] > best_prev:
                best_prev = dp[t - 1, s - 1]
                best_state = s - 1
            if s > 1 and extended[s] != extended[s - 2] and dp[t - 1, s - 2] > best_prev:
                best_prev = dp[t - 1, s - 2]
                best_state = s - 2
            dp[t, s] = best_prev + log_prob
            back[t, s] = best_state

    end_state = int(np.argmax(dp[-1]))
    alignment_states = [0] * time_steps
    state = end_state
    for t in range(time_steps - 1, -1, -1):
        alignment_states[t] = state
        if t > 0:
            state = int(back[t, state])
    return [extended[state] for state in alignment_states]


__all__ = [
    "BeamSearchConfig",
    "BeamSearchResult",
    "PhoneLanguageModel",
    "TokenCatalog",
    "beam_search_ctc",
    "build_token_catalog",
    "load_phone_language_model",
]
