"""ARPAbet列をスタイル可変のカタカナ列へ変換する。"""
from __future__ import annotations

import os
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field

from core.normalize import canonical_to_arpabet

SILENCE_LABELS = {"<SP>", "SP", "SIL", "NSN"}
VOWELS = {
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "EH",
    "ER",
    "EY",
    "IH",
    "IY",
    "OW",
    "OY",
    "UH",
    "UW",
}
DIPHTHONGS = {"AW", "AY", "EY", "OW", "OY"}
SEMI_VOWELS = {"Y", "W"}
FLAP_TARGETS = {"T", "D"}
VOICELESS_STOPS = {"K", "T", "P"}
VOICE_STOPS = {"B", "D", "G"}

CANONICAL_CONSONANT_KANA = {
    "p": "プ",
    "b": "ブ",
    "t": "ト",
    "d": "ド",
    "k": "ク",
    "ɡ": "グ",
    "g": "グ",
    "m": "ム",
    "n": "ン",
    "ŋ": "ング",
    "f": "フ",
    "v": "ヴ",
    "θ": "ス",
    "ð": "ズ",
    "s": "ス",
    "z": "ズ",
    "ʃ": "シ",
    "ʒ": "ジ",
    "tʃ": "チ",
    "dʒ": "ジ",
    "h": "ハ",
    "j": "イ",
    "w": "ウ",
    "ɹ": "ル",
    "r": "ル",
    "l": "ル",
    "ɾ": "ラ",
    "y": "イ",
}


READABLE_CV_FUSIONS = {
    ("ト", "ェイ"): "テイ",
    ("プ", "オー"): "ポー",
    ("ド", "ォウ"): "ドウ",
    ("グ", "ァ"): "ガ",
    ("ニ", "イー"): "ニー",
}

CV_TOKEN_FUSIONS = {
    ("ク", "ア"): "カ",
    ("ク", "ァ"): "カ",
    ("ク", "アー"): "カー",
    ("ク", "イ"): "キ",
    ("ク", "ィ"): "キ",
    ("ク", "イー"): "キー",
    ("グ", "ア"): "ガ",
    ("グ", "ァ"): "ガ",
    ("グ", "アー"): "ガー",
    ("グ", "イ"): "ギ",
    ("グ", "ィ"): "ギ",
    ("グ", "イー"): "ギー",
    ("ス", "ア"): "サ",
    ("ス", "ァ"): "サ",
    ("ス", "アー"): "サー",
    ("ス", "イ"): "シ",
    ("ス", "ィ"): "シ",
    ("ス", "イー"): "シー",
    ("ズ", "ア"): "ザ",
    ("ズ", "ァ"): "ザ",
    ("ズ", "アー"): "ザー",
    ("ズ", "イ"): "ジ",
    ("ズ", "ィ"): "ジ",
    ("ズ", "イー"): "ジー",
    ("ト", "ア"): "タ",
    ("ト", "ァ"): "タ",
    ("ト", "アー"): "ター",
    ("ト", "イ"): "ティ",
    ("ト", "ィ"): "ティ",
    ("ト", "イー"): "ティー",
    ("ド", "ア"): "ダ",
    ("ド", "ァ"): "ダ",
    ("ド", "アー"): "ダー",
    ("ド", "イ"): "ディ",
    ("ド", "ィ"): "ディ",
    ("ド", "イー"): "ディー",
    ("プ", "ア"): "パ",
    ("プ", "ァ"): "パ",
    ("プ", "アー"): "パー",
    ("プ", "イ"): "ピ",
    ("プ", "ィ"): "ピ",
    ("プ", "イー"): "ピー",
    ("ブ", "ア"): "バ",
    ("ブ", "ァ"): "バ",
    ("ブ", "アー"): "バー",
    ("ブ", "イ"): "ビ",
    ("ブ", "ィ"): "ビ",
    ("ブ", "イー"): "ビー",
    ("フ", "ア"): "ファ",
    ("フ", "ァ"): "ファ",
    ("フ", "アー"): "ファー",
    ("フ", "イ"): "フィ",
    ("フ", "ィ"): "フィ",
    ("フ", "イー"): "フィー",
    ("ヴ", "ア"): "ヴァ",
    ("ヴ", "ァ"): "ヴァ",
    ("ヴ", "アー"): "ヴァー",
    ("ヴ", "イ"): "ヴィ",
    ("ヴ", "ィ"): "ヴィ",
    ("ヴ", "イー"): "ヴィー",
}

IPA_VOWEL_PATTERNS = (
    "a",
    "æ",
    "ɑ",
    "e",
    "ɛ",
    "i",
    "ɪ",
    "o",
    "ɔ",
    "u",
    "ʊ",
    "ʌ",
    "ə",
    "ɚ",
    "ɜ",
    "ɒ",
    "ɐ",
    "y",
    "ø",
    "ɯ",
    "ɤ",
)

AUTO_LONG_VOWEL_MS = 140.0


@dataclass(frozen=True)
class KanaConversionOptions:
    """パラメータ付きの変換設定。"""

    reading_style: str = "balanced"
    long_vowel_level: int = 1
    sokuon_level: int = 1
    r_coloring: bool = False
    th_style: str = "su"
    final_c_handling: str = "xtsu"
    auto_long_vowel_ms: float = AUTO_LONG_VOWEL_MS

    def clamp(self) -> KanaConversionOptions:
        long_vowel_ms = self.auto_long_vowel_ms
        try:
            long_vowel_ms = float(long_vowel_ms)
        except (TypeError, ValueError):  # noqa: PERF203
            long_vowel_ms = AUTO_LONG_VOWEL_MS
        long_vowel_ms = max(0.0, long_vowel_ms)
        return KanaConversionOptions(
            reading_style=(
                self.reading_style
                if self.reading_style in {"raw", "balanced", "natural"}
                else "balanced"
            ),
            long_vowel_level=_clamp(self.long_vowel_level, 0, 2),
            sokuon_level=_clamp(self.sokuon_level, 0, 2),
            r_coloring=bool(self.r_coloring),
            th_style=self.th_style if self.th_style in {"su", "zu"} else "su",
            final_c_handling=(
                self.final_c_handling
                if self.final_c_handling in {"xtsu", "tsu"}
                else "xtsu"
            ),
            auto_long_vowel_ms=long_vowel_ms,
        )


@dataclass(frozen=True)
class KanaConversionResult:
    """変換結果。"""

    tokens: list[str]
    text: str
    strict_text: str = ""
    readable_text: str = ""
    ops: list[dict[str, str | int]] = field(default_factory=list)


@dataclass(frozen=True)
class _PhoneContext:
    symbol: str
    canonical_symbol: str
    duration: float | None
    confidence: float | None
    index: int
    prev_symbol: str | None
    next_symbol: str | None
    prev_content: str | None
    next_content: str | None
    prev_canonical: str | None
    next_canonical: str | None
    is_pause: bool
    is_first_in_segment: bool
    is_last_in_segment: bool


def _coerce_phone(entry: object) -> tuple[str, float | None, float | None]:
    if isinstance(entry, str):
        return entry, None, None
    if isinstance(entry, dict):
        symbol = entry.get("sym") or entry.get("p") or entry.get("symbol") or ""
        start = entry.get("start")
        end = entry.get("end")
        duration = None
        try:
            if start is not None and end is not None:
                duration = float(end) - float(start)
                if duration < 0:
                    duration = None
        except (TypeError, ValueError):  # noqa: BLE001
            duration = None
        confidence = entry.get("conf")
        try:
            confidence = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):  # noqa: BLE001
            confidence = None
        return str(symbol), duration, confidence
    if isinstance(entry, tuple) and entry:
        symbol = str(entry[0])
        duration = None
        if len(entry) > 1 and entry[1] is not None:
            try:
                duration = float(entry[1])
            except (TypeError, ValueError):  # noqa: PERF203
                duration = None
        confidence = None
        if len(entry) > 2 and entry[2] is not None:
            try:
                confidence = float(entry[2])
            except (TypeError, ValueError):  # noqa: PERF203
                confidence = None
        return symbol, duration, confidence
    symbol = getattr(entry, "symbol", None)
    if symbol is not None:
        start = getattr(entry, "start", None)
        end = getattr(entry, "end", None)
        duration = None
        try:
            if start is not None and end is not None:
                duration = float(end) - float(start)
                if duration < 0:
                    duration = None
        except (TypeError, ValueError):  # noqa: BLE001
            duration = None
        confidence = getattr(entry, "confidence", None)
        try:
            confidence = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):  # noqa: BLE001
            confidence = None
        return str(symbol), duration, confidence
    return str(entry), None, None


def _looks_like_vowel(symbol: str) -> bool:
    upper = symbol.upper()
    if upper in VOWELS:
        return True
    if any(ch in upper for ch in {"A", "E", "I", "O", "U"}):
        return True
    lowered = symbol.lower()
    return any(lowered.startswith(pattern) for pattern in IPA_VOWEL_PATTERNS)


def to_kana_sequence(
    phones: Iterable[object],
    *,
    options: KanaConversionOptions,
) -> KanaConversionResult:
    """音素配列をカタカナ配列と結合文字列へ変換する。"""

    symbols_with_props = [_coerce_phone(entry) for entry in phones]
    canonical_list = [symbol for symbol, _, _ in symbols_with_props]
    phone_list = [canonical_to_arpabet(symbol).upper() for symbol in canonical_list]
    durations = [duration for _, duration, _ in symbols_with_props]
    confidences = [confidence for _, _, confidence in symbols_with_props]
    if not phone_list:
        return KanaConversionResult(tokens=[], text="", strict_text="", readable_text="", ops=[])

    opts = options.clamp()
    disabled_rules = _parse_disabled_rules()
    contexts = list(_build_context(phone_list, canonical_list, durations, confidences))
    tokens = [_map_symbol(ctx, opts) for ctx in contexts]
    strict_tokens = tokens[:]
    processed_tokens, ops = _postprocess_with_ops(tokens, contexts, opts, disabled_rules)
    kana_text_readable = _build_text(processed_tokens, contexts)
    kana_text_strict = _build_text(strict_tokens, contexts)
    return KanaConversionResult(
        tokens=processed_tokens,
        text=kana_text_readable,
        strict_text=kana_text_strict,
        readable_text=kana_text_readable,
        ops=ops,
    )


def _build_context(
    phones: Sequence[str],
    canonical: Sequence[str],
    durations: Sequence[float | None],
    confidences: Sequence[float | None],
) -> Iterator[_PhoneContext]:
    prev_content = None
    prev_canonical = None
    next_content_cache, next_canonical_cache = _next_content_cache(phones, canonical)
    segment_start = True

    for idx, symbol in enumerate(phones):
        is_pause = symbol in SILENCE_LABELS
        next_symbol = phones[idx + 1] if idx + 1 < len(phones) else None
        next_content = next_content_cache[idx]
        canonical_symbol = canonical[idx] if idx < len(canonical) else symbol
        next_canonical = next_canonical_cache[idx]
        is_last_in_segment = False
        if not is_pause:
            look_ahead = idx + 1
            is_last_in_segment = True
            while look_ahead < len(phones):
                candidate = phones[look_ahead]
                if candidate in SILENCE_LABELS:
                    break
                if candidate not in SILENCE_LABELS:
                    is_last_in_segment = False
                    break
                look_ahead += 1

        yield _PhoneContext(
            symbol=symbol,
            canonical_symbol=canonical_symbol,
            duration=durations[idx] if idx < len(durations) else None,
            confidence=confidences[idx] if idx < len(confidences) else None,
            index=idx,
            prev_symbol=phones[idx - 1] if idx > 0 else None,
            next_symbol=next_symbol,
            prev_content=prev_content,
            next_content=next_content,
            prev_canonical=prev_canonical,
            next_canonical=next_canonical,
            is_pause=is_pause,
            is_first_in_segment=segment_start and not is_pause,
            is_last_in_segment=is_last_in_segment,
        )

        if is_pause:
            segment_start = True
        else:
            segment_start = False
            prev_content = symbol
            prev_canonical = canonical_symbol


def _next_content_cache(
    phones: Sequence[str],
    canonical: Sequence[str],
) -> tuple[list[str | None], list[str | None]]:
    next_content: list[str | None] = [None] * len(phones)
    next_canonical: list[str | None] = [None] * len(phones)
    ahead_symbol: str | None = None
    ahead_canonical: str | None = None
    for idx in range(len(phones) - 1, -1, -1):
        symbol = phones[idx]
        next_content[idx] = ahead_symbol
        next_canonical[idx] = ahead_canonical
        if symbol not in SILENCE_LABELS:
            ahead_symbol = symbol
            ahead_canonical = canonical[idx] if idx < len(canonical) else None
    return next_content, next_canonical


def _map_symbol(context: _PhoneContext, options: KanaConversionOptions) -> str:
    symbol = context.symbol
    if context.is_pause:
        return ""
    if symbol in SILENCE_LABELS:
        return ""
    if symbol == "TH":
        return "ス" if options.th_style == "su" else "ズ"
    if symbol == "DH":
        return "ズ"
    if symbol == "ZH":
        return "ジ"
    if symbol == "JH":
        return "ジ"
    if symbol == "CH":
        return "チ"
    if symbol == "HH":
        return "ハ"

    if symbol == "ER":
        return _map_er(context, options)
    if symbol in DIPHTHONGS:
        return _map_diphthong(symbol, context, options)
    if symbol in VOWELS:
        return _map_vowel(symbol, context, options)
    if symbol in FLAP_TARGETS and _is_flap(context):
        return _map_flap(context)
    if symbol == "R":
        return _map_r(context, options)

    if symbol in VOICELESS_STOPS | VOICE_STOPS:
        return _map_stop(symbol, context, options)

    mapped = _BASE_MAPPING.get(symbol)
    if mapped is not None:
        return mapped

    canonical = context.canonical_symbol or ""
    direct = CANONICAL_CONSONANT_KANA.get(canonical)
    if direct is None:
        direct = CANONICAL_CONSONANT_KANA.get(canonical.lower())
    if direct is not None:
        return direct

    if _looks_like_vowel(canonical) or _looks_like_vowel(symbol):
        return "ア"
    return "ッ"


def _map_vowel(symbol: str, context: _PhoneContext, options: KanaConversionOptions) -> str:
    base = _VOWEL_BASE.get(symbol, "")
    if not base:
        return "ア" if _looks_like_vowel(symbol) else symbol

    duration_ms = (context.duration or 0.0) * 1000.0
    auto_long = (
        options.long_vowel_level >= 1
        and duration_ms >= options.auto_long_vowel_ms
    )
    effective_last = context.is_last_in_segment or (
        context.next_symbol == "R" and not options.r_coloring
    )
    canonical = context.canonical_symbol

    if canonical in {"iː"} and auto_long and options.long_vowel_level >= 1:
        return "イー"
    if canonical in {"u", "uː"} and auto_long and options.long_vowel_level >= 1:
        return "ウー"

    if symbol == "IY":
        if effective_last and options.long_vowel_level >= 1:
            return "イー"
        if auto_long and options.long_vowel_level >= 1:
            return "イー"
        if options.long_vowel_level == 0:
            return "イ"
    if symbol == "UW":
        if effective_last and options.long_vowel_level >= 1:
            return "ウー"
        if auto_long and options.long_vowel_level >= 1:
            return "ウー"
        if options.long_vowel_level == 0:
            return "ウ"
    if symbol == "AA" and effective_last and options.long_vowel_level >= 1:
        return "アー"
    if symbol == "AO" and effective_last and options.long_vowel_level >= 1:
        return "オー"
    if symbol == "AH" and effective_last and options.long_vowel_level >= 2:
        return "アー"

    return base


def _map_er(context: _PhoneContext, options: KanaConversionOptions) -> str:
    if context.is_last_in_segment:
        return "アー" if options.long_vowel_level >= 1 else "ア"
    if options.r_coloring:
        return "アル"
    return "ア"


def _map_diphthong(symbol: str, context: _PhoneContext, options: KanaConversionOptions) -> str:
    final = context.is_last_in_segment
    level = options.long_vowel_level
    if symbol == "OW":
        if final and level >= 1:
            return "オー"
        if level == 2 and not final:
            return "オー"
        return "オウ"
    if symbol == "EY":
        if final and level >= 1:
            return "エー"
        if level == 2 and not final:
            return "エー"
        return "ェイ"
    if symbol == "AY":
        if not final and not context.is_first_in_segment:
            return "ァイ"
        return "アイ"
    if symbol == "OY":
        return "オイ"
    if symbol == "AW":
        return "アウ"
    return _VOWEL_BASE.get(symbol, symbol)


def _map_r(context: _PhoneContext, options: KanaConversionOptions) -> str:
    next_content = context.next_content
    prev_content = context.prev_content

    if prev_content and prev_content in VOWELS:
        return "ル" if options.r_coloring else ""

    if next_content and next_content in VOWELS:
        if next_content in {"IY", "IH", "EY"}:
            return "リ"
        if next_content in {"AA", "AE", "AH", "AY"}:
            return "ラ"
        if next_content in {"AO", "OW", "UH", "UW", "OY"}:
            return "ル"
    return "ル" if options.r_coloring else "ル"


def _map_flap(context: _PhoneContext) -> str:
    next_content = context.next_content
    if next_content in {"IY", "IH", "EY"}:
        return "リ"
    return "ラ"


def _map_stop(symbol: str, context: _PhoneContext, options: KanaConversionOptions) -> str:
    is_coda = context.is_last_in_segment or (context.next_content is None)
    if not is_coda:
        return _BASE_MAPPING.get(symbol, symbol)

    if symbol in VOICELESS_STOPS:
        base = _BASE_MAPPING.get(symbol, "")
        if options.sokuon_level == 0:
            return base or ("ツ" if options.final_c_handling == "tsu" else base)
        if options.sokuon_level >= 1 or options.final_c_handling == "xtsu":
            return f"ッ{base[:1]}" if base else "ッ"
        if options.final_c_handling == "tsu":
            return base or "ツ"
    if symbol in VOICE_STOPS:
        if options.sokuon_level >= 2:
            return "ッ"
        return _BASE_MAPPING.get(symbol, symbol)

    return _BASE_MAPPING.get(symbol, symbol)


def _is_flap(context: _PhoneContext) -> bool:
    if context.symbol == "DX":
        return True
    return context.canonical_symbol == "ɾ"


def _postprocess(
    tokens: list[str],
    contexts: Sequence[_PhoneContext],
    options: KanaConversionOptions,
    disabled_rules: set[str],
) -> tuple[list[str], list[dict[str, str | int]]]:
    result = tokens[:]
    ops: list[dict[str, str | int]] = []
    _adjust_leading_n(result, contexts, options)
    for idx in range(len(result) - 1):
        current_ctx = contexts[idx]
        next_ctx = contexts[idx + 1]
        following = result[idx + 1]
        if current_ctx.symbol == "M" and next_ctx.symbol == "EY":
            result[idx] = "メ"
            result[idx + 1] = "イ" if following else "イ"
        if current_ctx.symbol == "K" and next_ctx.symbol in {"IH", "IY"}:
            result[idx] = "キ" if next_ctx.symbol == "IH" else "キー"
            result[idx + 1] = ""
        if current_ctx.symbol == "G" and next_ctx.symbol == "EH":
            result[idx] = "ゲ"
            result[idx + 1] = ""
        if (
            current_ctx.symbol == "D"
            and current_ctx.canonical_symbol != "ɾ"
            and next_ctx.symbol in {"IH", "IY"}
        ):
            long_target = (
                next_ctx.canonical_symbol in {"iː"}
                and options.long_vowel_level >= 1
            )
            result[idx] = "ディー" if (next_ctx.symbol == "IY" and long_target) else "ディ"
            result[idx + 1] = ""
        if current_ctx.symbol == "TH" and next_ctx.symbol == "AE":
            result[idx] = "サ"
            result[idx + 1] = "ァ" if not next_ctx.is_last_in_segment else "ァー"
        if current_ctx.symbol == "NG" and next_ctx.symbol == "K":
            result[idx] = "ング"
            if result[idx + 1] == "ク":
                result[idx + 1] = ""
        if current_ctx.symbol == "K" and next_ctx.symbol == "Y":
            result[idx] = "キ"
            result[idx + 1] = "" if next_ctx.next_content in {"UW", "UH"} else result[idx + 1]
        if current_ctx.symbol == "Y" and next_ctx.symbol in {"UW", "UH"}:
            result[idx] = "ュ"
            result[idx + 1] = "ウー" if options.long_vowel_level >= 1 else "ウ"

    for idx, token in enumerate(result):
        if token != "ッ":
            continue
        next_index = _next_nonempty_index(result, idx + 1, contexts)
        if next_index is None:
            continue
        follower = result[next_index]
        if follower.startswith("チ"):
            result[next_index] = "ッチ" + follower[1:]
            result[idx] = ""
        elif follower.startswith("シュ"):
            result[next_index] = "ッシュ" + follower[1:]
            result[idx] = ""

    idx = 0
    while idx < len(result) - 1:
        pair = (result[idx], result[idx + 1])
        fused = READABLE_CV_FUSIONS.get(pair)
        if fused:
            ops.append({"type": "cv_fuse", "at": [idx, idx + 1], "from": [pair[0], pair[1]], "to": fused})
            result[idx] = fused
            result[idx + 1] = ""
            idx += 2
        else:
            idx += 1

    result = _apply_r_split(result, contexts, ops, options, disabled_rules)
    result = _apply_flap_visual(result, contexts, ops, disabled_rules)
    result = _apply_glottal_suppress(result, contexts, ops, disabled_rules)
    result = _apply_long_vowel_override(result, contexts, ops, disabled_rules)

    return result, ops


def _postprocess_with_ops(
    tokens: list[str],
    contexts: Sequence[_PhoneContext],
    options: KanaConversionOptions,
    disabled_rules: set[str],
) -> tuple[list[str], list[dict[str, str | int]]]:
    processed, ops = _postprocess(tokens[:], contexts, options, disabled_rules)
    return processed, ops


def _apply_r_split(
    tokens: Sequence[str],
    contexts: Sequence[_PhoneContext],
    ops: list[dict[str, str | int]],
    options: KanaConversionOptions,
    disabled_rules: set[str],
) -> list[str]:
    active = "r_split" not in disabled_rules and not options.r_coloring
    updated: list[str] = []
    token_count = len(tokens)
    for idx, ctx in enumerate(contexts):
        token = tokens[idx] if idx < token_count else ""
        new_token = token
        if active and token and ctx.canonical_symbol in {"ɹ", "r"} and idx > 0:
            prev_ctx = contexts[idx - 1]
            prev_vowel = _looks_like_vowel(prev_ctx.canonical_symbol)
            choice = "ラ" if prev_vowel else "ル"
            new_token = choice
            ops.append({"type": "r_split", "at": idx, "choice": choice})
        elif ctx.canonical_symbol in {"ɹ", "r"} and idx == 0 and not token:
            new_token = "ル"
        updated.append(new_token)
    return updated


def _apply_flap_visual(
    tokens: Sequence[str],
    contexts: Sequence[_PhoneContext],
    ops: list[dict[str, str | int]],
    disabled_rules: set[str],
) -> list[str]:
    active = "flap_visual" not in disabled_rules
    updated: list[str] = []
    token_count = len(tokens)
    limit = len(contexts)
    for idx, ctx in enumerate(contexts):
        token = tokens[idx] if idx < token_count else ""
        new_token = token
        if (
            active
            and ctx.canonical_symbol == "ɾ"
            and idx + 1 < limit
        ):
            duration = ctx.duration if ctx.duration is not None else 0.0
            try:
                duration_ms = float(duration) * 1000.0
            except (TypeError, ValueError):
                duration_ms = 0.0
            if duration_ms < 80.0:
                new_token = "ラ"
                ops.append({"type": "flap_visual", "span": [idx, idx + 1]})
        updated.append(new_token)
    return updated


def _apply_glottal_suppress(
    tokens: Sequence[str],
    contexts: Sequence[_PhoneContext],
    ops: list[dict[str, str | int]],
    disabled_rules: set[str],
) -> list[str]:
    active = "glottal_suppress" not in disabled_rules
    updated: list[str] = []
    token_count = len(tokens)
    last_index = len(contexts) - 1
    for idx, ctx in enumerate(contexts):
        token = tokens[idx] if idx < token_count else ""
        new_token = token
        if (
            active
            and token
            and ctx.canonical_symbol == "ʔ"
            and (idx == 0 or idx == last_index)
        ):
            confidence = ctx.confidence if ctx.confidence is not None else 1.0
            try:
                conf_value = float(confidence)
            except (TypeError, ValueError):
                conf_value = 1.0
            duration = ctx.duration if ctx.duration is not None else 0.0
            try:
                duration_ms = float(duration) * 1000.0
            except (TypeError, ValueError):
                duration_ms = 0.0
            if conf_value < 0.45 and duration_ms < 60.0:
                new_token = ""
                ops.append({"type": "glottal_suppress", "at": idx})
        updated.append(new_token)
    return updated


def _apply_long_vowel_override(
    tokens: Sequence[str],
    contexts: Sequence[_PhoneContext],
    ops: list[dict[str, str | int]],
    disabled_rules: set[str],
) -> list[str]:
    active = "long_vowel" not in disabled_rules
    updated: list[str] = []
    token_count = len(tokens)
    long_map = {"iː": "イー", "uː": "ウー", "ɑː": "アー"}
    for idx, ctx in enumerate(contexts):
        token = tokens[idx] if idx < token_count else ""
        new_token = token
        mapped = long_map.get(ctx.canonical_symbol)
        if not active or mapped is None:
            updated.append(new_token)
            continue
        if token == "ー":
            ops.append({"type": "long_vowel", "at": idx, "value": mapped})
        elif token != mapped:
            new_token = mapped
            ops.append({"type": "long_vowel", "at": idx, "value": mapped})
        updated.append(new_token)
    return updated


def _parse_disabled_rules() -> set[str]:
    raw = os.getenv("KANA_DISABLE", "")
    if not raw:
        return set()
    return {entry.strip() for entry in raw.split(",") if entry.strip()}


def _next_nonempty_index(
    tokens: Sequence[str],
    start: int,
    contexts: Sequence[_PhoneContext],
) -> int | None:
    for idx in range(start, len(tokens)):
        if contexts[idx].is_pause:
            continue
        if tokens[idx]:
            return idx
    return None



def _build_text(tokens: Sequence[str], contexts: Sequence[_PhoneContext]) -> str:
    units: list[tuple[str, _PhoneContext | None]] = []
    for token, ctx in zip(tokens, contexts):
        if ctx.is_pause:
            if units and units[-1][0] == "・":
                continue
            units.append(("・", None))
            continue
        if token:
            units.append((token, ctx))

    fused_units = _fuse_cv_units(units)
    text = "".join(token for token, _ in fused_units if token)
    text = text.strip("・")
    text = text.replace("ッッ", "ッ")
    while "・ ・" in text:
        text = text.replace("・ ・", "・")
    while "・・" in text:
        text = text.replace("・・", "・")
    return text


def _fuse_cv_units(units: Sequence[tuple[str, _PhoneContext | None]]) -> list[tuple[str, _PhoneContext | None]]:
    result: list[tuple[str, _PhoneContext | None]] = []
    idx = 0
    while idx < len(units):
        token, ctx = units[idx]
        if token == "・":
            if result and result[-1][0] == "・":
                idx += 1
                continue
            result.append((token, None))
            idx += 1
            continue
        if ctx is None:
            result.append((token, ctx))
            idx += 1
            continue
        if idx + 1 < len(units):
            next_token, next_ctx = units[idx + 1]
            fused = _attempt_cv_fusion(token, ctx, next_token, next_ctx)
            if fused is not None:
                result.append((fused, ctx))
                idx += 2
                continue
        result.append((token, ctx))
        idx += 1
    return result


def _attempt_cv_fusion(
    consonant_token: str,
    consonant_ctx: _PhoneContext | None,
    vowel_token: str,
    vowel_ctx: _PhoneContext | None,
) -> str | None:
    if consonant_ctx is None or vowel_ctx is None:
        return None
    if vowel_token == "・":
        return None
    consonant = consonant_ctx.canonical_symbol or ""
    vowel = vowel_ctx.canonical_symbol or ""
    if _looks_like_vowel(consonant):
        return None
    if not _looks_like_vowel(vowel):
        return None
    return CV_TOKEN_FUSIONS.get((consonant_token, vowel_token))


def _adjust_leading_n(
    tokens: list[str],
    contexts: Sequence[_PhoneContext],
    options: KanaConversionOptions,
) -> None:
    first_idx = _next_nonempty_index(tokens, 0, contexts)
    if first_idx is None:
        return
    first_ctx = contexts[first_idx]
    if first_ctx.symbol != "N" or not first_ctx.is_first_in_segment:
        return
    if tokens[first_idx] != "ン":
        return

    next_idx = _next_nonempty_index(tokens, first_idx + 1, contexts)
    if next_idx is None:
        return

    follower = tokens[next_idx]
    if not follower:
        return

    mapping = {
        "ア": "ナ",
        "ァ": "ナ",
        "イ": "ニ",
        "ィ": "ニ",
        "ウ": "ヌ",
        "ゥ": "ヌ",
        "エ": "ネ",
        "ェ": "ネ",
        "オ": "ノ",
        "ォ": "ノ",
    }

    for head, syllable in mapping.items():
        if follower.startswith(head):
            tokens[first_idx] = syllable
            remainder = follower[len(head) :]
            follower_ctx = contexts[next_idx]
            if not remainder:
                duration_ms = (follower_ctx.duration or 0.0) * 1000.0
                auto_long = (
                    options.long_vowel_level >= 1
                    and duration_ms >= options.auto_long_vowel_ms
                )
                if follower_ctx.symbol in {"IY", "UW"} and (
                    options.long_vowel_level >= 1 or auto_long
                ):
                    tokens[next_idx] = "ー"
                elif follower_ctx.symbol in {"AA", "AO"} and options.long_vowel_level >= 1:
                    tokens[next_idx] = "ー"
                else:
                    tokens[next_idx] = ""
            else:
                tokens[next_idx] = remainder
            return


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))


_BASE_MAPPING: dict[str, str] = {
    "B": "ブ",
    "CH": "チ",
    "D": "ド",
    "DH": "ズ",
    "F": "フ",
    "G": "グ",
    "HH": "ハ",
    "JH": "ジ",
    "K": "ク",
    "L": "ル",
    "M": "ム",
    "N": "ン",
    "NG": "ング",
    "P": "プ",
    "R": "ル",
    "S": "ス",
    "SH": "シュ",
    "T": "ト",
    "TH": "ス",
    "V": "ヴ",
    "W": "ウ",
    "Y": "イ",
    "Z": "ズ",
    "ZH": "ジ",
}

_VOWEL_BASE: dict[str, str] = {
    "AA": "ア",
    "AE": "ア",
    "AH": "ア",
    "AO": "オ",
    "AW": "アウ",
    "AY": "アイ",
    "EH": "エ",
    "ER": "ア",
    "EY": "ェイ",
    "IH": "イ",
    "IY": "イ",
    "OW": "オウ",
    "OY": "オイ",
    "UH": "ウ",
    "UW": "ウ",
}

__all__ = [
    "KanaConversionOptions",
    "KanaConversionResult",
    "to_kana_sequence",
]
