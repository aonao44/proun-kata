"""ARPAbet列をスタイル可変のカタカナ列へ変換する。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

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


@dataclass(frozen=True)
class KanaConversionOptions:
    """パラメータ付きの変換設定。"""

    reading_style: str = "balanced"
    long_vowel_level: int = 1
    sokuon_level: int = 1
    r_coloring: bool = False
    th_style: str = "su"
    final_c_handling: str = "xtsu"

    def clamp(self) -> KanaConversionOptions:
        return KanaConversionOptions(
            reading_style=self.reading_style if self.reading_style in {"raw", "balanced", "natural"} else "balanced",
            long_vowel_level=_clamp(self.long_vowel_level, 0, 2),
            sokuon_level=_clamp(self.sokuon_level, 0, 2),
            r_coloring=bool(self.r_coloring),
            th_style=self.th_style if self.th_style in {"su", "zu"} else "su",
            final_c_handling=self.final_c_handling if self.final_c_handling in {"xtsu", "tsu"} else "xtsu",
        )


@dataclass(frozen=True)
class KanaConversionResult:
    """変換結果。"""

    tokens: list[str]
    text: str


@dataclass(frozen=True)
class _PhoneContext:
    symbol: str
    index: int
    prev_symbol: str | None
    next_symbol: str | None
    prev_content: str | None
    next_content: str | None
    is_pause: bool
    is_first_in_segment: bool
    is_last_in_segment: bool


def to_kana_sequence(phones: Iterable[str], *, options: KanaConversionOptions) -> KanaConversionResult:
    """音素配列をカタカナ配列と結合文字列へ変換する。"""

    phone_list = [symbol.upper() for symbol in phones]
    if not phone_list:
        return KanaConversionResult(tokens=[], text="")

    opts = options.clamp()
    contexts = list(_build_context(phone_list))
    tokens = [_map_symbol(ctx, opts) for ctx in contexts]
    tokens = _postprocess(tokens, contexts, opts)
    kana_text = _build_text(tokens, contexts)
    return KanaConversionResult(tokens=tokens, text=kana_text)


def _build_context(phones: Sequence[str]) -> Iterator[_PhoneContext]:
    prev_content = None
    next_content_cache = _next_content_cache(phones)
    segment_start = True

    for idx, symbol in enumerate(phones):
        is_pause = symbol in SILENCE_LABELS
        next_symbol = phones[idx + 1] if idx + 1 < len(phones) else None
        next_content = next_content_cache[idx]
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
            index=idx,
            prev_symbol=phones[idx - 1] if idx > 0 else None,
            next_symbol=next_symbol,
            prev_content=prev_content,
            next_content=next_content,
            is_pause=is_pause,
            is_first_in_segment=segment_start and not is_pause,
            is_last_in_segment=is_last_in_segment,
        )

        if is_pause:
            segment_start = True
        else:
            segment_start = False
            prev_content = symbol


def _next_content_cache(phones: Sequence[str]) -> list[str | None]:
    next_content: list[str | None] = [None] * len(phones)
    ahead: str | None = None
    for idx in range(len(phones) - 1, -1, -1):
        symbol = phones[idx]
        next_content[idx] = ahead
        if symbol not in SILENCE_LABELS:
            ahead = symbol
    return next_content


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

    return _BASE_MAPPING.get(symbol, symbol)


def _map_vowel(symbol: str, context: _PhoneContext, options: KanaConversionOptions) -> str:
    base = _VOWEL_BASE.get(symbol, "")
    if not base:
        return symbol

    if symbol == "IY":
        if context.is_last_in_segment and options.long_vowel_level >= 1:
            return "イー"
        if options.long_vowel_level == 0:
            return "イ"
    if symbol == "UW":
        if context.is_last_in_segment and options.long_vowel_level >= 1:
            return "ウー"
        if options.long_vowel_level == 0:
            return "ウ"
    if symbol == "AA" and context.is_last_in_segment and options.long_vowel_level >= 1:
        return "アー"
    if symbol == "AO" and context.is_last_in_segment and options.long_vowel_level >= 1:
        return "オー"
    if symbol == "AH" and context.is_last_in_segment and options.long_vowel_level >= 2:
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
    return (
        context.prev_content is not None
        and context.next_content is not None
        and context.prev_content in VOWELS | SEMI_VOWELS
        and context.next_content in VOWELS | SEMI_VOWELS
    )


def _postprocess(tokens: list[str], contexts: Sequence[_PhoneContext], options: KanaConversionOptions) -> list[str]:
    result = tokens[:]
    for idx in range(len(result) - 1):
        current_ctx = contexts[idx]
        next_ctx = contexts[idx + 1]
        current = result[idx]
        following = result[idx + 1]
        if current_ctx.symbol == "M" and next_ctx.symbol == "EY":
            result[idx] = "メ"
            result[idx + 1] = "イ" if following else "イ"
        if current_ctx.symbol == "K" and next_ctx.symbol in {"IH", "IY"}:
            result[idx] = "キ" if next_ctx.symbol == "IH" else "キー"
            result[idx + 1] = "" if next_ctx.symbol == "IH" else ""
        if current_ctx.symbol == "TH" and next_ctx.symbol == "AE":
            result[idx] = "サ"
            result[idx + 1] = "ァ" if not next_ctx.is_last_in_segment else "ァー"
        if current_ctx.symbol == "NG" and next_ctx.symbol == "K":
            result[idx] = "ング"
            result[idx + 1] = "" if result[idx + 1] == "ク" else result[idx + 1]
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
    return result


def _next_nonempty_index(tokens: Sequence[str], start: int, contexts: Sequence[_PhoneContext]) -> int | None:
    for idx in range(start, len(tokens)):
        if contexts[idx].is_pause:
            continue
        if tokens[idx]:
            return idx
    return None


def _build_text(tokens: Sequence[str], contexts: Sequence[_PhoneContext]) -> str:
    pieces: list[str] = []
    for token, ctx in zip(tokens, contexts, strict=True):
        if ctx.is_pause:
            if pieces and pieces[-1] == "・":
                continue
            pieces.append("・")
            continue
        if token:
            pieces.append(token)
    text = "".join(pieces)
    text = text.strip("・")
    text = text.replace("ッッ", "ッ")
    while "・ ・" in text:
        text = text.replace("・ ・", "・")
    while "・・" in text:
        text = text.replace("・・", "・")
    return text


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
