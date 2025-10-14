"""Compatibility layer for legacy imports."""
from __future__ import annotations

from core.normalize import canonical_to_arpabet, normalize_symbol


SILENCE_TOKENS = {"SIL", "SP", "NSN"}

__all__ = ["normalize_symbol", "canonical_to_arpabet", "SILENCE_TOKENS"]
