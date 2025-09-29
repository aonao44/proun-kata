"""Application-level dependency wiring."""
from functools import lru_cache

from phoneme.pipeline import PhonemePipeline


@lru_cache(maxsize=1)
def get_pipeline() -> PhonemePipeline:
    """Provide a shared phoneme pipeline instance."""

    return PhonemePipeline()
