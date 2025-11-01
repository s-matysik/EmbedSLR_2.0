from importlib import metadata as _m
from .embeddings import get_embeddings, list_models
from .similarity import rank_by_cosine
from .bibliometrics import full_report, indicators
from .colab_app import run as colab_run

# New MCDA and ranking modules
from .mcda import l_scoring, z_scoring, l_scoring_plus, mcda_report
from .ranking import (
    rank_by_keywords, 
    rank_by_references, 
    rank_by_citations,
    compute_keyword_frequency,
    compute_reference_frequency,
    detailed_frequency_report
)

try:
    __version__ = _m.version(__name__)
except _m.PackageNotFoundError:
    __version__ = "0.6.0"

__all__ = [
    # Original functions
    "get_embeddings", "list_models", "rank_by_cosine",
    "full_report", "indicators", "colab_run",
    # MCDA functions
    "l_scoring", "z_scoring", "l_scoring_plus", "mcda_report",
    # Ranking functions
    "rank_by_keywords", "rank_by_references", "rank_by_citations",
    "compute_keyword_frequency", "compute_reference_frequency",
    "detailed_frequency_report",
]
