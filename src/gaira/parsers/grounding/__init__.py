"""Grounding-layer parser scaffolds."""

from gaira.parsers.grounding.serum_ag_colloids_grounding_parser import (
    SerumAgColloidsGroundingParser,
)
from gaira.parsers.grounding.serum_ag_colloids_literature_grounding_parser import (
    SerumAgColloidsLiteratureGroundingParser,
)

__all__ = [
    "SerumAgColloidsGroundingParser",
    "SerumAgColloidsLiteratureGroundingParser",
]
