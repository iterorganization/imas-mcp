"""Content extractors for entity recognition during ingestion.

Extractors scan chunk text for domain-specific references:
- IDS: IMAS IDS references (equilibrium, core_profiles, etc.)
- MDSplus: Tree paths and TDI function calls
- Units: Physical units mentioned in text
- Conventions: Sign conventions and COCOS references
"""

from .ids import extract_ids_references, get_known_ids
from .mdsplus import MDSplusReference, extract_mdsplus_paths
from .units import extract_conventions, extract_units

__all__ = [
    "MDSplusReference",
    "extract_conventions",
    "extract_ids_references",
    "extract_mdsplus_paths",
    "extract_units",
    "get_known_ids",
]
