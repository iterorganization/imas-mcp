"""Physics domain enum — canonical source is imas-standard-names.

This module re-exports PhysicsDomain so that all imas-codex code
continues to import from the same path:
    from imas_codex.core.physics_domain import PhysicsDomain
"""

from imas_standard_names.grammar.tag_types import PhysicsDomain

__all__ = ["PhysicsDomain"]
