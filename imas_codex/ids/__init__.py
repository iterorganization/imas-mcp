"""IDS assembly from facility graph data.

Assembles complete IMAS IDS instances by querying the knowledge graph
for DataNode properties and populating imas-python IDS objects.

Two modes:
  - Graph-driven: IDSRecipe + IMASMapping nodes in the graph define
    structure and field transformations.
  - YAML fallback: YAML recipe files with embedded Cypher queries.

Usage:
    from imas_codex.ids import IDSAssembler

    assembler = IDSAssembler("jet", "pf_active")
    ids = assembler.assemble(epoch="p68613")
    assembler.export(Path("jet_pf_active.h5"), epoch="p68613")
"""

from imas_codex.ids.assembler import IDSAssembler

__all__ = ["IDSAssembler"]
