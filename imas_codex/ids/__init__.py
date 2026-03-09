"""IDS assembly from facility graph data.

Assembles complete IMAS IDS instances by querying the knowledge graph
for DataNode properties and populating imas-python IDS objects.

Usage:
    from imas_codex.ids import IDSAssembler

    assembler = IDSAssembler("jet", "pf_active")
    ids = assembler.assemble(epoch="p68613")
    assembler.export(Path("jet_pf_active.h5"), epoch="p68613")
"""

from imas_codex.ids.assembler import IDSAssembler

__all__ = ["IDSAssembler"]
