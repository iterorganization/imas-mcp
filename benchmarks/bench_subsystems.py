"""Offline subsystem benchmarks.

No graph dependency — measures pure Python subsystem performance:
COCOS computation, unit normalization, schema context generation.
"""

from __future__ import annotations

from imas_codex.cocos import cocos_to_parameters, determine_cocos
from imas_codex.cocos.transforms import path_needs_cocos_transform
from imas_codex.units import normalize_unit_symbol

try:
    from imas_codex.graph.schema_context import schema_for
except ImportError:
    schema_for = None

UNIT_BATCH = [
    "eV",
    "m",
    "m^-3",
    "T",
    "Pa",
    "s",
    "A",
    "V",
    "W",
    "m.s^-1",
    "T.m^2",
    "keV",
    "m^-2.s^-1",
    "ohm.m",
    "kg",
    "J",
    "Hz",
    "N",
    "C",
    "F",
    "H",
    "S",
    "Wb",
    "lm",
    "lx",
    "Bq",
    "Gy",
    "Sv",
    "kat",
    "mol",
    "rad",
    "sr",
    "cd",
    "K",
    "kg.m^-3",
    "J.s",
    "W.m^-2",
    "A.m^-1",
    "V.m^-1",
    "C.m^-2",
    "F.m^-1",
    "H.m^-1",
    "T.m",
    "Wb.m^-2",
    "J.kg^-1",
    "J.mol^-1",
    "W.m^-1.K^-1",
    "Pa.s",
    "m^2.s^-1",
    "kg.m^-1.s^-1",
]


class SubsystemBenchmarks:
    """Benchmark pure Python subsystems on the critical path."""

    timeout = 60

    def setup(self):
        """Warmup each subsystem."""
        cocos_to_parameters(11)
        normalize_unit_symbol("eV")
        if schema_for is not None:
            schema_for(task="overview")

    # -- COCOS ---------------------------------------------------------------

    def time_cocos_determine(self):
        """COCOS computation from equilibrium quantities."""
        determine_cocos(
            psi_axis=-0.5,
            psi_edge=0.1,
            ip=1e6,
            b0=5.3,
            q=1.5,
        )

    def time_cocos_to_params(self):
        """COCOS lookup table."""
        cocos_to_parameters(11)

    def time_cocos_transform_check(self):
        """Path classification for COCOS transform."""
        path_needs_cocos_transform(
            "equilibrium", "time_slice/global_quantities/psi_axis"
        )

    # -- Unit normalization --------------------------------------------------

    def time_unit_normalize_cached(self):
        """LRU cache hit."""
        normalize_unit_symbol("m.s^-1")

    def time_unit_normalize_cold(self):
        """LRU miss + pint parse."""
        normalize_unit_symbol.cache_clear()
        normalize_unit_symbol("m.s^-1")

    def time_unit_normalize_batch(self):
        """Batch throughput (50 distinct units)."""
        normalize_unit_symbol.cache_clear()
        for unit in UNIT_BATCH:
            normalize_unit_symbol(unit)

    # -- Schema context ------------------------------------------------------

    def time_schema_for_overview(self):
        """Schema slice — overview scope."""
        if schema_for is None:
            raise NotImplementedError("schema_for not available")
        schema_for(task="overview")

    def time_schema_for_signals(self):
        """Schema slice — task-scoped."""
        if schema_for is None:
            raise NotImplementedError("schema_for not available")
        schema_for(task="signals")

    def time_schema_for_labels(self):
        """Schema slice — label-scoped."""
        schema_for("Facility", "DataSource")
