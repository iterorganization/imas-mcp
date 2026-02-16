"""
TDI function discovery for signal registration.

This module provides discovery of TDI functions as a preferred data access method.
TDI functions like tcv_eq() and tcv_get() provide:
- Physics-level abstraction over raw MDSplus paths
- Built-in versioning and source selection (LIUQE vs FBTE etc.)
- Sign convention handling
- Shot-conditional logic

Strategy:
- Parse .fun files to extract function metadata and supported quantities
- Create TDIFunction nodes for each physics accessor function
- Create FacilitySignal nodes for each quantity with TDI accessor format
- Optionally link to existing MDSplus-path signals as backing data

The TDI access layer is facility-specific (TCV) but the schema is generic.
Other facilities may have equivalent layers (JET PPF, ITER IMAS, etc.).
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field

from imas_codex.graph import GraphClient
from imas_codex.graph.models import (
    DataAccess,
    FacilitySignal,
    FacilitySignalStatus,
    PhysicsDomain,
)
from imas_codex.remote.executor import run_python_script

logger = logging.getLogger(__name__)


def get_tdi_exclude_functions(facility: str) -> frozenset[str]:
    """Load TDI function exclusion set from facility config.

    Reads data_sources.tdi.exclude_functions from the facility YAML config.
    Function names are case-insensitive (lowered on load).

    Args:
        facility: Facility ID (e.g., "tcv")

    Returns:
        Frozenset of lowercase function names to exclude from signal discovery.
        Returns empty frozenset if no excludes configured.
    """
    from imas_codex.discovery.base.facility import get_facility

    try:
        config = get_facility(facility)
        data_sources = config.get("data_sources", {})
        tdi_config = data_sources.get("tdi", {})
        excludes = tdi_config.get("exclude_functions", [])
        return frozenset(name.lower() for name in excludes)
    except Exception:
        logger.debug("No TDI exclude_functions configured for %s", facility)
        return frozenset()


@dataclass
class TDIQuantity:
    """A quantity accessible via a TDI function."""

    name: str
    function_name: str
    function_path: str
    description: str = ""
    mdsplus_paths: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)


@dataclass
class TDIFunctionMeta:
    """Metadata for a TDI function."""

    name: str
    path: str
    description: str = ""
    signature: str = ""
    source_code: str = ""  # Full .fun file content for LLM context
    parameters: list[str] = field(default_factory=list)
    quantities: list[dict] = field(default_factory=list)
    mdsplus_trees: list[str] = field(default_factory=list)
    tdi_dependencies: list[str] = field(default_factory=list)
    has_shot_conditional: bool = False
    shot_conditionals: list[str] = field(default_factory=list)


async def extract_tdi_functions(
    ssh_host: str,
    tdi_path: str,
) -> list[TDIFunctionMeta]:
    """Extract TDI function metadata from a remote facility.

    Args:
        ssh_host: SSH host alias for the facility
        tdi_path: Path to TDI function directory (e.g., /usr/local/CRPP/tdi/tcv)

    Returns:
        List of TDIFunctionMeta objects
    """
    # Run the extraction script via run_python_script
    result = await asyncio.to_thread(
        run_python_script,
        "extract_tdi_functions.py",
        {"tdi_path": tdi_path},
        ssh_host,
        120,  # timeout
    )

    # Parse the JSON output
    functions = json.loads(result)
    return [TDIFunctionMeta(**f) for f in functions]


def classify_tdi_quantity(
    quantity_name: str,
    function_name: str,
) -> PhysicsDomain:
    """Classify a TDI quantity into a physics domain.

    This is a heuristic classification based on naming patterns.
    LLM enrichment will refine this later.
    Uses PhysicsDomain enum values from the schema.
    """
    name_lower = quantity_name.lower()

    # Equilibrium quantities
    if any(
        kw in name_lower
        for kw in [
            "psi",
            "q_",
            "kappa",
            "delta",
            "axis",
            "xpt",
            "contour",
            "area",
            "volume",
            "ip",
            "i_p",
            "beta",
            "flux",
            "rmag",
            "zmag",
            "li",
            "elongation",
            "separatrix",
        ]
    ):
        return PhysicsDomain.equilibrium

    # Density / particle measurements
    if any(kw in name_lower for kw in ["nel", "ne", "density", "fir", "particle"]):
        return PhysicsDomain.particle_measurement_diagnostics

    # Temperature / Thomson scattering
    if any(kw in name_lower for kw in ["te", "temperature", "thomson", "ece"]):
        return PhysicsDomain.particle_measurement_diagnostics

    # Radiation / bolometry
    if any(kw in name_lower for kw in ["radiat", "bolo", "zeff"]):
        return PhysicsDomain.radiation_measurement_diagnostics

    # Spectroscopy / H-alpha
    if any(kw in name_lower for kw in ["ha", "alpha", "spec", "emission"]):
        return PhysicsDomain.radiation_measurement_diagnostics

    # Stored energy / MHD
    if any(kw in name_lower for kw in ["we", "energy", "mhd", "w_"]):
        return PhysicsDomain.magnetohydrodynamics

    # X-ray
    if any(kw in name_lower for kw in ["x_", "xray", "softx"]):
        return PhysicsDomain.radiation_measurement_diagnostics

    # Time / pulse
    if any(kw in name_lower for kw in ["tim", "time", "date", "shot", "pulse"]):
        return PhysicsDomain.machine_operations

    # Magnetics
    if any(
        kw in name_lower for kw in ["magnetic", "coil", "bphi", "btor", "bpol", "rbt"]
    ):
        return PhysicsDomain.magnetic_field_diagnostics

    # Gaps / geometry
    if "gap" in name_lower:
        return PhysicsDomain.equilibrium

    # Ohmic power / heating
    if "pohm" in name_lower or "heat" in name_lower or "power" in name_lower:
        return PhysicsDomain.auxiliary_heating

    # Default
    return PhysicsDomain.general


def build_signal_id(
    facility: str,
    physics_domain: PhysicsDomain,
    quantity_name: str,
) -> str:
    """Build a FacilitySignal ID in standard format.

    Format: facility:physics_domain/signal_name
    """
    # Normalize quantity name to lowercase with underscores
    signal_name = quantity_name.lower().replace(" ", "_")
    return f"{facility}:{physics_domain.value}/{signal_name}"


def build_tdi_accessor(function_name: str, quantity_name: str) -> str:
    """Build a TDI accessor expression.

    Examples:
        tcv_get, IP -> tcv_get('IP')
        tcv_eq, I_P -> tcv_eq('I_P')
    """
    return f"{function_name}('{quantity_name}')"


async def discover_tdi_signals(
    facility: str,
    ssh_host: str,
    tdi_path: str,
    data_access_id: str | None = None,
    filter_functions: list[str] | None = None,
) -> tuple[list[FacilitySignal], list[TDIFunctionMeta]]:
    """Discover signals from TDI functions.

    Loads exclude_functions from facility config (data_sources.tdi.exclude_functions)
    to filter out hardware/operational functions that don't return physics data.

    Args:
        facility: Facility ID (e.g., "tcv")
        ssh_host: SSH host alias
        tdi_path: Path to TDI function directory
        data_access_id: DataAccess to link signals to (default: facility:tdi:functions)
        filter_functions: Only process these function names (e.g., ["t"])

    Returns:
        Tuple of (FacilitySignal list, TDIFunctionMeta list with source code)
    """
    # Load exclude set from facility config
    exclude_functions = get_tdi_exclude_functions(facility)
    if exclude_functions:
        logger.info(
            "Loaded %d TDI exclude_functions from %s config",
            len(exclude_functions),
            facility,
        )

    # Extract TDI function metadata
    logger.info("Extracting TDI functions from %s:%s", ssh_host, tdi_path)
    functions = await extract_tdi_functions(ssh_host, tdi_path)
    logger.info("Found %d TDI functions", len(functions))

    # Filter to physics functions if specified
    if filter_functions:
        functions = [f for f in functions if f.name in filter_functions]

    # Default data access
    if not data_access_id:
        data_access_id = f"{facility}:tdi:functions"

    # Build signals from quantities
    signals: list[FacilitySignal] = []
    seen_quantities: set[str] = set()
    functions_with_signals: list[TDIFunctionMeta] = []

    for func in functions:
        # Skip internal/utility functions (start with _)
        if func.name.startswith("_"):
            continue

        # Skip operational/hardware functions that don't return physics data
        if func.name.lower() in exclude_functions:
            logger.debug("Skipping operational TDI function: %s", func.name)
            continue

        func_has_signals = False
        for q in func.quantities:
            quantity_name = q["name"]

            # Skip source selectors (these are function modes, not quantities)
            if quantity_name in {
                "FBTE",
                "FBTE.M",
                "LIUQE",
                "LIUQE.M",
                "LIUQE2",
                "LIUQE.M2",
                "LIUQE.M3",
                "LIUQE3",
                "FLAT",
                "FLAT.M",
                "RAMP",
                "RAMP.M",
                "RUNS",
                "RUNS.M",
                "MAGNETICS",
                "PCS",
            }:
                continue

            # Skip duplicates (same quantity from multiple functions)
            accessor = build_tdi_accessor(func.name, quantity_name)
            if accessor in seen_quantities:
                continue
            seen_quantities.add(accessor)

            # Classify physics domain
            physics_domain = classify_tdi_quantity(quantity_name, func.name)

            # Build signal ID
            signal_id = build_signal_id(facility, physics_domain, quantity_name)

            # Create signal
            signal = FacilitySignal(
                id=signal_id,
                status=FacilitySignalStatus.discovered,
                facility_id=facility,
                physics_domain=physics_domain,
                accessor=accessor,
                data_access=data_access_id,
                tdi_function=func.name,
                tdi_quantity=quantity_name,
                discovery_source="tdi_introspection",
            )

            signals.append(signal)
            func_has_signals = True

        # Keep functions that contribute signals (for LLM enrichment context)
        if func_has_signals:
            functions_with_signals.append(func)

    return signals, functions_with_signals


async def create_tdi_data_access(
    gc: GraphClient,
    facility: str,
) -> DataAccess:
    """Create or get the TDI data access node for a facility.

    This creates a generic TDI data access that works with any TDI function.
    The accessor field on FacilitySignal contains the full TDI call.
    """
    am_id = f"{facility}:tdi:functions"

    # Check if already exists
    existing = gc.query(
        "MATCH (da:DataAccess {id: $id}) RETURN da",
        id=am_id,
    )
    if existing:
        return DataAccess(**existing[0]["da"])

    # Create new data access node
    am = DataAccess(
        id=am_id,
        facility_id=facility,
        name="TDI Function Access",
        method_type="tdi",
        library="MDSplus",
        access_type="local",
        data_source="tcv_shot",  # Default tree for TDI context
        imports_template="import MDSplus",
        connection_template="tree = MDSplus.Tree('{data_source}', {shot}, 'readonly')",
        data_template="data = tree.tdiExecute('{accessor}').data()",
        time_template="time = tree.tdiExecute('dim_of({accessor})').data()",
        cleanup_template="tree.close()",
        full_example="""import MDSplus

tree = MDSplus.Tree('tcv_shot', {shot}, 'readonly')

# Using TDI function for high-level access
ip = tree.tdiExecute("tcv_ip()").data()
time = tree.tdiExecute("dim_of(tcv_ip())").data()

# Or via tcv_get for registry access
ip = tree.tdiExecute("tcv_get('IP')").data()

# tcv_eq for equilibrium with source selection
psi = tree.tdiExecute("tcv_eq('PSI', 'LIUQE')").data()

tree.close()
""",
    )

    # Insert via GraphClient using proper signature
    props = am.model_dump(exclude_none=True, by_alias=True)
    gc.create_node("DataAccess", am_id, props)
    logger.info("Created TDI data access: %s", am_id)

    return am


async def ingest_tdi_functions(
    gc: GraphClient,
    facility: str,
    functions: list[TDIFunctionMeta],
) -> int:
    """Ingest TDI function metadata into the graph.

    Creates TDIFunction nodes with source_code for LLM enrichment context.
    """
    if not functions:
        return 0

    func_dicts = []
    for func in functions:
        func_id = f"{facility}:tdi:{func.name}"
        func_dicts.append(
            {
                "id": func_id,
                "facility_id": facility,
                "name": func.name,
                "path": func.path,
                "description": func.description,
                "signature": func.signature,
                "source_code": func.source_code,
                "parameters": func.parameters,
                "mdsplus_trees": func.mdsplus_trees,
                "tdi_dependencies": func.tdi_dependencies,
                "has_shot_conditional": func.has_shot_conditional,
                "quantity_count": len(func.quantities),
            }
        )

    gc.query(
        """
        UNWIND $functions AS f
        MERGE (tf:TDIFunction {id: f.id})
        SET tf += f,
            tf.updated_at = datetime()
        WITH tf, f
        MATCH (fac:Facility {id: f.facility_id})
        MERGE (tf)-[:AT_FACILITY]->(fac)
        """,
        functions=func_dicts,
    )

    logger.info("Ingested %d TDI functions", len(func_dicts))
    return len(func_dicts)


async def ingest_tdi_signals(
    gc: GraphClient,
    signals: list[FacilitySignal],
    batch_size: int = 100,
) -> int:
    """Ingest TDI signals into the graph.

    Uses MERGE to avoid duplicates - signals with same ID are updated.
    """
    total = 0

    for i in range(0, len(signals), batch_size):
        batch = signals[i : i + batch_size]

        # Convert to dicts
        signal_dicts = [s.model_dump(exclude_none=True, by_alias=True) for s in batch]

        # Batch merge with AT_FACILITY edge
        gc.query(
            """
            UNWIND $signals AS s
            MERGE (fs:FacilitySignal {id: s.id})
            SET fs += s
            WITH fs, s
            MATCH (f:Facility {id: s.facility_id})
            MERGE (fs)-[:AT_FACILITY]->(f)
            """,
            signals=signal_dicts,
        )

        total += len(batch)
        logger.info("Ingested %d/%d TDI signals", total, len(signals))

    return total


async def run_tdi_discovery(
    facility: str,
    ssh_host: str,
    tdi_path: str,
    filter_functions: list[str] | None = None,
) -> int:
    """Main entry point for TDI signal discovery.

    Returns number of signals created.
    """
    gc = GraphClient()

    # Create data access node
    am = await create_tdi_data_access(gc, facility)

    # Discover signals
    signals = await discover_tdi_signals(
        facility=facility,
        ssh_host=ssh_host,
        tdi_path=tdi_path,
        data_access_id=am.id,
        filter_functions=filter_functions,
    )

    logger.info("Discovered %d TDI signals", len(signals))

    # Ingest to graph
    count = await ingest_tdi_signals(gc, signals)

    return count
