"""JPF (JET Processing Facility) scanner plugin for JET.

JPF provides raw hardware diagnostic signals organized by subsystem code
and signal name. Data is accessed via MDSplus thin-client TDI functions:
  dpf("{subsystem}/{signal}", {shot})

Discovery strategy:
  1. SSH to JET, connect to MDSplus server
  2. Enumerate JPF subsystem signal nodes
  3. Create FacilitySignal per subsystem/signal with JPF accessor format
  4. Create DataAccess node with dpf() TDI template

Remote execution uses the shared run_python_script() infrastructure
with scripts in imas_codex/remote/scripts/ (enumerate_jpf.py, check_jpf.py).

Config key: data_systems.mdsplus.jpf_subsystems (subsystem codes list)
Facility: JET
"""

from __future__ import annotations

import json
import logging
from typing import Any

from imas_codex.discovery.signals.scanners.base import (
    ScanResult,
    register_scanner,
)
from imas_codex.graph.models import (
    DataAccess,
    FacilitySignal,
    FacilitySignalStatus,
)

logger = logging.getLogger(__name__)

# Physics domain hints from JPF subsystem codes
_SUBSYSTEM_DOMAIN_HINTS: dict[str, str] = {
    "DA": "magnetic_field_diagnostics",
    "DB": "particle_measurement_diagnostics",
    "DC": "electromagnetic_wave_diagnostics",
    "DD": "radiation_measurement_diagnostics",
    "DE": "particle_measurement_diagnostics",
    "DF": "electromagnetic_wave_diagnostics",
    "DG": "particle_measurement_diagnostics",
    "DH": "electromagnetic_wave_diagnostics",
    "DI": "radiation_measurement_diagnostics",
    "PF": "magnetic_field_diagnostics",
    "TF": "magnetic_field_diagnostics",
    "PL": "magnetics",
    "GS": "plant_systems",
    "VC": "plant_systems",
    "AH": "auxiliary_heating",
    "RF": "auxiliary_heating",
    "CC": "plant_systems",
    "LH": "auxiliary_heating",
    "MC": "plant_systems",
    "NM": "radiation_measurement_diagnostics",
    "SA": "plant_systems",
}


class JPFScanner:
    """Discover signals from JET JPF (JET Processing Facility) system.

    JPF discovery strategy:
    1. SSH to JET host, connect to MDSplus server
    2. Enumerate JPF signal nodes per subsystem via TDI functions
    3. Create FacilitySignal per subsystem/signal with dpf() accessor
    4. Heuristic physics domain from subsystem code

    Config (data_systems.mdsplus):
        server: str - MDSplus server hostname
        jpf_subsystems: list[str] - 2-char subsystem codes
        reference_shot: int - Shot for signal enumeration
        setup_commands: list[str] - Module loads for MDSplus
    """

    scanner_type: str = "jpf"

    async def scan(
        self,
        facility: str,
        ssh_host: str,
        config: dict[str, Any],
        reference_shot: int | None = None,
    ) -> ScanResult:
        """Discover signals from JPF via SSH MDSplus enumeration.

        Uses remote/scripts/enumerate_jpf.py via async_run_python_script()
        for proper SSH execution with JSON encode/decode.
        """
        from imas_codex.remote.executor import async_run_python_script

        # JPF config is nested under data_systems.mdsplus
        mdsplus_config = config
        server = mdsplus_config.get("server")
        ref_shot = reference_shot or mdsplus_config.get("reference_shot")
        subsystems = mdsplus_config.get("jpf_subsystems", [])

        if not server:
            logger.warning("JPF scanner: no MDSplus server configured for %s", facility)
            return ScanResult(stats={"error": "no server configured"})
        if not ref_shot:
            logger.warning("JPF scanner: no reference_shot configured for %s", facility)
            return ScanResult(stats={"error": "no reference_shot"})
        if not subsystems:
            logger.warning("JPF scanner: no jpf_subsystems configured for %s", facility)
            return ScanResult(stats={"error": "no jpf_subsystems configured"})

        logger.info(
            "JPF scanner: enumerating %d subsystems on %s (shot %d)",
            len(subsystems),
            server,
            ref_shot,
        )

        try:
            output = await async_run_python_script(
                "enumerate_jpf.py",
                {
                    "server": server,
                    "shot": ref_shot,
                    "subsystems": subsystems,
                },
                ssh_host=ssh_host,
                timeout=180,
                python_command=mdsplus_config.get("python_command", "python3"),
                setup_commands=mdsplus_config.get("setup_commands"),
            )
            data = json.loads(output.strip().split("\n")[-1])
        except Exception as e:
            logger.error("JPF enumeration failed on %s: %s", ssh_host, e)
            return ScanResult(stats={"error": str(e)[:300]})

        if "error" in data:
            logger.error("JPF enumeration error: %s", data["error"])
            return ScanResult(stats={"error": data["error"]})

        raw_signals = data.get("signals", [])

        # Create DataAccess node for JPF
        data_access = DataAccess(
            id=f"{facility}:jpf:standard",
            facility_id=facility,
            name="JPF Standard Access",
            method_type="jpf",
            library="MDSplus",
            access_type="remote",
            data_source="jpf",
            connection_template=(
                f"import MDSplus\nconn = MDSplus.Connection('{server}')"
            ),
            data_template=(
                "data = conn.get('dpf(\"{signal_path}\", {shot})')\n"
                "time = conn.get('dpf(\"{signal_path}:t\", {shot})')"
            ),
            setup_commands=mdsplus_config.get("setup_commands"),
        )

        # Convert to FacilitySignal nodes
        signals = []
        for raw in raw_signals:
            subsystem = raw["subsystem"]
            signal = raw.get("signal", "")
            path = raw.get("path", "")

            # Skip placeholder entries
            if raw.get("enumeration_pending"):
                continue

            # Heuristic physics domain from subsystem code
            domain = _SUBSYSTEM_DOMAIN_HINTS.get(subsystem, "general")

            signal_name = f"{subsystem}/{signal}"
            signal_id = f"{facility}:{domain}/{subsystem.lower()}_{signal.lower()}"
            accessor = f'dpf("{path}", {ref_shot})'

            signals.append(
                FacilitySignal(
                    id=signal_id,
                    facility_id=facility,
                    status=FacilitySignalStatus.discovered,
                    physics_domain=domain,
                    name=signal_name,
                    accessor=accessor,
                    data_access=data_access.id,
                    discovery_source="jpf_enumeration",
                    example_shot=ref_shot,
                    node_path=path,
                )
            )

        logger.info(
            "JPF scanner: discovered %d signals from %d subsystems",
            len(signals),
            data.get("subsystems_scanned", 0),
        )

        errors = data.get("errors", [])
        if errors:
            logger.warning("JPF enumeration had %d errors: %s", len(errors), errors[:3])

        return ScanResult(
            signals=signals,
            data_access=data_access,
            metadata={
                "reference_shot": ref_shot,
                "server": server,
                "subsystems": subsystems,
            },
            stats={
                "signals_discovered": len(signals),
                "subsystems_scanned": data.get("subsystems_scanned", 0),
                "subsystems_failed": data.get("subsystems_failed", 0),
                "reference_shot": ref_shot,
            },
        )

    async def check(
        self,
        facility: str,
        ssh_host: str,
        signals: list[FacilitySignal],
        config: dict[str, Any],
        reference_shot: int | None = None,
    ) -> list[dict[str, Any]]:
        """Validate JPF signals return data for reference pulse.

        Uses remote/scripts/check_jpf.py via async_run_python_script().
        """
        from imas_codex.remote.executor import async_run_python_script

        mdsplus_config = config
        server = mdsplus_config.get("server")
        ref_shot = reference_shot or mdsplus_config.get("reference_shot")

        if not server or not ref_shot:
            return [
                {
                    "signal_id": s.id,
                    "valid": False,
                    "error": "no server or reference_shot",
                }
                for s in signals
            ]

        batch = []
        for s in signals:
            batch.append({"id": s.id, "path": s.node_path or ""})

        try:
            output = await async_run_python_script(
                "check_jpf.py",
                {
                    "signals": batch,
                    "server": server,
                    "shot": ref_shot,
                },
                ssh_host=ssh_host,
                timeout=120,
                python_command=mdsplus_config.get("python_command", "python3"),
                setup_commands=mdsplus_config.get("setup_commands"),
            )
            response = json.loads(output.strip().split("\n")[-1])
            return [
                {
                    "signal_id": r["id"],
                    "valid": r.get("success", False),
                    "shape": r.get("shape"),
                    "dtype": r.get("dtype"),
                    "error": r.get("error"),
                }
                for r in response.get("results", [])
            ]
        except Exception as e:
            logger.error("JPF check failed: %s", e)
            return [
                {"signal_id": s.id, "valid": False, "error": str(e)[:200]}
                for s in signals
            ]


# Auto-register on import
register_scanner(JPFScanner())
