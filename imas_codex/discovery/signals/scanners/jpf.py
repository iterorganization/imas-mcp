"""JPF (JET Processing Facility) scanner plugin for JET.

JPF provides raw hardware diagnostic signals organized by subsystem code
and signal name. Data is accessed via MDSplus thin-client TDI functions:
  dpf("{subsystem}/{signal}", {shot})

Discovery strategy:
  1. SSH to JET, use getdat module to enumerate JPF signal nodes
  2. getdat.zadop() opens subsystem pulse file, adlnod() lists nodes
  3. Create FacilitySignal per subsystem/signal with JPF accessor format
  4. Create DataAccess node with dpf() TDI template

Remote execution uses the shared run_python_script() infrastructure
with scripts in imas_codex/remote/scripts/ (enumerate_jpf.py, check_jpf.py).

Config key: data_systems.jpf.subsystems (structured subsystem definitions)
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


class JPFScanner:
    """Discover signals from JET JPF (JET Processing Facility) system.

    JPF discovery strategy:
    1. SSH to JET host, use getdat module to enumerate signal nodes
    2. getdat.zadop() opens subsystem pulse file per subsystem
    3. getdat.adlnod() lists all signal nodes within each subsystem
    4. Create FacilitySignal per subsystem/signal with dpf() accessor
    5. Physics domain from per-subsystem config in jet.yaml

    Config (data_systems.jpf):
        server: str - MDSplus server hostname (for data access template)
        subsystems: list[JPFSubsystem] - structured subsystem definitions
        reference_shot: int - Shot for signal enumeration
        setup_commands: list[str] - Module loads for getdat
    """

    scanner_type: str = "jpf"

    async def scan(
        self,
        facility: str,
        ssh_host: str,
        config: dict[str, Any],
        reference_shot: int | None = None,
    ) -> ScanResult:
        """Discover signals from JPF via SSH getdat enumeration.

        Uses remote/scripts/enumerate_jpf.py via async_run_python_script()
        for proper SSH execution with JSON encode/decode.
        """
        from imas_codex.remote.executor import async_run_python_script

        jpf_config = config
        server = jpf_config.get("server", "mdsplus.jet.uk")
        ref_shot = reference_shot or jpf_config.get("reference_shot")
        subsystem_defs = jpf_config.get("subsystems", [])

        # Build domain lookup from config
        domain_lookup: dict[str, str] = {}
        subsystem_codes: list[str] = []
        for sub in subsystem_defs:
            if isinstance(sub, dict):
                code = sub["code"]
                domain_lookup[code] = sub.get("physics_domain", "general")
                subsystem_codes.append(code)
            else:
                # Backward compat: plain string codes
                subsystem_codes.append(str(sub))

        if not ref_shot:
            logger.warning("JPF scanner: no reference_shot configured for %s", facility)
            return ScanResult(stats={"error": "no reference_shot"})
        if not subsystem_codes:
            logger.warning("JPF scanner: no subsystems configured for %s", facility)
            return ScanResult(stats={"error": "no subsystems configured"})

        logger.info(
            "JPF scanner: enumerating %d subsystems via getdat (shot %d)",
            len(subsystem_codes),
            ref_shot,
        )

        try:
            output = await async_run_python_script(
                "enumerate_jpf.py",
                {
                    "shot": ref_shot,
                    "subsystems": subsystem_codes,
                },
                ssh_host=ssh_host,
                timeout=300,
                python_command=jpf_config.get("python_command", "python3"),
                setup_commands=jpf_config.get("setup_commands"),
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
            setup_commands=jpf_config.get("setup_commands"),
        )

        # Convert to FacilitySignal nodes
        signals = []
        for raw in raw_signals:
            subsystem = raw["subsystem"]
            signal = raw.get("signal", "")
            path = raw.get("path", "")
            description = raw.get("description", "")

            # Physics domain from config, fallback to general
            domain = domain_lookup.get(subsystem, "general")

            signal_name = f"{subsystem}/{signal}"
            signal_id = f"{facility}:{domain}/{subsystem.lower()}_{signal.lower()}"
            accessor = f'dpf("{path}", {ref_shot})'

            sig = FacilitySignal(
                id=signal_id,
                facility_id=facility,
                status=FacilitySignalStatus.discovered,
                physics_domain=domain,
                name=signal_name,
                accessor=accessor,
                data_access=data_access.id,
                data_source_name="jpf",
                data_source_path=path,
                discovery_source="jpf",
                example_shot=ref_shot,
                node_path=path,
            )
            if description:
                sig.description = description

            signals.append(sig)

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
                "subsystems": subsystem_codes,
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

        jpf_config = config
        server = jpf_config.get("server", "mdsplus.jet.uk")
        ref_shot = reference_shot or jpf_config.get("reference_shot")

        if not ref_shot:
            return [
                {
                    "signal_id": s.id,
                    "valid": False,
                    "error": "no reference_shot",
                }
                for s in signals
            ]

        batch = []
        for s in signals:
            # JPF signals use accessor as the signal identifier (e.g.,
            # "DA/C1D-IPLA"), not node_path or data_source_path.  When those
            # fields are empty the path falls through to "", causing 100% failure.
            # Prefer data_source_path (raw path), then node_path, then try to
            # extract from accessor (which wraps as dpf("path", shot)).
            path = s.data_source_path or s.node_path or ""
            if not path and s.accessor:
                # Extract path from dpf("DA/C1D-IPLA", 99896) format
                acc = s.accessor
                if 'dpf("' in acc:
                    path = acc.split('dpf("', 1)[1].split('"', 1)[0]
                else:
                    path = acc
            batch.append({"id": s.id, "path": path})

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
                python_command=jpf_config.get("python_command", "python3"),
                setup_commands=jpf_config.get("setup_commands"),
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
