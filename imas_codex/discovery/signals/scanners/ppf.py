"""PPF (Processed Pulse File) scanner plugin for JET.

PPF is JET's primary data access system. Data is organized as:
  Owner / DDA (Diagnostic Data Area) / Dtype (Data Type)

Discovery strategy:
  1. SSH to JET, call ppfdda(pulse) to enumerate DDAs for a reference pulse
  2. For each DDA, call ppfdti(pulse, dda) to enumerate Dtypes
  3. Create FacilitySignal per DDA/Dtype with PPF accessor format
  4. Use default_owner (typically "jetppf") for standard processed data

Remote execution uses the shared run_python_script() infrastructure
with scripts in imas_codex/remote/scripts/ (enumerate_ppf.py, check_ppf.py).

Config key: data_sources.ppf
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

# Physics domain hints from DDA name patterns
_DDA_DOMAIN_HINTS: dict[str, str] = {
    "EFIT": "equilibrium",
    "EHTR": "equilibrium",
    "MAGN": "magnetic_field_diagnostics",
    "HRTS": "particle_measurement_diagnostics",
    "KK3": "electromagnetic_wave_diagnostics",
    "KG1V": "particle_measurement_diagnostics",
    "KG1L": "particle_measurement_diagnostics",
    "KG10": "particle_measurement_diagnostics",
    "BOLO": "radiation_measurement_diagnostics",
    "BOLP": "radiation_measurement_diagnostics",
    "B5NN": "radiation_measurement_diagnostics",
    "NBI": "auxiliary_heating",
    "ICRH": "auxiliary_heating",
    "LHCD": "auxiliary_heating",
    "SXR": "radiation_measurement_diagnostics",
    "KS3": "spectroscopic_diagnostics",
    "EDG7": "spectroscopic_diagnostics",
    "EDG8": "spectroscopic_diagnostics",
    "GASH": "gas_injection",
    "CXSE": "particle_measurement_diagnostics",
    "KAD": "spectroscopic_diagnostics",
}


class PPFScanner:
    """Discover signals from JET PPF (Processed Pulse File) system.

    PPF discovery strategy:
    1. SSH to JET host, use ppfdda/ppfdti to enumerate DDAs and Dtypes
    2. Create FacilitySignal per DDA/Dtype with PPF accessor format
    3. Heuristic physics domain from DDA name (refined by LLM enrichment)

    Config (data_sources.ppf):
        reference_pulse: int - Pulse for signal enumeration
        default_owner: str - Default PPF owner (e.g., "jetppf")
        exclude_ddas: list[str] - DDAs to skip
    """

    scanner_type: str = "ppf"

    async def scan(
        self,
        facility: str,
        ssh_host: str,
        config: dict[str, Any],
        reference_shot: int | None = None,
    ) -> ScanResult:
        """Discover signals from PPF via SSH ppfdda/ppfdti enumeration.

        Uses remote/scripts/enumerate_ppf.py via async_run_python_script()
        for proper SSH execution with JSON encode/decode.
        """
        from imas_codex.remote.executor import async_run_python_script

        ref_pulse = reference_shot or config.get("reference_pulse")
        default_owner = config.get("default_owner", "jetppf")
        exclude_ddas = set(config.get("exclude_ddas", []))

        if not ref_pulse:
            logger.warning(
                "PPF scanner: no reference_pulse configured for %s", facility
            )
            return ScanResult(stats={"error": "no reference_pulse configured"})

        logger.info(
            "PPF scanner: enumerating DDAs for pulse %d (owner=%s) on %s",
            ref_pulse,
            default_owner,
            ssh_host,
        )

        try:
            output = await async_run_python_script(
                "enumerate_ppf.py",
                {
                    "pulse": ref_pulse,
                    "owner": default_owner,
                    "exclude_ddas": list(exclude_ddas),
                },
                ssh_host=ssh_host,
                timeout=120,
                python_command=config.get("python_command", "python3"),
                setup_commands=config.get("setup_commands"),
            )
            data = json.loads(output.strip().split("\n")[-1])
        except Exception as e:
            logger.error("PPF enumeration failed on %s: %s", ssh_host, e)
            return ScanResult(stats={"error": str(e)[:300]})

        if "error" in data:
            logger.error("PPF enumeration error: %s", data["error"])
            return ScanResult(stats={"error": data["error"]})

        raw_signals = data.get("signals", [])

        # Create DataAccess node
        data_access = DataAccess(
            id=f"{facility}:ppf:standard",
            facility_id=facility,
            name="PPF Standard Access",
            method_type="ppf",
            library="ppf",
            access_type="local",
            template_python=(
                "import ppf\n"
                "ppf.ppfuid('{owner}', rw='R')\n"
                "ppf.ppfgo({pulse}, seq=0)\n"
                "data, x, t, ier = ppf.ppfdata("
                "{pulse}, '{dda}', '{dtype}', uid='{owner}')"
            ),
        )

        # Convert to FacilitySignal nodes
        signals = []
        for raw in raw_signals:
            dda = raw["dda"]
            dtype = raw["dtype"]

            # Heuristic physics domain from DDA
            domain = _DDA_DOMAIN_HINTS.get(dda, "general")

            signal_id = f"{facility}:{domain}/{dda.lower()}_{dtype.lower()}"
            accessor = f"ppfdata({ref_pulse}, '{dda}', '{dtype}')"

            signals.append(
                FacilitySignal(
                    id=signal_id,
                    facility_id=facility,
                    status=FacilitySignalStatus.discovered,
                    physics_domain=domain,
                    name=f"{dda}/{dtype}",
                    accessor=accessor,
                    data_access=data_access.id,
                    discovery_source="ppf_enumeration",
                    example_shot=ref_pulse,
                )
            )

        logger.info(
            "PPF scanner: discovered %d signals from %d DDAs (pulse %d)",
            len(signals),
            data.get("ndda", 0),
            ref_pulse,
        )

        return ScanResult(
            signals=signals,
            data_access=data_access,
            metadata={
                "reference_pulse": ref_pulse,
                "default_owner": default_owner,
                "ndda": data.get("ndda", 0),
                "exclude_ddas": list(exclude_ddas),
            },
            stats={
                "signals_discovered": len(signals),
                "ddas_found": data.get("ndda", 0),
                "reference_pulse": ref_pulse,
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
        """Validate PPF signals return data for reference pulse.

        Uses remote/scripts/check_ppf.py via async_run_python_script().
        """
        from imas_codex.remote.executor import async_run_python_script

        ref_pulse = reference_shot or config.get("reference_pulse")
        default_owner = config.get("default_owner", "jetppf")

        if not ref_pulse:
            return [
                {"signal_id": s.id, "valid": False, "error": "no reference_pulse"}
                for s in signals
            ]

        # Parse DDA/Dtype from signal name (format: "DDA/DTYPE")
        batch = []
        for s in signals:
            parts = (s.name or "").split("/")
            if len(parts) == 2:
                batch.append({"id": s.id, "dda": parts[0], "dtype": parts[1]})
            else:
                batch.append({"id": s.id, "dda": "", "dtype": ""})

        try:
            output = await async_run_python_script(
                "check_ppf.py",
                {
                    "signals": batch,
                    "pulse": ref_pulse,
                    "owner": default_owner,
                },
                ssh_host=ssh_host,
                timeout=120,
                python_command=config.get("python_command", "python3"),
                setup_commands=config.get("setup_commands"),
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
            logger.error("PPF check failed: %s", e)
            return [
                {"signal_id": s.id, "valid": False, "error": str(e)[:200]}
                for s in signals
            ]


# Auto-register on import
register_scanner(PPFScanner())
