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

Config key: data_systems.ppf
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
    "KS3": "electromagnetic_wave_diagnostics",
    "EDG7": "electromagnetic_wave_diagnostics",
    "EDG8": "electromagnetic_wave_diagnostics",
    "GASH": "plant_systems",
    "CXSE": "particle_measurement_diagnostics",
    "KAD": "electromagnetic_wave_diagnostics",
}


class PPFScanner:
    """Discover signals from JET PPF (Processed Pulse File) system.

    PPF discovery strategy:
    1. SSH to JET host, use ppfdda/ppfdti to enumerate DDAs and Dtypes
    2. Create FacilitySignal per DDA/Dtype with PPF accessor format
    3. Heuristic physics domain from DDA name (refined by LLM enrichment)

    Config (data_systems.ppf):
        reference_pulse: int - Pulse for signal enumeration
        default_owner: str - Default PPF owner (e.g., "jetppf")
        exclude_ddas: list[str] - DDAs to skip
        sal_endpoint: str - Reserved for future SAL REST access (currently unreachable)
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
            data_source="ppf",
            connection_template=(
                "import ppf\nppf.ppfuid('{owner}', rw='R')\nppf.ppfgo({shot}, seq=0)"
            ),
            data_template=(
                "result = ppf.ppfdata({shot}, '{dda}', '{dtype}', uid='{owner}')\n"
                "data = result[0]  # data array\n"
                "ier = result[-1]  # error code"
            ),
            setup_commands=config.get("setup_commands"),
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
                    data_source_name="ppf",
                    data_source_path=f"{dda}/{dtype}",
                    discovery_source="ppf",
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
        If the primary shot returns ier=210002 (data not available for shot),
        retries with additional check_shots from the config before marking
        as not_available_for_shot.
        """
        from imas_codex.remote.executor import async_run_python_script

        ref_pulse = reference_shot or config.get("reference_pulse")
        default_owner = config.get("default_owner", "jetppf")
        # Additional shots to try when ier=210002 (shot-specific absence)
        check_shots = config.get("check_shots", [])

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

        # Build ordered list of shots to try: primary first, then fallbacks
        shots_to_try = [ref_pulse]
        for shot in check_shots:
            if shot != ref_pulse and shot not in shots_to_try:
                shots_to_try.append(shot)

        async def _run_check(pulse: int) -> dict:
            output = await async_run_python_script(
                "check_ppf.py",
                {
                    "signals": batch,
                    "pulse": pulse,
                    "owner": default_owner,
                },
                ssh_host=ssh_host,
                timeout=120,
                python_command=config.get("python_command", "python3"),
                setup_commands=config.get("setup_commands"),
            )
            return json.loads(output.strip().split("\n")[-1])

        try:
            response = await _run_check(shots_to_try[0])
            results_by_id: dict[str, dict] = {}
            shot_unavail_ids: list[str] = []

            for r in response.get("results", []):
                error = r.get("error", "")
                if r.get("success", False):
                    results_by_id[r["id"]] = {
                        "signal_id": r["id"],
                        "valid": True,
                        "shape": r.get("shape"),
                        "dtype": r.get("dtype"),
                        "shot": shots_to_try[0],
                    }
                elif "ier=210002" in error:
                    # Shot-specific absence — candidate for retry
                    shot_unavail_ids.append(r["id"])
                    results_by_id[r["id"]] = {
                        "signal_id": r["id"],
                        "valid": False,
                        "error": error,
                        "checked_shots": [shots_to_try[0]],
                    }
                else:
                    results_by_id[r["id"]] = {
                        "signal_id": r["id"],
                        "valid": False,
                        "error": error,
                    }

            # Retry shot-unavailable signals with fallback shots
            for fallback_shot in shots_to_try[1:]:
                if not shot_unavail_ids:
                    break

                retry_batch = [b for b in batch if b["id"] in shot_unavail_ids]
                try:
                    retry_response = await async_run_python_script(
                        "check_ppf.py",
                        {
                            "signals": retry_batch,
                            "pulse": fallback_shot,
                            "owner": default_owner,
                        },
                        ssh_host=ssh_host,
                        timeout=120,
                        python_command=config.get("python_command", "python3"),
                        setup_commands=config.get("setup_commands"),
                    )
                    retry_data = json.loads(retry_response.strip().split("\n")[-1])

                    still_unavail = []
                    for r in retry_data.get("results", []):
                        rid = r["id"]
                        prev = results_by_id.get(rid, {})
                        checked = prev.get("checked_shots", [])
                        checked.append(fallback_shot)

                        if r.get("success", False):
                            results_by_id[rid] = {
                                "signal_id": rid,
                                "valid": True,
                                "shape": r.get("shape"),
                                "dtype": r.get("dtype"),
                                "shot": fallback_shot,
                                "checked_shots": checked,
                            }
                        elif "ier=210002" in r.get("error", ""):
                            results_by_id[rid]["checked_shots"] = checked
                            still_unavail.append(rid)
                        else:
                            results_by_id[rid] = {
                                "signal_id": rid,
                                "valid": False,
                                "error": r.get("error", ""),
                                "checked_shots": checked,
                            }

                    shot_unavail_ids = still_unavail
                except Exception as e:
                    logger.warning(
                        "PPF retry check for shot %d failed: %s", fallback_shot, e
                    )

            # Mark remaining unavailable signals with descriptive error
            for rid in shot_unavail_ids:
                prev = results_by_id.get(rid, {})
                checked = prev.get("checked_shots", shots_to_try)
                results_by_id[rid] = {
                    "signal_id": rid,
                    "valid": False,
                    "error": f"not_available_for_shot (checked {len(checked)} shots)",
                    "checked_shots": checked,
                }

            return list(results_by_id.values())
        except Exception as e:
            logger.error("PPF check failed: %s", e)
            return [
                {"signal_id": s.id, "valid": False, "error": str(e)[:200]}
                for s in signals
            ]


# Auto-register on import
register_scanner(PPFScanner())
