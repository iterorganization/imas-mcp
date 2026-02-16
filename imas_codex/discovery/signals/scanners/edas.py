"""EDAS (Experiment Data Access System) scanner plugin for JT-60SA.

EDAS is JT-60SA's primary data access system. Data is organized by:
  (shot, category, data_name) — e.g., ('E012345', 'EDDB', 'tesTime')

Discovery strategy:
  1. SSH to JT-60SA, use eddbreadCatTable() to enumerate categories
  2. For each category, use eddbreadTable() to enumerate data names
     with their units, descriptions, aliases, and shot ranges
  3. Create FacilitySignal per (category, data_name) with EDAS accessor

Key insight: eddbreadTable returns a self-describing catalog — far richer
than TDI .fun parsing. It provides units, descriptions (Japanese/English),
aliases, data class, and shot validity ranges.

Remote execution uses the shared run_python_script() infrastructure
with scripts in imas_codex/remote/scripts/ (enumerate_edas.py, check_edas.py).

Config key: data_sources.edas
Facility: JT-60SA
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


class EDASScanner:
    """Discover signals from JT-60SA EDAS system.

    EDAS discovery strategy:
    1. SSH to JT-60SA, use eddbreadCatTable + eddbreadTable API
    2. The database is self-describing: returns data names, units,
       descriptions (Japanese/English), aliases, and shot ranges
    3. Create FacilitySignal per (category, data_name) with EDAS accessor
    4. Japanese descriptions provide LLM enrichment context

    Config (data_sources.edas):
        api_path: str - Path to EDAS API source files
        header_path: str - Path to C headers with signal definitions
        reference_shot: int - Shot for validation (E-prefix format)
    """

    scanner_type: str = "edas"

    async def scan(
        self,
        facility: str,
        ssh_host: str,
        config: dict[str, Any],
        reference_shot: int | None = None,
    ) -> ScanResult:
        """Discover signals from EDAS via SSH eddbreadTable enumeration.

        Uses remote/scripts/enumerate_edas.py via async_run_python_script()
        for proper SSH execution with JSON encode/decode.
        """
        from imas_codex.remote.executor import async_run_python_script

        ref_shot = reference_shot or config.get("reference_shot")

        if not ref_shot:
            logger.warning(
                "EDAS scanner: no reference_shot configured for %s", facility
            )
            return ScanResult(stats={"error": "no reference_shot configured"})

        # Format shot as E-prefix string if numeric
        shot_str = str(ref_shot)
        if not shot_str.startswith("E"):
            shot_str = f"E{ref_shot:06d}"

        logger.info(
            "EDAS scanner: enumerating signals for shot %s on %s",
            shot_str,
            ssh_host,
        )

        try:
            output = await async_run_python_script(
                "enumerate_edas.py",
                {"ref_shot": shot_str},
                ssh_host=ssh_host,
                timeout=180,
            )
            data = json.loads(output.strip().split("\n")[-1])
        except Exception as e:
            logger.error("EDAS enumeration failed on %s: %s", ssh_host, e)
            return ScanResult(stats={"error": str(e)[:300]})

        if "error" in data:
            logger.error("EDAS enumeration error: %s", data["error"])
            return ScanResult(stats={"error": data["error"]})

        raw_signals = data.get("signals", [])

        # Create DataAccess node
        data_access = DataAccess(
            id=f"{facility}:edas:eddb",
            facility_id=facility,
            name="EDAS EDDB Access",
            method_type="edas",
            library="eddb_pwrapper",
            access_type="local",
            template_python=(
                "from eddb_pwrapper import eddbWrapper\n"
                "db = eddbWrapper()\n"
                "db.opendb('EDDB')\n"
                "data = db.get_time_seriese_data("
                "'{shot}', t1, t2, '{data_name}', '{category}')\n"
                "db.closedb()"
            ),
        )

        # Convert to FacilitySignal nodes
        signals = []
        for raw in raw_signals:
            cat = raw["category"]
            dname = raw["data_name"]
            units = raw.get("units", "")
            description = raw.get("description", "")

            signal_id = f"{facility}:general/{cat.lower()}_{dname.lower()}"
            accessor = f"eddbreadTime('{shot_str}', '{cat}', '{dname}', t1, t2)"

            signals.append(
                FacilitySignal(
                    id=signal_id,
                    facility_id=facility,
                    status=FacilitySignalStatus.discovered,
                    physics_domain="general",  # Enriched by LLM
                    name=f"{cat}/{dname}",
                    accessor=accessor,
                    data_access=data_access.id,
                    units=units,
                    description=description,  # May be Japanese
                    discovery_source="edas_enumeration",
                    example_shot=ref_shot,
                )
            )

        logger.info(
            "EDAS scanner: discovered %d signals from %d categories (shot %s)",
            len(signals),
            data.get("ncats", 0),
            shot_str,
        )

        return ScanResult(
            signals=signals,
            data_access=data_access,
            metadata={
                "reference_shot": shot_str,
                "categories": data.get("categories", []),
                "ncats": data.get("ncats", 0),
            },
            stats={
                "signals_discovered": len(signals),
                "categories_found": data.get("ncats", 0),
                "reference_shot": shot_str,
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
        """Validate EDAS signals return data for reference shot.

        Uses remote/scripts/check_edas.py via async_run_python_script().
        """
        from imas_codex.remote.executor import async_run_python_script

        ref_shot = reference_shot or config.get("reference_shot")
        if not ref_shot:
            return [
                {"signal_id": s.id, "valid": False, "error": "no reference_shot"}
                for s in signals
            ]

        shot_str = str(ref_shot)
        if not shot_str.startswith("E"):
            shot_str = f"E{ref_shot:06d}"

        # Parse category/data_name from signal name
        batch = []
        for s in signals:
            parts = (s.name or "").split("/")
            if len(parts) == 2:
                batch.append({"id": s.id, "category": parts[0], "data_name": parts[1]})
            else:
                batch.append({"id": s.id, "category": "", "data_name": ""})

        try:
            output = await async_run_python_script(
                "check_edas.py",
                {
                    "signals": batch,
                    "ref_shot": shot_str,
                },
                ssh_host=ssh_host,
                timeout=180,
            )
            response = json.loads(output.strip().split("\n")[-1])
            return [
                {
                    "signal_id": r["id"],
                    "valid": r.get("success", False),
                    "dtype": r.get("dtype"),
                    "error": r.get("error"),
                }
                for r in response.get("results", [])
            ]
        except Exception as e:
            logger.error("EDAS check failed: %s", e)
            return [
                {"signal_id": s.id, "valid": False, "error": str(e)[:200]}
                for s in signals
            ]


# Auto-register on import
register_scanner(EDASScanner())
