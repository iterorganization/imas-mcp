"""MetadataContext builder for IDS properties & code metadata population.

Provides programmatic gathering of deterministic metadata fields that
populate ``ids_properties/*`` and ``code/*`` IMAS IDS sections.

Typical usage::

    from imas_codex.ids.metadata import build_metadata_context, populate_deterministic_fields
    from imas_codex.graph.client import GraphClient

    with GraphClient.from_profile() as gc:
        ctx = build_metadata_context("jet", "pf_active", gc=gc, dd_version="4.0.0")
    fields = populate_deterministic_fields(ctx, "pf_active")
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from imas_codex.graph.client import GraphClient

# settings imported at call-site to avoid circular imports

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_KEY_DEPS = ["imas", "numpy", "pydantic", "litellm"]


def _pkg_version(name: str) -> str:
    """Return installed version of *name*, or ``'unknown'``."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _git_output(*args: str) -> str:
    """Run a git sub-command and return stripped stdout, or ``'unknown'``."""
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:  # noqa: BLE001 — subprocess errors are non-fatal
        pass
    return "unknown"


def _collect_library_deps() -> list[dict[str, str]]:
    """Return a list of metadata dicts for key pipeline dependencies."""
    deps: list[dict[str, str]] = []
    for name in _KEY_DEPS:
        version = _pkg_version(name)
        if version == "unknown":
            continue
        try:
            meta = importlib.metadata.metadata(name)
            repo = (
                meta.get("Home-page")
                or next(
                    (
                        url.split(", ")[1]
                        for url in (meta.get_all("Project-URL") or [])
                        if url.lower().startswith("source")
                    ),
                    "unknown",
                )
                or "unknown"
            )
            description = meta.get("Summary") or ""
        except Exception:  # noqa: BLE001
            repo = "unknown"
            description = ""
        deps.append(
            {
                "name": name,
                "version": version,
                "repository": repo,
                "description": description,
                "commit": "unknown",
            }
        )
    return deps


# ---------------------------------------------------------------------------
# MetadataContext dataclass
# ---------------------------------------------------------------------------


@dataclass
class MetadataContext:
    """Context for populating ``ids_properties`` and ``code`` fields.

    Attributes:
        dd_version: IMAS Data Dictionary version string (e.g. ``"4.0.0"``).
        access_layer_version: Installed ``imas`` package version or ``"unknown"``.
        creation_date: ISO 8601 timestamp of context creation.
        provider: Facility name / operator string.
        source: Human-readable source tag, e.g. ``"imas-codex v4.1.0"``.
        pipeline_version: ``imas-codex`` package version.
        pipeline_commit: Git HEAD commit hash of the running codebase.
        pipeline_repo: URL of the ``origin`` remote.
        pipeline_description: Package summary / description string.
        pipeline_config: Mapping configuration dict passed by the caller.
        library_deps: Key dependency records with name/version/repository/
            description/commit entries.
    """

    dd_version: str
    access_layer_version: str
    creation_date: str
    provider: str
    source: str
    pipeline_version: str
    pipeline_commit: str
    pipeline_repo: str
    pipeline_description: str
    pipeline_config: dict[str, Any] = field(default_factory=dict)
    library_deps: list[dict[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_metadata_context(
    facility: str,
    ids_name: str,  # noqa: ARG001 — reserved for future per-IDS specialisation
    *,
    gc: GraphClient,
    dd_version: str,
) -> MetadataContext:
    """Gather all known values and return a populated :class:`MetadataContext`.

    Args:
        facility: Facility identifier (e.g. ``"jet"``).
        ids_name: Target IDS name (e.g. ``"pf_active"``).  Currently unused
            but included for future per-IDS specialisation.
        gc: Open :class:`~imas_codex.graph.client.GraphClient` instance.
        dd_version: Requested DD version string.  The function attempts to
            resolve it against the graph; if not found it uses the supplied
            string as-is.

    Returns:
        A fully populated :class:`MetadataContext`.
    """
    # --- DD version (validate / normalise via graph) -----------------------
    resolved_dd: str = dd_version
    try:
        rows = gc.query(
            "MATCH (v:DDVersion) WHERE v.id = $ver OR v.version = $ver RETURN v.version AS ver LIMIT 1",
            ver=dd_version,
        )
        if rows:
            resolved_dd = rows[0].get("ver", dd_version)
    except Exception:  # noqa: BLE001 — graph may be unavailable; non-fatal
        logger.debug("Could not resolve DDVersion for %r from graph", dd_version)

    # --- Package / build metadata ------------------------------------------
    pipeline_version = _pkg_version("imas-codex")

    try:
        meta = importlib.metadata.metadata("imas-codex")
        pipeline_description = (
            meta.get("Summary") or "An IMAS Data Dictionary MCP server"
        )
    except importlib.metadata.PackageNotFoundError:
        pipeline_description = "An IMAS Data Dictionary MCP server"

    pipeline_commit = _git_output("rev-parse", "HEAD")
    pipeline_repo = _git_output("config", "--get", "remote.origin.url")

    # --- Access layer version -----------------------------------------------
    access_layer_version = _pkg_version("imas")

    # --- Library deps -------------------------------------------------------
    library_deps = _collect_library_deps()

    # --- Timestamps & provenance --------------------------------------------
    creation_date = datetime.now(UTC).isoformat()
    source = f"imas-codex v{pipeline_version}"

    return MetadataContext(
        dd_version=resolved_dd,
        access_layer_version=access_layer_version,
        creation_date=creation_date,
        provider=facility,
        source=source,
        pipeline_version=pipeline_version,
        pipeline_commit=pipeline_commit,
        pipeline_repo=pipeline_repo,
        pipeline_description=pipeline_description,
        library_deps=library_deps,
    )


# ---------------------------------------------------------------------------
# Field population
# ---------------------------------------------------------------------------


def populate_deterministic_fields(
    ctx: MetadataContext,
    ids_name: str,  # noqa: ARG001 — reserved for future per-IDS specialisation
) -> dict[str, Any]:
    """Return a mapping of IDS-relative paths → deterministic values.

    All paths are relative to the IDS root (i.e. no leading IDS name prefix).

    A special key ``"_library_entries"`` is included whose value is a
    :class:`list` of dicts, each representing one ``code/library`` array
    element with keys ``name``, ``version``, ``repository``, ``description``,
    and ``commit``.

    Args:
        ctx: Populated :class:`MetadataContext`.
        ids_name: Target IDS name — reserved for future specialisation.

    Returns:
        Dict mapping IMAS relative paths to values, plus ``"_library_entries"``.
    """
    fields: dict[str, Any] = {
        # ids_properties/version_put -----------------------------------------
        "ids_properties/version_put/data_dictionary": ctx.dd_version,
        "ids_properties/version_put/access_layer": ctx.access_layer_version,
        "ids_properties/version_put/access_layer_language": "python",
        # ids_properties metadata --------------------------------------------
        "ids_properties/creation_date": ctx.creation_date,
        "ids_properties/provider": ctx.provider,
        "ids_properties/source": ctx.source,
        # code/* -------------------------------------------------------------
        "code/name": "imas-codex",
        "code/version": ctx.pipeline_version,
        "code/repository": ctx.pipeline_repo,
        "code/commit": ctx.pipeline_commit,
        "code/description": ctx.pipeline_description,
        "code/parameters": json.dumps(ctx.pipeline_config, separators=(",", ":")),
        # Library entries — returned as a list (array-of-structures) ---------
        "_library_entries": list(ctx.library_deps),
    }
    return fields


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class IDSMetadataResult(BaseModel):
    """Complete metadata population result for one IDS.

    Attributes:
        facility: Facility identifier.
        ids_name: Target IDS name.
        dd_version: Data Dictionary version used.
        deterministic_fields: Programmatically derived path → value mapping.
        llm_fields: LLM-generated path → value mapping (populated later).
        cost_usd: Total LLM cost in USD (default ``0.0``).
        tokens: Total LLM tokens consumed (default ``0``).
    """

    facility: str
    ids_name: str
    dd_version: str
    deterministic_fields: dict[str, Any]
    llm_fields: dict[str, Any]
    cost_usd: float = 0.0
    tokens: int = 0


# ---------------------------------------------------------------------------
# Signals summary helper
# ---------------------------------------------------------------------------


def _format_signals_summary(signals: list[dict] | None) -> str:
    """Format a list of mapped signal dicts as a human-readable summary.

    Caps output at 50 entries to keep prompts within token budgets.

    Args:
        signals: List of signal mapping dicts with optional keys
            ``source_id``, ``target_id``, and ``confidence``.

    Returns:
        Multi-line string, or ``"No signal mappings available."`` if
        *signals* is empty or ``None``.
    """
    if not signals:
        return "No signal mappings available."
    lines: list[str] = []
    for sig in signals[:50]:
        src = sig.get("source_id", "unknown")
        tgt = sig.get("target_id", "unknown")
        conf = sig.get("confidence", 0)
        lines.append(f"  - {src} → {tgt} (confidence: {conf:.2f})")
    summary = "\n".join(lines)
    if len(signals) > 50:
        summary += f"\n  ... and {len(signals) - 50} more"
    return summary


# ---------------------------------------------------------------------------
# Main entry point — Stage 3 of the mapping pipeline
# ---------------------------------------------------------------------------


def populate_metadata(
    facility: str,
    ids_name: str,
    *,
    gc: GraphClient,
    dd_version: str,
    mapped_signals: list[dict[str, Any]] | None = None,
    model: str | None = None,
    pipeline_config: dict[str, Any] | None = None,
) -> IDSMetadataResult:
    """Orchestrate both programmatic and LLM-based metadata population.

    This is the main entry point for Stage 3 of the mapping pipeline.
    It builds a :class:`MetadataContext`, populates deterministic fields,
    then calls the LLM to infer fields that require reasoning about the
    mapped signals (``comment``, ``occurrence_type``, ``provenance_sources``,
    ``homogeneous_time``).

    Args:
        facility: Facility identifier (e.g. ``"jet"``).
        ids_name: Target IDS name (e.g. ``"pf_active"``).
        gc: Open :class:`~imas_codex.graph.client.GraphClient` instance.
        dd_version: IMAS Data Dictionary version string.
        mapped_signals: Optional list of signal mapping dicts produced by
            earlier pipeline stages.  Each dict may contain ``source_id``,
            ``target_id``, and ``confidence`` keys.
        model: LLM model identifier to use.  Falls back to
            ``settings.get_model("language")`` when ``None``.
        pipeline_config: Arbitrary pipeline configuration dict stored in
            ``code/parameters``.

    Returns:
        Fully populated :class:`IDSMetadataResult`.  If the LLM call fails,
        ``llm_fields`` is empty and the result is still returned (with a
        warning logged).
    """
    # 1 — Build context
    ctx = build_metadata_context(facility, ids_name, gc=gc, dd_version=dd_version)
    if pipeline_config:
        ctx.pipeline_config = pipeline_config

    # 2 — Deterministic fields
    deterministic = populate_deterministic_fields(ctx, ids_name)

    # 3 — Fetch IDS description from graph
    try:
        rows = gc.query(
            "MATCH (ids:IDS {id: $ids_name}) RETURN ids.documentation AS doc",
            ids_name=ids_name,
        )
        ids_description: str = rows[0]["doc"] if rows else f"IMAS IDS: {ids_name}"
    except Exception:  # noqa: BLE001 — non-fatal graph lookup
        logger.debug("Could not fetch IDS description for %r from graph", ids_name)
        ids_description = f"IMAS IDS: {ids_name}"

    # 4 — Format signals summary
    signals_summary = _format_signals_summary(mapped_signals)

    # 5 — Format deterministic fields summary (skip internal keys)
    det_summary_lines: list[str] = []
    for path, value in deterministic.items():
        if path.startswith("_"):
            continue  # e.g. _library_entries
        det_summary_lines.append(f"  - {path}: {value}")
    det_summary = "\n".join(det_summary_lines)

    # 6 — Render prompts and call LLM (lazy imports to avoid circular deps)
    llm_fields: dict[str, Any] = {}
    cost: float = 0.0
    tokens: int = 0
    try:
        from imas_codex.discovery.base.llm import call_llm_structured
        from imas_codex.ids.models import MetadataPopulationResponse
        from imas_codex.llm.prompt_loader import render_prompt
        from imas_codex.settings import get_model

        llm_model = model or get_model("language")

        system_prompt = render_prompt("mapping/metadata_population_system")
        user_prompt = render_prompt(
            "mapping/metadata_population",
            {
                "facility": facility,
                "ids_name": ids_name,
                "ids_description": ids_description,
                "mapped_signals_summary": signals_summary,
                "deterministic_fields_summary": det_summary,
                "dd_version": dd_version,
            },
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        llm_response, cost, tokens = call_llm_structured(
            model=llm_model,
            messages=messages,
            response_model=MetadataPopulationResponse,
        )

        # 7 — Convert LLM response to IMAS path → value dict
        llm_fields = {
            "ids_properties/comment": llm_response.comment,
            "ids_properties/occurrence_type/name": llm_response.occurrence_type_name,
            "ids_properties/occurrence_type/index": llm_response.occurrence_type_index,
            "ids_properties/occurrence_type/description": llm_response.occurrence_type_description,
            "ids_properties/homogeneous_time": llm_response.homogeneous_time,
        }
        if llm_response.provenance_sources:
            llm_fields["ids_properties/provenance/node/sources"] = (
                llm_response.provenance_sources
            )

    except Exception:  # noqa: BLE001
        logger.warning(
            "LLM metadata population failed for %s/%s — returning empty llm_fields",
            facility,
            ids_name,
            exc_info=True,
        )

    # 8 — Return result
    return IDSMetadataResult(
        facility=facility,
        ids_name=ids_name,
        dd_version=dd_version,
        deterministic_fields=deterministic,
        llm_fields=llm_fields,
        cost_usd=cost,
        tokens=tokens,
    )


def persist_metadata(
    result: IDSMetadataResult,
    mapping_id: str,
    *,
    gc: GraphClient,
) -> None:
    """Persist IDS metadata on the IMASMapping node.

    Stores the complete metadata (deterministic + LLM fields) as JSON
    properties on the already-existing IMASMapping node created by
    ``persist_mapping_result()``.

    Args:
        result: The metadata population result.
        mapping_id: The IMASMapping node id (e.g. "jet:pf_active").
        gc: Open GraphClient instance.
    """
    # Combine all fields (deterministic + LLM), excluding internal keys
    all_fields: dict[str, Any] = {}
    for k, v in result.deterministic_fields.items():
        if not k.startswith("_"):
            all_fields[k] = v
    all_fields.update(result.llm_fields)

    # Separate into ids_properties and code categories
    ids_properties = {
        k: v for k, v in all_fields.items() if k.startswith("ids_properties/")
    }
    code_fields = {k: v for k, v in all_fields.items() if k.startswith("code/")}
    library_entries = result.deterministic_fields.get("_library_entries", [])

    gc.query(
        """
        MATCH (m:IMASMapping {id: $mapping_id})
        SET m.ids_properties_metadata = $ids_props,
            m.code_metadata = $code_meta,
            m.library_metadata = $library_meta,
            m.metadata_populated = true
        """,
        mapping_id=mapping_id,
        ids_props=json.dumps(ids_properties),
        code_meta=json.dumps(code_fields),
        library_meta=json.dumps(library_entries),
    )

    logger.info(
        "Persisted metadata for %s: %d ids_properties fields, %d code fields, %d libraries",
        mapping_id,
        len(ids_properties),
        len(code_fields),
        len(library_entries),
    )
