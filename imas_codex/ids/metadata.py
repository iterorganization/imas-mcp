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
