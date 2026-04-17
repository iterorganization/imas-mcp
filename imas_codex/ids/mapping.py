"""IMAS signal mapping pipeline orchestrator.

Multi-step LLM pipeline that generates signal-level IMAS mappings
from facility signal sources:

  gather_context:        Fetch signal sources + DD context (programmatic)
  assign_targets:        LLM assigns sources to IDS target paths
  map_signals:           For each target, LLM generates signal mappings
  discover_assembly:     For each target, LLM discovers assembly patterns
  validate_mappings:     Programmatic validation (source/target existence, transforms, units)
  derive_error_mappings: Derive error field mappings via HAS_ERROR graph traversal (no LLM)
  populate_metadata:     Populate ids_properties and code metadata (programmatic + LLM)
  persist:               Write to graph

Usage:
    from imas_codex.ids.mapping import generate_mapping

    result = generate_mapping("jet", "pf_active")
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from imas_codex.graph.client import GraphClient
from imas_codex.ids.metadata import (
    IDSMetadataResult,
    persist_metadata,
    populate_metadata,
)
from imas_codex.ids.models import (
    AssemblyBatch,
    AssemblyConfig,
    AssemblyPattern,
    EscalationFlag,
    SignalMappingBatch,
    TargetAssignmentBatch,
    TargetType,
    UnmappedSignal,
    ValidatedMappingResult,
    ValidatedSignalMapping,
    persist_mapping_result,
)
from imas_codex.ids.tools import (
    _run_async,
    analyze_units,
    compute_semantic_matches,
    fetch_code_context,
    fetch_cross_facility_mappings,
    fetch_imas_fields,
    fetch_imas_subtree,
    fetch_source_code_refs,
    fetch_wiki_context,
    get_sign_flip_paths,
    query_ids_physics_domains,
    query_signal_sources,
    search_existing_mappings,
    search_imas_semantic,
)
from imas_codex.settings import get_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cost tracker
# ---------------------------------------------------------------------------


@dataclass
class PipelineCost:
    """Accumulated cost across pipeline steps."""

    steps: dict[str, float] = field(default_factory=dict)
    total_tokens: int = 0

    @property
    def total_usd(self) -> float:
        return sum(self.steps.values())

    def add(self, step: str, cost: float, tokens: int) -> None:
        self.steps[step] = self.steps.get(step, 0) + cost
        self.total_tokens += tokens


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


def _render_prompt(name: str, **context: Any) -> str:
    """Render a mapping prompt template with Jinja2 substitutions."""
    from imas_codex.llm.prompt_loader import render_prompt

    return render_prompt(f"mapping/{name}", context)


def _render_system_prompt(name: str) -> str:
    """Render a static system-level mapping prompt (no context variables)."""
    from imas_codex.llm.prompt_loader import render_prompt

    return render_prompt(f"mapping/{name}")


def _build_messages(system_name: str, user_prompt: str) -> list[dict[str, str]]:
    """Build [system, user] message pair for LLM calls.

    Uses a dedicated static system prompt (cacheable across calls)
    and the already-rendered dynamic user prompt.
    """
    return [
        {"role": "system", "content": _render_system_prompt(system_name)},
        {"role": "user", "content": user_prompt},
    ]


def _format_subtree(rows: list[dict[str, Any]]) -> str:
    """Format IMAS subtree rows into a readable tree summary."""
    lines: list[str] = []
    for r in rows:
        path = r.get("id", "")
        dtype = r.get("data_type", "")
        units = r.get("units", "")
        doc = r.get("documentation", "")
        parts = [path]
        if dtype:
            parts.append(f"({dtype})")
        if units:
            parts.append(f"[{units}]")
        line = " ".join(parts)
        if doc:
            line += f" — {doc}"
        lines.append(line)
    return "\n".join(lines) if lines else "(no paths)"


def _format_section_clusters(clusters: list[dict[str, Any]]) -> str:
    """Format section clusters for the section assignment prompt."""
    if not clusters:
        return "(no cluster data available)"
    lines: list[str] = []
    for c in clusters:
        label = c.get("label", "")
        desc = c.get("description", "")
        paths = c.get("paths", [])
        line = f"- **{label}**"
        if desc:
            line += f": {desc}"
        if paths:
            line += f"\n  Paths: {', '.join(paths)}"
        lines.append(line)
    return "\n".join(lines)


def _format_cross_facility_mappings(rows: list[dict[str, Any]]) -> str:
    """Format cross-facility mapping precedent for the prompt."""
    if not rows:
        return ""
    by_facility: dict[str, list[str]] = {}
    for r in rows:
        fac = r.get("facility", "?")
        path = r.get("target_path", "")
        by_facility.setdefault(fac, []).append(path)
    lines: list[str] = []
    for fac, paths in sorted(by_facility.items()):
        lines.append(f"- **{fac}**: {', '.join(sorted(paths))}")
    return "\n".join(lines)


def _format_sources(groups: list[dict[str, Any]]) -> str:
    """Format signal sources into a readable summary."""
    lines: list[str] = []
    for g in groups:
        gid = g.get("id", "")
        key = g.get("group_key", "")
        desc = g.get("description", "")
        members = g.get("member_count", 0)
        domain = g.get("physics_domain", "")
        existing = g.get("imas_mappings", [])
        mapped = [m for m in existing if m.get("target_id")]
        line = f"- {gid}"
        if domain:
            line += f" (domain={domain}, members={members})"
        else:
            line += f" (key={key}, members={members})"
        if desc:
            line += f": {desc}"
        # Enriched metadata from representative signal
        rep_desc = g.get("rep_description")
        if rep_desc and rep_desc != desc:
            line += f"\n  Representative: {rep_desc}"
        rep_unit = g.get("rep_unit")
        if rep_unit:
            line += f"\n  Unit: {rep_unit}"
        rep_cocos = g.get("rep_cocos")
        if rep_cocos:
            line += f"\n  COCOS: {rep_cocos}"
        rep_sign = g.get("rep_sign_convention")
        if rep_sign:
            line += f"\n  Sign convention: {rep_sign}"
        accessors = g.get("sample_accessors")
        if accessors:
            line += f"\n  Accessors: {', '.join(str(a) for a in accessors)}"
        if mapped:
            targets = ", ".join(m["target_id"] for m in mapped)
            line += f"\n  [already mapped → {targets}]"
        lines.append(line)
    return "\n".join(lines) if lines else "(no sources)"


def _format_source_detail(source: dict[str, Any]) -> str:
    """Format a single signal source as structured markdown for the prompt."""
    parts: list[str] = []
    sid = source.get("id", "unknown")
    parts.append(f"**Source ID**: {sid}")
    desc = source.get("description", "")
    if desc:
        parts.append(f"**Description**: {desc}")
    rep_desc = source.get("rep_description")
    if rep_desc and rep_desc != desc:
        parts.append(f"**Representative Signal**: {rep_desc}")
    domain = source.get("physics_domain")
    if domain:
        parts.append(f"**Physics Domain**: {domain}")
    parts.append(f"**Units**: {source.get('rep_unit') or 'unknown'}")
    sign = source.get("rep_sign_convention")
    parts.append(f"**Sign Convention**: {sign or 'unknown'}")
    cocos = source.get("rep_cocos")
    parts.append(f"**COCOS**: {cocos or 'not set'}")
    members = source.get("member_count", 0)
    parts.append(f"**Members**: {members} signals")
    key = source.get("group_key", "")
    if key:
        parts.append(f"**Accessor Pattern**: {key}")
    accessors = source.get("sample_accessors")
    if accessors:
        parts.append(f"**Sample Accessors**: {', '.join(str(a) for a in accessors)}")
    existing = source.get("imas_mappings", [])
    mapped = [m for m in existing if m.get("target_id")]
    if mapped:
        targets = ", ".join(m["target_id"] for m in mapped)
        parts.append(f"**Existing Mappings**: {targets}")
    return "\n".join(parts)


def _format_fields(fields: list[dict[str, Any]]) -> str:
    """Format IMAS field details for the field mapping prompt."""
    lines: list[str] = []
    for f in fields:
        path = f.get("id", "") or f.get("path", "")
        dtype = f.get("data_type", "")
        units = f.get("units", "")
        doc = f.get("documentation", "")
        ndim = f.get("ndim")
        parts = [f"- {path} ({dtype})"]
        if units:
            parts.append(f"[{units}]")
        if ndim is not None:
            parts.append(f"ndim={ndim}")
        line = " ".join(parts)
        if doc:
            line += f"\n  {doc}"
        lines.append(line)
    return "\n".join(lines) if lines else "(no fields)"


def _format_identifier_schemas(fields: list[dict[str, Any]]) -> str:
    """Extract and format identifier schemas from field data."""
    lines: list[str] = []
    for f in fields:
        schema = f.get("identifier_schema")
        if not schema:
            continue
        path = f.get("id", "") or f.get("path", "")
        if isinstance(schema, dict):
            schema_path = schema.get("schema_path", "")
            doc = schema.get("documentation", "")
            options = schema.get("options", [])
        else:
            # Pydantic model
            schema_path = schema.schema_path if hasattr(schema, "schema_path") else ""
            doc = schema.documentation if hasattr(schema, "documentation") else ""
            options = schema.options if hasattr(schema, "options") else []

        line = f"- **{path}** (schema: {schema_path})"
        if doc:
            line += f"\n  {doc}"
        if options:
            opt_lines = []
            for opt in options:
                if isinstance(opt, dict):
                    name = opt.get("name", "")
                    idx = opt.get("index", "")
                    desc = opt.get("description", "")
                else:
                    name = opt.name if hasattr(opt, "name") else ""
                    idx = opt.index if hasattr(opt, "index") else ""
                    desc = opt.description if hasattr(opt, "description") else ""
                opt_lines.append(
                    f"    - {idx}: {name}" + (f" — {desc}" if desc else "")
                )
            line += "\n  Valid values:\n" + "\n".join(opt_lines)
        lines.append(line)
    return "\n".join(lines) if lines else "(no identifier schemas)"


def _format_version_context(version_ctx: dict[str, Any]) -> str:
    """Format version change context for the signal mapping prompt."""
    from imas_codex.models.error_models import ToolError

    if isinstance(version_ctx, ToolError):
        return f"(version change history unavailable: {version_ctx.error})"

    paths_data = version_ctx.get("paths", {})
    not_found = version_ctx.get("not_found", [])
    paths_without_changes = version_ctx.get("paths_without_changes", [])

    if not paths_data and not not_found:
        return "(no version change history)"

    lines: list[str] = []
    for path_id, ctx in paths_data.items():
        changes = ctx.get("notable_changes", [])
        if not changes:
            continue
        lines.append(f"- **{path_id}** ({len(changes)} change(s)):")
        for c in changes:
            version = c.get("version", "?")
            ctype = c.get("type", "?")
            summary = c.get("summary", "")
            lines.append(f"  - v{version} [{ctype}]: {summary}")

    if paths_without_changes:
        lines.append(
            "Paths without notable changes: " + ", ".join(paths_without_changes)
        )
    if not_found:
        lines.append("Paths not found in DD graph: " + ", ".join(not_found))

    return "\n".join(lines) if lines else "(no notable version changes)"


def _format_coordinate_context(fields: list[dict[str, Any]]) -> str:
    """Format coordinate spec data from fields for the assembly prompt.

    Includes coordinate axis references, timebase info, and shared grid
    relationships so the LLM can determine array sizing and dimension ordering.
    """
    lines: list[str] = []
    shared_grids: dict[str, list[str]] = {}  # coordinate_ref → [field_paths]

    for f in fields:
        path = f.get("id", "") or f.get("path", "")
        data_type = f.get("data_type", "")
        coords = f.get("coordinates", [])
        coord1 = f.get("coordinate1")
        coord2 = f.get("coordinate2")
        timebase = f.get("timebase")

        # Skip fields with no dimensionality info
        if not coords and not coord1 and not coord2 and not timebase:
            continue

        parts = [f"- **{path}** ({data_type})"]

        # Dimension-specific coordinate axes
        dim_info: list[str] = []
        if coord1:
            dim_info.append(f"dim1={coord1}")
            shared_grids.setdefault(coord1, []).append(path)
        if coord2:
            dim_info.append(f"dim2={coord2}")
            shared_grids.setdefault(coord2, []).append(path)
        if dim_info:
            parts.append(f"  axes: {', '.join(dim_info)}")

        # Index-based coordinate specs (bounded/unbounded)
        if coords:
            parts.append(f"  specs: {', '.join(coords)}")

        # Timebase reference
        if timebase:
            parts.append(f"  timebase: {timebase}")

        lines.append("\n".join(parts))

    # Append shared grid summary — fields sharing coordinate axes
    # should be sized consistently
    shared = {ref: paths for ref, paths in shared_grids.items() if len(paths) > 1}
    if shared:
        lines.append("\n**Shared coordinate grids** (size these consistently):")
        for ref, paths in sorted(shared.items()):
            lines.append(f"- `{ref}` shared by: {', '.join(paths)}")

    return "\n".join(lines) if lines else "(no coordinate spec data)"


def _format_unit_analysis(
    groups: list[dict[str, Any]], fields: list[dict[str, Any]]
) -> str:
    """Run unit analysis between signal sources and IMAS fields.

    Uses the rep_unit field (dot-exp format from normalize_unit_symbol)
    instead of extracting units from keywords.
    """
    from imas_codex.units import normalize_unit_symbol

    lines: list[str] = []
    for g in groups:
        signal_unit = g.get("rep_unit")
        if not signal_unit:
            continue
        # Normalize to dot-exp
        signal_unit = normalize_unit_symbol(signal_unit) or signal_unit
        for f in fields:
            imas_unit = f.get("units")
            if imas_unit:
                imas_unit = normalize_unit_symbol(imas_unit) or imas_unit
                result = analyze_units(signal_unit, imas_unit)
                if result.get("compatible"):
                    factor = result.get("conversion_factor", 1.0)
                    lines.append(
                        f"  {signal_unit} → {imas_unit}: compatible"
                        + (f" (×{factor})" if factor != 1.0 else "")
                    )
                elif result.get("error"):
                    lines.append(f"  {signal_unit} → {imas_unit}: {result['error']}")
    return "\n".join(lines) if lines else "(no unit analysis needed)"


def _format_wiki_context(wiki_items: list[dict[str, Any]]) -> str:
    """Format wiki context for the signal mapping prompt."""
    if not wiki_items:
        return ""
    lines: list[str] = []
    for item in wiki_items:
        title = item.get("page_title", "")
        text = item.get("text", "")
        score = item.get("score_imas_relevance", 0)
        if title:
            lines.append(f"**{title}** (IMAS relevance: {score:.2f})")
        if text:
            # Truncate long chunks
            snippet = text[:500] + "..." if len(text) > 500 else text
            lines.append(snippet)
        lines.append("")
    return "\n".join(lines)


def _format_code_context(code_items: list[dict[str, Any]]) -> str:
    """Format code context for the signal mapping prompt."""
    if not code_items:
        return ""
    lines: list[str] = []
    for item in code_items:
        path = item.get("source_file", "")
        func = item.get("function_name", "")
        text = item.get("text", "")
        lang = item.get("language", "")
        score = item.get("score_data_access", 0)
        header = path
        if func:
            header += f"::{func}"
        if score:
            header += f" (data_access: {score:.2f})"
        lines.append(header)
        if text:
            snippet = text[:800] + "..." if len(text) > 800 else text
            lines.append(f"```{lang}\n{snippet}\n```")
        lines.append("")
    return "\n".join(lines)


def _format_semantic_match_matrix(
    matrix: dict[str, list[dict[str, Any]]],
    source_id: str,
) -> str:
    """Format the semantic match matrix for a single source."""
    matches = matrix.get(source_id, [])
    if not matches:
        return ""
    lines: list[str] = []
    for m in matches:
        content_type = m.get("content_type", "")
        target = m.get("target_id", "")
        score = m.get("score", 0)
        excerpt = m.get("excerpt", "")
        type_label = {"imas": "IMAS", "wiki": "Wiki", "code": "Code"}.get(
            content_type, content_type.upper()
        )
        line = f"  - {type_label}: {target} ({score:.3f})"
        if excerpt:
            line += f" — {excerpt}"
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def _call_llm(
    messages: list[dict[str, str]],
    response_model: type,
    *,
    model: str | None = None,
    step_name: str = "",
    cost: PipelineCost | None = None,
) -> Any:
    """Call LLM with structured output parsing."""
    from imas_codex.discovery.base.llm import call_llm_structured

    llm_model = model or get_model("language")
    logger.info("Step %s: calling %s", step_name, llm_model)

    result, usd, tokens = call_llm_structured(
        llm_model,
        messages,
        response_model,
        service="imas-mapping",
    )

    if cost:
        cost.add(step_name, usd, tokens)

    logger.info("Step %s: %d tokens, $%.4f", step_name, tokens, usd)
    return result


async def _acall_llm(
    messages: list[dict[str, str]],
    response_model: type,
    *,
    model: str | None = None,
    step_name: str = "",
    cost: PipelineCost | None = None,
) -> Any:
    """Async LLM call with structured output parsing."""
    from imas_codex.discovery.base.llm import acall_llm_structured

    llm_model = model or get_model("language")
    logger.info("Step %s: calling %s (async)", step_name, llm_model)

    result, usd, tokens = await acall_llm_structured(
        llm_model,
        messages,
        response_model,
        service="imas-mapping",
    )

    if cost:
        cost.add(step_name, usd, tokens)

    logger.info("Step %s: %d tokens, $%.4f", step_name, tokens, usd)
    return result


def gather_shared_context(
    facility: str,
    all_ids_names: list[str],
    *,
    gc: GraphClient,
    dd_version: int | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Gather domain-level context shared across all IDS targets.

    This is the expensive part — batch embedding + source queries — done
    ONCE regardless of how many IDS are being mapped.
    """
    import time as _time

    def _emit(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    t0 = _time.monotonic()

    # Collect ALL unique physics domains across all target IDS
    _emit("querying physics domains")
    all_domains: set[str] = set()
    ids_domains: dict[str, list[str]] = {}
    for ids_name in all_ids_names:
        domains = query_ids_physics_domains(ids_name, gc=gc, dd_version=dd_version)
        ids_domains[ids_name] = domains
        all_domains.update(domains)

    domain_list = sorted(all_domains)
    _emit(f"{len(domain_list)} domains across {len(all_ids_names)} IDS")

    # Query ALL signal sources for the union of all domains (ONCE)
    if domain_list:
        groups = query_signal_sources(
            facility,
            gc=gc,
            physics_domains=domain_list,
            status_filter="enriched",
        )
    else:
        groups = query_signal_sources(facility, gc=gc, status_filter="enriched")

    t_sources = _time.monotonic()
    _emit(f"{len(groups)} sources ({t_sources - t0:.1f}s)")

    # Batch embed ALL source descriptions ONCE
    source_descs = [
        (g["id"], g.get("rep_description") or g.get("description") or "")
        for g in groups
        if g.get("rep_description") or g.get("description")
    ]

    embeddings = None
    if source_descs:
        from imas_codex.embeddings.encoder import Encoder

        encoder = Encoder()
        _emit(f"embedding {len(source_descs)} sources")
        texts = [desc for _, desc in source_descs]
        embeddings = encoder.embed_texts(texts)
        t_embed = _time.monotonic()
        _emit(f"embedded {len(source_descs)} sources ({t_embed - t_sources:.1f}s)")

    # Cluster searcher (loaded once, used per-IDS)
    cluster_searcher = None
    try:
        from imas_codex.clusters.search import ClusterSearcher

        cluster_searcher = ClusterSearcher.load()
    except Exception:
        logger.debug("Cluster searcher unavailable")

    # Wiki + code context (domain-scoped, shared across IDS)
    _emit("wiki + code context")
    wiki_context: list[dict[str, Any]] = []
    code_context: list[dict[str, Any]] = []
    if domain_list:
        try:
            wiki_context = fetch_wiki_context(
                facility,
                domain_list,
                min_imas_relevance=0.5,
                k=15,
                gc=gc,
            )
        except Exception:
            logger.debug("Wiki context fetch failed")
        try:
            code_context = fetch_code_context(
                facility,
                domain_list,
                score_dimension="score_data_access",
                min_score=0.5,
                k=15,
                gc=gc,
            )
        except Exception:
            logger.debug("Code context fetch failed")

    # DD COCOS convention
    dd_cocos: int | None = None
    try:
        cocos_rows = gc.query(
            """
            MATCH (v:DDVersion)
            WHERE v.major = $dd_major
            RETURN v.cocos AS cocos
            LIMIT 1
            """,
            dd_major=dd_version,
        )
        if cocos_rows and cocos_rows[0].get("cocos"):
            dd_cocos = cocos_rows[0]["cocos"]
    except Exception:
        pass

    # Per-source vector queries — run ONCE with NO IDS filter.
    # Results are post-filtered by IDS prefix in gather_ids_context.
    _emit(f"vector search ({len(source_descs)} sources)")
    semantic_match_matrix: dict[str, list[dict[str, Any]]] = {}
    if source_descs and embeddings is not None:
        try:
            semantic_match_matrix = compute_semantic_matches(
                source_descs,
                "",  # No IDS filter — get matches across ALL IDS
                gc=gc,
                k_per_source=20,
                include_wiki=True,
                include_code=True,
                dd_version=dd_version,
                on_progress=on_progress,
                precomputed_embeddings=embeddings,
            )
        except Exception:
            logger.debug("Semantic match matrix failed", exc_info=True)

    t_total = _time.monotonic()
    _emit(
        f"{len(groups)} sources, {len(semantic_match_matrix)} matched, "
        f"{len(wiki_context)} wiki, {len(code_context)} code "
        f"({t_total - t0:.1f}s)"
    )

    return {
        "groups": groups,
        "source_descs": source_descs,
        "embeddings": embeddings,
        "ids_domains": ids_domains,
        "cluster_searcher": cluster_searcher,
        "wiki_context": wiki_context,
        "code_context": code_context,
        "dd_version": dd_version,
        "dd_cocos": dd_cocos,
        "semantic_match_matrix": semantic_match_matrix,
    }


def gather_ids_context(
    facility: str,
    ids_name: str,
    shared: dict[str, Any],
    *,
    gc: GraphClient,
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Gather IDS-specific context using pre-computed shared data.

    Uses the pre-computed ``semantic_match_matrix`` from
    ``gather_shared_context`` — no vector queries here.  Only cheap
    graph lookups (subtree, existing mappings, clusters, COCOS).
    """
    import time as _time

    def _emit(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    dd_version = shared["dd_version"]
    groups = shared["groups"]
    cluster_searcher = shared["cluster_searcher"]

    t0 = _time.monotonic()

    # IDS subtree
    _emit("IDS subtree")
    subtree = fetch_imas_subtree(ids_name, gc=gc, dd_version=dd_version)

    # Global semantic search (1 embed + 1 vector query — cheap)
    semantic_hits: list[dict[str, Any]] = []
    try:
        semantic_hits = search_imas_semantic(
            f"{facility} {ids_name}",
            ids_name,
            gc=gc,
            k=20,
            dd_version=dd_version,
        )
    except Exception:
        logger.warning("Semantic search unavailable")

    # Post-filter pre-computed match matrix by IDS prefix
    _emit("filtering matches")
    ids_prefix = f"{ids_name}/"
    full_matrix = shared.get("semantic_match_matrix", {})
    semantic_match_matrix: dict[str, list[dict[str, Any]]] = {}
    for source_id, matches in full_matrix.items():
        filtered = [
            m
            for m in matches
            if m["content_type"] != "imas" or m["target_id"].startswith(ids_prefix)
        ]
        if filtered:
            semantic_match_matrix[source_id] = filtered

    # Build source_candidates from IMAS hits in the filtered matrix
    source_candidates: dict[str, list[dict[str, Any]]] = {}
    for source_id, matches in semantic_match_matrix.items():
        imas_hits = [
            {
                "id": m["target_id"],
                "score": m["score"],
                "documentation": m["excerpt"],
            }
            for m in matches
            if m["content_type"] == "imas"
        ]
        if imas_hits:
            source_candidates[source_id] = imas_hits

    # Cluster enrichment (using pre-loaded searcher)
    if cluster_searcher:
        for _source_id, candidates in source_candidates.items():
            cluster_additions: list[dict[str, Any]] = []
            existing_ids = {c["id"] for c in candidates}
            for cand in candidates[:3]:
                try:
                    cluster_hits = cluster_searcher.search_by_path(cand["id"])
                    for hit in cluster_hits:
                        for member_path in hit.paths:
                            if member_path not in existing_ids:
                                cluster_additions.append(
                                    {
                                        "id": member_path,
                                        "score": hit.similarity_score
                                        * cand.get("score", 0.5),
                                        "via_cluster": hit.label,
                                        "documentation": f"Cluster member: {hit.description}",
                                    }
                                )
                                existing_ids.add(member_path)
                except Exception:
                    pass
            candidates.extend(cluster_additions)

    # IDS-specific: existing mappings, COCOS, cross-facility, section clusters
    existing = search_existing_mappings(facility, ids_name, gc=gc)
    cocos_paths = get_sign_flip_paths(ids_name)
    cross_mappings = fetch_cross_facility_mappings(ids_name, facility, gc=gc)

    section_clusters: list[dict[str, Any]] = []
    try:
        from imas_codex.tools.graph_search import GraphClustersTool

        clusters_tool = GraphClustersTool(gc)
        cluster_result = _run_async(
            clusters_tool.search_dd_clusters(
                ids_filter=ids_name,
                section_only=True,
                dd_version=dd_version,
            )
        )
        section_clusters = cluster_result.get("clusters", [])
    except Exception:
        pass

    target_domains = shared["ids_domains"].get(ids_name, [])
    t_total = _time.monotonic()
    _emit(
        f"{len(source_candidates)} candidates, "
        f"{sum(len(v) for v in semantic_match_matrix.values())} matches "
        f"({t_total - t0:.1f}s)"
    )

    return {
        "groups": groups,
        "subtree": subtree,
        "semantic": semantic_hits,
        "source_candidates": source_candidates,
        "semantic_match_matrix": semantic_match_matrix,
        "existing": existing,
        "cross_mappings": cross_mappings,
        "cocos_paths": cocos_paths,
        "dd_version": dd_version,
        "dd_cocos": shared["dd_cocos"],
        "section_clusters": section_clusters,
        "target_domains": target_domains,
        "wiki_context": shared["wiki_context"],
        "code_context": shared["code_context"],
    }


def gather_context(
    facility: str,
    ids_name: str,
    *,
    gc: GraphClient,
    dd_version: int | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Gather all context needed for the signal mapping pipeline.

    Convenience wrapper that calls ``gather_shared_context`` +
    ``gather_ids_context`` for a single IDS.  For multi-IDS mapping,
    use the split functions directly to share the expensive batch
    embedding across IDS targets.
    """
    shared = gather_shared_context(
        facility,
        [ids_name],
        gc=gc,
        dd_version=dd_version,
        on_progress=on_progress,
    )
    return gather_ids_context(
        facility,
        ids_name,
        shared,
        gc=gc,
        on_progress=on_progress,
    )


def _fetch_ids_description(ids_name: str, gc: GraphClient) -> str:
    """Fetch the enriched IDS description from the graph."""
    rows = gc.query(
        """
        MATCH (i:IDS {id: $ids_name})
        RETURN i.description AS desc
        """,
        ids_name=ids_name,
    )
    if rows and rows[0].get("desc"):
        return rows[0]["desc"]
    return ""


def assign_targets(
    facility: str,
    ids_name: str,
    context: dict[str, Any],
    *,
    gc: GraphClient | None = None,
    model: str | None = None,
    cost: PipelineCost,
) -> TargetAssignmentBatch:
    """Assign signal sources to IDS target paths."""
    logger.info("Assigning signal sources to IDS target paths")

    ids_description = ""
    if gc is not None:
        ids_description = _fetch_ids_description(ids_name, gc)

    prompt = _render_prompt(
        "target_assignment",
        facility=facility,
        ids_name=ids_name,
        ids_description=ids_description,
        signal_sources=_format_sources(context["groups"]),
        imas_subtree=_format_subtree(context["subtree"]),
        semantic_results=_format_subtree(context["semantic"]),
        section_clusters=_format_section_clusters(context.get("section_clusters", [])),
        cross_facility_mappings=_format_cross_facility_mappings(
            context.get("cross_mappings", [])
        ),
    )

    messages = _build_messages("target_assignment_system", prompt)

    return _call_llm(
        messages,
        TargetAssignmentBatch,
        model=model,
        step_name="assign_targets",
        cost=cost,
    )


async def aassign_targets(
    facility: str,
    ids_name: str,
    context: dict[str, Any],
    *,
    gc: GraphClient | None = None,
    model: str | None = None,
    cost: PipelineCost,
) -> TargetAssignmentBatch:
    """Async version of assign_targets."""
    logger.info("Assigning signal sources to IDS target paths (async)")

    ids_description = ""
    if gc is not None:
        ids_description = _fetch_ids_description(ids_name, gc)

    prompt = _render_prompt(
        "target_assignment",
        facility=facility,
        ids_name=ids_name,
        ids_description=ids_description,
        signal_sources=_format_sources(context["groups"]),
        imas_subtree=_format_subtree(context["subtree"]),
        semantic_results=_format_subtree(context["semantic"]),
        section_clusters=_format_section_clusters(context.get("section_clusters", [])),
        cross_facility_mappings=_format_cross_facility_mappings(
            context.get("cross_mappings", [])
        ),
    )

    messages = _build_messages("target_assignment_system", prompt)

    return await _acall_llm(
        messages,
        TargetAssignmentBatch,
        model=model,
        step_name="assign_targets",
        cost=cost,
    )


def map_signals(
    facility: str,
    ids_name: str,
    sections: TargetAssignmentBatch,
    context: dict[str, Any],
    *,
    gc: GraphClient,
    model: str | None = None,
    cost: PipelineCost,
) -> list[SignalMappingBatch]:
    """Generate signal mappings per section."""
    logger.info("Generating signal mappings for %d targets", len(sections.assignments))

    batches: list[SignalMappingBatch] = []
    dd_ver = context.get("dd_version")

    for assignment in sections.assignments:
        target_path = assignment.imas_target_path

        # Get detailed fields for this target
        fields = fetch_imas_fields(
            ids_name,
            [target_path],
            gc=gc,
            dd_version=dd_ver,
        )
        # Also get leaf fields under this target
        subtree_fields = fetch_imas_subtree(
            ids_name,
            target_path.removeprefix(f"{ids_name}/"),
            gc=gc,
            leaf_only=True,
            dd_version=dd_ver,
        )

        # Find the signal source details
        sg_detail = next(
            (g for g in context["groups"] if g["id"] == assignment.source_id),
            {},
        )

        # Fetch code references for this source
        code_refs = fetch_source_code_refs(assignment.source_id, gc=gc)
        code_context = ""
        if code_refs:
            snippets = []
            for ref in code_refs:
                if ref.get("code"):
                    lang = ref.get("language", "")
                    snippets.append(f"```{lang}\n{ref['code']}\n```")
            if snippets:
                code_context = "\n".join(snippets)

        # Build per-source COCOS context
        source_cocos = sg_detail.get("rep_cocos")
        dd_cocos = context.get("dd_cocos")
        cocos_context = ""
        if source_cocos or dd_cocos:
            parts = []
            if source_cocos:
                parts.append(f"Signal COCOS convention: {source_cocos}")
            if dd_cocos:
                parts.append(f"Target DD COCOS convention: {dd_cocos}")
            flip_paths = [
                p for p in context["cocos_paths"] if p.startswith(target_path)
            ]
            if flip_paths:
                parts.append(
                    "Target IMAS paths requiring sign flip:\n"
                    + "\n".join(f"- {p}" for p in flip_paths)
                )
            cocos_context = "\n".join(parts)

        # Combine fields for identifier schema extraction
        all_fields = subtree_fields or fields

        # Fetch version change history for section fields
        version_context_str = "(no version change history)"
        try:
            from imas_codex.tools.version_tool import VersionTool

            vt = VersionTool(gc)
            field_paths = [
                f.get("id", "") or f.get("path", "")
                for f in all_fields
                if (f.get("id") or f.get("path"))
            ]
            if field_paths:
                version_ctx = _run_async(vt.get_dd_version_context(paths=field_paths))
                version_context_str = _format_version_context(version_ctx)
        except Exception:
            logger.debug("Version context fetch failed — continuing without it")

        # Per-source semantic candidates
        source_id = assignment.source_id
        source_semantic = context.get("source_candidates", {}).get(source_id, [])
        semantic_context = ""
        cluster_context = ""
        if source_semantic:
            semantic_lines = []
            cluster_lines = []
            for sc in source_semantic:
                path = sc.get("id", "")
                score = sc.get("score", 0)
                doc = sc.get("documentation", "")
                via = sc.get("via_cluster")
                if via:
                    cluster_lines.append(
                        f"  - {path} (score={score:.2f}, cluster={via}): {doc}"
                    )
                else:
                    semantic_lines.append(f"  - {path} (score={score:.2f}): {doc}")
            if semantic_lines:
                semantic_context = "Semantic search candidates:\n" + "\n".join(
                    semantic_lines
                )
            if cluster_lines:
                cluster_context = "Cluster-derived candidates:\n" + "\n".join(
                    cluster_lines
                )

        # Wiki and code context from gather_context (Phase 3 enrichment)
        wiki_ctx = _format_wiki_context(context.get("wiki_context", []))
        code_ctx = _format_code_context(context.get("code_context", []))

        # Semantic match matrix from gather_context (Phase 4 enrichment)
        match_matrix_ctx = _format_semantic_match_matrix(
            context.get("semantic_match_matrix", {}),
            source_id,
        )

        prompt = _render_prompt(
            "signal_mapping",
            facility=facility,
            ids_name=ids_name,
            section_path=target_path,
            target_type=assignment.target_type.value,
            signal_source_detail=_format_source_detail(sg_detail),
            imas_fields=_format_fields(all_fields),
            identifier_schemas=_format_identifier_schemas(all_fields),
            coordinate_context=_format_coordinate_context(fields),
            version_context=version_context_str,
            unit_analysis=_format_unit_analysis(
                [sg_detail] if sg_detail else [], all_fields
            ),
            cocos_paths="\n".join(f"- {p}" for p in context["cocos_paths"]) or "(none)",
            existing_mappings=json.dumps(context["existing"], indent=2, default=str),
            code_references=code_context or "(no code references available)",
            source_cocos=cocos_context or "(no COCOS context)",
            semantic_candidates=semantic_context or "(no semantic candidates)",
            cluster_candidates=cluster_context or "(no cluster candidates)",
            wiki_context=wiki_ctx,
            code_data_access=code_ctx,
            semantic_match_matrix=match_matrix_ctx,
        )

        messages = _build_messages("signal_mapping_system", prompt)

        batch = _call_llm(
            messages,
            SignalMappingBatch,
            model=model,
            step_name=f"map_signals_{target_path}",
            cost=cost,
        )
        batches.append(batch)

    return batches


async def amap_signals(
    facility: str,
    ids_name: str,
    sections: TargetAssignmentBatch,
    context: dict[str, Any],
    *,
    gc: GraphClient,
    model: str | None = None,
    cost: PipelineCost,
):
    """Async generator that yields (assignment, batch) per section.

    Identical logic to ``map_signals`` but uses async LLM calls and
    yields each batch as it completes for streaming display.
    """
    import asyncio

    logger.info(
        "Generating signal mappings for %d targets (async)", len(sections.assignments)
    )

    dd_ver = context.get("dd_version")

    for assignment in sections.assignments:
        target_path = assignment.imas_target_path

        # Prepare context — all sync graph/CPU work, run in thread
        prep = await asyncio.to_thread(
            _prepare_section_context,
            facility,
            ids_name,
            assignment,
            context,
            gc=gc,
            dd_version=dd_ver,
        )

        messages = _build_messages("signal_mapping_system", prep["prompt"])

        batch = await _acall_llm(
            messages,
            SignalMappingBatch,
            model=model,
            step_name=f"map_signals_{target_path}",
            cost=cost,
        )
        yield assignment, batch


def _prepare_section_context(
    facility: str,
    ids_name: str,
    assignment,
    context: dict[str, Any],
    *,
    gc: GraphClient,
    dd_version: int | None = None,
) -> dict[str, Any]:
    """Prepare all context for a single section mapping (sync, thread-safe).

    Returns a dict with the rendered prompt string.
    """
    target_path = assignment.imas_target_path

    fields = fetch_imas_fields(
        ids_name,
        [target_path],
        gc=gc,
        dd_version=dd_version,
    )
    subtree_fields = fetch_imas_subtree(
        ids_name,
        target_path.removeprefix(f"{ids_name}/"),
        gc=gc,
        leaf_only=True,
        dd_version=dd_version,
    )

    sg_detail = next(
        (g for g in context["groups"] if g["id"] == assignment.source_id),
        {},
    )

    code_refs = fetch_source_code_refs(assignment.source_id, gc=gc)
    code_context = ""
    if code_refs:
        snippets = []
        for ref in code_refs:
            if ref.get("code"):
                lang = ref.get("language", "")
                snippets.append(f"```{lang}\n{ref['code']}\n```")
        if snippets:
            code_context = "\n".join(snippets)

    source_cocos = sg_detail.get("rep_cocos")
    dd_cocos = context.get("dd_cocos")
    cocos_context = ""
    if source_cocos or dd_cocos:
        parts = []
        if source_cocos:
            parts.append(f"Signal COCOS convention: {source_cocos}")
        if dd_cocos:
            parts.append(f"Target DD COCOS convention: {dd_cocos}")
        flip_paths = [p for p in context["cocos_paths"] if p.startswith(target_path)]
        if flip_paths:
            parts.append(
                "Target IMAS paths requiring sign flip:\n"
                + "\n".join(f"- {p}" for p in flip_paths)
            )
        cocos_context = "\n".join(parts)

    all_fields = subtree_fields or fields

    version_context_str = "(no version change history)"
    try:
        from imas_codex.tools.version_tool import VersionTool

        vt = VersionTool(gc)
        field_paths = [
            f.get("id", "") or f.get("path", "")
            for f in all_fields
            if (f.get("id") or f.get("path"))
        ]
        if field_paths:
            version_ctx = _run_async(vt.get_dd_version_context(paths=field_paths))
            version_context_str = _format_version_context(version_ctx)
    except Exception:
        logger.debug("Version context fetch failed — continuing without it")

    source_id = assignment.source_id
    source_semantic = context.get("source_candidates", {}).get(source_id, [])
    semantic_context = ""
    cluster_context = ""
    if source_semantic:
        semantic_lines = []
        cluster_lines = []
        for sc in source_semantic:
            path = sc.get("id", "")
            score = sc.get("score", 0)
            doc = sc.get("documentation", "")
            via = sc.get("via_cluster")
            if via:
                cluster_lines.append(
                    f"  - {path} (score={score:.2f}, cluster={via}): {doc}"
                )
            else:
                semantic_lines.append(f"  - {path} (score={score:.2f}): {doc}")
        if semantic_lines:
            semantic_context = "Semantic search candidates:\n" + "\n".join(
                semantic_lines
            )
        if cluster_lines:
            cluster_context = "Cluster-derived candidates:\n" + "\n".join(cluster_lines)

    wiki_ctx = _format_wiki_context(context.get("wiki_context", []))
    code_ctx = _format_code_context(context.get("code_context", []))

    match_matrix_ctx = _format_semantic_match_matrix(
        context.get("semantic_match_matrix", {}),
        source_id,
    )

    prompt = _render_prompt(
        "signal_mapping",
        facility=facility,
        ids_name=ids_name,
        section_path=target_path,
        target_type=assignment.target_type.value,
        signal_source_detail=_format_source_detail(sg_detail),
        imas_fields=_format_fields(all_fields),
        identifier_schemas=_format_identifier_schemas(all_fields),
        coordinate_context=_format_coordinate_context(fields),
        version_context=version_context_str,
        unit_analysis=_format_unit_analysis(
            [sg_detail] if sg_detail else [], all_fields
        ),
        cocos_paths="\n".join(f"- {p}" for p in context["cocos_paths"]) or "(none)",
        existing_mappings=json.dumps(context["existing"], indent=2, default=str),
        code_references=code_context or "(no code references available)",
        source_cocos=cocos_context or "(no COCOS context)",
        semantic_candidates=semantic_context or "(no semantic candidates)",
        cluster_candidates=cluster_context or "(no cluster candidates)",
        wiki_context=wiki_ctx,
        code_data_access=code_ctx,
        semantic_match_matrix=match_matrix_ctx,
    )

    return {"prompt": prompt}


def _format_signal_mappings(batch: SignalMappingBatch) -> str:
    """Format signal mappings for the assembly prompt."""
    lines: list[str] = []
    for m in batch.mappings:
        line = f"- {m.source_id}.{m.source_property} → {m.target_id}"
        if m.transform_expression != "value":
            line += f" (transform: {m.transform_expression})"
        if m.source_units or m.target_units:
            line += f" [{m.source_units or '?'} → {m.target_units or '?'}]"
        lines.append(line)
    return "\n".join(lines) if lines else "(no mappings)"


def _format_source_metadata(
    assignment: Any,
    context: dict[str, Any],
) -> str:
    """Format source metadata for the assembly prompt."""
    sg = next(
        (g for g in context["groups"] if g["id"] == assignment.source_id),
        {},
    )
    if not sg:
        return "(no source metadata)"
    parts = [
        f"Source: {sg.get('id', '')}",
        f"Key: {sg.get('group_key', '')}",
        f"Members: {sg.get('member_count', 0)}",
        f"Domain: {sg.get('physics_domain', '')}",
    ]
    desc = sg.get("rep_description") or sg.get("description")
    if desc:
        parts.append(f"Description: {desc}")
    return "\n".join(parts)


def discover_assembly(
    facility: str,
    ids_name: str,
    sections: TargetAssignmentBatch,
    signal_batches: list[SignalMappingBatch],
    context: dict[str, Any],
    *,
    gc: GraphClient,
    model: str | None = None,
    cost: PipelineCost,
) -> AssemblyBatch:
    """Discover assembly patterns for each target."""
    logger.info(
        "Discovering assembly patterns for %d targets",
        len(sections.assignments),
    )

    configs: list[AssemblyConfig] = []
    dd_ver = context.get("dd_version")

    for assignment, batch in zip(sections.assignments, signal_batches, strict=False):
        target_path = assignment.imas_target_path

        # For SCALAR and PROFILE targets, auto-generate a DIRECT config
        # without an LLM call — these have trivial assembly patterns
        if assignment.target_type in (TargetType.SCALAR, TargetType.PROFILE):
            logger.info(
                "Auto-generating DIRECT assembly for %s target %s",
                assignment.target_type,
                target_path,
            )
            configs.append(
                AssemblyConfig(
                    target_path=target_path,
                    pattern=AssemblyPattern.DIRECT,
                    reasoning=f"Auto-generated: {assignment.target_type} target uses direct write",
                    confidence=1.0,
                )
            )
            continue

        target_structure = fetch_imas_subtree(
            ids_name,
            target_path.removeprefix(f"{ids_name}/"),
            gc=gc,
            dd_version=dd_ver,
        )

        # Fetch fields with identifier schema data for assembly context
        target_fields = fetch_imas_fields(
            ids_name,
            [target_path],
            gc=gc,
            dd_version=dd_ver,
        )

        prompt = _render_prompt(
            "assembly",
            facility=facility,
            ids_name=ids_name,
            section_path=target_path,
            signal_mappings=_format_signal_mappings(batch),
            imas_section_structure=_format_subtree(target_structure),
            source_metadata=_format_source_metadata(assignment, context),
            identifier_schemas=_format_identifier_schemas(target_fields),
            coordinate_context=_format_coordinate_context(target_fields),
        )

        messages = _build_messages("assembly_system", prompt)

        config = _call_llm(
            messages,
            AssemblyConfig,
            model=model,
            step_name=f"discover_assembly_{target_path}",
            cost=cost,
        )
        configs.append(config)

    return AssemblyBatch(ids_name=ids_name, configs=configs)


async def adiscover_assembly(
    facility: str,
    ids_name: str,
    sections: TargetAssignmentBatch,
    signal_batches: list[SignalMappingBatch],
    context: dict[str, Any],
    *,
    gc: GraphClient,
    model: str | None = None,
    cost: PipelineCost,
):
    """Async generator that yields (assignment, config) per section."""
    import asyncio

    logger.info(
        "Discovering assembly patterns for %d targets (async)",
        len(sections.assignments),
    )

    dd_ver = context.get("dd_version")

    for assignment, batch in zip(sections.assignments, signal_batches, strict=False):
        target_path = assignment.imas_target_path

        if assignment.target_type in (TargetType.SCALAR, TargetType.PROFILE):
            logger.info(
                "Auto-generating DIRECT assembly for %s target %s",
                assignment.target_type,
                target_path,
            )
            config = AssemblyConfig(
                target_path=target_path,
                pattern=AssemblyPattern.DIRECT,
                reasoning=f"Auto-generated: {assignment.target_type} target uses direct write",
                confidence=1.0,
            )
            yield assignment, config
            continue

        target_structure = await asyncio.to_thread(
            fetch_imas_subtree,
            ids_name,
            target_path.removeprefix(f"{ids_name}/"),
            gc=gc,
            dd_version=dd_ver,
        )

        target_fields = await asyncio.to_thread(
            fetch_imas_fields,
            ids_name,
            [target_path],
            gc=gc,
            dd_version=dd_ver,
        )

        prompt = _render_prompt(
            "assembly",
            facility=facility,
            ids_name=ids_name,
            section_path=target_path,
            signal_mappings=_format_signal_mappings(batch),
            imas_section_structure=_format_subtree(target_structure),
            source_metadata=_format_source_metadata(assignment, context),
            identifier_schemas=_format_identifier_schemas(target_fields),
            coordinate_context=_format_coordinate_context(target_fields),
        )

        messages = _build_messages("assembly_system", prompt)

        config = await _acall_llm(
            messages,
            AssemblyConfig,
            model=model,
            step_name=f"discover_assembly_{target_path}",
            cost=cost,
        )
        yield assignment, config


def validate_mappings(
    facility: str,
    ids_name: str,
    dd_version: str,
    sections: TargetAssignmentBatch,
    field_batches: list[SignalMappingBatch],
    *,
    gc: GraphClient,
) -> ValidatedMappingResult:
    """Assemble and programmatically validate all signal mappings.

    Runs concrete checks via ``validate_mapping()``: source/target existence,
    transform execution, unit compatibility, and duplicate target detection.
    """
    from imas_codex.ids.validation import validate_mapping

    logger.info("Running programmatic validation")

    # Assemble bindings + escalations + unmapped from Step 2 batches
    all_bindings: list[ValidatedSignalMapping] = []
    all_escalations: list[EscalationFlag] = []
    all_unmapped: list[UnmappedSignal] = []
    for batch in field_batches:
        for m in batch.mappings:
            all_bindings.append(
                ValidatedSignalMapping(
                    source_id=m.source_id,
                    source_property=m.source_property,
                    target_id=m.target_id,
                    transform_expression=m.transform_expression,
                    source_units=m.source_units,
                    target_units=m.target_units,
                    cocos_label=m.cocos_label,
                    confidence=m.confidence,
                    mapping_type="direct",
                )
            )
        all_unmapped.extend(batch.unmapped)
        all_escalations.extend(batch.escalations)

    # Run programmatic validation
    from imas_codex.ids.tools import get_sign_flip_paths
    from imas_codex.ids.validation import check_coverage_threshold

    flip_paths = get_sign_flip_paths(ids_name)
    report = validate_mapping(all_bindings, gc=gc, sign_flip_paths=flip_paths)
    all_escalations.extend(report.escalations)

    # Coverage threshold check
    coverage_escalations = check_coverage_threshold(
        ids_name,
        all_bindings,
        gc=gc,
    )
    all_escalations.extend(coverage_escalations)

    corrections: list[str] = []
    if report.duplicate_targets:
        corrections.append(
            f"Duplicate targets detected: {', '.join(report.duplicate_targets)}"
        )
    if not report.all_passed:
        failed = sum(
            1
            for c in report.binding_checks
            if not (
                c.source_exists
                and c.target_exists
                and c.transform_executes
                and c.units_compatible
            )
        )
        corrections.append(f"{failed} binding(s) failed validation checks")

    return ValidatedMappingResult(
        facility=facility,
        ids_name=ids_name,
        dd_version=dd_version,
        sections=sections.assignments,
        bindings=all_bindings,
        unmapped=all_unmapped,
        escalations=all_escalations,
        corrections=corrections,
    )


# ---------------------------------------------------------------------------
# Stage 2: Error field derivation (zero LLM cost)
# ---------------------------------------------------------------------------

# Patterns indicating a signal represents measurement uncertainty
_ERROR_SIGNAL_PATTERNS = re.compile(
    r"\b(?:error|uncertainty|sigma|std[_ ]?dev|err[_ ]?bar|"
    r"confidence[_ ]?interval|error[_ ]?bar|"
    r"delta|±|\bunc\b|rms[_ ]?error)\b",
    re.IGNORECASE,
)

# Patterns indicating "error field" in the physics sense (NOT uncertainty)
# These should NOT be classified as error signals
_PHYSICS_ERROR_PATTERNS = re.compile(
    r"\b(?:error[_ ]?field[_ ]?coil|error[_ ]?field[_ ]?correction|"
    r"resonant[_ ]?magnetic[_ ]?perturbation|rmp|"
    r"error[_ ]?field[_ ]?(?:amplitude|phase|spectrum))\b",
    re.IGNORECASE,
)

# Patterns to strip from a signal description/group_key to find the base
_STRIP_ERROR_SUFFIX = re.compile(
    r"[_ ]*(?:error|uncertainty|sigma|err|unc|std[_ ]?dev|rms)[_ ]*$",
    re.IGNORECASE,
)


def classify_error_signals(
    facility: str,
    *,
    gc: GraphClient | None = None,
) -> list[dict]:
    """Identify facility signal sources that represent measurement uncertainties.

    Scans signal source group_keys and descriptions for error/uncertainty
    patterns, excluding physics "error field" concepts (like error field
    correction coils which are actual magnetic devices, not uncertainties).

    Args:
        facility: Facility identifier.
        gc: GraphClient (created if None).

    Returns:
        List of dicts with keys: signal_id, group_key, description,
        physics_domain, probable_error_type ("upper", "lower", or "symmetric").
    """
    if gc is None:
        gc = GraphClient()

    # Query all enriched signal sources with representative member name
    results = gc.query(
        """
        MATCH (sg:SignalSource {facility_id: $facility})
        WHERE sg.status = 'enriched'
        OPTIONAL MATCH (rep:FacilitySignal {id: sg.representative_id})
        RETURN sg.id AS id, sg.group_key AS group_key,
               sg.description AS description,
               sg.physics_domain AS physics_domain,
               coalesce(rep.name, '') AS rep_name
        """,
        facility=facility,
    )

    error_signals: list[dict] = []
    for r in results:
        group_key = r.get("group_key", "") or ""
        desc = r.get("description", "") or ""
        rep_name = r.get("rep_name", "") or ""
        text = f"{group_key} {desc} {rep_name}"

        # Skip physics error field concepts
        if _PHYSICS_ERROR_PATTERNS.search(text):
            continue

        # Match uncertainty patterns
        if _ERROR_SIGNAL_PATTERNS.search(text):
            # Determine probable error type from text
            lower_text = text.lower()
            if any(w in lower_text for w in ("lower", "minimum", "min_err", "low")):
                error_type = "lower"
            elif any(w in lower_text for w in ("upper", "maximum", "max_err", "high")):
                error_type = "upper"
            else:
                error_type = "symmetric"  # Default: could be upper or lower

            error_signals.append(
                {
                    "signal_id": r["id"],
                    "group_key": group_key,
                    "description": desc,
                    "physics_domain": r.get("physics_domain"),
                    "probable_error_type": error_type,
                }
            )

    logger.info(
        "Classified %d error signals out of %d total for %s",
        len(error_signals),
        len(results),
        facility,
    )
    return error_signals


def match_error_signals_to_imas(
    facility: str,
    error_signals: list[dict],
    *,
    gc: GraphClient | None = None,
) -> list[ValidatedSignalMapping]:
    """Match error-related signal sources directly to IMAS error fields.

    Cross-references error signals with existing data mappings:
    if signal source "X Error" exists and signal source "X" maps to
    data_path, then "X Error" maps to data_path_error_upper/lower.

    Uses group_key and description similarity to find the parent data
    signal source, then traverses HAS_ERROR to find the target error field.

    Args:
        facility: Facility identifier.
        error_signals: Output from classify_error_signals().
        gc: GraphClient (created if None).

    Returns:
        List of ValidatedSignalMapping for direct error signal matches.
    """
    if gc is None:
        gc = GraphClient()

    if not error_signals:
        return []

    # Get all existing direct MAPS_TO_IMAS relationships for this facility
    existing = gc.query(
        """
        MATCH (sg:SignalSource {facility_id: $facility})-[r:MAPS_TO_IMAS]->(ip:IMASNode)
        WHERE coalesce(r.mapping_type, 'direct') = 'direct'
        RETURN sg.id AS source_id, sg.group_key AS source_group_key,
               ip.id AS target_id,
               r.source_units AS source_units,
               r.target_units AS target_units,
               r.confidence AS confidence
        """,
        facility=facility,
    )

    if not existing:
        logger.info("No existing data mappings to cross-reference for %s", facility)
        return []

    # Build lookup: group_key (normalized) -> list of mappings
    key_to_mappings: dict[str, list[dict]] = defaultdict(list)
    for e in existing:
        src_key = (e.get("source_group_key") or "").strip().lower()
        if src_key:
            key_to_mappings[src_key].append(e)

    # Build lookup: target_path -> error children
    target_paths = list({e["target_id"] for e in existing})
    error_children: dict[str, list[dict]] = {}
    batch_size = 500
    for i in range(0, len(target_paths), batch_size):
        batch = target_paths[i : i + batch_size]
        errors = gc.query(
            """
            MATCH (d:IMASNode)-[r:HAS_ERROR]->(e:IMASNode)
            WHERE d.id IN $paths
            RETURN d.id AS data_path, e.id AS error_path,
                   r.error_type AS error_type
            """,
            paths=batch,
        )
        for row in errors:
            error_children.setdefault(row["data_path"], []).append(
                {
                    "error_path": row["error_path"],
                    "error_type": row["error_type"],
                }
            )

    # Match error signals to error fields
    mappings: list[ValidatedSignalMapping] = []

    for esig in error_signals:
        sig_key = (esig.get("group_key") or "").strip().lower()
        if not sig_key:
            continue

        # Try to find the base signal by stripping error suffixes
        base_key = _STRIP_ERROR_SUFFIX.sub("", sig_key).strip()

        # Look up the base signal's data mappings
        parent_mappings = key_to_mappings.get(base_key, [])

        if not parent_mappings:
            # Try partial match — base_key might be a substring
            for known_key, known_mappings in key_to_mappings.items():
                if base_key and (base_key in known_key or known_key in base_key):
                    parent_mappings = known_mappings
                    break

        if not parent_mappings:
            continue

        for pm in parent_mappings:
            errors = error_children.get(pm["target_id"], [])
            if not errors:
                continue

            # Determine which error type to map to
            prob_type = esig.get("probable_error_type", "symmetric")

            for err in errors:
                # For symmetric errors, map to both upper and lower
                # For specific types, only map to matching type
                if prob_type == "symmetric" or prob_type == err["error_type"]:
                    mappings.append(
                        ValidatedSignalMapping(
                            source_id=esig["signal_id"],
                            target_id=err["error_path"],
                            transform_expression="value",
                            source_units=pm.get("source_units"),
                            target_units=pm.get("target_units"),
                            confidence=min((pm.get("confidence") or 0.5) * 0.9, 1.0),
                            evidence=(
                                f"Direct error signal match via parent "
                                f"{pm['source_id']} -> {pm['target_id']}"
                            ),
                            mapping_type="error_derived",
                            error_type=err["error_type"],
                            derived_from=pm["target_id"],
                        )
                    )

    logger.info(
        "Matched %d error signals to %d error field mappings for %s",
        len(
            [
                e
                for e in error_signals
                if any(m.source_id == e["signal_id"] for m in mappings)
            ]
        ),
        len(mappings),
        facility,
    )
    return mappings


def derive_error_mappings(
    data_mappings: list[ValidatedSignalMapping],
    *,
    gc: GraphClient | None = None,
    facility: str | None = None,
    include_direct_error_signals: bool = True,
) -> list[ValidatedSignalMapping]:
    """Derive error field mappings from validated data mappings via HAS_ERROR.

    For each data mapping to an IMASNode, traverses HAS_ERROR relationships
    to find associated error fields (_error_upper, _error_lower, _error_index).
    Creates error_derived mappings inheriting transform, units, and confidence
    from the parent data mapping.

    When *facility* is provided and *include_direct_error_signals* is True,
    also runs Phase 2b: identifies facility signal sources that directly
    represent measurement uncertainties (e.g. "HRTS Electron Density Error")
    and matches them to IMAS error fields via cross-reference with existing
    data mappings.

    Cost: Zero LLM tokens. Graph queries only (~10ms).

    Args:
        data_mappings: Validated data mappings from Stage 1.
        gc: GraphClient (created if None).
        facility: Facility identifier (needed for direct error signal matching).
        include_direct_error_signals: Whether to run Phase 2b direct error
            signal matching. Default True.

    Returns:
        List of error-derived ValidatedSignalMapping instances.
    """
    if gc is None:
        gc = GraphClient()

    # Only process direct data mappings (not already-derived error mappings)
    direct_mappings = [m for m in data_mappings if m.mapping_type == "direct"]

    if not direct_mappings:
        return []

    # Batch query: get all error fields for all data mapping targets at once
    target_paths = list({m.target_id for m in direct_mappings})

    error_map: dict[str, list[dict]] = {}
    # Batch in groups to avoid overly large IN clauses
    batch_size = 500
    for i in range(0, len(target_paths), batch_size):
        batch = target_paths[i : i + batch_size]
        results = gc.query(
            """
            MATCH (d:IMASNode)-[r:HAS_ERROR]->(e:IMASNode)
            WHERE d.id IN $paths
            RETURN d.id AS data_path, e.id AS error_path, r.error_type AS error_type
            """,
            paths=batch,
        )
        for row in results:
            error_map.setdefault(row["data_path"], []).append(
                {
                    "error_path": row["error_path"],
                    "error_type": row["error_type"],
                }
            )

    # Create derived mappings
    error_mappings: list[ValidatedSignalMapping] = []
    for dm in direct_mappings:
        errors = error_map.get(dm.target_id, [])
        for err in errors:
            error_mappings.append(
                ValidatedSignalMapping(
                    source_id=dm.source_id,
                    source_property=dm.source_property,
                    target_id=err["error_path"],
                    transform_expression=dm.transform_expression,
                    source_units=dm.source_units,
                    target_units=dm.target_units,
                    cocos_label=dm.cocos_label,
                    confidence=dm.confidence,
                    disposition=dm.disposition,
                    evidence=f"Derived from data mapping to {dm.target_id}",
                    mapping_type="error_derived",
                    error_type=err["error_type"],
                    derived_from=dm.target_id,
                )
            )

    logger.info(
        "Derived %d error mappings from %d data mappings (%d targets with errors)",
        len(error_mappings),
        len(direct_mappings),
        len(error_map),
    )

    # Phase 2b: Direct error signal matching
    if include_direct_error_signals and facility:
        error_signals = classify_error_signals(facility, gc=gc)
        if error_signals:
            direct_error_mappings = match_error_signals_to_imas(
                facility, error_signals, gc=gc
            )
            # Deduplicate: don't create direct mappings that overlap with derived
            existing_targets = {(m.source_id, m.target_id) for m in error_mappings}
            added = 0
            for dm in direct_error_mappings:
                if (dm.source_id, dm.target_id) not in existing_targets:
                    error_mappings.append(dm)
                    existing_targets.add((dm.source_id, dm.target_id))
                    added += 1

            logger.info(
                "Added %d direct error signal mappings (after dedup) for %s",
                added,
                facility,
            )

    return error_mappings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_error_derivation_only(
    facility: str,
    ids_name: str,
    *,
    gc: GraphClient | None = None,
    dry_run: bool = False,
) -> list[ValidatedSignalMapping]:
    """Run Stage 2 error derivation against existing data mappings.

    Fetches existing MAPS_TO_IMAS relationships with mapping_type='direct',
    then derives error mappings via HAS_ERROR graph traversal.

    Args:
        facility: Facility identifier (e.g. "tcv", "jet").
        ids_name: IDS name to derive error mappings for.
        gc: GraphClient (created if None).
        dry_run: If True, skip persisting derived mappings to the graph.

    Returns:
        List of error-derived ValidatedSignalMapping instances.
    """
    if gc is None:
        gc = GraphClient()

    # Fetch existing direct mappings from graph
    results = gc.query(
        """
        MATCH (sg:SignalSource {facility_id: $facility})-[r:MAPS_TO_IMAS]->(ip:IMASNode)
        WHERE ip.ids = $ids_name
          AND coalesce(r.mapping_type, 'direct') = 'direct'
        RETURN sg.id AS source_id,
               r.source_property AS source_property,
               ip.id AS target_id,
               r.transform_expression AS transform_expression,
               r.source_units AS source_units,
               r.target_units AS target_units,
               r.cocos_label AS cocos_label,
               r.confidence AS confidence
        """,
        facility=facility,
        ids_name=ids_name,
    )

    if not results:
        logger.warning("No existing direct mappings for %s:%s", facility, ids_name)
        return []

    # Reconstruct ValidatedSignalMapping objects
    data_mappings = [
        ValidatedSignalMapping(
            source_id=r["source_id"],
            source_property=r.get("source_property") or "value",
            target_id=r["target_id"],
            transform_expression=r.get("transform_expression") or "value",
            source_units=r.get("source_units"),
            target_units=r.get("target_units"),
            cocos_label=r.get("cocos_label"),
            confidence=r.get("confidence") or 0.5,
            mapping_type="direct",
        )
        for r in results
    ]

    error_bindings = derive_error_mappings(data_mappings, gc=gc, facility=facility)

    if error_bindings and not dry_run:
        for fm in error_bindings:
            gc.query(
                """
                MATCH (sg:SignalSource {id: $sg_id})
                MATCH (ip:IMASNode {id: $target_id})
                MERGE (sg)-[r:MAPS_TO_IMAS]->(ip)
                SET r.source_property = $source_property,
                    r.transform_expression = $transform_expression,
                    r.source_units = $source_units,
                    r.target_units = $target_units,
                    r.cocos_label = $cocos_label,
                    r.confidence = $confidence,
                    r.evidence = $evidence,
                    r.mapping_type = $mapping_type,
                    r.error_type = $error_type,
                    r.derived_from = $derived_from
                """,
                sg_id=fm.source_id,
                target_id=fm.target_id,
                source_property=fm.source_property,
                transform_expression=fm.transform_expression,
                source_units=fm.source_units,
                target_units=fm.target_units,
                cocos_label=fm.cocos_label,
                confidence=fm.confidence,
                evidence=fm.evidence,
                mapping_type=fm.mapping_type,
                error_type=fm.error_type,
                derived_from=fm.derived_from,
            )
        logger.info(
            "Persisted %d error mappings for %s:%s",
            len(error_bindings),
            facility,
            ids_name,
        )

    return error_bindings


@dataclass
class MappingResult:
    """Result of the mapping pipeline."""

    mapping_id: str
    validated: ValidatedMappingResult
    assembly: AssemblyBatch | None
    cost: PipelineCost
    persisted: bool = False
    unassigned_groups: list[str] = field(default_factory=list)
    metadata: IDSMetadataResult | None = None


def generate_mapping(
    facility: str,
    ids_name: str,
    *,
    model: str | None = None,
    reasoning_model: str | None = None,
    dd_version: str | None = None,
    persist: bool = True,
    activate: bool = True,
    gc: GraphClient | None = None,
) -> MappingResult:
    """Generate IMAS signal mapping via multi-step LLM pipeline.

    Steps:
        1. Gather signal sources + DD context (programmatic)
        2. LLM assigns sources to IMAS sections
        3. For each section, LLM generates signal mappings
        4. Programmatic validation (source/target existence, transforms, units)
        5. (Optional) Persist to graph

    Args:
        facility: Facility name (e.g., "jet").
        ids_name: IDS name (e.g., "pf_active").
        model: LLM model override for classification steps (default: language tier).
        reasoning_model: LLM model override for signal mapping (default: reasoning tier).
        dd_version: DD version override (default: from settings).
        persist: Whether to persist results to graph.
        activate: Whether to promote status to 'active' after persisting.
        gc: GraphClient instance (created if None).

    Returns:
        MappingResult with validated mappings and cost breakdown.
    """
    if gc is None:
        gc = GraphClient()

    if dd_version is None:
        # Try to find the latest DD version ≥ 4.x in the graph
        try:
            rows = gc.query(
                """
                MATCH (v:DDVersion)
                WHERE v.major >= 4
                RETURN v.id AS id
                ORDER BY v.major DESC, v.minor DESC, v.patch DESC
                LIMIT 1
                """
            )
            if rows:
                dd_version = rows[0]["id"]
                logger.info("Auto-detected DD version from graph: %s", dd_version)
        except Exception:
            pass

    if dd_version is None:
        from imas_codex import dd_version as default_dd

        dd_version = default_dd

    # Extract major version number for DD filtering queries
    dd_major: int | None = None
    if dd_version:
        try:
            dd_major = int(str(dd_version).split(".")[0])
        except (ValueError, IndexError):
            pass

    cost = PipelineCost()

    # Step 0: Gather context
    context = gather_context(facility, ids_name, gc=gc, dd_version=dd_major)

    if not context["groups"]:
        raise ValueError(
            f"No signal sources found for {facility}/{ids_name}. "
            "Run signal discovery first."
        )

    # Step 1: Assign targets
    sections = assign_targets(
        facility, ids_name, context, gc=gc, model=model, cost=cost
    )

    if not sections.assignments:
        raise ValueError(
            f"LLM could not assign any signal sources to IDS target paths "
            f"for {facility}/{ids_name}."
        )

    # Step 2: Signal mappings (reasoning tier — highest accuracy needed)
    mapping_model = reasoning_model or get_model("reasoning")
    field_batches = map_signals(
        facility,
        ids_name,
        sections,
        context,
        gc=gc,
        model=mapping_model,
        cost=cost,
    )

    # Step 3: Assembly discovery (LLM)
    assembly = discover_assembly(
        facility,
        ids_name,
        sections,
        field_batches,
        context,
        gc=gc,
        model=model,
        cost=cost,
    )

    # Step 4: Programmatic validation (no LLM call)
    validated = validate_mappings(
        facility,
        ids_name,
        dd_version,
        sections,
        field_batches,
        gc=gc,
    )

    # Step 5: Derive error field mappings (no LLM call)
    error_bindings = derive_error_mappings(validated.bindings, gc=gc, facility=facility)
    if error_bindings:
        validated.bindings.extend(error_bindings)
        logger.info(
            "Derived %d error mappings for %s/%s",
            len(error_bindings),
            facility,
            ids_name,
        )

    # Step 6: Populate IDS metadata (programmatic + LLM)
    metadata_result: IDSMetadataResult | None = None
    try:
        # Convert validated bindings to signal summary dicts for the LLM
        mapped_signals = [
            {
                "source_id": b.source_id,
                "target_id": b.target_id,
                "confidence": b.confidence,
            }
            for b in validated.bindings
        ]
        metadata_result = populate_metadata(
            facility,
            ids_name,
            gc=gc,
            dd_version=dd_version,
            mapped_signals=mapped_signals,
            model=model,
        )
        if metadata_result.cost_usd > 0:
            cost.add("metadata", metadata_result.cost_usd, metadata_result.tokens)
        logger.info(
            "Populated metadata for %s/%s: %d deterministic, %d LLM fields",
            facility,
            ids_name,
            len(metadata_result.deterministic_fields),
            len(metadata_result.llm_fields),
        )
    except Exception:
        logger.warning(
            "Metadata population failed for %s/%s — continuing without metadata",
            facility,
            ids_name,
            exc_info=True,
        )

    # Persist
    mapping_id = f"{facility}:{ids_name}"
    persisted = False
    if persist:
        status = "active" if activate else "generated"
        mapping_id = persist_mapping_result(
            validated, assembly=assembly, gc=gc, status=status
        )
        persisted = True
        logger.info("Persisted mapping %s with status '%s'", mapping_id, status)
        # Persist metadata if available
        if metadata_result is not None:
            persist_metadata(metadata_result, mapping_id, gc=gc)

    logger.info(
        "Pipeline complete: %d bindings, %d escalations, $%.4f total",
        len(validated.bindings),
        len(validated.escalations),
        cost.total_usd,
    )

    return MappingResult(
        mapping_id=mapping_id,
        validated=validated,
        assembly=assembly,
        cost=cost,
        persisted=persisted,
        unassigned_groups=sections.unassigned_groups,
        metadata=metadata_result,
    )
