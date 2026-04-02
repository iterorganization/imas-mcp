"""Formatters for unified MCP search tool output.

Pure functions that take structured query results and produce
formatted text reports for agent consumption. Output size is
controlled by the caller via the ``k`` parameter — formatters
render all results they receive without artificial truncation.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any


def _get_value(result: Any, key: str, default: Any = None) -> Any:
    """Read a field from a dict-like or attribute-based result object."""
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


def _format_tool_error(result: Any) -> str | None:
    """Render ToolError-like results consistently across formatters."""
    error = _get_value(result, "error")
    if not error:
        return None

    lines = [f"Error: {error}"]

    suggestions = _get_value(result, "suggestions", []) or []
    if suggestions:
        lines.append("")
        lines.append("Suggestions:")
        for suggestion in suggestions:
            lines.append(f"- {suggestion}")

    fallback_data = _get_value(result, "fallback_data")
    if isinstance(fallback_data, dict) and fallback_data:
        message = fallback_data.get("message")
        if message:
            lines.append("")
            lines.append(f"Fallback: {message}")

        fallback_suggestions = fallback_data.get("suggestions") or []
        if fallback_suggestions:
            lines.append("Fallback suggestions:")
            for item in fallback_suggestions:
                if isinstance(item, dict):
                    tool = item.get("tool") or "unknown"
                    reason = item.get("reason") or ""
                    description = item.get("description") or ""
                    detail = " — ".join(part for part in (reason, description) if part)
                    lines.append(f"- {tool}" + (f": {detail}" if detail else ""))
                else:
                    lines.append(f"- {item}")

    return "\n".join(lines)


def _stringify_cluster_labels(values: Any) -> list[str]:
    """Normalize cluster labels for rendering."""
    if not values:
        return []

    if not isinstance(values, Iterable) or isinstance(values, str):
        values = [values]

    labels: list[str] = []
    for value in values:
        if not value:
            continue
        if isinstance(value, str):
            labels.append(value)
            continue
        if isinstance(value, dict):
            label = value.get("label") or value.get("name") or value.get("id")
            if label:
                labels.append(str(label))
                continue
        labels.append(str(value))

    return labels


# ---------------------------------------------------------------------------
# search_signals formatter
# ---------------------------------------------------------------------------


def format_signals_report(
    signals: list[dict[str, Any]],
    data_nodes: list[dict[str, Any]],
    scores: dict[str, float],
) -> str:
    """Format signal search results into a readable report.

    Handles deduplicated access methods (collected into arrays by
    ``_enrich_signals``), interpolates template placeholders with
    actual signal properties, and filters noisy data nodes.

    Args:
        signals: Enriched signal records with ``access_methods`` arrays.
        data_nodes: Data node results from vector search.
        scores: Map of signal ID → similarity score.

    Returns:
        Formatted text report.
    """
    if not signals and not data_nodes:
        return "No signals found."

    parts: list[str] = []

    if signals:
        parts.append(f"## Signals ({len(signals)} matches)\n")

        for sig in signals:
            sid = sig.get("id", "?")
            score = scores.get(sid)
            score_str = f" (score: {score:.2f})" if score is not None else ""
            parts.append(f"### {sid}{score_str}")

            desc = sig.get("description") or ""
            if desc:
                parts.append(f"  {desc}")

            # Metadata line
            meta_parts: list[str] = []
            diag = sig.get("diagnostic_name")
            if diag:
                cat = sig.get("diagnostic_category")
                meta_parts.append(f"Diagnostic: {diag}" + (f" ({cat})" if cat else ""))
            domain = sig.get("physics_domain")
            if domain:
                meta_parts.append(f"Domain: {domain}")
            unit = sig.get("unit")
            unit_symbol = sig.get("unit_symbol")
            if unit:
                unit_str = unit
                if unit_symbol and unit_symbol != unit:
                    unit_str = f"{unit} ({unit_symbol})"
                meta_parts.append(f"Unit: {unit_str}")
            elif unit_symbol:
                meta_parts.append(f"Unit: {unit_symbol}")
            sign_conv = sig.get("sign_convention")
            if sign_conv:
                meta_parts.append(f"Sign: {sign_conv}")
            cocos = sig.get("cocos")
            if cocos is not None:
                meta_parts.append(f"COCOS: {cocos}")
            checked = sig.get("checked")
            shot = sig.get("example_shot")
            if checked and shot:
                meta_parts.append(f"Checked: shot {shot}")
            if meta_parts:
                parts.append("  " + " | ".join(meta_parts))

            # Extended metadata
            keywords = sig.get("keywords")
            if keywords:
                kw_str = ", ".join(keywords) if isinstance(keywords, list) else keywords
                parts.append(f"  Keywords: {kw_str}")
            aliases = sig.get("aliases")
            if aliases:
                al_str = ", ".join(aliases) if isinstance(aliases, list) else aliases
                parts.append(f"  Aliases: {al_str}")
            analysis_code = sig.get("analysis_code")
            if analysis_code:
                parts.append(f"  Analysis code: {analysis_code}")

            # Signal source context
            sg_key = sig.get("signal_source_key")
            if sg_key:
                sg_desc = sig.get("signal_source_description") or ""
                sg_count = sig.get("signal_source_member_count")
                sg_line = f"  Signal source: {sg_key}"
                if sg_count:
                    sg_line += f" ({sg_count} members)"
                if sg_desc:
                    sg_line += f" — {sg_desc}"
                parts.append(sg_line)

            # Wiki cross-references
            wiki = sig.get("wiki_mentions")
            if wiki:
                wiki = [w for w in wiki if w]
                if wiki:
                    parts.append(f"  Wiki refs: {', '.join(wiki)}")

            # Data access section — handles multiple access methods
            access_methods = sig.get("access_methods") or []
            # Filter out empty/null access methods from OPTIONAL MATCH
            access_methods = [
                am
                for am in access_methods
                if isinstance(am, dict) and am.get("access_template")
            ]

            # Also support legacy single-access format for backward compat
            if not access_methods and sig.get("access_template"):
                access_methods = [
                    {
                        "access_template": sig["access_template"],
                        "access_type": sig.get("access_type"),
                        "imports_template": sig.get("imports_template"),
                        "connection_template": sig.get("connection_template"),
                        "imas_path": sig.get("imas_path"),
                        "imas_docs": sig.get("imas_docs"),
                        "imas_unit": sig.get("imas_unit"),
                    }
                ]

            if access_methods:
                for am in access_methods:
                    access_type = am.get("access_type") or "unknown"
                    method_type = am.get("method_type")
                    template = am.get("access_template") or ""

                    # Interpolate template placeholders with signal properties
                    template = _interpolate_template(template, sig)

                    label = access_type
                    if method_type:
                        label = f"{access_type}/{method_type}"
                    parts.append(f"\n  **Data access** ({label}):")
                    imports = am.get("imports_template")
                    if imports:
                        parts.append(f"    {imports}")
                    connection = am.get("connection_template")
                    if connection:
                        parts.append(f"    {connection}")
                    parts.append(f"    {template}")

                    # IMAS mapping from this access method
                    imas_path = am.get("imas_path")
                    if imas_path:
                        parts.append(f"\n  **IMAS mapping**: {imas_path}")
                        imas_docs = am.get("imas_docs")
                        if imas_docs:
                            parts.append(f'    "{imas_docs}"')
                        imas_unit = am.get("imas_unit")
                        if imas_unit:
                            parts.append(f"    Unit: {imas_unit}")

            # Data node section
            tree_path = sig.get("tree_path")
            data_source_name = sig.get("data_source_name")
            if tree_path:
                parts.append(
                    f"\n  **Data node**: {tree_path}"
                    + (f" (source: {data_source_name})" if data_source_name else "")
                )

            parts.append("")  # blank line between signals

    if data_nodes:
        # Filter out noisy STATIC data nodes and low-score results
        filtered_data_nodes = _filter_data_nodes(data_nodes, signals)
        if filtered_data_nodes:
            parts.append(
                f"\n## Related Data Nodes ({len(filtered_data_nodes)} matches)"
            )
            for tn in filtered_data_nodes:
                path = tn.get("path", "?")
                desc = tn.get("description") or ""
                tree = tn.get("data_source_name") or ""
                tn_unit = tn.get("unit") or ""
                score = tn.get("score")
                meta = []
                if tree:
                    meta.append(f"tree: {tree}")
                if tn_unit:
                    meta.append(f"unit: {tn_unit}")
                if score is not None:
                    meta.append(f"score: {score:.2f}")
                meta_str = f" ({', '.join(meta)})" if meta else ""
                line = f"  {path}{meta_str}"
                if desc:
                    line += f" — {desc}"
                parts.append(line)

    return "\n".join(parts)


def _interpolate_template(template: str, sig: dict[str, Any]) -> str:
    """Interpolate known placeholders in data access templates.

    Substitutes ``{node_path}``, ``{accessor}``, ``{data_source}``
    from signal properties. Keeps ``{shot}`` as a user-parameterized
    placeholder since it's session-specific.
    """
    substitutions = {
        "node_path": sig.get("node_path") or sig.get("tree_path"),
        "accessor": sig.get("accessor"),
        "data_source": sig.get("data_source_name"),
    }
    for key, val in substitutions.items():
        if val:
            template = template.replace(f"{{{key}}}", str(val))
    return template


def _filter_data_nodes(
    data_nodes: list[dict[str, Any]],
    signals: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Filter data nodes to remove noise.

    - Excludes STATIC data nodes (calibration/geometry data)
    - Only keeps data nodes that are connected to matched signals
      or have high relevance scores (>= 0.92)
    """
    # Collect tree paths already shown in signal results
    signal_tree_paths = {
        sig.get("tree_path") for sig in signals if sig.get("tree_path")
    }

    filtered = []
    for tn in data_nodes:
        path = tn.get("path", "")
        source_name = (tn.get("data_source_name") or "").upper()

        # Skip STATIC data nodes (calibration/geometry noise)
        if "STATIC" in path.upper() or "STATIC" in source_name:
            continue

        # Skip nodes already shown in signal results
        if path in signal_tree_paths:
            continue

        # Keep only high-score nodes
        score = tn.get("score")
        if score is not None and score < 0.92:
            continue

        filtered.append(tn)

    return filtered


# ---------------------------------------------------------------------------
# search_docs formatter
# ---------------------------------------------------------------------------


def format_docs_report(
    chunks: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    scores: dict[str, float],
) -> str:
    """Format documentation search results into a readable report.

    Groups chunks by parent page for readability.

    Args:
        chunks: Enriched wiki chunk records.
        documents: Document/image results.
        scores: Map of chunk/document ID → similarity score.

    Returns:
        Formatted text report.
    """
    if not chunks and not documents:
        return "No documentation found."

    parts: list[str] = []

    if chunks:
        # Deduplicate chunks by text content hash
        seen_hashes: set[str] = set()
        unique_chunks: list[dict[str, Any]] = []
        for chunk in chunks:
            text = chunk.get("text", "")
            text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                unique_chunks.append(chunk)
        chunks = unique_chunks

        # Group by page
        pages: dict[str, list[dict[str, Any]]] = {}
        for chunk in chunks:
            page_title = chunk.get("page_title") or "Unknown Page"
            pages.setdefault(page_title, []).append(chunk)

        page_count = len(pages)
        chunk_count = sum(len(v) for v in pages.values())
        parts.append(
            f"## Wiki Documentation ({chunk_count} chunks from {page_count} pages)\n"
        )

        for title, page_chunks in pages.items():
            url = page_chunks[0].get("page_url") or ""
            page_id = page_chunks[0].get("page_id") or ""
            url_str = f" ({url})" if url else ""
            parts.append(f'### Page: "{title}"{url_str}')
            if page_id:
                parts.append(f"  fetch('{page_id}') for full content")

            for chunk in page_chunks:
                section = chunk.get("section") or "General"
                cid = chunk.get("id", "")
                score = scores.get(cid)
                score_str = f" [score: {score:.2f}]" if score is not None else ""
                parts.append(f"**Section: {section}**{score_str}")

                text = chunk.get("text") or ""
                parts.append(f"  {text}")

                # Cross-links
                linked_signals = chunk.get("linked_signals") or []
                if linked_signals:
                    parts.append(f"  Signals: {', '.join(linked_signals)}")

                imas_refs = chunk.get("imas_refs") or []
                if imas_refs:
                    parts.append(f"  IMAS: {', '.join(imas_refs)}")

                linked_data_nodes = chunk.get("linked_data_nodes") or []
                if linked_data_nodes:
                    parts.append(f"  Data nodes: {', '.join(linked_data_nodes)}")

                tool_mentions = chunk.get("tool_mentions") or []
                if tool_mentions:
                    parts.append(f"  Tools: {', '.join(tool_mentions)}")

                parts.append("")

    if documents:
        parts.append(f"\n## Related Documents ({len(documents)} items)")
        for art in documents:
            aid = art.get("id", "?")
            title = art.get("title") or aid
            page = art.get("page_title") or ""
            desc = art.get("description") or ""
            score = scores.get(aid)
            score_str = f" [score: {score:.2f}]" if score is not None else ""
            line = f'  - "{title}"{score_str}'
            if page:
                line += f' — from "{page}"'
            if desc:
                line += f" ({desc})"
            parts.append(line)
            parts.append(f"    fetch('{aid}') for full content")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# search_code formatter
# ---------------------------------------------------------------------------


def format_code_report(
    code_results: list[dict[str, Any]],
    scores: dict[str, float],
) -> str:
    """Format code search results into a readable report.

    Args:
        code_results: Enriched code chunk records.
        scores: Map of chunk ID → similarity score.

    Returns:
        Formatted text report.
    """
    if not code_results:
        return "No code examples found."

    parts: list[str] = []
    parts.append(f"## Code Examples ({len(code_results)} matches)\n")

    for chunk in code_results:
        cid = chunk.get("id", "?")
        func_name = chunk.get("function_name") or "module-level"
        source = chunk.get("source_file") or "unknown"
        facility = chunk.get("facility_id") or ""
        score = scores.get(cid)
        score_str = f" (score: {score:.2f})" if score is not None else ""

        source_id = chunk.get("source_file_id") or ""
        parts.append(f"### {func_name} — {source}{score_str}")
        if source_id:
            parts.append(f"  fetch('{source_id}') for full file")
        if facility:
            parts.append(f"  Facility: {facility}")

        text = chunk.get("text") or ""
        if text:
            lang = chunk.get("language") or "python"
            parts.append(f"  ```{lang}\n  {text}\n  ```")

        # Data references
        data_refs = chunk.get("data_refs") or []
        if data_refs:
            parts.append("  **Data references**:")
            for ref in data_refs:
                if isinstance(ref, dict):
                    ref_type = ref.get("type") or "unknown"
                    raw = ref.get("raw") or ""
                    tree = ref.get("tree") or ""
                    imas = ref.get("imas") or ""
                    tdi = ref.get("tdi") or ""
                    line = f"    {ref_type}: {raw}"
                    if tree:
                        line += f" → tree: {tree}"
                    if imas:
                        line += f" → IMAS: {imas}"
                    if tdi:
                        line += f" → TDI: {tdi}"
                    parts.append(line)

        # Directory info
        directory = chunk.get("directory")
        dir_desc = chunk.get("dir_description")
        if directory:
            line = f"  **Directory**: {directory}"
            if dir_desc:
                line += f' — "{dir_desc}"'
            parts.append(line)

        parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# search_imas formatter
# ---------------------------------------------------------------------------


def format_imas_report(
    paths: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
    facility_xrefs: dict[str, dict[str, Any]],
    version_context: dict[str, dict[str, Any]],
    scores: dict[str, float],
) -> str:
    """Format IMAS DD search results into a readable report.

    Args:
        paths: Enriched IMAS path records.
        clusters: Cluster search results.
        facility_xrefs: Facility cross-references keyed by path ID.
        version_context: Version context keyed by path ID.
        scores: Map of path ID → similarity score.

    Returns:
        Formatted text report.
    """
    if not paths and not clusters:
        return "No IMAS paths found."

    parts: list[str] = []

    if paths:
        parts.append(f"## IMAS Paths ({len(paths)} matches)\n")

        for p in paths:
            pid = p.get("id", "?")
            score = scores.get(pid)
            score_str = f" (score: {score:.2f})" if score is not None else ""
            parts.append(f"### {pid}{score_str}")

            doc = p.get("documentation") or ""
            if doc:
                parts.append(f'  "{doc}"')

            # Metadata line
            meta_parts: list[str] = []
            ids_name = p.get("ids")
            if ids_name:
                meta_parts.append(f"IDS: {ids_name}")
            dtype = p.get("data_type")
            if dtype:
                meta_parts.append(f"Type: {dtype}")
            unit = p.get("unit")
            if unit:
                meta_parts.append(f"Unit: {unit}")
            if meta_parts:
                parts.append("  " + " | ".join(meta_parts))

            domain = p.get("physics_domain")
            if domain:
                parts.append(f"  Physics domain: {domain}")

            clusters_list = p.get("clusters") or []
            if clusters_list:
                parts.append(
                    f"  Clusters: {', '.join(f'"{c}"' for c in clusters_list)}"
                )

            coords = p.get("coordinates") or []
            if coords:
                parts.append(f"  Coordinates: {', '.join(coords)}")

            introduced = p.get("introduced_in")
            if introduced:
                parts.append(f"  Introduced: DD {introduced}")

            cocos = p.get("cocos_label_transformation")
            if cocos:
                parts.append(f"  COCOS: {cocos}")

            lifecycle = p.get("lifecycle_status")
            if lifecycle and lifecycle != "active":
                parts.append(f"  Lifecycle: {lifecycle}")

            structure_ref = p.get("structure_reference")
            if structure_ref:
                parts.append(f"  Structure: {structure_ref}")

            # Facility cross-references
            xref = facility_xrefs.get(pid, {})
            if any(
                xref.get(k) for k in ("facility_signals", "wiki_mentions", "code_files")
            ):
                parts.append("")
                facility_sigs = xref.get("facility_signals") or []
                if facility_sigs:
                    parts.append(f"  Signals: {', '.join(facility_sigs)}")
                wiki_mentions = xref.get("wiki_mentions") or []
                if wiki_mentions:
                    parts.append(
                        f"  Wiki: mentioned in {', '.join(f'"{s}"' for s in wiki_mentions)}"
                    )
                code_files = xref.get("code_files") or []
                if code_files:
                    parts.append(f"  Code: {', '.join(code_files)}")

            # Version context
            vctx = version_context.get(pid, {})
            changes = vctx.get("notable_changes") or []
            if changes:
                parts.append(
                    f"\n  **Version history** ({vctx.get('change_count', 0)} changes):"
                )
                for ch in changes:
                    if isinstance(ch, dict):
                        ver = ch.get("version", "?")
                        ctype = ch.get("type", "")
                        summary = ch.get("summary", "")
                        parts.append(f"    DD {ver} [{ctype}]: {summary}")

            parts.append("")

    if clusters:
        # Deduplicate clusters by label (same cluster can appear with
        # different scopes: ids, global, domain)
        deduped: dict[str, dict[str, Any]] = {}
        for cl in clusters:
            label = cl.get("label") or cl.get("id", "?")
            if label not in deduped:
                deduped[label] = {
                    "label": label,
                    "id": cl.get("id", ""),
                    "scopes": [],
                    "path_count": cl.get("path_count"),
                    "sample_paths": cl.get("sample_paths") or [],
                    "score": scores.get(cl.get("id", "")),
                }
            scope = cl.get("scope")
            if scope and scope not in deduped[label]["scopes"]:
                deduped[label]["scopes"].append(scope)
            # Keep highest score
            cl_score = scores.get(cl.get("id", ""))
            if cl_score and (
                deduped[label]["score"] is None or cl_score > deduped[label]["score"]
            ):
                deduped[label]["score"] = cl_score

        unique_clusters = list(deduped.values())
        parts.append(f"\n## Related Clusters ({len(unique_clusters)} matches)")
        for cl in unique_clusters:
            label = cl["label"]
            score = cl.get("score")
            score_str = f" (score: {score:.2f})" if score is not None else ""
            parts.append(f'  "{label}"{score_str}')
            scopes = cl.get("scopes") or []
            path_count = cl.get("path_count")
            meta = []
            if scopes:
                meta.append(f"Scope: {', '.join(scopes)}")
            if path_count:
                meta.append(f"{path_count} paths")
            if meta:
                parts.append(f"    {' | '.join(meta)}")
            sample = cl.get("sample_paths") or []
            if sample:
                parts.append(f"    Sample: {', '.join(sample)}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# fetch formatter
# ---------------------------------------------------------------------------


def format_fetch_report(chunks: list[dict[str, Any]]) -> str:
    """Format fetched resource chunks into a full-content report.

    Used by the ``fetch`` tool for WikiPage, Document, and CodeFile
    results. Chunks are expected to be pre-sorted by chunk_index.

    Args:
        chunks: Ordered chunk records with keys: source_type, title,
            url, source_id, section, text, chunk_index, plus optional
            mdsplus_paths and imas_paths.

    Returns:
        Formatted text report with all chunks in reading order.
    """
    if not chunks:
        return "No content found."

    first = chunks[0]
    source_type = first.get("source_type", "document")
    title = first.get("title") or first.get("source_id") or "Untitled"
    url = first.get("url") or ""
    source_id = first.get("source_id") or ""

    type_label = {
        "wiki_page": "Wiki Page",
        "document": "Wiki Document",
        "code": "Code File",
    }.get(source_type, source_type.title())

    parts: list[str] = [f"## {type_label}: {title}"]
    if source_id:
        parts.append(f"ID: {source_id}")
    if url:
        parts.append(f"URL: {url}")
    parts.append(f"Chunks: {len(chunks)}")

    # Collect cross-referenced paths across all chunks
    all_mdsplus: set[str] = set()
    all_imas: set[str] = set()

    current_section = None
    for chunk in chunks:
        section = chunk.get("section") or ""
        text = chunk.get("text") or ""
        idx = chunk.get("chunk_index")

        if section and section != current_section:
            current_section = section
            parts.append(f"\n### {section}")

        if source_type == "code":
            fn_name = section or ""
            line = idx or 0
            header = f"\n#### {fn_name}" if fn_name else ""
            if header:
                parts.append(header)
            if line:
                parts.append(f"Line {line}:")
            parts.append(f"```\n{text}\n```")
        else:
            parts.append(f"\n{text}")

        # Accumulate cross-references
        mds = chunk.get("mdsplus_paths") or []
        if isinstance(mds, str):
            mds = [mds]
        all_mdsplus.update(p for p in mds if p)

        imas = chunk.get("imas_paths") or []
        if isinstance(imas, str):
            imas = [imas]
        all_imas.update(p for p in imas if p)

    if all_mdsplus or all_imas:
        parts.append("\n---\n### Cross-references")
        if all_mdsplus:
            parts.append(f"MDSplus paths: {', '.join(sorted(all_mdsplus))}")
        if all_imas:
            parts.append(f"IMAS paths: {', '.join(sorted(all_imas))}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# IMAS tool formatters (for promoted MCP tools)
# ---------------------------------------------------------------------------


def format_check_report(result: Any) -> str:
    """Format CheckPathsResult into a readable validation report."""
    tool_error = _format_tool_error(result)
    if tool_error:
        return tool_error

    if result.error:
        return f"Error: {result.error}"

    parts: list[str] = []
    summary = result.summary
    parts.append(
        f"## Path Validation ({summary.get('total', 0)} paths: "
        f"{summary.get('found', 0)} found, "
        f"{summary.get('not_found', 0)} not found)\n"
    )

    for item in result.results:
        status = "FOUND" if item.exists else "NOT FOUND"
        parts.append(f"  {item.path}: **{status}**")
        if item.exists:
            meta = []
            if item.data_type:
                meta.append(f"Type: {item.data_type}")
            if item.units:
                meta.append(f"Units: {item.units}")
            if item.ids_name:
                meta.append(f"IDS: {item.ids_name}")
            if meta:
                parts.append(f"    {' | '.join(meta)}")
        else:
            if item.suggestion:
                parts.append(f"    Suggestion: {item.suggestion}")
            if item.migration:
                parts.append(f"    Migration: {item.migration}")

    return "\n".join(parts)


def format_fetch_paths_report(result: Any) -> str:
    """Format FetchPathsResult into a detailed path documentation report."""
    tool_error = _format_tool_error(result)
    if tool_error:
        return tool_error

    parts: list[str] = []
    summary = _get_value(result, "summary", {}) or {}
    fetched = summary.get("fetched", 0)
    not_found_count = summary.get("not_found", 0)
    nf_str = f", {not_found_count} not found" if not_found_count else ""
    parts.append(f"## IMAS Path Details ({fetched} fetched{nf_str})\n")

    for node in _get_value(result, "nodes", []) or []:
        parts.append(f"### {node.path}")
        # Prefer enriched description over documentation
        if hasattr(node, "description") and node.description:
            parts.append(f'  "{node.description}"')
        elif node.documentation:
            parts.append(f'  "{node.documentation}"')

        meta = []
        if node.ids_name:
            meta.append(f"IDS: {node.ids_name}")
        if node.data_type:
            meta.append(f"Type: {node.data_type}")
        if node.units:
            meta.append(f"Units: {node.units}")
        if meta:
            parts.append(f"  {' | '.join(meta)}")

        if node.physics_domain:
            parts.append(f"  Physics domain: {node.physics_domain}")
        if node.coordinates:
            parts.append(f"  Coordinates: {', '.join(node.coordinates)}")
        labels = _stringify_cluster_labels(getattr(node, "cluster_labels", None))
        if labels:
            parts.append(f"  Clusters: {', '.join(f'"{c}"' for c in labels)}")
        parts.append("")

    for nf in _get_value(result, "not_found_paths", []) or []:
        parts.append(f"  {nf.path}: NOT FOUND ({nf.reason})")
        if nf.suggestion:
            parts.append(f"    Suggestion: {nf.suggestion}")

    for dep in _get_value(result, "deprecated_paths", []) or []:
        parts.append(f"  {dep.path}: DEPRECATED (since DD {dep.deprecated_in})")
        if dep.new_path:
            parts.append(f"    Replacement: {dep.new_path}")

    return "\n".join(parts)


def format_list_report(result: Any) -> str:
    """Format ListPathsResult into a path listing report."""
    tool_error = _format_tool_error(result)
    if tool_error:
        return tool_error

    parts: list[str] = []
    summary = result.summary
    parts.append(
        f"## IMAS Path Listing ({summary.get('total_paths', 0)} total paths)\n"
    )

    for item in result.results:
        if item.error:
            parts.append(f"### {item.query}: ERROR — {item.error}")
            continue

        header = f"### {item.query} ({item.path_count} paths)"
        if item.truncated_to:
            header += f" — showing first {item.truncated_to}"
        parts.append(header)

        if item.path_details:
            for d in item.path_details:
                dtype = d.get("data_type", "")
                units = d.get("units", "")
                doc = d.get("documentation", "")
                line = f"  {d['id']}"
                if dtype:
                    line += f" ({dtype})"
                if units:
                    line += f" [{units}]"
                if doc:
                    line += f" — {doc[:100]}"
                parts.append(line)
        elif isinstance(item.paths, list):
            for p in item.paths:
                parts.append(f"  {p}")
        elif isinstance(item.paths, str):
            parts.append(item.paths)
        parts.append("")

    return "\n".join(parts)


def format_overview_report(result: Any) -> str:
    """Format GetOverviewResult into an overview report."""
    tool_error = _format_tool_error(result)
    if tool_error:
        return tool_error

    parts: list[str] = [result.content, ""]

    if result.physics_domains:
        parts.append(f"**Physics domains**: {', '.join(result.physics_domains)}")

    if result.ids_statistics:
        parts.append(f"\n### IDS Summary ({len(result.available_ids)} IDS)\n")
        # Sort by path count descending
        sorted_ids = sorted(
            result.ids_statistics.items(),
            key=lambda kv: kv[1].get("path_count", 0),
            reverse=True,
        )
        for ids_name, stats in sorted_ids:
            count = stats.get("path_count", 0)
            desc = stats.get("description", "")
            domain = stats.get("physics_domain", "")
            line = f"  {ids_name} ({count} paths)"
            if domain:
                line += f" [{domain}]"
            parts.append(line)
            if desc:
                parts.append(f"    {desc[:120]}")

    if result.mcp_tools:
        parts.append(f"\n**Available tools**: {', '.join(result.mcp_tools)}")

    return "\n".join(parts)


def format_identifiers_report(result: Any) -> str:
    """Format GetIdentifiersResult into an identifiers report."""
    tool_error = _format_tool_error(result)
    if tool_error:
        return tool_error

    parts: list[str] = []
    analytics = result.analytics
    parts.append(
        f"## IMAS Identifier Schemas ({analytics.get('total_schemas', 0)} schemas, "
        f"{analytics.get('enumeration_space', 0)} total options)\n"
    )

    for schema in result.schemas:
        name = schema.get("path", "?")
        option_count = schema.get("option_count", 0)
        significance = schema.get("branching_significance", "")
        desc = schema.get("description", "")

        parts.append(f"### {name} ({option_count} options) [{significance}]")
        if desc:
            parts.append(f"  {desc}")

        options = schema.get("options", [])
        if options:
            for opt in options[:15]:
                if isinstance(opt, dict):
                    parts.append(
                        f"    {opt.get('index', '?')}: {opt.get('name', '?')}"
                        f" — {opt.get('description', '')}"
                    )
                else:
                    parts.append(f"    {opt}")
            if len(options) > 15:
                parts.append(f"    ... and {len(options) - 15} more")
        parts.append("")

    return "\n".join(parts)


def format_cluster_report(result: Any) -> str:
    """Format cluster search result dict into a readable report."""
    tool_error = _format_tool_error(result)
    if tool_error:
        return tool_error

    parts: list[str] = []
    clusters = _get_value(result, "clusters", []) or []

    parts.append(
        f"## IMAS Clusters ({_get_value(result, 'clusters_found', 0)} found)\n"
    )

    for cl in clusters:
        label = cl.get("label", "?")
        desc = cl.get("description", "")
        scope = cl.get("scope", "")
        cl_type = cl.get("type", "")
        relevance = cl.get("relevance_score")

        header = f"### {label}"
        if relevance:
            header += f" (relevance: {relevance:.2f})"
        parts.append(header)

        if desc:
            parts.append(f"  {desc}")

        meta = []
        if scope:
            meta.append(f"Scope: {scope}")
        if cl_type:
            meta.append(f"Type: {cl_type}")
        ids_list = cl.get("ids", [])
        if ids_list:
            meta.append(f"IDS: {', '.join(ids_list)}")
        if meta:
            parts.append(f"  {' | '.join(meta)}")

        paths = cl.get("paths", [])
        if paths:
            total = cl.get("total_paths", len(paths))
            shown = min(len(paths), 10)
            parts.append(f"  Paths ({shown} of {total}):")
            for p in paths[:10]:
                parts.append(f"    {p}")
            if total > 10:
                parts.append(f"    ... and {total - 10} more")
        parts.append("")

    return "\n".join(parts)


def format_search_imas_report(result: Any, cluster_result: Any | None = None) -> str:
    """Format SearchPathsResult + optional clusters into a combined report.

    This is the typed-result version of format_imas_report(), used when
    the Codex MCP delegates to the shared GraphSearchTool.
    """
    parts: list[str] = []

    tool_error = _format_tool_error(result)
    if tool_error:
        parts.append(tool_error)

    hits = _get_value(result, "hits", []) or []
    if hits:
        parts.append(f"## IMAS Paths ({len(hits)} matches)\n")

        for hit in hits:
            score_str = f" (score: {hit.score:.2f})" if hit.score else ""
            parts.append(f"### {hit.path}{score_str}")

            # Prefer enriched description over raw documentation
            if hit.description:
                parts.append(f'  "{hit.description}"')
            elif hit.documentation:
                parts.append(f'  "{hit.documentation}"')

            meta = []
            if hit.ids_name:
                meta.append(f"IDS: {hit.ids_name}")
            if hit.data_type:
                meta.append(f"Type: {hit.data_type}")
            if hit.units:
                meta.append(f"Unit: {hit.units}")
            if meta:
                parts.append(f"  {' | '.join(meta)}")

            if hit.physics_domain:
                parts.append(f"  Physics domain: {hit.physics_domain}")
            if hit.coordinates:
                parts.append(f"  Coordinates: {', '.join(hit.coordinates)}")
            if hit.lifecycle_status and hit.lifecycle_status != "active":
                parts.append(f"  Lifecycle: {hit.lifecycle_status}")
            if hit.structure_reference:
                parts.append(f"  Structure: {hit.structure_reference}")
            if hit.introduced_after_version:
                parts.append(f"  Introduced: DD {hit.introduced_after_version}")
            if hit.keywords:
                parts.append(f"  Keywords: {', '.join(hit.keywords)}")

            # Facility cross-references
            xref = hit.facility_xrefs or {}
            if any(
                xref.get(k) for k in ("facility_signals", "wiki_mentions", "code_files")
            ):
                parts.append("")
                sigs = xref.get("facility_signals") or []
                if sigs:
                    parts.append(f"  Signals: {', '.join(sigs)}")
                wiki = xref.get("wiki_mentions") or []
                if wiki:
                    parts.append(
                        f"  Wiki: mentioned in {', '.join(f'"{s}"' for s in wiki)}"
                    )
                code = xref.get("code_files") or []
                if code:
                    parts.append(f"  Code: {', '.join(code)}")

            # Version context
            vctx = hit.version_context or {}
            changes = vctx.get("notable_changes") or []
            if changes:
                parts.append(
                    f"\n  **Version history** ({vctx.get('change_count', 0)} changes):"
                )
                for ch in changes:
                    if isinstance(ch, dict):
                        ver = ch.get("version", "?")
                        ctype = ch.get("type", "")
                        summary = ch.get("summary", "")
                        parts.append(f"    DD {ver} [{ctype}]: {summary}")

            parts.append("")

    cluster_hits = _get_value(cluster_result, "clusters", []) if cluster_result else []
    if cluster_result and cluster_hits:
        parts.append(format_cluster_report(cluster_result))
    elif cluster_result:
        cluster_error = _format_tool_error(cluster_result)
        if cluster_error:
            parts.append(cluster_error)

    if not parts:
        return "No IMAS paths found."

    return "\n".join(parts)


def format_path_context_report(result: dict[str, Any]) -> str:
    """Format get_imas_path_context result into readable text."""
    tool_error = _format_tool_error(result)
    if tool_error:
        return tool_error

    parts: list[str] = []
    path = result.get("path", "")
    sections = result.get("sections", {})
    total = result.get("total_connections", 0)

    parts.append(f"## Path Context: {path}")
    parts.append(f"Total cross-IDS connections: {total}\n")

    if "cluster_siblings" in sections:
        siblings = sections["cluster_siblings"]
        parts.append(f"### Cluster Siblings ({len(siblings)} paths)")
        by_cluster: dict[str, list[dict[str, Any]]] = {}
        for s in siblings:
            by_cluster.setdefault(s["cluster"], []).append(s)
        for cluster, items in by_cluster.items():
            parts.append(f"\n**{cluster}**")
            for item in items:
                doc = f" — {item['doc']}" if item.get("doc") else ""
                parts.append(f"  - `{item['path']}`{doc}")
        parts.append("")

    if "coordinate_partners" in sections:
        partners = sections["coordinate_partners"]
        parts.append(f"### Coordinate Partners ({len(partners)} paths)")
        by_coord: dict[str, list[dict[str, Any]]] = {}
        for p in partners:
            by_coord.setdefault(p["coordinate"], []).append(p)
        for coord, items in by_coord.items():
            parts.append(f"\n**{coord}**")
            for item in items:
                dtype = f" [{item['data_type']}]" if item.get("data_type") else ""
                parts.append(f"  - `{item['path']}`{dtype}")
        parts.append("")

    if "unit_companions" in sections:
        companions = sections["unit_companions"]
        parts.append(f"### Unit Companions ({len(companions)} paths)")
        by_unit: dict[str, list[dict[str, Any]]] = {}
        for c in companions:
            by_unit.setdefault(c["unit"], []).append(c)
        for unit, items in by_unit.items():
            parts.append(f"\n**{unit}**")
            for item in items:
                doc = f" — {item['doc']}" if item.get("doc") else ""
                parts.append(f"  - `{item['path']}`{doc}")
        parts.append("")

    if "identifier_links" in sections:
        links = sections["identifier_links"]
        parts.append(f"### Identifier Schema Links ({len(links)} paths)")
        by_schema: dict[str, list[dict[str, Any]]] = {}
        for lnk in links:
            by_schema.setdefault(lnk["schema"], []).append(lnk)
        for schema, items in by_schema.items():
            parts.append(f"\n**{schema}**")
            for item in items:
                parts.append(f"  - `{item['path']}`")
        parts.append("")

    if not sections:
        parts.append("No cross-IDS connections found for this path.")

    return "\n".join(parts)


def format_structure_report(result: dict[str, Any]) -> str:
    """Format analyze_imas_structure result into readable text."""
    tool_error = _format_tool_error(result)
    if tool_error:
        return tool_error

    parts: list[str] = []
    ids_name = result.get("ids_name", "")
    dd_version = result.get("dd_version")

    header = f"## IDS Structure Analysis: {ids_name}"
    if dd_version is not None:
        header += f" (DD v{dd_version})"
    parts.append(header + "\n")

    # Version context note when filtered
    version_ctx = result.get("version_context")
    if version_ctx:
        parts.append(f"> {version_ctx['note']}")
        dep = version_ctx.get("deprecated_in_or_before", 0)
        ren = version_ctx.get("renamed_paths", 0)
        if dep or ren:
            ctx_parts = []
            if dep:
                ctx_parts.append(f"{dep} deprecated")
            if ren:
                ctx_parts.append(f"{ren} renamed")
            parts.append(
                f"> Version changes: {', '.join(ctx_parts)} paths in this IDS."
            )
        parts.append("")

    parts.append(f"- Total paths: {result.get('total_paths', 0)}")
    parts.append(f"- Leaf fields: {result.get('leaf_count', 0)}")
    parts.append(f"- Structures: {result.get('structure_count', 0)}")
    parts.append(f"- Max depth: {result.get('max_depth', 0)}")
    parts.append(f"- Avg depth: {result.get('avg_depth', 0)}")

    domains = result.get("physics_domains", [])
    if domains:
        parts.append("\n### Physics Domains")
        for d in domains:
            parts.append(f"  - {d['domain']}: {d['count']} paths")

    types = result.get("data_types", [])
    if types:
        parts.append("\n### Data Types")
        for t in types:
            parts.append(f"  - {t['type']}: {t['count']}")

    arrays = result.get("array_structures", [])
    if arrays:
        parts.append(f"\n### Array Structures ({len(arrays)})")
        for a in arrays[:20]:
            coords = ", ".join(a.get("coordinates", []))
            parts.append(f"  - `{a['path']}` → [{coords}]")
        if len(arrays) > 20:
            parts.append(f"  ... and {len(arrays) - 20} more")

    cocos = result.get("cocos_fields", [])
    if cocos:
        parts.append(f"\n### COCOS-Labeled Fields ({len(cocos)})")
        for c in cocos:
            parts.append(f"  - `{c['path']}` ({c['label']})")

    return "\n".join(parts)


def format_export_ids_report(result: dict[str, Any]) -> str:
    """Format export_imas_ids result into readable text."""
    tool_error = _format_tool_error(result)
    if tool_error:
        return tool_error

    parts: list[str] = []
    ids_name = result.get("ids_name", "")
    path_count = result.get("path_count", 0)
    leaf_only = result.get("leaf_only", False)

    label = "leaf fields" if leaf_only else "paths"
    parts.append(f"## IDS Export: {ids_name} ({path_count} {label})\n")

    for p in result.get("paths", []):
        path = p.get("path", "")
        doc = p.get("documentation", "")
        dtype = p.get("data_type", "")
        units = p.get("units", "")

        meta = []
        if dtype:
            meta.append(dtype)
        if units:
            meta.append(units)
        meta_str = f" [{', '.join(meta)}]" if meta else ""

        parts.append(f"- `{path}`{meta_str}")
        if doc:
            parts.append(f"  {doc}")

        coords = p.get("coordinates", [])
        if coords and any(coords):
            parts.append(f"  Coordinates: {', '.join(c for c in coords if c)}")
        clusters = p.get("clusters", [])
        if clusters and any(clusters):
            parts.append(f"  Clusters: {', '.join(c for c in clusters if c)}")

    return "\n".join(parts)


def format_export_domain_report(result: Any) -> str:
    """Format export_imas_domain result into readable text."""
    parts: list[str] = []
    if isinstance(result, dict):
        domain = result.get("domain", "")
        total = result.get("total_paths", 0)
        ids_count = result.get("ids_count", 0)
        resolved_domains = result.get("resolved_domains", []) or []
        resolution = result.get("resolution") or ""
        error = result.get("error")

        parts.append(f"## Physics Domain: {domain}")
        if resolved_domains:
            parts.append(f"Resolved domains: {', '.join(resolved_domains)}")
        if resolution:
            parts.append(f"Resolution: {resolution}")
        parts.append(f"Total paths: {total} across {ids_count} IDS\n")

        by_ids = result.get("by_ids", {}) or {}
        if error and not by_ids:
            parts.append(error)
            return "\n".join(parts)

        if not by_ids:
            parts.append("No paths matched the resolved domain query.")
            return "\n".join(parts)

        for ids_name, paths in sorted(by_ids.items()):
            parts.append(f"### {ids_name} ({len(paths)} paths)")
            for p in paths:
                path = p.get("path", "")
                doc = p.get("documentation", "")
                units = p.get("units", "")
                units_str = f" [{units}]" if units else ""
                parts.append(f"  - `{path}`{units_str}")
                if doc:
                    parts.append(f"    {doc}")
            parts.append("")

        return "\n".join(parts)

    tool_error = _format_tool_error(result)
    if tool_error:
        return tool_error

    domain = _get_value(result, "domain", "")
    total = _get_value(result, "total_paths", 0)
    ids_count = _get_value(result, "ids_count", 0)

    parts.append(f"## Physics Domain: {domain}")
    parts.append(f"Total paths: {total} across {ids_count} IDS\n")

    by_ids = _get_value(result, "by_ids", {}) or {}
    for ids_name, paths in sorted(by_ids.items()):
        parts.append(f"### {ids_name} ({len(paths)} paths)")
        for p in paths:
            path = p.get("path", "")
            doc = p.get("documentation", "")
            units = p.get("units", "")
            units_str = f" [{units}]" if units else ""
            parts.append(f"  - `{path}`{units_str}")
            if doc:
                parts.append(f"    {doc}")
        parts.append("")

    return "\n".join(parts)


def format_explain_report(result: Any) -> str:
    """Format explain_concept result into readable markdown."""
    parts: list[str] = []
    if isinstance(result, dict):
        concept = result.get("concept", "")
        detail_level = result.get("detail_level", "intermediate")
        sections = result.get("sections", [])

        parts.append(f"# {concept}")
        parts.append(f"Detail level: {detail_level}\n")

        if not sections:
            parts.append("No information found for this concept.")
            return "\n".join(parts)

        for section in sections:
            sec_type = section.get("type", "")
            title = section.get("title", sec_type)
            parts.append(f"## {title}\n")

            if sec_type == "clusters":
                for cluster in section.get("clusters", []):
                    label = cluster.get("label", "")
                    desc = cluster.get("description", "")
                    scope = cluster.get("scope", "")
                    score = cluster.get("score")
                    ids_list = cluster.get("ids", [])
                    parts.append(f"**{label}** (scope: {scope})")
                    if score is not None:
                        parts.append(f"  Relevance: {score}")
                    if desc:
                        parts.append(f"  {desc}")
                    if ids_list:
                        parts.append(f"  IDSs: {', '.join(str(i) for i in ids_list)}")
                    example_paths = cluster.get("example_paths", [])
                    if example_paths:
                        parts.append("  Example paths:")
                        for p in example_paths:
                            parts.append(f"    - `{p}`")
                    parts.append("")

            elif sec_type == "cocos":
                for v in section.get("versions", []):
                    version = v.get("version", "")
                    cocos_id = v.get("cocos_id", "")
                    parts.append(f"  - DD {version}: COCOS {cocos_id}")
                parts.append("")

            elif sec_type == "cocos_paths":
                for p in section.get("paths", []):
                    path = p.get("path", "")
                    ids_name = p.get("ids", "")
                    summary = p.get("summary", "")
                    parts.append(f"  - `{path}` ({ids_name})")
                    if summary:
                        parts.append(f"    {summary}")
                parts.append("")

            elif sec_type == "identifiers":
                for schema in section.get("schemas", []):
                    sid = schema.get("id", "")
                    sdesc = schema.get("description", "")
                    parts.append(f"**{sid}**")
                    if sdesc:
                        parts.append(f"  {sdesc}")
                    options = schema.get("options", [])
                    if options:
                        for opt in options:
                            oname = (
                                opt.get("name", "")
                                if isinstance(opt, dict)
                                else str(opt)
                            )
                            oidx = opt.get("index", "") if isinstance(opt, dict) else ""
                            odesc = (
                                opt.get("description", "")
                                if isinstance(opt, dict)
                                else ""
                            )
                            idx_str = f" ({oidx})" if oidx else ""
                            desc_str = f" — {odesc}" if odesc else ""
                            parts.append(f"  - {oname}{idx_str}{desc_str}")
                    parts.append("")

            elif sec_type == "ids":
                for ids_item in section.get("ids_list", []):
                    name = ids_item.get("name", "")
                    desc = ids_item.get("description", "")
                    domain = ids_item.get("physics_domain", "")
                    domain_str = f" [{domain}]" if domain else ""
                    parts.append(f"  - **{name}**{domain_str}")
                    if desc:
                        parts.append(f"    {desc}")
                parts.append("")

            elif sec_type == "paths":
                for p in section.get("paths", []):
                    path = p.get("path", "")
                    doc = p.get("documentation", "")
                    dtype = p.get("data_type", "")
                    units = p.get("units", "")
                    meta_parts = []
                    if dtype:
                        meta_parts.append(dtype)
                    if units:
                        meta_parts.append(units)
                    meta_str = f" ({', '.join(meta_parts)})" if meta_parts else ""
                    parts.append(f"  - `{path}`{meta_str}")
                    if doc:
                        parts.append(f"    {doc}")
                parts.append("")

            else:
                # Fallback for unknown section types
                parts.append(f"  {section}")
                parts.append("")

        return "\n".join(parts)

    return str(result) if result else "No results found."
