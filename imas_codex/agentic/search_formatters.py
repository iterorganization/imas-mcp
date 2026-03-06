"""Formatters for unified MCP search tool output.

Pure functions that take structured query results and produce
formatted text reports for agent consumption. Output size is
controlled by the caller via the ``k`` parameter — formatters
render all results they receive without artificial truncation.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# search_signals formatter
# ---------------------------------------------------------------------------


def format_signals_report(
    signals: list[dict[str, Any]],
    tree_nodes: list[dict[str, Any]],
    scores: dict[str, float],
) -> str:
    """Format signal search results into a readable report.

    Args:
        signals: Enriched signal records from the enrichment query.
        tree_nodes: Tree node results from vector search.
        scores: Map of signal ID → similarity score.

    Returns:
        Formatted text report.
    """
    if not signals and not tree_nodes:
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
            if unit:
                meta_parts.append(f"Unit: {unit}")
            checked = sig.get("checked")
            shot = sig.get("example_shot")
            if checked and shot:
                meta_parts.append(f"Checked: shot {shot}")
            if meta_parts:
                parts.append("  " + " | ".join(meta_parts))

            # Data access section
            access_template = sig.get("access_template")
            access_type = sig.get("access_type")
            if access_template:
                parts.append(f"\n  **Data access** ({access_type or 'unknown'}):")
                imports = sig.get("imports_template")
                if imports:
                    parts.append(f"    {imports}")
                connection = sig.get("connection_template")
                if connection:
                    parts.append(f"    {connection}")
                parts.append(f"    {access_template}")

            # IMAS mapping section
            imas_path = sig.get("imas_path")
            if imas_path:
                parts.append(f"\n  **IMAS mapping**: {imas_path}")
                imas_docs = sig.get("imas_docs")
                if imas_docs:
                    parts.append(f'    "{imas_docs}"')
                imas_unit = sig.get("imas_unit")
                if imas_unit:
                    parts.append(f"    Unit: {imas_unit}")

            # Tree node section
            tree_path = sig.get("tree_path")
            data_source_name = sig.get("data_source_name")
            if tree_path:
                parts.append(
                    f"\n  **Tree node**: {tree_path}"
                    + (f" (tree: {data_source_name})" if data_source_name else "")
                )

            parts.append("")  # blank line between signals

    if tree_nodes:
        parts.append(f"\n## Related Tree Nodes ({len(tree_nodes)} matches)")
        for tn in tree_nodes:
            path = tn.get("path", "?")
            desc = tn.get("description") or ""
            tree = tn.get("data_source_name") or ""
            unit = tn.get("unit") or ""
            meta = []
            if tree:
                meta.append(f"tree: {tree}")
            if unit:
                meta.append(f"unit: {unit}")
            meta_str = f" ({', '.join(meta)})" if meta else ""
            line = f"  {path}{meta_str}"
            if desc:
                line += f" — {desc}"
            parts.append(line)

    return "\n".join(parts)


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

                linked_tree_nodes = chunk.get("linked_tree_nodes") or []
                if linked_tree_nodes:
                    parts.append(f"  Tree nodes: {', '.join(linked_tree_nodes)}")

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
            parts.append(f"  ```python\n  {text}\n  ```")

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
        parts.append(f"\n## Related Clusters ({len(clusters)} matches)")
        for cl in clusters:
            label = cl.get("label") or cl.get("id", "?")
            score = scores.get(cl.get("id", ""))
            score_str = f" (score: {score:.2f})" if score is not None else ""
            parts.append(f'  "{label}"{score_str}')
            scope = cl.get("scope")
            path_count = cl.get("path_count")
            if scope or path_count:
                meta = []
                if scope:
                    meta.append(f"Scope: {scope}")
                if path_count:
                    meta.append(f"{path_count} paths")
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
