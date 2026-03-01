"""LLM-based enrichment for static tree nodes.

Provides batch descriptions for MDSplus static tree nodes using
physics context from the tree path hierarchy, tags, and node types.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from imas_codex.discovery.base.llm import call_llm_structured
from imas_codex.settings import get_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class StaticNodeResult(BaseModel):
    """Enrichment result for a single static tree node."""

    path: str = Field(description="The node path (echo from input)")
    description: str = Field(
        description="Concise physics description of what this node stores (1-2 sentences)"
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Searchable keywords for this node (max 5)",
    )
    category: str = Field(
        default="",
        description=(
            "Physical category: geometry, coil, vessel, diagnostic, "
            "magnetic_probe, flux_loop, mesh, green_function, heating, or other"
        ),
    )


class StaticNodeBatch(BaseModel):
    """Batch of static node enrichment results from LLM."""

    results: list[StaticNodeResult]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _build_system_prompt(facility: str, tree_name: str) -> str:
    """Build system prompt for static tree enrichment."""
    from imas_codex.agentic.prompt_loader import render_prompt

    return render_prompt(
        "discovery/static-enricher",
        context={"facility": facility, "tree_name": tree_name},
    )


def _build_user_prompt(
    nodes: list[dict[str, Any]],
    version_descriptions: dict[int, str] | None = None,
    tree_context: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Build user prompt with nodes grouped by parent structure.

    Nodes sharing the same parent STRUCTURE are presented together so
    the LLM sees the full parameter set of each component (e.g., all
    R, Z, W, H, INOM for a single coil) in one block.

    Args:
        nodes: List of node dicts with path, node_type, tags, units.
        version_descriptions: Optional map of version number to description.
        tree_context: Optional dict mapping node path to hierarchy context
            from ``fetch_enrichment_context`` (parent info, sibling nodes).
    """
    lines: list[str] = []
    if version_descriptions:
        lines.append("## Version Context")
        for ver, desc in sorted(version_descriptions.items()):
            lines.append(f"- Version {ver}: {desc}")
        lines.append("")

    # Group nodes by parent path for coherent presentation
    ctx = tree_context or {}
    groups: dict[str | None, list[dict[str, Any]]] = {}
    for node in nodes:
        parent = ctx.get(node["path"], {}).get("parent_path")
        groups.setdefault(parent, []).append(node)

    lines.append("## Nodes to Describe")
    lines.append("")

    node_num = 0
    for parent_path, group_nodes in groups.items():
        # Emit parent header when available
        if parent_path:
            parent_info = ctx.get(group_nodes[0]["path"], {})
            parent_label = parent_path
            if parent_info.get("parent_tags"):
                ptags = parent_info["parent_tags"]
                if isinstance(ptags, list):
                    parent_label += f" (tags: {', '.join(ptags)})"
                else:
                    parent_label += f" (tag: {ptags})"
            lines.append(f"### Parent: {parent_label}")
            lines.append("")

            # Show sibling parameters not in this batch for context
            siblings = parent_info.get("siblings", [])
            if siblings:
                batch_paths = {n["path"] for n in group_nodes}
                other_sibs = [s for s in siblings if s["path"] not in batch_paths]
                if other_sibs:
                    sib_parts = []
                    for s in other_sibs:
                        sib_str = s["path"].rsplit(".", 1)[-1]
                        extras = []
                        if s.get("units"):
                            extras.append(s["units"])
                        if s.get("tags"):
                            t = s["tags"]
                            if isinstance(t, list):
                                extras.append(", ".join(t))
                            else:
                                extras.append(str(t))
                        if extras:
                            sib_str += f" [{'; '.join(extras)}]"
                        sib_parts.append(sib_str)
                    lines.append(f"other parameters: {', '.join(sib_parts)}")
                    lines.append("")

        for node in group_nodes:
            node_num += 1
            # Use leaf name as concise identifier
            leaf = (
                node["path"].rsplit(".", 1)[-1] if "." in node["path"] else node["path"]
            )
            lines.append(f"#### Node {node_num}: {leaf}")
            lines.append(f"path: {node['path']}")
            if node.get("node_type"):
                lines.append(f"type: {node['node_type']}")
            if node.get("tags"):
                tags = (
                    node["tags"] if isinstance(node["tags"], list) else [node["tags"]]
                )
                lines.append(f"tags: {', '.join(tags)}")
            if node.get("units"):
                lines.append(f"units: {node['units']}")
            if node.get("dtype"):
                lines.append(f"dtype: {node['dtype']}")
            if node.get("shape"):
                lines.append(f"shape: {node['shape']}")
            if node.get("scalar_value") is not None:
                lines.append(f"value: {node['scalar_value']}")
            # Pattern context: this node is a representative of an indexed group
            if node.get("_pattern_count"):
                lines.append(
                    f"pattern: {node['_pattern_count']} indexed instances "
                    f"under {node['_pattern_grandparent']} "
                    f"(leaf: {node['_pattern_leaf']})"
                )
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Enrichment entry point
# ---------------------------------------------------------------------------


def enrich_static_nodes(
    nodes: list[dict[str, Any]],
    facility: str,
    tree_name: str,
    version_descriptions: dict[int, str] | None = None,
    batch_size: int = 40,
) -> tuple[list[StaticNodeResult], float]:
    """Enrich a list of static tree nodes with LLM-generated descriptions.

    Args:
        nodes: List of node dicts with at least 'path' key.
        facility: Facility identifier.
        tree_name: MDSplus tree name (e.g., "static").
        version_descriptions: Map of version number to description.
        batch_size: Number of nodes per LLM call.

    Returns:
        Tuple of (results, total_cost_usd).
    """
    model = get_model("language")
    system_prompt = _build_system_prompt(facility, tree_name)
    all_results: list[StaticNodeResult] = []
    total_cost = 0.0

    for i in range(0, len(nodes), batch_size):
        batch = nodes[i : i + batch_size]
        user_prompt = _build_user_prompt(batch, version_descriptions)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            parsed, cost, _tokens = call_llm_structured(
                model=model,
                messages=messages,
                response_model=StaticNodeBatch,
            )
            all_results.extend(parsed.results)
            total_cost += cost
            logger.info(
                "Enriched batch %d-%d (%d results, $%.4f)",
                i + 1,
                i + len(batch),
                len(parsed.results),
                cost,
            )
        except Exception:
            logger.exception("Failed to enrich batch %d-%d", i + 1, i + len(batch))

    return all_results, total_cost
