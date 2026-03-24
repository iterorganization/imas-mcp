"""Graph CLI audit command — reference integrity and data quality checks."""

from __future__ import annotations

import click

from imas_codex.graph.client import GraphClient

_SCORE_DIMS = [
    "score_data_documentation",
    "score_physics_content",
    "score_code_documentation",
    "score_data_access",
    "score_calibration",
    "score_imas_relevance",
]


def _facility_filter(alias: str = "n") -> tuple[str, dict]:
    """Return (cypher_fragment, params) for optional facility filtering."""
    return f"{alias}.facility_id = $facility", {}


def _run_check(
    gc: GraphClient,
    name: str,
    cypher: str,
    *,
    facility: str | None = None,
    detail_cypher: str | None = None,
) -> list[dict]:
    """Run a single audit check, print result, return issues."""
    params: dict = {}
    if facility:
        params["facility"] = facility
    results = gc.query(cypher, **params)
    count = results[0]["cnt"] if results else 0
    if count:
        click.echo(click.style(f"  FAIL  {name}: {count}", fg="red"))
        if detail_cypher:
            details = gc.query(detail_cypher, **params)
            for d in details[:10]:
                parts = [f"{k}={v}" for k, v in d.items()]
                click.echo(f"         {', '.join(parts)}")
    else:
        click.echo(click.style(f"  OK    {name}", fg="green"))
    return results


@click.command("audit")
@click.option("--facility", "-f", default=None, help="Scope to a single facility.")
@click.option("--fix", is_flag=True, default=False, help="Auto-fix repairable issues.")
@click.option("--graph", "-g", "graph_name", default=None, help="Target graph name.")
def graph_audit(facility: str | None, fix: bool, graph_name: str | None) -> None:
    """Audit graph data for reference integrity and data quality.

    Checks WikiPages, Documents, Images, and WikiChunks for missing
    URLs, orphaned nodes, and missing score composites.

    Use --fix to auto-repair computable issues (e.g. score_composite
    from individual dimensions).
    """
    if graph_name:
        import os

        os.environ["IMAS_CODEX_GRAPH"] = graph_name

    fac_clause = "AND n.facility_id = $facility" if facility else ""
    params: dict = {"facility": facility} if facility else {}

    header = "Graph Audit"
    if facility:
        header += f" [{facility}]"
    click.echo(click.style(f"\n{header}", bold=True))
    click.echo("=" * len(header))

    with GraphClient.from_profile() as gc:
        # --- 1. WikiPages missing url ---
        _run_check(
            gc,
            "WikiPages missing url",
            f"""
            MATCH (n:WikiPage)
            WHERE (n.url IS NULL OR n.url = '') {fac_clause}
            RETURN count(n) AS cnt
            """,
            facility=facility,
            detail_cypher=f"""
            MATCH (n:WikiPage)
            WHERE (n.url IS NULL OR n.url = '') {fac_clause}
            RETURN n.id AS id, n.status AS status, n.facility_id AS facility
            ORDER BY n.facility_id, n.id
            LIMIT 10
            """,
        )

        # --- 2. Documents missing url ---
        _run_check(
            gc,
            "Documents missing url",
            f"""
            MATCH (n:Document)
            WHERE (n.url IS NULL OR n.url = '') {fac_clause}
            RETURN count(n) AS cnt
            """,
            facility=facility,
            detail_cypher=f"""
            MATCH (n:Document)
            WHERE (n.url IS NULL OR n.url = '') {fac_clause}
            RETURN n.id AS id, n.status AS status, n.document_type AS type,
                   n.facility_id AS facility
            ORDER BY n.facility_id, n.id
            LIMIT 10
            """,
        )

        # --- 3. Images missing source_url ---
        _run_check(
            gc,
            "Images missing source_url",
            f"""
            MATCH (n:Image)
            WHERE (n.source_url IS NULL OR n.source_url = '') {fac_clause}
            RETURN count(n) AS cnt
            """,
            facility=facility,
            detail_cypher=f"""
            MATCH (n:Image)
            WHERE (n.source_url IS NULL OR n.source_url = '') {fac_clause}
            RETURN n.id AS id, n.status AS status, n.facility_id AS facility
            ORDER BY n.facility_id, n.id
            LIMIT 10
            """,
        )

        # --- 4. Images with score dimensions but no score_composite ---
        img_fac = "AND img.facility_id = $facility" if facility else ""
        dims_present = " OR ".join(f"img.{d} IS NOT NULL" for d in _SCORE_DIMS)
        missing_composite_q = f"""
        MATCH (img:Image)
        WHERE img.score_composite IS NULL
          AND ({dims_present})
          {img_fac}
        RETURN count(img) AS cnt
        """
        results = _run_check(
            gc,
            "Images missing score_composite (have dimensions)",
            missing_composite_q,
            facility=facility,
            detail_cypher=f"""
            MATCH (img:Image)
            WHERE img.score_composite IS NULL
              AND ({dims_present})
              {img_fac}
            RETURN img.facility_id AS facility, count(img) AS cnt
            ORDER BY facility
            """,
        )

        missing_count = results[0]["cnt"] if results else 0
        if fix and missing_count > 0:
            # Compute score_composite as average of non-null dimensions
            dim_sum = " + ".join(
                f"CASE WHEN img.{d} IS NOT NULL THEN img.{d} ELSE 0 END"
                for d in _SCORE_DIMS
            )
            dim_count = " + ".join(
                f"CASE WHEN img.{d} IS NOT NULL THEN 1 ELSE 0 END" for d in _SCORE_DIMS
            )
            fix_q = f"""
            MATCH (img:Image)
            WHERE img.score_composite IS NULL
              AND ({dims_present})
              {img_fac}
            WITH img, ({dim_sum}) AS total, ({dim_count}) AS n_dims
            WHERE n_dims > 0
            SET img.score_composite = total / toFloat(n_dims)
            RETURN count(img) AS fixed
            """
            fix_results = gc.query(fix_q, **params)
            fixed = fix_results[0]["fixed"] if fix_results else 0
            click.echo(
                click.style(f"  FIXED score_composite on {fixed} images", fg="yellow")
            )

        # --- 5. Stray 'score' property on Image nodes (old bug residue) ---
        stray_q = f"""
        MATCH (img:Image)
        WHERE img.score IS NOT NULL {img_fac}
        RETURN count(img) AS cnt
        """
        stray_results = _run_check(
            gc,
            "Images with stray 'score' property (non-schema)",
            stray_q,
            facility=facility,
        )

        stray_count = stray_results[0]["cnt"] if stray_results else 0
        if fix and stray_count > 0:
            fix_stray_q = f"""
            MATCH (img:Image)
            WHERE img.score IS NOT NULL {img_fac}
            REMOVE img.score
            RETURN count(img) AS fixed
            """
            fix_results = gc.query(fix_stray_q, **params)
            fixed = fix_results[0]["fixed"] if fix_results else 0
            click.echo(
                click.style(
                    f"  FIXED removed stray 'score' from {fixed} images", fg="yellow"
                )
            )

        # --- 6. Orphan WikiChunks (no parent reference) ---
        chunk_fac = "AND wc.facility_id = $facility" if facility else ""
        _run_check(
            gc,
            "Orphan WikiChunks (no parent reference)",
            f"""
            MATCH (wc:WikiChunk)
            WHERE wc.wiki_page_id IS NULL AND wc.document_id IS NULL {chunk_fac}
            WITH wc
            WHERE NOT EXISTS {{ (parent)-[:HAS_CHUNK]->(wc) }}
            RETURN count(wc) AS cnt
            """,
            facility=facility,
            detail_cypher=f"""
            MATCH (wc:WikiChunk)
            WHERE wc.wiki_page_id IS NULL AND wc.document_id IS NULL {chunk_fac}
            WITH wc
            WHERE NOT EXISTS {{ (parent)-[:HAS_CHUNK]->(wc) }}
            RETURN wc.id AS id, wc.facility_id AS facility
            LIMIT 10
            """,
        )

        # --- 7. Ingested WikiPages with no chunks ---
        _run_check(
            gc,
            "Ingested WikiPages with no chunks",
            f"""
            MATCH (n:WikiPage)
            WHERE n.status = 'ingested' {fac_clause}
            AND NOT EXISTS {{ (n)-[:HAS_CHUNK]->(:WikiChunk) }}
            RETURN count(n) AS cnt
            """,
            facility=facility,
            detail_cypher=f"""
            MATCH (n:WikiPage)
            WHERE n.status = 'ingested' {fac_clause}
            AND NOT EXISTS {{ (n)-[:HAS_CHUNK]->(:WikiChunk) }}
            RETURN n.id AS id, n.facility_id AS facility
            ORDER BY n.facility_id, n.id
            LIMIT 10
            """,
        )

        # --- 8. Ingested Documents with no chunks ---
        _run_check(
            gc,
            "Ingested Documents with no chunks",
            f"""
            MATCH (n:Document)
            WHERE n.status = 'ingested' {fac_clause}
            AND NOT EXISTS {{ (n)-[:HAS_CHUNK]->(:WikiChunk) }}
            RETURN count(n) AS cnt
            """,
            facility=facility,
            detail_cypher=f"""
            MATCH (n:Document)
            WHERE n.status = 'ingested' {fac_clause}
            AND NOT EXISTS {{ (n)-[:HAS_CHUNK]->(:WikiChunk) }}
            RETURN n.id AS id, n.document_type AS type, n.facility_id AS facility
            ORDER BY n.facility_id, n.id
            LIMIT 10
            """,
        )

        # --- Summary ---
        click.echo("")
        if fix:
            click.echo(click.style("Audit complete (with fixes applied).", bold=True))
        else:
            click.echo(
                click.style(
                    "Audit complete. Use --fix to auto-repair computable issues.",
                    bold=True,
                )
            )
