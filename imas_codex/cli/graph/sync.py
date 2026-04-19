"""Graph CLI sync commands — sync external specs (ISN grammar) to Neo4j."""

from __future__ import annotations

import dataclasses
from typing import Any

import click

from imas_codex.graph.client import GraphClient

# ---------------------------------------------------------------------------
# Post-sync finalisation (imas-codex pipeline state)
# ---------------------------------------------------------------------------
#
# ISN's ``sync_grammar`` is authoritative for the grammar *spec* nodes
# (segments, tokens, templates, NEXT/DEFINES/HAS_TOKEN edges). Two pieces
# of state are owned by imas-codex rather than ISN and live here:
#
# 1. Composite ``id`` properties — the LinkML schema in
#    ``imas_codex/schemas/grammar_graph.yaml`` declares ``id`` as the
#    identifier slot for every grammar node using a deterministic
#    composite format (e.g. ``{version}:{segment}:{value}``). ISN's sync
#    keys nodes on the natural composite ``(version, name)`` etc. but
#    does not project that into a single ``id`` slot, so we project it
#    here idempotently.
# 2. ``active`` flag rotation — "which grammar version is the running
#    composition pipeline using" is imas-codex pipeline state (ADR-8),
#    not ISN spec state. The rotation sets ``active=true`` on the
#    current sync target and ``active=false`` on every other version.
#
# All statements are MERGE-free idempotent SETs keyed by the composite
# natural key, so re-running is a no-op at the database level.


_FINALISE_STATEMENTS: tuple[tuple[str, str], ...] = (
    # id backfill covers *all* versions — composite ids are required by the
    # LinkML schema for every ISN grammar node regardless of active state,
    # and additive version bumps leave historical nodes in place (per ISN
    # sync_grammar docstring).
    (
        "set ISNGrammarVersion.id",
        "MATCH (v:ISNGrammarVersion) "
        "WHERE v.id IS NULL AND v.version IS NOT NULL "
        "SET v.id = v.version",
    ),
    (
        "set GrammarSegment.id",
        "MATCH (s:GrammarSegment) "
        "WHERE s.id IS NULL AND s.version IS NOT NULL AND s.name IS NOT NULL "
        "SET s.id = s.version + ':' + s.name",
    ),
    (
        "set GrammarToken.id",
        "MATCH (t:GrammarToken) "
        "WHERE t.id IS NULL AND t.version IS NOT NULL "
        "  AND t.segment IS NOT NULL AND t.value IS NOT NULL "
        "SET t.id = t.version + ':' + t.segment + ':' + t.value",
    ),
    (
        "set GrammarTemplate.id",
        "MATCH (tpl:GrammarTemplate) "
        "WHERE tpl.id IS NULL AND tpl.version IS NOT NULL AND tpl.name IS NOT NULL "
        "SET tpl.id = tpl.version + ':template:' + tpl.name",
    ),
    # Active rotation is *version-scoped*: exactly one ISNGrammarVersion
    # node should have active=true (the current sync target).
    (
        "rotate ISNGrammarVersion.active flag",
        "MATCH (v:ISNGrammarVersion) SET v.active = (v.version = $version)",
    ),
)


def _finalise_active_version(
    gc: GraphClient, version: str, dry_run: bool
) -> dict[str, Any]:
    """Set composite ``id`` props + rotate ``active`` flag to ``version``.

    Idempotent: safe to re-run. Returns a report dict for the CLI to
    surface to the user.
    """
    report: dict[str, Any] = {"target_version": version, "applied": not dry_run}

    if dry_run:
        report["planned_statements"] = [
            (label, cypher) for label, cypher in _FINALISE_STATEMENTS
        ]
        return report

    for label, cypher in _FINALISE_STATEMENTS:
        gc.query(cypher, version=version)
        report[label] = "ok"
    return report


@click.command("sync-isn-grammar")
@click.option("--dry-run", is_flag=True, help="Log Cypher without executing.")
@click.option(
    "--skip-grammar-sync",
    is_flag=True,
    help="Skip sync entirely (release-CLI hook only; prints warning).",
)
@click.option("-v", "--verbose", is_flag=True, help="Show planned Cypher statements.")
def sync_isn_grammar(dry_run: bool, skip_grammar_sync: bool, verbose: bool) -> None:
    """Sync ISN grammar spec (segments, tokens, templates, NEXT edges) to Neo4j.

    Reads the grammar spec from the installed ``imas_standard_names`` package
    and writes the ``ISNGrammarVersion``, ``GrammarSegment``, ``GrammarToken``,
    and ``GrammarTemplate`` nodes (plus ``NEXT``/``DEFINES``/``HAS_TOKEN``
    edges) to the active Neo4j graph.

    Hard-fails on Neo4j unreachable (plan 29 M3) unless ``--skip-grammar-sync``
    is provided.
    """
    if skip_grammar_sync:
        click.secho(
            "⚠ Grammar sync skipped (--skip-grammar-sync). "
            "Run `imas-codex graph sync-isn-grammar` manually once Neo4j is reachable.",
            fg="yellow",
        )
        return

    from imas_standard_names import __version__ as isn_version
    from imas_standard_names.graph.spec import get_grammar_graph_spec
    from imas_standard_names.graph.sync import sync_grammar

    spec = get_grammar_graph_spec()
    click.echo(
        f"ISN version: {isn_version}  "
        f"spec_version: {spec.get('version')}  "
        f"segments: {len(spec['segments'])}  "
        f"templates: {len(spec['templates'])}"
    )

    try:
        with GraphClient() as gc:
            report = sync_grammar(gc, active_version=isn_version, dry_run=dry_run)
            finalise_report = _finalise_active_version(
                gc, version=isn_version, dry_run=dry_run
            )
    except click.ClickException:
        raise
    except Exception as exc:  # noqa: BLE001 — hard-fail per M3
        raise click.ClickException(
            f"Failed to sync grammar to Neo4j: {exc}. "
            "Pass --skip-grammar-sync to bypass in release CLI."
        ) from exc

    click.secho(f"✓ Grammar sync complete (dry_run={dry_run})", fg="green")
    if dataclasses.is_dataclass(report):
        report_dict = dataclasses.asdict(report)
    elif hasattr(report, "__dict__"):
        report_dict = dict(report.__dict__)
    else:
        report_dict = dict(report)

    planned = report_dict.pop("planned_statements", None)
    for key, val in report_dict.items():
        click.echo(f"  {key}: {val}")

    if verbose and planned:
        click.echo(f"  planned_statements: {len(planned)}")
        for cypher, params in planned:
            click.echo(f"    {cypher}  params={params}")
    elif planned:
        click.echo(f"  planned_statements: {len(planned)} (use -v to show)")

    # Surface finalisation report (composite-id backfill + active flag rotation)
    click.secho(
        f"✓ Finalised active grammar version → {finalise_report['target_version']} "
        f"(applied={finalise_report['applied']})",
        fg="green",
    )
    if verbose and dry_run:
        for label, cypher in finalise_report.get("planned_statements", []):
            click.echo(f"    [{label}] {cypher}")
