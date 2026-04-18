"""Graph CLI sync commands — sync external specs (ISN grammar) to Neo4j."""

from __future__ import annotations

import dataclasses

import click

from imas_codex.graph.client import GraphClient


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
