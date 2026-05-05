"""Plan 40 A13 — AST-level naming alignment test.

Enforces §17 naming policy: public functions, classes, and module
attributes in ``imas_codex/standard_names/`` and ``imas_codex/llm/``
must not use the abbreviation ``sn``/``sns`` outside the §17.1
retain-list.

Internal/private patterns retained verbatim (§17.1):
- ``_sn_ids``, ``seen_per_sn``, ``max_per_sn`` — local variable names
- ``total_sn``, ``orphan_sn``, ``orphan_src``, ``stale_token_sn`` —
  Cypher result aliases
- Deprecated alias-bridge functions explicitly listed below.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# §17 alias-bridge — these names are explicitly retained for one
# release with a DeprecationWarning. They are NOT violations.
ALIAS_BRIDGE_NAMES: frozenset[str] = frozenset(
    {
        "search_similar_names",
        "search_similar_sns_with_full_docs",
        "fetch_docs_review_feedback_for_sns",
        "_segment_filter_search_sn",
        "_vector_search_sn",
        "_keyword_search_sn",
        "_fetch_nearby_sns",
    }
)

# Pre-existing SN-run / SN-id helpers outside Plan 40 §17 rename scope.
# Plan 40 only mandates renames for the search-facility surface; these
# infrastructure helpers (run lifecycle, claim atomicity, id regex) keep
# their abbreviations until a future plan covers them.
OUT_OF_SCOPE_NAMES: frozenset[str] = frozenset(
    {
        "clear_sn_subsystem",
        "create_sn_run_open",
        "finalize_sn_run",
        "update_sn_per_phase_costs",
        "bump_sn_run_counter",
        "update_sn_run_progress",
        "backfill_sn_run_telemetry",
        "_claim_sn_atomic",
        "_SN_ID_RE",
        "_fetch_existing_sn_names",
    }
)

EXEMPT: frozenset[str] = ALIAS_BRIDGE_NAMES | OUT_OF_SCOPE_NAMES

# Public modules under audit. The "sn" token is allowed in import paths
# (e.g. ``standard_names``) but not in defined names within these modules.
AUDITED_MODULES: tuple[Path, ...] = (
    Path("imas_codex/standard_names/search.py"),
    Path("imas_codex/standard_names/graph_ops.py"),
    Path("imas_codex/standard_names/enrich_workers.py"),
    Path("imas_codex/standard_names/workers.py"),
    Path("imas_codex/standard_names/review/audits.py"),
    Path("imas_codex/standard_names/review/enrichment.py"),
    Path("imas_codex/llm/sn_tools.py"),
)


def _name_uses_sn(name: str) -> bool:
    """Return True if *name* contains a forbidden ``sn``/``sns`` token.

    Detects whole-word ``sn`` or ``sns`` segments in snake_case names.
    Allows ``standard_name`` (legitimate full word).
    """
    parts = name.lower().split("_")
    return any(p in {"sn", "sns"} for p in parts)


def _collect_def_names(tree: ast.Module) -> list[tuple[str, int]]:
    """Return (name, lineno) for every top-level def/class in *tree*."""
    out: list[tuple[str, int]] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            out.append((node.name, node.lineno))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out.append((target.id, node.lineno))
    return out


@pytest.mark.parametrize("module_path", AUDITED_MODULES)
def test_no_sn_abbreviation_in_public_names(module_path: Path) -> None:
    """A13: top-level def/class/assignment names must use ``standard_name``.

    Exemptions (§17.1 retain-list): explicit alias-bridge names.
    """
    repo_root = Path(__file__).resolve().parents[2]
    full_path = repo_root / module_path
    source = full_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(full_path))

    violations: list[tuple[str, int]] = []
    for name, lineno in _collect_def_names(tree):
        if name in EXEMPT:
            continue
        if _name_uses_sn(name):
            violations.append((name, lineno))

    assert not violations, (
        f"§17 naming violation in {module_path}: {violations}. "
        f"Public names must spell out 'standard_name'. "
        f"Alias-bridge: {sorted(ALIAS_BRIDGE_NAMES)}. "
        f"Out-of-scope: {sorted(OUT_OF_SCOPE_NAMES)}."
    )
