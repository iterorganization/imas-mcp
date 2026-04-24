"""Compute a versioned hash of pipeline inputs that affect generated names.

This hash is written onto SNRun nodes at cycle start. On the NEXT cycle
start, if the hash has changed vs the last SNRun, the CLI warns and
exits with a non-zero code unless ``--skip-clear-gate`` is passed.

The gate prevents silent name-quality drift when prompts, classifier
code, or the ISN vocab version change between runs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# Prompt/config files that directly influence name generation quality.
# Path is relative to the project root (two parents up from this file).
PIPELINE_PROMPTS: list[Path] = [
    Path("imas_codex/llm/prompts/sn/compose_system.md"),
    Path("imas_codex/llm/prompts/sn/compose_dd.md"),
    Path("imas_codex/llm/prompts/sn/compose_dd_names.md"),
    Path("imas_codex/llm/prompts/sn/_grammar_reference.md"),
    Path("imas_codex/llm/prompts/sn/_scoring_rubric.md"),
    Path("imas_codex/llm/config/sn_review_criteria.yaml"),
]

# Core pipeline code that affects generated names
PIPELINE_CODE: list[Path] = [
    Path("imas_codex/standard_names/classifier.py"),
    Path("imas_codex/standard_names/enrichment.py"),
    Path("imas_codex/standard_names/consolidation.py"),
    Path("imas_codex/standard_names/vocab_token_filter.py"),
]


def compute_pipeline_hash() -> dict[str, str]:
    """Return a dict of ``{key: hash}`` for each pipeline input.

    Keys are the relative path strings.  A ``_composite`` key is
    appended with a hex digest of all individual hashes joined in
    sorted order — this is the single value stored on ``SNRun.pipeline_hash``.

    Files that do not exist are skipped (e.g. first-run or partial
    install).  The ``isn_version`` key reflects the installed
    ``imas-standard-names`` package version; falls back to ``"unknown"``.

    Returns
    -------
    dict[str, str]
        ``{relative_path: 16-hex-digest, ..., "isn_version": "x.y.z",
        "_composite": 16-hex-digest}``
    """
    root = Path(__file__).parents[2]
    result: dict[str, str] = {}

    for p in PIPELINE_PROMPTS + PIPELINE_CODE:
        fp = root / p
        if fp.exists():
            result[str(p)] = hashlib.sha256(fp.read_bytes()).hexdigest()[:16]

    try:
        from importlib.metadata import version

        result["isn_version"] = version("imas-standard-names")
    except Exception:  # noqa: BLE001
        result["isn_version"] = "unknown"

    # Composite — deterministic join of all leaf values sorted by key
    leaf_pairs = "|".join(
        f"{k}:{v}" for k, v in sorted(result.items()) if k != "_composite"
    )
    result["_composite"] = hashlib.sha256(leaf_pairs.encode()).hexdigest()[:16]
    return result


def diff_pipeline_hashes(old: dict[str, str], new: dict[str, str]) -> list[str]:
    """Return list of keys that differ between two hash dicts.

    Ignores the ``_composite`` key — callers use the list of changed
    leaf keys for human-readable warnings.
    """
    changed = []
    all_keys = (set(old) | set(new)) - {"_composite"}
    for k in sorted(all_keys):
        if old.get(k) != new.get(k):
            changed.append(k)
    return changed
