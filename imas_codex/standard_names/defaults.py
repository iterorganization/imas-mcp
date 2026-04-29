"""Centralized defaults for the standard-names refine pipeline (Phase 8.1).

All tunable thresholds and timing knobs live here. Override at the CLI or in
``pyproject.toml`` under ``[tool.imas-codex.sn]`` rather than editing this
file.
"""

# Reviewer-score thresholds
DEFAULT_MIN_SCORE: float = 0.75
"""Minimum reviewer score (0-1) for a name or docs revision to be marked accepted."""

# Refine rotation cap
DEFAULT_REFINE_ROTATIONS: int = 3
"""Maximum REFINED_FROM (or DOCS_REVISION_OF) chain depth before exhaustion."""

# Escalation model — used on the final refine attempt before exhaustion
DEFAULT_ESCALATION_MODEL: str = "openrouter/anthropic/claude-opus-4.6"
"""Higher-capability model used on the final refine attempt (chain_length == cap-1)."""

# Orphan sweep timing
DEFAULT_ORPHAN_SWEEP_INTERVAL_S: int = 30
"""How often the orphan sweep coroutine runs (seconds)."""

DEFAULT_ORPHAN_SWEEP_TIMEOUT_S: int = 300
"""How long a *_stage='refining' claim may sit before being reverted (seconds)."""

# ── Backlog throttle caps ─────────────────────────────────────────────
# Upstream generators pause when downstream review queues exceed these
# counts, preventing unbounded backlog growth and wasted budget.

REVIEW_NAME_BACKLOG_CAP: int = 200
"""Max review_name pending items before generate_name / refine_name pause."""

REVIEW_DOCS_BACKLOG_CAP: int = 200
"""Max review_docs pending items before generate_docs / refine_docs pause."""
