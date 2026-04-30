"""Structured fan-out for the SN compose pipeline (plan 39).

A bounded, schema-validated, two-stage LLM pattern that pulls targeted
DD context for the ``refine_name`` worker:

1. **Stage A — Proposer LLM** emits a closed-catalog
   :class:`~imas_codex.standard_names.fanout.schemas.FanoutPlan`
   (Pydantic discriminated union on ``fn_id``; all bounds enforced at
   parse time).
2. **Stage B — Pure-Python executor** runs the plan in parallel via
   ``asyncio.to_thread`` + per-call/total ``wait_for``.
3. **Stage C — Synthesizer** is the call-site's existing LLM call,
   with the rendered evidence block injected into its prompt.

No agentic loop, no runtime function generation.  Default off; when
``[tool.imas-codex.sn.fanout].enabled`` is ``False`` (the default),
:func:`run_fanout` is a true no-op that returns ``""`` and writes
nothing to the graph.

Public API:
    :func:`run_fanout` — async orchestrator (entry point for the
    refine-name worker).
    :class:`FanoutSettings` — pyproject-loaded settings model.
    :func:`load_settings` — loader that returns :class:`FanoutSettings`.
    :class:`FanoutScope` — caller-injected scope.
    :class:`CandidateContext` — refine-site candidate metadata.
    :data:`CATALOG_VERSION` — sha256 of the rendered proposer prompt
    body (plan 39 §6.1 I4).
"""

from __future__ import annotations

from .config import (
    CATALOG_VERSION,
    FanoutSettings,
    load_settings,
    render_proposer_system_prompt,
)
from .dispatcher import run_fanout
from .schemas import (
    CandidateContext,
    FanoutHit,
    FanoutOutcome,
    FanoutPlan,
    FanoutResult,
    FanoutScope,
)

__all__ = [
    "CATALOG_VERSION",
    "CandidateContext",
    "FanoutHit",
    "FanoutOutcome",
    "FanoutPlan",
    "FanoutResult",
    "FanoutScope",
    "FanoutSettings",
    "load_settings",
    "render_proposer_system_prompt",
    "run_fanout",
]
