"""Plan 39 Phase 0 (b) smoke test — verify ``related_dd_search`` signature.

The catalog runner ``find_related_dd_paths`` imports
``imas_codex.graph.dd_search.related_dd_search`` directly.  This test
locks down the public signature so a future rename or kwarg shuffle
trips immediately.
"""

from __future__ import annotations

import inspect


def test_related_dd_search_signature() -> None:
    """``related_dd_search(gc, path, *, max_results, ...)`` per plan 39 §3.1."""
    from imas_codex.graph.dd_search import related_dd_search

    sig = inspect.signature(related_dd_search)
    params = sig.parameters

    # Positional
    assert "gc" in params
    assert "path" in params
    gc_param = params["gc"]
    path_param = params["path"]
    assert gc_param.kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    assert path_param.kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )

    # Keyword-only
    assert "max_results" in params
    assert params["max_results"].kind == inspect.Parameter.KEYWORD_ONLY
    # Default per plan §3.2 is 12 from the catalog; helper keeps a higher
    # default of 20 since it's the general-purpose entry point.
    assert isinstance(params["max_results"].default, int)

    # Optional companions
    assert "relationship_types" in params
    assert "dd_version" in params


def test_related_dd_search_is_sync() -> None:
    """The catalog requires a sync helper (runner wraps in ``asyncio.to_thread``)."""
    from imas_codex.graph.dd_search import related_dd_search

    assert not inspect.iscoroutinefunction(related_dd_search), (
        "related_dd_search must remain sync; fan-out runners wrap it in "
        "asyncio.to_thread (plan 39 §4.2)."
    )
