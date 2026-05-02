"""Regression tests for refine-model dispatch.

Pin the contract that ``process_refine_name_batch`` and
``process_refine_docs_batch`` resolve their model via
``get_model("refine")`` — *not* ``get_model("language")`` (the
generate_docs / bulk-content tier).

This guards against the failure mode that triggered E3's collapse in
acceptance rate (cl=0 ~42% → cl=1+ ~5%): flash-lite refines could not
lift reviewer-critiqued names.  Peeling refine onto its own pyproject
section (default Sonnet 4.6) gives the refine pass a model with
sufficient capability to recover from feedback, and keeps the section
free to diverge from both compose (``sn-run``) and bulk content
(``language``) without coupling.
"""

from __future__ import annotations

import pytest

from imas_codex import settings

# ─── Section registration ────────────────────────────────────────────


def test_refine_section_registered() -> None:
    """``"refine"`` is a valid model section (so ``get_model`` accepts it)."""
    assert "refine" in settings.MODEL_SECTIONS


def test_refine_default_is_sonnet_class(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default refine model is in the Sonnet/Opus capability tier.

    E3 audit: flash-lite cannot lift critiqued names.  We assert the
    default contains 'sonnet' or 'opus' so a future config edit that
    drops it back to flash/haiku/mini will fail this guard.
    """
    monkeypatch.delenv("IMAS_CODEX_REFINE_MODEL", raising=False)
    # Bypass any pyproject override — we want to verify the *default*
    # baked into settings.py protects against accidental downgrades.
    model = settings._MODEL_DEFAULTS["refine"].lower()
    assert any(tier in model for tier in ("sonnet", "opus")), (
        f"Default refine model {model!r} must be Sonnet or Opus tier "
        f"(flash-lite refines accept at ~5% per E3 audit)."
    )


def test_refine_env_override_independent_of_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``IMAS_CODEX_REFINE_MODEL`` is honoured without leaking to language."""
    monkeypatch.setenv("IMAS_CODEX_REFINE_MODEL", "test/refine-only")
    monkeypatch.setenv("IMAS_CODEX_LANGUAGE_MODEL", "test/language-only")
    assert settings.get_model("refine") == "test/refine-only"
    assert settings.get_model("language") == "test/language-only"


def test_refine_does_not_alias_language(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dispatch contract: refine and language resolve via *different* keys.

    Future regressions where refine is silently routed back through
    ``get_model("language")`` would re-introduce the E3 failure mode.
    Override only the refine env var and assert language is unaffected
    (this is the property we care about — distinct dispatch keys).
    """
    monkeypatch.setenv("IMAS_CODEX_REFINE_MODEL", "test/refine-canary")
    monkeypatch.delenv("IMAS_CODEX_LANGUAGE_MODEL", raising=False)
    refine = settings.get_model("refine")
    language = settings.get_model("language")
    assert refine == "test/refine-canary"
    assert language != "test/refine-canary"


# ─── Worker-level dispatch ───────────────────────────────────────────


def test_refine_name_worker_imports_refine_section() -> None:
    """``process_refine_name_batch`` resolves its non-escalation model
    via ``get_model("refine")``.

    Source-level check: the worker must reference the ``"refine"``
    settings key (not ``"language"`` or ``"sn-run"``) for the
    non-escalation branch.  This is cheaper and more stable than
    spinning up an asyncio mock harness with the full GraphClient +
    fanout fakes that the worker requires at runtime.
    """
    import inspect

    from imas_codex.standard_names import workers

    src = inspect.getsource(workers.process_refine_name_batch)
    assert 'get_model("refine")' in src, (
        "process_refine_name_batch must dispatch via get_model('refine'); "
        "see commit history for E3 acceptance-rate audit."
    )
    # And it must NOT silently route through language tier.
    assert 'get_model("language")' not in src, (
        "process_refine_name_batch must not use get_model('language') — "
        "flash-lite refines accept critiqued names at ~5% (E3 audit)."
    )


def test_refine_docs_worker_imports_refine_section() -> None:
    """``process_refine_docs_batch`` resolves its non-escalation model
    via ``get_model("refine")`` (same contract as refine_name)."""
    import inspect

    from imas_codex.standard_names import workers

    src = inspect.getsource(workers.process_refine_docs_batch)
    assert 'get_model("refine")' in src
    assert 'get_model("language")' not in src
