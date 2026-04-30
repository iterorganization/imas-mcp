"""Config + catalog-version tests (plan 39 §6.1, §9, §12.2)."""

from __future__ import annotations

from imas_codex.standard_names.fanout import config as fanout_config
from imas_codex.standard_names.fanout.config import (
    FanoutSettings,
    load_settings,
    render_proposer_system_prompt,
)


class TestFanoutSettings:
    def test_defaults(self) -> None:
        s = FanoutSettings()
        assert s.enabled is False
        assert s.max_fan_degree == 3
        assert s.function_timeout_s == 5.0
        assert s.total_timeout_s == 12.0
        assert s.result_hit_cap == 8
        assert s.evidence_token_cap_baseline == 2000
        assert s.evidence_token_cap_escalation == 800
        # Tier helpers.
        assert (
            s.cap_for_charge(escalate=False) == s.fanout_max_charge_per_cycle_baseline
        )
        assert (
            s.cap_for_charge(escalate=True) == s.fanout_max_charge_per_cycle_escalation
        )
        assert s.evidence_token_cap_for(escalate=False) == 2000
        assert s.evidence_token_cap_for(escalate=True) == 800
        assert s.cost_estimate_for(escalate=False) == s.fanout_cost_estimate_baseline
        assert s.cost_estimate_for(escalate=True) == s.fanout_cost_estimate_escalation

    def test_load_settings_from_pyproject(self) -> None:
        # The shipped pyproject.toml has the section in place — load
        # it and confirm the master switch is False (Phase 1A).
        s = load_settings()
        assert isinstance(s, FanoutSettings)
        assert s.enabled is False
        assert s.sites == {"refine_name": False}


class TestCatalogVersion:
    def test_first_line_contains_hash(self) -> None:
        prompt = render_proposer_system_prompt()
        first_line = prompt.split("\n", 1)[0]
        assert first_line.startswith("catalog_version=")
        assert len(first_line) == len("catalog_version=") + 64  # sha256 hex

    def test_hash_covers_body(self, tmp_path, monkeypatch) -> None:
        """Mutating the prompt body flips the hash (plan 39 §6.1 I4)."""
        # Snapshot.
        original_path = fanout_config._prompt_path()
        original_text = original_path.read_text(encoding="utf-8")
        original_hash = fanout_config._compute_catalog_version()

        try:
            # Mutate one help-text character.  The change is below
            # the frontmatter so it lands in the post-frontmatter body.
            mutated = original_text.replace(
                "AT MOST 3", "AT MOST 3 (mutated for catalog-version test)"
            )
            assert mutated != original_text
            original_path.write_text(mutated, encoding="utf-8")

            fanout_config._reset_catalog_version_cache()
            new_hash = fanout_config._compute_catalog_version()
            assert new_hash != original_hash
        finally:
            # Restore.
            original_path.write_text(original_text, encoding="utf-8")
            fanout_config._reset_catalog_version_cache()
            assert fanout_config._compute_catalog_version() == original_hash

    def test_hash_stable_across_calls(self) -> None:
        h1 = fanout_config._compute_catalog_version()
        h2 = fanout_config._compute_catalog_version()
        assert h1 == h2
