"""DD unit override engine.

The IMAS Data Dictionary emits several invalid or context-dependent unit
strings that break the standard-name pipeline (prose names like
``"Elementary Charge Unit"``, unresolved Jinja templates like
``"m^dimension"``, and sentinel ``"1"`` values under
``pulse_schedule/*/reference``).  The DD team has confirmed these are
by-design; we handle them entirely here.

See ``plans/research/standard-names/dd-unit-bugs.md`` for the defect catalog.

The overrides are declared in ``config/unit_overrides.yaml``. Each rule is
a (path_pattern, dd_unit) → action mapping with one of two strategies:

- ``override``: replace the reported dd_unit with ``override_unit``; the
  candidate then flows through the pipeline normally.
- ``skip``: mark the DD path as a skipped ``StandardNameSource`` with a
  documented ``skip_reason``; the candidate is removed from extraction.

Path patterns use glob-style syntax where ``*`` matches a single path
segment and ``**`` matches zero or more segments.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

_CONFIG = Path(__file__).parent / "config" / "unit_overrides.yaml"


@dataclass(frozen=True)
class OverrideRule:
    """A single DD unit override/skip rule."""

    path_pattern: str
    dd_unit: str
    strategy: str  # 'override' | 'skip'
    override_unit: str | None
    skip_reason: str | None
    reason: str

    def matches(self, path: str, unit: str) -> bool:
        if self.dd_unit != unit:
            return False
        return _glob_match(self.path_pattern, path)


def _glob_match(pattern: str, path: str) -> bool:
    """Glob-match a DD path against a pattern.

    - ``*``  matches a single path segment (no ``/``)
    - ``**`` matches zero or more path segments (may contain ``/``)
    - All other segment characters are compared literally.
    """
    parts = pattern.split("/")
    rx_segments: list[str] = []
    for p in parts:
        if p == "**":
            rx_segments.append("__DOUBLESTAR__")
        elif p == "*":
            rx_segments.append("[^/]+")
        elif "*" in p:
            # Single-segment wildcard mixed with literals (rare, but supported).
            escaped = re.escape(p).replace(r"\*", "[^/]*")
            rx_segments.append(escaped)
        else:
            rx_segments.append(re.escape(p))

    # Join with literal "/" — then rewrite "__DOUBLESTAR__" boundaries so
    # that a "**" can absorb its flanking slashes ("" zero-segment match).
    joined = "/".join(rx_segments)
    # "/__DOUBLESTAR__/" → "(?:/.*)?/" (match zero or more segments)
    joined = joined.replace("__DOUBLESTAR__/", "(?:.*/)?")
    joined = joined.replace("/__DOUBLESTAR__", "(?:/.*)?")
    # Standalone "**" (whole-path pattern)
    joined = joined.replace("__DOUBLESTAR__", ".*")

    regex = f"^{joined}$"
    return re.match(regex, path) is not None


@lru_cache(maxsize=1)
def _load_rules() -> tuple[OverrideRule, ...]:
    """Load and cache the override rule set from YAML."""
    if not _CONFIG.exists():
        return ()
    doc = yaml.safe_load(_CONFIG.read_text()) or {}
    rules: list[OverrideRule] = []
    for r in doc.get("overrides", []):
        strategy = r.get("strategy")
        if strategy not in ("override", "skip"):
            raise ValueError(
                f"unit_overrides.yaml: invalid strategy '{strategy}' in rule {r}"
            )
        if strategy == "override" and not r.get("override_unit"):
            raise ValueError(
                f"unit_overrides.yaml: override rule missing override_unit: {r}"
            )
        if strategy == "skip" and not r.get("skip_reason"):
            raise ValueError(f"unit_overrides.yaml: skip rule missing skip_reason: {r}")
        rules.append(
            OverrideRule(
                path_pattern=r["path_pattern"],
                dd_unit=r["dd_unit"],
                strategy=strategy,
                override_unit=r.get("override_unit"),
                skip_reason=r.get("skip_reason"),
                reason=r["reason"],
            )
        )
    return tuple(rules)


def resolve_unit(path: str, dd_unit: str | None) -> tuple[str | None, dict | None]:
    """Return ``(effective_unit, metadata)`` for a (path, unit) pair.

    - If no rule matches: returns ``(dd_unit, None)`` — pass-through.
    - Override rule: returns ``(override_unit, {"rule": "override", ...})``.
    - Skip rule: returns ``(None, {"rule": "skip", "skip_reason": ..., ...})``.

    ``metadata`` is safe to persist on a ``StandardNameSource`` record when
    the path is skipped (keys ``skip_reason`` and ``skip_reason_detail``).
    """
    if dd_unit is None:
        return None, None
    for rule in _load_rules():
        if rule.matches(path, dd_unit):
            if rule.strategy == "override":
                return rule.override_unit, {
                    "rule": "override",
                    "original_unit": dd_unit,
                    "reason": rule.reason,
                    "path_pattern": rule.path_pattern,
                }
            if rule.strategy == "skip":
                return None, {
                    "rule": "skip",
                    "skip_reason": rule.skip_reason,
                    "skip_reason_detail": f"{rule.reason} (dd_unit={dd_unit!r})",
                    "original_unit": dd_unit,
                    "path_pattern": rule.path_pattern,
                }
    return dd_unit, None


def _reload_rules_for_tests() -> None:
    """Clear the rule cache. Only for use in tests."""
    _load_rules.cache_clear()
