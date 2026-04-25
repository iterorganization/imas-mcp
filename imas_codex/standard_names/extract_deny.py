"""Extract-phase deny filter for DD paths.

Filters out DD paths that pass the ``SN_SOURCE_CATEGORIES`` gate but should
not receive standard names.  Examples: boolean constraint selectors
(``use_exact_*``), engineering coil geometry, control-system force matrices.

Rules are declared in ``config/extract_deny.yaml``.  Each rule is a
(path_pattern) → skip mapping with a machine-readable ``skip_reason``.

Architecture choice: **Option B** from the W19A plan.  The DD classifier's
``node_category`` is correct (these ARE geometry/quantity leaves in the DD
sense), but the standard-name pipeline needs a finer gate.  Rather than a
full DD rebuild (Option A/C), we filter at extraction time and record
``StandardNameSource`` nodes with ``status='skipped'`` for the audit trail.

Path patterns use the same glob syntax as ``unit_overrides.yaml``:

- ``*``  matches a single path segment (no ``/``)
- ``**`` matches zero or more path segments (may contain ``/``)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

_CONFIG = Path(__file__).parent / "config" / "extract_deny.yaml"


@dataclass(frozen=True)
class DenyRule:
    """A single extract-deny rule."""

    path_pattern: str
    skip_reason: str
    reason: str

    def matches(self, path: str) -> bool:
        """Return True if *path* matches this rule's glob pattern."""
        return _glob_match(self.path_pattern, path)


def _glob_match(pattern: str, path: str) -> bool:
    """Glob-match a DD path against a pattern.

    - ``*``  matches a single path segment (no ``/``)
    - ``**`` matches zero or more path segments (may contain ``/``)
    - All other segment characters are compared literally.

    Same algorithm as ``unit_overrides._glob_match``.
    """
    parts = pattern.split("/")
    rx_segments: list[str] = []
    for p in parts:
        if p == "**":
            rx_segments.append("__DOUBLESTAR__")
        elif p == "*":
            rx_segments.append("[^/]+")
        elif "*" in p:
            # Single-segment wildcard mixed with literals.
            escaped = re.escape(p).replace(r"\*", "[^/]*")
            rx_segments.append(escaped)
        else:
            rx_segments.append(re.escape(p))

    joined = "/".join(rx_segments)
    joined = joined.replace("__DOUBLESTAR__/", "(?:.*/)?")
    joined = joined.replace("/__DOUBLESTAR__", "(?:/.*)?")
    joined = joined.replace("__DOUBLESTAR__", ".*")

    regex = f"^{joined}$"
    return re.match(regex, path) is not None


@lru_cache(maxsize=1)
def _load_rules() -> tuple[DenyRule, ...]:
    """Load and cache deny rules from YAML."""
    if not _CONFIG.exists():
        return ()
    doc = yaml.safe_load(_CONFIG.read_text()) or {}
    rules: list[DenyRule] = []
    for r in doc.get("rules", []):
        if not r.get("path_pattern"):
            raise ValueError(f"extract_deny.yaml: rule missing path_pattern: {r}")
        if not r.get("skip_reason"):
            raise ValueError(f"extract_deny.yaml: rule missing skip_reason: {r}")
        rules.append(
            DenyRule(
                path_pattern=r["path_pattern"],
                skip_reason=r["skip_reason"],
                reason=r.get("reason", ""),
            )
        )
    return tuple(rules)


def match_deny_rule(path: str) -> DenyRule | None:
    """Return the first matching deny rule for *path*, or ``None``."""
    for rule in _load_rules():
        if rule.matches(path):
            return rule
    return None
