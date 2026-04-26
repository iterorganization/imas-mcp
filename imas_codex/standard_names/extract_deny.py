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
    """A single extract-deny rule.

    Rules with only ``path_pattern`` match purely on the DD path (backward
    compatible).  Optional attribute predicates (``data_type_in``,
    ``units_empty``, ``doc_contains_any``) refine the match using node
    metadata — all specified predicates must hold.

    ``status`` controls the ``StandardNameSource.status`` written on match
    (default ``'skipped'`` for backward compatibility; new classes may use
    ``'not_physical_quantity'``).
    """

    path_pattern: str
    skip_reason: str
    reason: str
    # Optional attribute predicates — all must match if specified.
    data_type_in: tuple[str, ...] = ()
    units_empty: bool = False
    doc_contains_any: tuple[str, ...] = ()
    status: str = "skipped"

    def matches(self, path: str, node_attrs: dict | None = None) -> bool:
        """Return True if *path* (and optional *node_attrs*) match this rule.

        Path pattern is checked first; if predicates are defined, *node_attrs*
        must also satisfy them.
        """
        if not _glob_match(self.path_pattern, path):
            return False
        return _predicates_match(self, node_attrs)


def _predicates_match(rule: DenyRule, attrs: dict | None) -> bool:
    """Return True if all attribute predicates on *rule* are satisfied.

    When no predicates are defined the match is path-only (always True).
    When predicates exist but *attrs* is ``None`` the match fails — the
    caller did not supply enough context to evaluate.
    """
    has_predicates = bool(
        rule.data_type_in or rule.units_empty or rule.doc_contains_any
    )
    if not has_predicates:
        return True  # path-only rule
    if attrs is None:
        return False  # predicates exist but no attrs supplied
    if rule.data_type_in and attrs.get("data_type") not in rule.data_type_in:
        return False
    if rule.units_empty:
        u = attrs.get("units")
        if u not in (None, "", "-"):
            return False
    if rule.doc_contains_any:
        doc = (attrs.get("documentation") or "").lower()
        if not any(s.lower() in doc for s in rule.doc_contains_any):
            return False
    return True


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
        # Parse optional attribute predicates
        data_type_in: tuple[str, ...] = ()
        if "data_type_in" in r:
            raw = r["data_type_in"]
            if isinstance(raw, str):
                data_type_in = (raw,)
            else:
                data_type_in = tuple(raw)
        doc_contains_any: tuple[str, ...] = ()
        if "doc_contains_any" in r:
            raw_doc = r["doc_contains_any"]
            if isinstance(raw_doc, str):
                doc_contains_any = (raw_doc,)
            else:
                doc_contains_any = tuple(raw_doc)
        rules.append(
            DenyRule(
                path_pattern=r["path_pattern"],
                skip_reason=r["skip_reason"],
                reason=r.get("reason", ""),
                data_type_in=data_type_in,
                units_empty=bool(r.get("units_empty", False)),
                doc_contains_any=doc_contains_any,
                status=r.get("status", "skipped"),
            )
        )
    return tuple(rules)


def match_deny_rule(
    path: str,
    node_attrs: dict | None = None,
) -> DenyRule | None:
    """Return the first matching deny rule for *path*, or ``None``.

    When *node_attrs* is supplied (dict with ``data_type``, ``units``,
    ``documentation`` keys), rules with attribute predicates can match.
    Without it, only path-only rules are evaluated.
    """
    for rule in _load_rules():
        if rule.matches(path, node_attrs):
            return rule
    return None
