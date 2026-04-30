"""Golden test: candidate schema in prompt must match StandardNameCandidate model.

This test is the regression gate for prompt ↔ schema drift.  It parses
the ``### Candidate Schema`` block in ``generate_name_system.md`` and
asserts the documented field set is exactly
``StandardNameCandidate.model_fields.keys()``.

Any addition or removal of a field in either the Pydantic model or the
prompt **must** be mirrored in the other; this test will catch the drift
at CI time.
"""

from __future__ import annotations

import re

import pytest

from imas_codex.llm.prompt_loader import PROMPTS_DIR
from imas_codex.standard_names.models import StandardNameCandidate

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _extract_candidate_schema_fields(prompt_text: str) -> set[str]:
    """Extract field names from the ``### Candidate Schema`` section.

    The section uses lines like:
        - `source_id`: full DD path …
        - `standard_name`: the composed name …

    We capture the backtick-delimited field name from each such line.
    """
    # Locate the section
    marker = "### Candidate Schema"
    idx = prompt_text.find(marker)
    if idx == -1:
        raise ValueError(f"'{marker}' section not found in prompt text")

    # Slice from the marker to the next heading or end-of-file
    rest = prompt_text[idx + len(marker) :]
    # Next heading: a line starting with '#'
    next_heading = re.search(r"(?m)^#{1,4}\s", rest)
    block = rest[: next_heading.start()] if next_heading else rest

    # Extract field names: lines matching ``- `field_name`:``
    fields: set[str] = set()
    for m in re.finditer(r"^-\s+`([a-z_]+)`\s*:", block, re.MULTILINE):
        fields.add(m.group(1))

    if not fields:
        raise ValueError(
            "No fields parsed from Candidate Schema block — check the prompt format"
        )
    return fields


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


class TestCandidateSchemaMatchesPromptSpec:
    """Prompt-documented fields must exactly match the Pydantic model."""

    @pytest.fixture(autouse=True)
    def _load(self) -> None:
        path = PROMPTS_DIR / "sn" / "generate_name_system.md"
        self.raw = path.read_text(encoding="utf-8")
        self.model_fields = set(StandardNameCandidate.model_fields.keys())

    def test_candidate_schema_matches_prompt_spec(self) -> None:
        """Field set in prompt == field set in StandardNameCandidate."""
        prompt_fields = _extract_candidate_schema_fields(self.raw)
        assert prompt_fields == self.model_fields, (
            f"Schema drift detected!\n"
            f"  In prompt but not model: {prompt_fields - self.model_fields}\n"
            f"  In model but not prompt: {self.model_fields - prompt_fields}"
        )

    def test_lean_prompt_also_matches(self) -> None:
        """Lean prompt schema must also match the model."""
        lean_path = PROMPTS_DIR / "sn" / "generate_name_system_lean.md"
        lean_raw = lean_path.read_text(encoding="utf-8")
        prompt_fields = _extract_candidate_schema_fields(lean_raw)
        assert prompt_fields == self.model_fields, (
            f"Lean prompt schema drift!\n"
            f"  In prompt but not model: {prompt_fields - self.model_fields}\n"
            f"  In model but not prompt: {self.model_fields - prompt_fields}"
        )
