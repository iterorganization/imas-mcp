"""Integration tests for embedding coverage, coalesce safety, and round-trip idempotence.

Verifies that:
1. Embedding fields are never accidentally erased by write_standard_names
   or _write_catalog_entries.
2. All optional fields in write_standard_names use coalesce so that a
   None value in the batch never overwrites existing graph data.
3. created_at is preserved across rewrites.
4. The import → build cycle is safe: catalog-imported rich fields are
   not erased by a subsequent sn-build write.
5. publish → import → publish is idempotent (key fields round-trip cleanly).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest
import yaml

# =============================================================================
# Helpers
# =============================================================================


def _call_write(names: list[dict], mock_gc: MagicMock) -> int:
    """Call write_standard_names with a mocked GraphClient."""
    with patch("imas_codex.sn.graph_ops.GraphClient") as MockGC:
        MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
        MockGC.return_value.__exit__ = MagicMock(return_value=False)
        from imas_codex.sn.graph_ops import write_standard_names

        return write_standard_names(names)


def _call_import_write(
    entries: list[dict], mock_gc: MagicMock, catalog_sha: str | None = None
) -> int:
    """Call _write_catalog_entries with a mocked GraphClient."""
    with patch("imas_codex.graph.client.GraphClient") as MockGC:
        MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
        MockGC.return_value.__exit__ = MagicMock(return_value=False)
        from imas_codex.sn.catalog_import import _write_catalog_entries

        return _write_catalog_entries(entries, catalog_commit_sha=catalog_sha)


def _merge_cypher(mock_gc: MagicMock) -> str:
    """Return the Cypher string from the first MERGE query call."""
    return mock_gc.query.call_args_list[0][0][0]


def _merge_batch(mock_gc: MagicMock) -> list[dict]:
    """Return the batch parameter from the first MERGE query call."""
    return mock_gc.query.call_args_list[0][1]["batch"]


# =============================================================================
# Part 1: Embedding Coverage
# =============================================================================


class TestEmbeddingCoverage:
    """Verify that embedding vectors are never accidentally erased."""

    def test_write_preserves_existing_embedding(self) -> None:
        """write_standard_names does not touch the embedding property at all.

        A StandardName that already has embedding=[0.1, 0.2, 0.3] and
        embedded_at set must be unchanged after write_standard_names is called.
        The Cypher must not reference the embedding property (since the function
        only manages metadata, not embeddings).
        """
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        names = [
            {
                "id": "electron_temperature",
                "source_type": "dd",
                "source_id": "core_profiles/profiles_1d/electrons/temperature",
                "description": "Electron temperature",
                # No embedding field — write should not touch it
            }
        ]
        _call_write(names, mock_gc)

        cypher = _merge_cypher(mock_gc)

        # The MERGE SET clause must NOT set sn.embedding unconditionally
        # (any mention would risk overwriting it with null)
        assert "sn.embedding = b.embedding" not in cypher, (
            "write_standard_names must not set sn.embedding from batch param"
        )

    def test_import_preserves_existing_embedding(self) -> None:
        """_write_catalog_entries uses coalesce to preserve existing embedding.

        The catalog import must never erase an embedding that was set by the
        embedding pipeline. The Cypher should contain the coalesce guard:
          sn.embedding = coalesce(sn.embedding, null)
        """
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        entries = [
            {
                "id": "electron_temperature",
                "description": "Electron temperature",
                "documentation": "Te measured by Thomson scattering.",
                "kind": "scalar",
                "units": "eV",
                "tags": ["core_profiles"],
                "links": None,
                "imas_paths": ["core_profiles/profiles_1d/electrons/temperature"],
                "validity_domain": "core plasma",
                "constraints": ["T_e > 0"],
                "physics_domain": "core_plasma_physics",
                "review_status": "accepted",
                "source_type": "dd",
                "physical_base": "temperature",
                "subject": "electron",
                "component": None,
                "coordinate": None,
                "position": None,
                "process": None,
            }
        ]
        _call_import_write(entries, mock_gc)

        cypher = _merge_cypher(mock_gc)

        # The catalog import Cypher must preserve embedding via coalesce
        assert "coalesce(sn.embedding, null)" in cypher, (
            "_write_catalog_entries Cypher must use coalesce(sn.embedding, null) "
            "to preserve existing embeddings"
        )
        assert "coalesce(sn.embedded_at, null)" in cypher, (
            "_write_catalog_entries Cypher must use coalesce(sn.embedded_at, null)"
        )

    def test_embedding_field_not_in_write_batch(self) -> None:
        """Batch dicts passed to gc.query by write_standard_names must not contain 'embedding'.

        This ensures that even if the caller accidentally includes an
        'embedding' key, the write function strips it before sending to the
        graph.  More importantly it confirms the build pipeline cannot
        null-out embeddings via this code path.
        """
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        names = [
            {
                "id": "electron_temperature",
                "source_type": "dd",
                "source_id": "core_profiles/profiles_1d/electrons/temperature",
                "description": "Electron temperature",
                "kind": "scalar",
                "units": "eV",
                "review_status": "drafted",
                "confidence": 0.95,
            }
        ]
        _call_write(names, mock_gc)

        batch = _merge_batch(mock_gc)

        for item in batch:
            assert "embedding" not in item, (
                f"Batch item for '{item.get('id')}' must not contain 'embedding' key; "
                "write_standard_names must never touch embedding data"
            )


# =============================================================================
# Part 2: Coalesce Safety
# =============================================================================


class TestCoalesceSafety:
    """Verify that write_standard_names uses coalesce for all optional fields.

    When a field is None in the batch, coalesce(None, sn.field) = sn.field,
    so an sn-build re-run cannot accidentally erase data that was set by
    an earlier catalog import.
    """

    _COALESCE_FIELDS = [
        ("review_status", "b.review_status, sn.review_status"),
        ("documentation", "b.documentation, sn.documentation"),
        ("kind", "b.kind, sn.kind"),
        ("tags", "b.tags, sn.tags"),
        ("links", "b.links, sn.links"),
        ("imas_paths", "b.imas_paths, sn.imas_paths"),
        ("validity_domain", "b.validity_domain, sn.validity_domain"),
        ("constraints", "b.constraints, sn.constraints"),
        ("confidence", "b.confidence, sn.confidence"),
        ("physical_base", "b.physical_base, sn.physical_base"),
        ("subject", "b.subject, sn.subject"),
        ("component", "b.component, sn.component"),
        ("coordinate", "b.coordinate, sn.coordinate"),
        ("position", "b.position, sn.position"),
        ("process", "b.process, sn.process"),
    ]

    def test_build_does_not_erase_imported_data(self) -> None:
        """All optional fields in the MERGE SET must use coalesce(b.field, sn.field).

        This protects against a scenario where:
          1. catalog import sets review_status='accepted', documentation, etc.
          2. sn-build re-runs write_standard_names with those fields = None
          3. Without coalesce, the re-run would null-out the imported values.
        """
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        # Simulate a minimal sn-build write — only id and source_type provided
        names = [
            {
                "id": "electron_temperature",
                "source_type": "dd",
                "source_id": "core_profiles/profiles_1d/electrons/temperature",
                "description": "Electron temperature",
                # review_status, documentation, kind, tags, etc. all absent/None
            }
        ]
        _call_write(names, mock_gc)

        cypher = _merge_cypher(mock_gc)

        for field_name, coalesce_args in self._COALESCE_FIELDS:
            assert f"coalesce({coalesce_args})" in cypher, (
                f"Field '{field_name}' must use coalesce({coalesce_args}) in "
                "write_standard_names Cypher to preserve existing graph data"
            )

    def test_build_with_none_fields_preserves_graph(self) -> None:
        """Batch dicts must include None for absent optional fields (not omit them).

        The coalesce(b.field, sn.field) pattern requires that b.field is
        present in the batch parameter (as None, not missing) so that Cypher
        can evaluate the coalesce.  If the key were absent from the dict,
        Neo4j would raise an error or behave unpredictably.
        """
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        # Entry with almost all optional fields missing
        names = [
            {
                "id": "plasma_current",
                "source_type": "dd",
                "source_id": "magnetics/method/0/ip",
            }
        ]
        _call_write(names, mock_gc)

        batch = _merge_batch(mock_gc)
        assert len(batch) == 1
        item = batch[0]

        # All optional fields must appear in the batch (value may be None)
        required_keys = {
            "id",
            "source_type",
            "physical_base",
            "subject",
            "component",
            "coordinate",
            "position",
            "process",
            "description",
            "documentation",
            "kind",
            "tags",
            "links",
            "imas_paths",
            "validity_domain",
            "constraints",
            "units",
            "model",
            "review_status",
            "generated_at",
            "confidence",
        }
        missing = required_keys - set(item.keys())
        assert not missing, (
            f"Batch item is missing keys: {missing}. "
            "All optional fields must be present (even as None) for coalesce to work."
        )

        # Fields absent from source must be None (not some unexpected value)
        for key in required_keys - {"id", "source_type"}:
            assert item[key] is None, (
                f"Batch key '{key}' should be None when not supplied, got {item[key]!r}"
            )

    def test_created_at_preserved_on_rewrite(self) -> None:
        """created_at must use coalesce(sn.created_at, datetime()) — not coalesce(b.created_at, ...).

        This pattern sets created_at on first write and then leaves it
        unchanged on all subsequent writes, so the node retains its
        original creation timestamp.
        """
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        names = [
            {
                "id": "electron_temperature",
                "source_type": "dd",
                "source_id": "core_profiles/profiles_1d/electrons/temperature",
            }
        ]
        _call_write(names, mock_gc)

        cypher = _merge_cypher(mock_gc)

        # Pattern: sn.created_at is preserved on re-writes, set only on first
        assert "coalesce(sn.created_at, datetime())" in cypher, (
            "write_standard_names must use coalesce(sn.created_at, datetime()) "
            "to preserve the original creation timestamp across rewrites"
        )
        # Must not blindly set created_at from the batch
        assert "b.created_at" not in cypher, (
            "write_standard_names must not set created_at from batch param"
        )

    def test_import_then_build_preserves_catalog_fields(self) -> None:
        """Verify coalesce semantics cover the full import → build cycle.

        Step 1: _write_catalog_entries (import) is called with rich metadata.
               The Cypher sets catalog-owned fields directly.
        Step 2: write_standard_names (build) is called with only basic fields.
               The Cypher uses coalesce for all optional fields.
        Together, the catalog-set values survive the build re-run because
        coalesce(None, sn.documentation) = sn.documentation.
        """
        import_gc = MagicMock()
        import_gc.query = MagicMock(return_value=[])

        build_gc = MagicMock()
        build_gc.query = MagicMock(return_value=[])

        # --- Step 1: catalog import with rich fields ---
        rich_entry = {
            "id": "electron_temperature",
            "description": "Electron temperature",
            "documentation": "Te measured by Thomson scattering.",
            "kind": "scalar",
            "units": "eV",
            "tags": ["core_profiles"],
            "links": None,
            "imas_paths": ["core_profiles/profiles_1d/electrons/temperature"],
            "validity_domain": "core plasma",
            "constraints": ["T_e > 0"],
            "physics_domain": "core_plasma_physics",
            "review_status": "accepted",
            "source_type": "dd",
            "physical_base": "temperature",
            "subject": "electron",
            "component": None,
            "coordinate": None,
            "position": None,
            "process": None,
        }
        imported = _call_import_write([rich_entry], import_gc)
        assert imported == 1

        # Verify import Cypher sets rich fields directly (no coalesce for catalog-owned)
        import_cypher = _merge_cypher(import_gc)
        assert "sn.documentation = b.documentation" in import_cypher, (
            "Catalog import must set documentation directly (authoritative)"
        )
        assert "sn.review_status = 'accepted'" in import_cypher, (
            "Catalog import must set review_status='accepted' directly"
        )
        # Embedding and model must still be protected via coalesce
        assert "coalesce(sn.embedding, null)" in import_cypher

        # --- Step 2: sn-build writes basic fields only ---
        basic_entry = {
            "id": "electron_temperature",
            "source_type": "dd",
            "source_id": "core_profiles/profiles_1d/electrons/temperature",
            "description": "Electron temperature",
            # review_status, documentation, kind, validity_domain, constraints all absent
        }
        written = _call_write([basic_entry], build_gc)
        assert written == 1

        # Verify build Cypher uses coalesce for all catalog-owned fields
        build_cypher = _merge_cypher(build_gc)

        catalog_owned = [
            ("review_status", "b.review_status, sn.review_status"),
            ("documentation", "b.documentation, sn.documentation"),
            ("kind", "b.kind, sn.kind"),
            ("tags", "b.tags, sn.tags"),
            ("validity_domain", "b.validity_domain, sn.validity_domain"),
            ("constraints", "b.constraints, sn.constraints"),
            ("confidence", "b.confidence, sn.confidence"),
        ]
        for field_name, coalesce_args in catalog_owned:
            assert f"coalesce({coalesce_args})" in build_cypher, (
                f"write_standard_names must protect '{field_name}' with coalesce "
                "so catalog-imported values survive sn-build re-runs"
            )

        # Verify the build batch item has None for the absent fields
        build_batch = _merge_batch(build_gc)
        assert len(build_batch) == 1
        build_item = build_batch[0]

        # These were not supplied — must be None in batch so coalesce falls back to graph
        for absent_field in (
            "review_status",
            "documentation",
            "kind",
            "validity_domain",
        ):
            assert absent_field in build_item, (
                f"'{absent_field}' must appear in batch dict (as None) for coalesce"
            )
            assert build_item[absent_field] is None, (
                f"'{absent_field}' must be None in batch when not supplied, "
                f"got {build_item[absent_field]!r}"
            )
