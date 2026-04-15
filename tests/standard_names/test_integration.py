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
6. Full E2E lifecycle: write_standard_names → get_validated_standard_names
   → graph_records_to_entries → generate_catalog_files → import_catalog.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest
import yaml

imas_sn = pytest.importorskip("imas_standard_names")

# =============================================================================
# Helpers
# =============================================================================


def _call_write(names: list[dict], mock_gc: MagicMock) -> int:
    """Call write_standard_names with a mocked GraphClient."""
    with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
        MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
        MockGC.return_value.__exit__ = MagicMock(return_value=False)
        from imas_codex.standard_names.graph_ops import write_standard_names

        return write_standard_names(names)


def _call_import_write(
    entries: list[dict], mock_gc: MagicMock, catalog_sha: str | None = None
) -> int:
    """Call _write_catalog_entries with a mocked GraphClient."""
    with patch("imas_codex.graph.client.GraphClient") as MockGC:
        MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
        MockGC.return_value.__exit__ = MagicMock(return_value=False)
        from imas_codex.standard_names.catalog_import import _write_catalog_entries

        return _write_catalog_entries(entries, catalog_commit_sha=catalog_sha)


def _merge_cypher(mock_gc: MagicMock) -> str:
    """Return the Cypher string from the first MERGE StandardName query call."""
    for c in mock_gc.query.call_args_list:
        cypher = c[0][0]
        if "MERGE (sn:StandardName" in cypher:
            return cypher
    raise AssertionError("No MERGE StandardName query found in calls")


def _merge_batch(mock_gc: MagicMock) -> list[dict]:
    """Return the batch parameter from the first MERGE StandardName query call."""
    for c in mock_gc.query.call_args_list:
        cypher = c[0][0]
        if "MERGE (sn:StandardName" in cypher:
            return c[1]["batch"]
    raise AssertionError("No MERGE StandardName query found in calls")


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
                "unit": "eV",
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
                "unit": "eV",
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
            "unit",
            "model",
            "review_status",
            "generated_at",
            "confidence",
            "reviewer_model",
            "reviewer_score",
            "reviewer_scores",
            "reviewer_comments",
            "reviewed_at",
            "review_tier",
            "vocab_gap_detail",
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
            "unit": "eV",
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


# =============================================================================
# Part 5: Round-trip idempotence
# =============================================================================

imas_sn = pytest.importorskip("imas_standard_names")

SAMPLE_GRAPH_RECORD: dict[str, Any] = {
    "name": "electron_temperature",
    "description": "Electron temperature profile",
    "documentation": "The $T_e$ profile measured by Thomson scattering.",
    "source": "dd",
    "source_path": "core_profiles/profiles_1d/electrons/temperature",
    "unit": "eV",
    "kind": "scalar",
    "tags": [],
    "links": [],
    "ids_paths": ["core_profiles/profiles_1d/electrons/temperature"],
    "constraints": ["T_e > 0"],
    "validity_domain": "core plasma",
    "confidence": 0.95,
    "model": "test/model",
    "ids_name": None,
    "physical_base": "temperature",
    "subject": "electron",
}

SAMPLE_CATALOG_ENTRY_RT: dict[str, Any] = {
    "name": "electron_temperature",
    "description": "Electron temperature",
    "documentation": "The electron temperature Te is measured by Thomson scattering.",
    "kind": "scalar",
    "unit": "eV",
    "tags": [],
    "links": [],
    "ids_paths": ["core_profiles/profiles_1d/electrons/temperature"],
    "validity_domain": "core plasma",
    "constraints": ["T_e > 0"],
    "physics_domain": "core_plasma_physics",
    "status": "active",
}


def _published_yaml_to_catalog(
    published_yaml: str, physics_domain: str = "unscoped"
) -> dict[str, Any]:
    """Convert a published YAML string to a catalog-importable dict.

    Strips provenance, adds ``physics_domain``, normalises ``status`` to
    ``"active"``, and converts links from ``[{name: …}]`` dicts to plain
    strings so that ``StandardNameEntry`` validation succeeds.
    """
    doc: dict[str, Any] = yaml.safe_load(published_yaml)
    # Provenance block is not part of the catalog schema
    doc.pop("provenance", None)
    # status must be a catalog-valid value
    doc["status"] = "active"
    # physics_domain is required by StandardNameEntry
    doc.setdefault("physics_domain", physics_domain)
    # Normalise links: published format uses [{name: link}] dicts
    raw_links = doc.get("links", [])
    if raw_links and isinstance(raw_links[0], dict):
        doc["links"] = [lnk.get("name", str(lnk)) for lnk in raw_links]
    # Ensure required list fields are present (even if empty)
    for list_field in ("tags", "links", "ids_paths", "constraints"):
        doc.setdefault(list_field, [])
    # Empty string validity_domain instead of None
    if doc.get("validity_domain") is None:
        doc["validity_domain"] = ""
    return doc


def _imported_dict_to_graph_record(d: dict[str, Any]) -> dict[str, Any]:
    """Normalise an imported graph dict for ``graph_records_to_entries``.

    ``import_catalog`` returns dicts with ``id`` / ``units`` / ``imas_paths``
    keys.  ``graph_records_to_entries`` looks for ``name``/``id``,
    ``unit``/``units``, and ``ids_paths``.  This helper adds the
    ``ids_paths`` alias so that the path list survives the round-trip.
    """
    rec = dict(d)
    # Alias imas_paths → ids_paths (graph_records_to_entries reads ids_paths)
    if "imas_paths" in rec and "ids_paths" not in rec:
        rec["ids_paths"] = rec["imas_paths"] or []
    return rec


def _key_fields(parsed_yaml: dict[str, Any]) -> dict[str, Any]:
    """Extract the semantic fields that must be preserved across a round-trip."""
    return {
        "name": parsed_yaml.get("name"),
        "kind": parsed_yaml.get("kind"),
        "unit": parsed_yaml.get("unit"),
        "description": parsed_yaml.get("description"),
        "documentation": parsed_yaml.get("documentation"),
        "ids_paths": sorted(parsed_yaml.get("ids_paths") or []),
        "validity_domain": parsed_yaml.get("validity_domain"),
        "constraints": sorted(parsed_yaml.get("constraints") or []),
    }


class TestRoundTripIdempotence:
    """Verify that publish → import → publish produces semantically identical YAML."""

    def test_publish_import_publish_idempotent(self, tmp_path: Path) -> None:
        """Round-trip: graph_records → YAML → catalog import → YAML should match.

        Key fields (name, kind, unit, ids_paths, validity_domain, constraints)
        must be identical after a full publish → import → publish cycle.
        Provenance and confidence fields are allowed to differ.
        """
        from imas_codex.standard_names.publish import (
            generate_catalog_files,
            graph_records_to_entries,
        )

        round1_dir = tmp_path / "round1"
        catalog_dir = tmp_path / "catalog"
        round2_dir = tmp_path / "round2"

        # --- Round 1: graph record → YAML files ---
        entries1 = graph_records_to_entries([SAMPLE_GRAPH_RECORD])
        assert len(entries1) == 1, "Expected one publish entry from graph record"
        generate_catalog_files(entries1, round1_dir)

        yaml_files1 = list(round1_dir.rglob("*.yaml"))
        assert len(yaml_files1) == 1, (
            f"Expected exactly 1 YAML file in round1, got {len(yaml_files1)}"
        )

        # --- Convert published YAML → catalog-importable format ---
        published_yaml_text = yaml_files1[0].read_text()
        catalog_doc = _published_yaml_to_catalog(
            published_yaml_text, physics_domain="core_plasma_physics"
        )
        catalog_dir.mkdir(parents=True, exist_ok=True)
        (catalog_dir / "electron_temperature.yaml").write_text(
            yaml.safe_dump(catalog_doc)
        )

        # --- Import catalog (dry run) → graph dicts ---
        from imas_codex.standard_names.catalog_import import import_catalog

        result = import_catalog(catalog_dir, dry_run=True)
        assert result.imported == 1, (
            f"Expected 1 imported entry, got {result.imported}; errors: {result.errors}"
        )
        assert not result.errors, f"Import errors: {result.errors}"

        # --- Normalise imported dicts and convert back to publish entries ---
        graph_records2 = [_imported_dict_to_graph_record(e) for e in result.entries]
        entries2 = graph_records_to_entries(graph_records2)
        assert len(entries2) == 1, "Expected one publish entry from imported dict"

        # --- Round 2: publish entries → YAML files ---
        generate_catalog_files(entries2, round2_dir)
        yaml_files2 = list(round2_dir.rglob("*.yaml"))
        assert len(yaml_files2) == 1, (
            f"Expected exactly 1 YAML file in round2, got {len(yaml_files2)}"
        )

        # --- Compare key fields (ignore provenance / confidence changes) ---
        parsed1 = yaml.safe_load(yaml_files1[0].read_text())
        parsed2 = yaml.safe_load(yaml_files2[0].read_text())

        fields1 = _key_fields(parsed1)
        fields2 = _key_fields(parsed2)

        assert fields2["name"] == fields1["name"], "name must be preserved"
        assert fields2["kind"] == fields1["kind"], "kind must be preserved"
        assert fields2["unit"] == fields1["unit"], "unit must be preserved"
        assert fields2["ids_paths"] == fields1["ids_paths"], (
            "ids_paths must be preserved"
        )
        assert fields2["validity_domain"] == fields1["validity_domain"], (
            "validity_domain must be preserved"
        )
        assert fields2["constraints"] == fields1["constraints"], (
            "constraints must be preserved"
        )

    def test_import_export_idempotent(self, tmp_path: Path) -> None:
        """Import a catalog entry then re-publish it — key fields must be unchanged.

        Tests the ``import_catalog`` → ``graph_records_to_entries`` →
        ``generate_yaml_entry`` path, asserting that the re-published YAML
        preserves every semantically significant field from the original catalog.
        """
        from imas_codex.standard_names.catalog_import import import_catalog
        from imas_codex.standard_names.publish import (
            generate_yaml_entry,
            graph_records_to_entries,
        )

        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()
        (catalog_dir / "electron_temperature.yaml").write_text(
            yaml.safe_dump(SAMPLE_CATALOG_ENTRY_RT)
        )

        # Import (dry run) → graph dicts
        result = import_catalog(catalog_dir, dry_run=True)
        assert result.imported == 1, (
            f"Expected 1 imported entry; errors: {result.errors}"
        )
        assert not result.errors, f"Import errors: {result.errors}"

        # Normalise dict keys and convert to StandardNamePublishEntry
        graph_records = [_imported_dict_to_graph_record(e) for e in result.entries]
        entries = graph_records_to_entries(graph_records)
        assert len(entries) == 1, "Expected one publish entry from imported graph dict"

        # Generate YAML and parse it back for comparison
        yaml_str = generate_yaml_entry(entries[0])
        published = yaml.safe_load(yaml_str)

        original = SAMPLE_CATALOG_ENTRY_RT
        assert published["name"] == original["name"], "name must round-trip"
        assert published["kind"] == original["kind"], "kind must round-trip"
        assert published.get("unit") == original["unit"], "unit must round-trip"
        assert published.get("description") == original["description"], (
            "description must round-trip"
        )
        assert published.get("documentation") == original["documentation"], (
            "documentation must round-trip"
        )
        assert sorted(published.get("ids_paths") or []) == sorted(
            original.get("ids_paths") or []
        ), "ids_paths must round-trip"
        if original.get("validity_domain"):
            assert published.get("validity_domain") == original["validity_domain"], (
                "validity_domain must round-trip"
            )
        assert sorted(published.get("constraints") or []) == sorted(
            original.get("constraints") or []
        ), "constraints must round-trip"

    def test_double_import_identical_entries(self, tmp_path: Path) -> None:
        """Importing the same catalog directory twice yields identical result entries.

        Verifies that ``import_catalog`` is deterministic: repeated calls on the
        same input produce identical graph dicts (same keys and values).
        """
        from imas_codex.standard_names.catalog_import import import_catalog

        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()
        (catalog_dir / "electron_temperature.yaml").write_text(
            yaml.safe_dump(SAMPLE_CATALOG_ENTRY_RT)
        )

        result1 = import_catalog(catalog_dir, dry_run=True)
        result2 = import_catalog(catalog_dir, dry_run=True)

        assert result1.imported == result2.imported, (
            "Both imports should report the same import count"
        )
        assert len(result1.entries) == len(result2.entries), (
            "Both imports should return the same number of entries"
        )

        compared_fields = (
            "id",
            "description",
            "documentation",
            "kind",
            "unit",
            "tags",
            "links",
            "imas_paths",
            "validity_domain",
            "constraints",
            "physics_domain",
            "review_status",
            "source_type",
        )
        for e1, e2 in zip(result1.entries, result2.entries, strict=True):
            for field in compared_fields:
                assert e1.get(field) == e2.get(field), (
                    f"Field {field!r} differs between imports: "
                    f"{e1.get(field)!r} != {e2.get(field)!r}"
                )


# =============================================================================
# Part 4: Full E2E Round-Trip (build → publish → edit → import)
# =============================================================================

# Rich sample data shared across E2E tests
_RICH_SN_RECORD = {
    "id": "electron_temperature",
    "source_type": "dd",
    "source_id": "core_profiles/profiles_1d/electrons/temperature",
    "ids_name": "core_profiles",
    "description": "Electron temperature in the core plasma",
    "documentation": (
        "The electron temperature $T_e$ is measured via Thomson scattering. "
        "It is a key parameter for transport modelling."
    ),
    "kind": "scalar",
    "unit": "eV",
    "tags": ["spatial-profile"],
    "links": ["name:ion_temperature"],
    "imas_paths": ["core_profiles/profiles_1d/electrons/temperature"],
    "validity_domain": "core plasma",
    "constraints": ["T_e > 0"],
    "physical_base": "temperature",
    "subject": "electron",
    "confidence": 0.95,
    "model": "gpt-4o",
    "review_status": "drafted",
}

# What get_validated_standard_names returns — graph-canonical keys
_GRAPH_QUERY_ROW = {
    "name": "electron_temperature",
    "description": "Electron temperature in the core plasma",
    "documentation": (
        "The electron temperature $T_e$ is measured via Thomson scattering. "
        "It is a key parameter for transport modelling."
    ),
    "kind": "scalar",
    "unit": "eV",
    "tags": ["spatial-profile"],
    "links": ["name:ion_temperature"],
    "ids_paths": ["core_profiles/profiles_1d/electrons/temperature"],
    "constraints": ["T_e > 0"],
    "validity_domain": "core plasma",
    "confidence": 0.95,
    "model": "gpt-4o",
    "source": "dd",
    "source_path": "core_profiles/profiles_1d/electrons/temperature",
    "ids_name": "core_profiles",
    "physical_base": "temperature",
    "subject": "electron",
    "component": None,
    "coordinate": None,
    "position": None,
    "process": None,
    "source_ids_names": ["core_profiles"],
}


class TestE2ERoundTrip:
    """Full lifecycle: build → publish → manual-edit → import.

    All graph operations are mocked — no live Neo4j required.
    """

    # ------------------------------------------------------------------
    # Phase helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mock_write_graph_client():
        """Mock GraphClient for write_standard_names (no return value needed)."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=None)
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_gc)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        return mock_ctx

    @staticmethod
    def _build_catalog_entry(publish_entry) -> dict:
        """Simulate a curator enriching a published entry into catalog format."""
        return {
            "name": publish_entry.name,
            "description": publish_entry.description,
            "documentation": publish_entry.documentation
            or "Enriched documentation by curator.",
            "kind": "scalar",
            "unit": publish_entry.unit,
            "tags": publish_entry.tags,
            "links": publish_entry.links,
            "ids_paths": publish_entry.ids_paths,
            "validity_domain": publish_entry.validity_domain or "",
            "constraints": publish_entry.constraints,
            "physics_domain": "core_plasma_physics",
            "status": "active",
        }

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_full_lifecycle_round_trip(self, tmp_path: Path) -> None:
        """Complete build → publish → manual-edit → import cycle."""
        from imas_codex.standard_names.catalog_import import import_catalog
        from imas_codex.standard_names.graph_ops import write_standard_names
        from imas_codex.standard_names.publish import (
            generate_catalog_files,
            graph_records_to_entries,
        )

        # Phase 1: write to graph (mocked)
        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value = self._mock_write_graph_client()
            count = write_standard_names([_RICH_SN_RECORD])
        assert count == 1

        # Phase 2: graph → publish entries → YAML files
        entries = graph_records_to_entries([_GRAPH_QUERY_ROW])
        assert len(entries) == 1

        written = generate_catalog_files(entries, tmp_path / "published")
        assert len(written) == 1
        assert written[0].exists()

        publish_entry = entries[0]
        assert publish_entry.name == "electron_temperature"
        assert publish_entry.unit == "eV"
        assert publish_entry.documentation is not None

        # Phase 3: simulate curator enrichment into catalog format
        catalog_entry = self._build_catalog_entry(publish_entry)
        catalog_dir = tmp_path / "reviewed_catalog"
        catalog_dir.mkdir()
        (catalog_dir / "electron_temperature.yaml").write_text(
            yaml.safe_dump(catalog_entry)
        )

        # Phase 4: import catalog (dry_run=True, no graph write)
        result = import_catalog(catalog_dir=catalog_dir, dry_run=True)

        assert result.imported == 1
        assert len(result.errors) == 0

        entry = result.entries[0]
        assert entry["id"] == "electron_temperature"
        assert entry["review_status"] == "accepted"
        assert entry["unit"] == "eV"
        assert entry["physics_domain"] == "core_plasma_physics"
        assert entry["physical_base"] == "temperature"
        assert entry["subject"] == "electron"

    def test_publish_generates_valid_yaml_for_import(self, tmp_path: Path) -> None:
        """Published YAML (after curator enrichment) is valid input for import_catalog."""
        from imas_codex.standard_names.catalog_import import import_catalog
        from imas_codex.standard_names.publish import graph_records_to_entries

        entries = graph_records_to_entries([_GRAPH_QUERY_ROW])
        assert len(entries) == 1

        # Curator enriches published entry into catalog format
        catalog_entry = self._build_catalog_entry(entries[0])
        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()
        (catalog_dir / "electron_temperature.yaml").write_text(
            yaml.safe_dump(catalog_entry)
        )

        result = import_catalog(catalog_dir=catalog_dir, dry_run=True)

        assert result.imported == 1
        assert len(result.errors) == 0

        entry = result.entries[0]
        # catalog 'unit' passes through as graph 'unit'
        assert entry["unit"] == "eV"
        # catalog 'ids_paths' → graph 'imas_paths'
        assert entry["imas_paths"] == [
            "core_profiles/profiles_1d/electrons/temperature"
        ]
        assert "ids_paths" not in entry
        # review_status always 'accepted' after import
        assert entry["review_status"] == "accepted"

    def test_field_preservation_across_lifecycle(self, tmp_path: Path) -> None:
        """Specific rich fields are verified at each stage of the lifecycle."""
        from imas_codex.standard_names.catalog_import import import_catalog
        from imas_codex.standard_names.publish import (
            generate_catalog_files,
            generate_yaml_entry,
            graph_records_to_entries,
        )

        # Stage A: graph_records_to_entries preserves rich fields
        entries = graph_records_to_entries([_GRAPH_QUERY_ROW])
        entry = entries[0]

        assert entry.name == "electron_temperature"
        assert entry.kind == "scalar"
        assert entry.unit == "eV"
        assert "spatial-profile" in entry.tags
        assert entry.links == ["name:ion_temperature"]
        assert entry.ids_paths == ["core_profiles/profiles_1d/electrons/temperature"]
        assert entry.validity_domain == "core plasma"
        assert entry.constraints == ["T_e > 0"]
        assert entry.documentation is not None
        assert "$T_e$" in (entry.documentation or "")
        assert entry.provenance.confidence == 0.95
        assert entry.provenance.source == "dd"
        assert entry.provenance.ids_name == "core_profiles"

        # Stage B: generate_yaml_entry serializes all fields
        yaml_str = generate_yaml_entry(entry)
        parsed = yaml.safe_load(yaml_str)

        assert parsed["name"] == "electron_temperature"
        assert parsed["kind"] == "scalar"
        assert parsed["unit"] == "eV"
        assert parsed["validity_domain"] == "core plasma"
        assert parsed["constraints"] == ["T_e > 0"]
        assert "documentation" in parsed
        assert parsed["ids_paths"] == [
            "core_profiles/profiles_1d/electrons/temperature"
        ]
        assert parsed["provenance"]["confidence"] == 0.95

        # Stage C: generate_catalog_files creates correct subdirectory structure
        written = generate_catalog_files(entries, tmp_path / "published")
        assert len(written) == 1
        # Primary tag is "core_profiles" → file lives in core_profiles/
        assert written[0].parent.name == "spatial-profile"

        # Stage D: import_catalog maps all fields correctly
        catalog_entry = self._build_catalog_entry(entry)
        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()
        (catalog_dir / "electron_temperature.yaml").write_text(
            yaml.safe_dump(catalog_entry)
        )

        result = import_catalog(catalog_dir=catalog_dir, dry_run=True)
        imported = result.entries[0]

        assert imported["id"] == "electron_temperature"
        assert imported["description"] == "Electron temperature in the core plasma"
        assert "documentation" in imported
        assert imported["kind"] == "scalar"
        assert imported["unit"] == "eV"
        assert imported["imas_paths"] == [
            "core_profiles/profiles_1d/electrons/temperature"
        ]
        assert imported["validity_domain"] == "core plasma"
        assert imported["constraints"] == ["T_e > 0"]
        assert imported["physics_domain"] == "core_plasma_physics"
        assert imported["review_status"] == "accepted"
        assert imported["physical_base"] == "temperature"
        assert imported["subject"] == "electron"

    def test_write_standard_names_called_with_all_fields(self) -> None:
        """write_standard_names receives all populated fields without losing any."""
        from imas_codex.standard_names.graph_ops import write_standard_names

        captured_calls: list = []

        def capture_query(cypher, **kwargs):
            if "batch" in kwargs:
                captured_calls.append({"cypher": cypher, "batch": kwargs["batch"]})
            return None

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(side_effect=capture_query)
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_gc)
        mock_ctx.__exit__ = MagicMock(return_value=False)

        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value = mock_ctx
            write_standard_names([_RICH_SN_RECORD])

        # Find the MERGE StandardName batch (not the conflict check batch)
        merge_batch = None
        for call_info in captured_calls:
            if "MERGE (sn:StandardName" in call_info["cypher"]:
                merge_batch = call_info["batch"]
                break
        assert merge_batch is not None, "MERGE StandardName query not found"
        node = merge_batch[0]
        assert node["id"] == "electron_temperature"
        assert node["description"] == "Electron temperature in the core plasma"
        assert node["kind"] == "scalar"
        assert node["unit"] == "eV"
        assert node["tags"] == ["spatial-profile"]
        assert node["constraints"] == ["T_e > 0"]
        assert node["validity_domain"] == "core plasma"
        assert node["review_status"] == "drafted"
        assert node["confidence"] == 0.95
        assert node["model"] == "gpt-4o"

    def test_import_dry_run_does_not_call_graph(self, tmp_path: Path) -> None:
        """dry_run=True must never invoke the graph write path."""
        from imas_codex.standard_names.catalog_import import import_catalog

        catalog_entry = {
            "name": "electron_temperature",
            "description": "Electron temperature",
            "documentation": "Te documentation.",
            "kind": "scalar",
            "unit": "eV",
            "tags": [],
            "links": [],
            "ids_paths": [],
            "validity_domain": "",
            "constraints": [],
            "physics_domain": "core_plasma_physics",
            "status": "active",
        }
        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()
        (catalog_dir / "electron_temperature.yaml").write_text(
            yaml.safe_dump(catalog_entry)
        )

        with patch(
            "imas_codex.standard_names.catalog_import._write_catalog_entries"
        ) as mock_write:
            result = import_catalog(catalog_dir=catalog_dir, dry_run=True)

        mock_write.assert_not_called()
        assert result.imported == 1
        assert result.entries[0]["review_status"] == "accepted"
