"""End-to-end round-trip tests: export → publish → import → protection.

Phase 6 of plan 35 — consolidated tests exercising the full pipeline
and CLI verbs together.

6a. Full round-trip with origin tracking.
6b. Protection regression: pipeline writers cannot overwrite catalog edits.
6c. Divergence injection: detect graph mutations bypassing protection.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

imas_sn = pytest.importorskip("imas_standard_names")

# ============================================================================
# Fixture data — 6 StandardName nodes across 3 domains
# ============================================================================

_FIXTURE_NODES: list[dict] = [
    {
        "id": "electron_temperature",
        "description": "Electron temperature profile",
        "documentation": "The electron temperature Te measured by Thomson scattering.",
        "kind": "scalar",
        "unit": "eV",
        "tags": ["time-dependent"],
        "links": ["name:ion_temperature"],
        "constraints": ["T_e > 0"],
        "validity_domain": "core plasma",
        "cocos_transformation_type": None,
        "status": "draft",
        "physics_domain": "kinetics",
        "pipeline_status": "reviewed",
        "reviewer_score": 0.85,
        "origin": "pipeline",
        "cocos": None,
        "deprecates": None,
        "superseded_by": None,
    },
    {
        "id": "ion_temperature",
        "description": "Ion temperature profile",
        "documentation": "The ion temperature Ti from charge exchange.",
        "kind": "scalar",
        "unit": "eV",
        "tags": ["time-dependent"],
        "links": ["name:electron_temperature"],
        "constraints": ["T_i > 0"],
        "validity_domain": "core plasma",
        "cocos_transformation_type": None,
        "status": "draft",
        "physics_domain": "kinetics",
        "pipeline_status": "reviewed",
        "reviewer_score": 0.82,
        "origin": "pipeline",
        "cocos": None,
        "deprecates": None,
        "superseded_by": None,
    },
    {
        "id": "electron_density",
        "description": "Electron density profile",
        "documentation": "Line-averaged electron density from interferometry.",
        "kind": "scalar",
        "unit": "m^-3",
        "tags": ["time-dependent"],
        "links": ["name:electron_temperature"],
        "constraints": ["n_e > 0"],
        "validity_domain": "core plasma",
        "cocos_transformation_type": None,
        "status": "draft",
        "physics_domain": "kinetics",
        "pipeline_status": "reviewed",
        "reviewer_score": 0.78,
        "origin": "pipeline",
        "cocos": None,
        "deprecates": None,
        "superseded_by": None,
    },
    {
        "id": "plasma_current",
        "description": "Total plasma current",
        "documentation": "Toroidal plasma current measured by Rogowski coil.",
        "kind": "scalar",
        "unit": "A",
        "tags": ["time-dependent"],
        "links": ["name:safety_factor"],
        "constraints": [],
        "validity_domain": "",
        "cocos_transformation_type": "ip_like",
        "status": "draft",
        "physics_domain": "magnetics",
        "pipeline_status": "reviewed",
        "reviewer_score": 0.90,
        "origin": "pipeline",
        "cocos": None,
        "deprecates": None,
        "superseded_by": None,
    },
    {
        "id": "safety_factor",
        "description": "Safety factor profile",
        "documentation": "The safety factor q from equilibrium reconstruction.",
        "kind": "scalar",
        "unit": "1",
        "tags": ["time-dependent"],
        "links": ["name:plasma_current"],
        "constraints": [],
        "validity_domain": "",
        "cocos_transformation_type": "q_like",
        "status": "draft",
        "physics_domain": "equilibrium",
        "pipeline_status": "reviewed",
        "reviewer_score": 0.88,
        "origin": "pipeline",
        "cocos": None,
        "deprecates": None,
        "superseded_by": None,
    },
    {
        "id": "toroidal_field",
        "description": "Vacuum toroidal magnetic field",
        "documentation": "Toroidal field B0 at the geometric axis.",
        "kind": "scalar",
        "unit": "T",
        "tags": ["time-dependent"],
        "links": [],
        "constraints": [],
        "validity_domain": "",
        "cocos_transformation_type": "b0_like",
        "status": "draft",
        "physics_domain": "magnetics",
        "pipeline_status": "reviewed",
        "reviewer_score": 0.75,
        "origin": "pipeline",
        "cocos": None,
        "deprecates": None,
        "superseded_by": None,
    },
]


# ============================================================================
# Git repo helpers
# ============================================================================


def _init_git_repo(d: Path) -> None:
    """Initialise a git repo with an initial commit."""
    subprocess.run(["git", "init"], cwd=str(d), capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=str(d),
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=str(d),
        capture_output=True,
        check=True,
    )
    (d / "README.md").write_text("initial\n")
    subprocess.run(["git", "add", "."], cwd=str(d), capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial commit"],
        cwd=str(d),
        capture_output=True,
        check=True,
    )


# ============================================================================
# Mock GraphClient for import
# ============================================================================


def _mock_gc_for_import(
    graph_state: dict[str, dict] | None = None,
):
    """Build a mock GraphClient that handles import lock/watermark/diff queries."""
    gc = MagicMock()

    def _query(cypher, **params):
        # Lock acquire
        if "ImportLock" in cypher and "holder IS NULL" in cypher:
            return [{"acquired": True}]
        # Lock release
        if "ImportLock" in cypher and "holder = $holder" in cypher:
            return []
        # Lock status
        if "ImportLock" in cypher:
            return [{"holder": None, "acquired_at": None}]
        # Watermark CAS set
        if "ImportWatermark" in cypher and "SET" in cypher:
            return [{"sha": "abc123"}]
        # Watermark read
        if "ImportWatermark" in cypher:
            return [
                {
                    "last_commit_sha": None,
                    "last_import_at": None,
                    "source_repo": None,
                }
            ]
        # Graph state for diff-based origin tracking
        if "StandardName" in cypher and "origin" in cypher and "RETURN" in cypher:
            if graph_state:
                return [
                    {"id": k, **v, "origin": v.get("origin", "pipeline")}
                    for k, v in graph_state.items()
                ]
            return []
        # Unit validation
        if "sn.unit <> b.unit" in cypher:
            return []
        # COCOS validation
        if "cocos_transformation_type" in cypher and "RETURN" in cypher:
            return []
        # HAS_UNIT relationship
        if "HAS_UNIT" in cypher:
            return []
        return []

    gc.query = MagicMock(side_effect=_query)
    return gc


def _patch_gc(gc):
    """Create a patch for GraphClient context manager."""
    mock_cls = MagicMock()
    mock_cls.return_value.__enter__ = MagicMock(return_value=gc)
    mock_cls.return_value.__exit__ = MagicMock(return_value=False)
    return patch("imas_codex.graph.client.GraphClient", mock_cls)


# ============================================================================
# 6a. Full round-trip test
# ============================================================================


class TestFullRoundTrip:
    """export → publish → import round-trip with origin tracking."""

    def test_export_publish_import_round_trip(self, tmp_path: Path) -> None:
        """Full round-trip: export, tweak description, publish, import.

        Asserts:
        - edited entry gets origin=catalog_edit
        - edited description is preserved in import report
        - non-edited entries keep origin=pipeline
        """
        from imas_codex.standard_names.export import run_export
        from imas_codex.standard_names.publish import run_publish

        staging = tmp_path / "staging"
        isnc = tmp_path / "isnc"
        isnc.mkdir()

        # ── Step 1: Export from mocked graph to staging ──────────
        with (
            patch(
                "imas_codex.standard_names.export._fetch_candidates",
                return_value=list(_FIXTURE_NODES),
            ),
            patch(
                "imas_codex.standard_names.export._get_codex_commit_sha",
                return_value="abc123test",
            ),
        ):
            report = run_export(
                staging,
                skip_gate=True,
                force=True,
                include_unreviewed=True,
                min_score=0.0,
            )

        assert report.exported_count == 6, (
            f"Expected 6 exports, got {report.exported_count}"
        )

        # Verify YAML files were written
        yml_files = list(staging.rglob("standard_names/**/*.yml"))
        assert len(yml_files) == 6

        # ── Step 2: Simulate manual YAML tweak ──────────────────
        et_path = staging / "standard_names" / "kinetics" / "electron_temperature.yml"
        assert et_path.exists(), f"Missing {et_path}"

        et_data = yaml.safe_load(et_path.read_text(encoding="utf-8"))
        et_data["description"] = "Improved electron temperature description (PR edit)"
        et_path.write_text(
            yaml.safe_dump(et_data, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )

        # ── Step 3: Publish staging → mock ISNC ─────────────────
        _init_git_repo(isnc)
        pub_report = run_publish(staging, isnc)

        assert pub_report.errors == [], f"Publish errors: {pub_report.errors}"
        assert pub_report.commit_sha is not None
        assert pub_report.files_copied >= 7  # 6 yml + catalog.yml

        # Verify ISNC has the files
        isnc_et = isnc / "standard_names" / "kinetics" / "electron_temperature.yml"
        assert isnc_et.exists()
        isnc_et_data = yaml.safe_load(isnc_et.read_text(encoding="utf-8"))
        assert "Improved" in isnc_et_data["description"]

        # ── Step 4: Import from ISNC back into (mocked) graph ───
        # Build graph state matching the original exported values
        graph_state = {}
        for node in _FIXTURE_NODES:
            graph_state[node["id"]] = {
                "description": node["description"],
                "documentation": node["documentation"],
                "kind": node["kind"],
                "tags": node["tags"],
                "links": node["links"],
                "status": node["status"],
                "deprecates": node.get("deprecates"),
                "superseded_by": node.get("superseded_by"),
                "validity_domain": node["validity_domain"],
                "constraints": node["constraints"],
                "origin": "pipeline",
            }

        gc = _mock_gc_for_import(graph_state)

        with _patch_gc(gc):
            from imas_codex.standard_names.catalog_import import run_import

            import_report = run_import(isnc)

        assert import_report.imported > 0, (
            f"Expected imports, got {import_report.imported}; "
            f"errors: {import_report.errors}"
        )

        # ── Step 5: Assert origin tracking ──────────────────────
        entries_by_id = {e["id"]: e for e in import_report.entries}

        # Edited entry should flip to catalog_edit
        et_entry = entries_by_id.get("electron_temperature")
        assert et_entry is not None, "electron_temperature not in import entries"
        assert et_entry["_origin"] == "catalog_edit", (
            f"Expected origin=catalog_edit for edited entry, got {et_entry['_origin']}"
        )
        assert "Improved" in et_entry.get("description", ""), (
            "Edited description not preserved"
        )

        # Non-edited entries should keep pipeline origin
        for name in ["ion_temperature", "plasma_current", "safety_factor"]:
            entry = entries_by_id.get(name)
            if entry is not None:
                assert entry["_origin"] == "pipeline", (
                    f"{name} should keep origin=pipeline, got {entry['_origin']}"
                )

        # Verify counts
        assert import_report.updated >= 1, "Should have at least 1 updated entry"

    def test_round_trip_no_unit_override_needed(self, tmp_path: Path) -> None:
        """Round-trip with no unit changes does not require --accept-unit-override."""
        from imas_codex.standard_names.export import run_export
        from imas_codex.standard_names.publish import run_publish

        staging = tmp_path / "staging"
        isnc = tmp_path / "isnc"
        isnc.mkdir()

        # Use single-node subset to keep it simple
        nodes = [_FIXTURE_NODES[0]]

        with (
            patch(
                "imas_codex.standard_names.export._fetch_candidates",
                return_value=list(nodes),
            ),
            patch(
                "imas_codex.standard_names.export._get_codex_commit_sha",
                return_value="test123",
            ),
        ):
            run_export(
                staging,
                skip_gate=True,
                force=True,
                include_unreviewed=True,
                min_score=0.0,
            )

        _init_git_repo(isnc)
        run_publish(staging, isnc)

        # Import with accept_unit_override=False (default)
        gc = _mock_gc_for_import(graph_state={})
        with _patch_gc(gc):
            from imas_codex.standard_names.catalog_import import run_import

            report = run_import(isnc, accept_unit_override=False)

        unit_errors = [e for e in report.errors if "unit" in e.lower()]
        assert len(unit_errors) == 0, f"Unexpected unit errors: {unit_errors}"


# ============================================================================
# 6b. Protection regression test
# ============================================================================


class TestProtectionRegression:
    """After import, pipeline writers must not overwrite catalog-edited fields."""

    def test_filter_protected_strips_edited_description(self) -> None:
        """filter_protected strips description from catalog_edit items."""
        from imas_codex.standard_names.protection import filter_protected

        items = [
            {
                "id": "electron_temperature",
                "description": "Pipeline wants to overwrite this",
                "documentation": "Pipeline docs",
                "kind": "vector",  # pipeline tries to change kind
                "pipeline_status": "enriched",  # NOT protected
                "confidence": 0.99,  # NOT protected
            },
            {
                "id": "ion_temperature",
                "description": "Pipeline generated desc",
                "pipeline_status": "enriched",
            },
        ]

        # Simulate: electron_temperature has origin=catalog_edit,
        # ion_temperature has origin=pipeline
        protected_names = {"electron_temperature"}

        filtered, skipped = filter_protected(
            items,
            override=False,
            protected_names=protected_names,
        )

        assert len(filtered) == 2

        # electron_temperature: protected fields stripped
        et = filtered[0]
        assert "description" not in et, "Protected description should be stripped"
        assert "documentation" not in et, "Protected documentation should be stripped"
        assert "kind" not in et, "Protected kind should be stripped"
        assert et["pipeline_status"] == "enriched", "Non-protected field preserved"
        assert et["confidence"] == 0.99, "Non-protected field preserved"

        # ion_temperature: all fields pass through
        it = filtered[1]
        assert it["description"] == "Pipeline generated desc"
        assert it["pipeline_status"] == "enriched"

        assert skipped == ["electron_temperature"]

    def test_override_edits_allows_specific_name(self) -> None:
        """override_names selectively bypasses protection for listed names."""
        from imas_codex.standard_names.protection import filter_protected

        items = [
            {
                "id": "electron_temperature",
                "description": "Override this specific name",
                "kind": "scalar",
            },
            {
                "id": "plasma_current",
                "description": "This stays protected",
                "kind": "scalar",
            },
        ]

        filtered, skipped = filter_protected(
            items,
            override=False,
            override_names={"electron_temperature"},
            protected_names={"electron_temperature", "plasma_current"},
        )

        assert len(filtered) == 2

        # electron_temperature: override lets all fields through
        et = filtered[0]
        assert et["description"] == "Override this specific name"
        assert et["kind"] == "scalar"

        # plasma_current: still protected
        pc = filtered[1]
        assert "description" not in pc
        assert "kind" not in pc

        assert skipped == ["plasma_current"]

    def test_embedding_and_graph_only_fields_pass_through_protection(self) -> None:
        """Graph-only fields (embedding, model, confidence) are never protected."""
        from imas_codex.standard_names.protection import PROTECTED_FIELDS

        # These fields should NOT be in PROTECTED_FIELDS
        graph_only = {
            "embedding",
            "model",
            "generated_at",
            "confidence",
            "pipeline_status",
            "reviewer_score",
            "source_types",
        }
        assert not graph_only & PROTECTED_FIELDS, (
            "Graph-only fields must not be in PROTECTED_FIELDS"
        )


# ============================================================================
# 6c. Divergence injection test
# ============================================================================


class TestDivergenceInjection:
    """Direct Cypher bypass → divergence detection flags the row."""

    def test_divergence_flagged_for_catalog_edit_with_sha(self) -> None:
        """A catalog-edited node with commit lineage is flagged."""
        from imas_codex.standard_names.export import detect_divergence

        # Simulate: node was imported from catalog (origin=catalog_edit)
        # and has a catalog_commit_sha. A direct Cypher SET bypassed
        # protection and modified the description.
        candidates = [
            {
                "id": "electron_temperature",
                "origin": "catalog_edit",
                "catalog_commit_sha": "deadbeef12345678",
                "description": "Description modified by direct Cypher SET",
                "documentation": "Original docs",
                "kind": "scalar",
                "tags": ["kinetics"],
                "links": [],
                "status": "draft",
            },
            {
                "id": "ion_temperature",
                "origin": "pipeline",
                "catalog_commit_sha": None,
                "description": "Pipeline generated",
                "documentation": "Pipeline docs",
                "kind": "scalar",
                "tags": [],
                "links": [],
                "status": "draft",
            },
        ]

        findings = detect_divergence(candidates)

        # Only catalog_edit with SHA should be flagged
        assert len(findings) == 1
        assert findings[0].name == "electron_temperature"
        assert findings[0].graph_hash  # non-empty hash
        assert "deadbeef" in findings[0].detail

    def test_divergence_not_flagged_for_pipeline_origin(self) -> None:
        """Pipeline-origin nodes are never flagged even with modifications."""
        from imas_codex.standard_names.export import detect_divergence

        candidates = [
            {
                "id": "plasma_current",
                "origin": "pipeline",
                "catalog_commit_sha": "abc123",
                "description": "Modified by pipeline",
                "kind": "scalar",
                "tags": [],
                "links": [],
            },
        ]

        findings = detect_divergence(candidates)
        assert len(findings) == 0

    def test_divergence_report_in_export(self, tmp_path: Path) -> None:
        """run_export with catalog-edit nodes includes divergence in report."""
        from imas_codex.standard_names.export import run_export

        # Inject a catalog-edited node with commit SHA
        modified_nodes = list(_FIXTURE_NODES)
        modified_nodes[0] = {
            **modified_nodes[0],
            "origin": "catalog_edit",
            "catalog_commit_sha": "abc123deadbeef",
            "description": "Manually modified via Cypher",
        }

        staging = tmp_path / "staging"

        with (
            patch(
                "imas_codex.standard_names.export._fetch_candidates",
                return_value=modified_nodes,
            ),
            patch(
                "imas_codex.standard_names.export._get_codex_commit_sha",
                return_value="test456",
            ),
        ):
            report = run_export(
                staging,
                force=True,
                include_unreviewed=True,
                min_score=0.0,
                gate_scope="domain",  # skip gate A (subprocess)
            )

        # Divergence entries should flag the modified node
        assert len(report.divergence_entries) >= 1
        flagged_names = {d.name for d in report.divergence_entries}
        assert "electron_temperature" in flagged_names
