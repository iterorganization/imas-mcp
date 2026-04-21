"""Tests for the transport-only publish module.

Plan 35 §3d: publish with skipped gate; verify commit+push.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from imas_codex.standard_names.publish import (
    PublishReport,
    _validate_staging_dir,
    run_publish,
)


@pytest.fixture()
def staging_dir(tmp_path: Path) -> Path:
    """Create a valid staging directory."""
    staging = tmp_path / "staging"
    staging.mkdir()

    # Write manifest
    manifest = {
        "catalog_name": "imas-standard-names-catalog",
        "cocos_convention": 17,
        "grammar_version": "0.7.0",
        "isn_model_version": "0.7.0",
        "dd_version_lineage": ["4.0.0"],
        "generated_by": "test",
        "generated_at": "2024-01-01T00:00:00Z",
        "candidate_count": 2,
        "published_count": 2,
        "excluded_below_score_count": 0,
        "excluded_unreviewed_count": 0,
    }
    (staging / "catalog.yml").write_text(yaml.safe_dump(manifest), encoding="utf-8")

    # Write entry files
    sn_dir = staging / "standard_names" / "equilibrium"
    sn_dir.mkdir(parents=True)
    entry = {
        "name": "electron_temperature",
        "description": "Te",
        "documentation": "Docs",
        "kind": "scalar",
        "unit": "eV",
        "tags": [],
        "links": [],
        "constraints": [],
        "status": "draft",
    }
    (sn_dir / "electron_temperature.yml").write_text(
        yaml.safe_dump(entry), encoding="utf-8"
    )

    return staging


@pytest.fixture()
def isnc_repo(tmp_path: Path) -> Path:
    """Create a mock ISNC git repository."""
    isnc = tmp_path / "isnc"
    isnc.mkdir()
    # Init git repo
    subprocess.run(["git", "init"], cwd=isnc, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=isnc,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=isnc,
        check=True,
        capture_output=True,
    )
    # Initial commit
    (isnc / "README.md").write_text("# ISNC\n")
    subprocess.run(["git", "add", "."], cwd=isnc, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=isnc,
        check=True,
        capture_output=True,
    )
    return isnc


class TestValidateStagingDir:
    """Staging directory validation."""

    def test_valid_staging_passes(self, staging_dir: Path) -> None:
        errors = _validate_staging_dir(staging_dir)
        assert errors == []

    def test_missing_dir_fails(self, tmp_path: Path) -> None:
        errors = _validate_staging_dir(tmp_path / "nonexistent")
        assert len(errors) == 1
        assert "does not exist" in errors[0]

    def test_missing_manifest_fails(self, tmp_path: Path) -> None:
        staging = tmp_path / "empty_staging"
        staging.mkdir()
        (staging / "standard_names").mkdir()
        errors = _validate_staging_dir(staging)
        assert any("catalog.yml" in e for e in errors)

    def test_missing_standard_names_fails(self, tmp_path: Path) -> None:
        staging = tmp_path / "no_sn"
        staging.mkdir()
        (staging / "catalog.yml").write_text(
            yaml.safe_dump({"catalog_name": "test"}), encoding="utf-8"
        )
        errors = _validate_staging_dir(staging)
        assert any("standard_names" in e for e in errors)


class TestRunPublish:
    """run_publish transport operation."""

    def test_dry_run_no_changes(self, staging_dir: Path, isnc_repo: Path) -> None:
        report = run_publish(staging_dir, isnc_repo, dry_run=True)
        assert report.dry_run is True
        assert report.files_copied > 0
        assert report.commit_sha is None
        assert not report.errors

    def test_publish_creates_commit(self, staging_dir: Path, isnc_repo: Path) -> None:
        report = run_publish(staging_dir, isnc_repo)
        assert not report.errors, f"Errors: {report.errors}"
        assert report.commit_sha is not None
        assert report.files_copied > 0

        # Verify files exist in ISNC
        assert (isnc_repo / "catalog.yml").is_file()
        sn_dir = isnc_repo / "standard_names" / "equilibrium"
        assert sn_dir.is_dir()
        assert (sn_dir / "electron_temperature.yml").is_file()

    def test_publish_commit_message(self, staging_dir: Path, isnc_repo: Path) -> None:
        run_publish(staging_dir, isnc_repo)

        result = subprocess.run(
            ["git", "log", "-1", "--format=%s"],
            cwd=isnc_repo,
            capture_output=True,
            text=True,
        )
        msg = result.stdout.strip()
        assert msg.startswith("chore(catalog): sync from imas-codex")

    def test_publish_clears_old_tree(self, staging_dir: Path, isnc_repo: Path) -> None:
        """Publishing should clear old standard_names/ before mirroring."""
        # Create pre-existing file in ISNC
        old_dir = isnc_repo / "standard_names" / "old_domain"
        old_dir.mkdir(parents=True)
        (old_dir / "old_name.yml").write_text("old content")
        subprocess.run(
            ["git", "add", "."], cwd=isnc_repo, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "old content"],
            cwd=isnc_repo,
            check=True,
            capture_output=True,
        )

        # Publish new content
        run_publish(staging_dir, isnc_repo)

        # Old file should be gone
        assert not old_dir.exists()

    def test_publish_invalid_staging_fails(
        self, tmp_path: Path, isnc_repo: Path
    ) -> None:
        report = run_publish(tmp_path / "nonexistent", isnc_repo)
        assert len(report.errors) > 0

    def test_publish_invalid_isnc_fails(
        self, staging_dir: Path, tmp_path: Path
    ) -> None:
        report = run_publish(staging_dir, tmp_path / "nonexistent")
        assert len(report.errors) > 0

    def test_publish_report_serialises(
        self, staging_dir: Path, isnc_repo: Path
    ) -> None:
        report = run_publish(staging_dir, isnc_repo)
        d = report.to_dict()
        assert "staging_dir" in d
        assert "isnc_path" in d
        assert "files_copied" in d
