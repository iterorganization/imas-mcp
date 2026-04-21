"""Tests that preview invokes the ISN catalog-site command.

Plan 35 §3d: stub subprocess, verify correct command.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from imas_codex.standard_names.preview import PreviewHandle, run_preview


@pytest.fixture()
def staging_dir(tmp_path: Path) -> Path:
    """Create a minimal staging directory for preview."""
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "catalog.yml").write_text(
        yaml.safe_dump({"catalog_name": "test"}), encoding="utf-8"
    )
    sn_dir = staging / "standard_names" / "test_domain"
    sn_dir.mkdir(parents=True)
    (sn_dir / "test_name.yml").write_text(
        yaml.safe_dump({"name": "test_name"}), encoding="utf-8"
    )
    return staging


class TestRunPreview:
    """run_preview delegates to ISN catalog-site serve."""

    @patch("imas_codex.standard_names.preview.subprocess.Popen")
    def test_invokes_correct_command(
        self, mock_popen: MagicMock, staging_dir: Path
    ) -> None:
        mock_process = MagicMock()
        mock_popen.return_value = mock_process

        run_preview(staging_dir)

        mock_popen.assert_called_once()
        cmd = mock_popen.call_args[0][0]
        assert "standard-names" in cmd
        assert "catalog-site" in cmd
        assert "serve" in cmd
        assert str(staging_dir) in cmd

    @patch("imas_codex.standard_names.preview.subprocess.Popen")
    def test_custom_port(self, mock_popen: MagicMock, staging_dir: Path) -> None:
        mock_popen.return_value = MagicMock()

        handle = run_preview(staging_dir, port=9090)

        cmd = mock_popen.call_args[0][0]
        assert "--port" in cmd
        assert "9090" in cmd
        assert handle.url == "http://localhost:9090"

    @patch("imas_codex.standard_names.preview.subprocess.Popen")
    def test_default_port_url(self, mock_popen: MagicMock, staging_dir: Path) -> None:
        mock_popen.return_value = MagicMock()

        handle = run_preview(staging_dir)
        assert handle.url == "http://localhost:8000"

    @patch("imas_codex.standard_names.preview.subprocess.Popen")
    def test_returns_handle(self, mock_popen: MagicMock, staging_dir: Path) -> None:
        mock_popen.return_value = MagicMock()

        handle = run_preview(staging_dir)
        assert isinstance(handle, PreviewHandle)
        assert handle.process is not None
        assert handle.staging_dir == str(staging_dir)

    def test_missing_staging_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            run_preview(tmp_path / "nonexistent")

    def test_missing_manifest_raises(self, tmp_path: Path) -> None:
        staging = tmp_path / "no_manifest"
        staging.mkdir()
        with pytest.raises(FileNotFoundError, match="catalog.yml"):
            run_preview(staging)

    @patch("imas_codex.standard_names.preview.subprocess.Popen")
    def test_stop_terminates_process(
        self, mock_popen: MagicMock, staging_dir: Path
    ) -> None:
        mock_process = MagicMock()
        mock_popen.return_value = mock_process

        handle = run_preview(staging_dir)
        handle.stop()

        mock_process.terminate.assert_called_once()

    @patch(
        "imas_codex.standard_names.preview.subprocess.Popen",
        side_effect=FileNotFoundError("standard-names not found"),
    )
    def test_missing_cli_returns_none_url(
        self, mock_popen: MagicMock, staging_dir: Path
    ) -> None:
        handle = run_preview(staging_dir)
        assert handle.process is None
        assert handle.url is None
