from __future__ import annotations

from types import SimpleNamespace

from imas_codex.config import discovery_config as discovery_config_module
from imas_codex.config.discovery_config import ExclusionConfig
from imas_codex.discovery.base import facility as facility_module


def test_get_facility_reads_public_yaml_only(tmp_path, monkeypatch):
    facilities_dir = tmp_path / "facilities"
    facilities_dir.mkdir()
    (facilities_dir / "tcv.yaml").write_text(
        """
facility: tcv
ssh_host: tcv
data_systems:
  mdsplus:
    reference_shot: 85000
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (facilities_dir / "tcv_private.yaml").write_text(
        """
ssh_host: should-not-win
data_systems:
  uda:
    available: false
tools:
  rg:
    available: true
""".strip()
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(facility_module, "get_facilities_dir", lambda: facilities_dir)
    facility_module.get_facility.cache_clear()
    facility_module.get_facility_with_infrastructure.cache_clear()

    config = facility_module.get_facility("tcv")

    assert config["ssh_host"] == "tcv"
    assert "tools" not in config
    assert set(config["data_systems"]) == {"mdsplus"}


def test_get_facility_with_infrastructure_merges_private_yaml(tmp_path, monkeypatch):
    facilities_dir = tmp_path / "facilities"
    facilities_dir.mkdir()
    (facilities_dir / "iter.yaml").write_text(
        """
facility: iter
ssh_host: iter
data_systems:
  imas:
    db_name: iter
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (facilities_dir / "iter_private.yaml").write_text(
        """
local_hosts:
  - iter
tools:
  rg:
    available: true
""".strip()
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(facility_module, "get_facilities_dir", lambda: facilities_dir)
    facility_module.get_facility.cache_clear()
    facility_module.get_facility_with_infrastructure.cache_clear()

    config = facility_module.get_facility_with_infrastructure("iter")

    assert config["ssh_host"] == "iter"
    assert config["local_hosts"] == ["iter"]
    assert config["tools"]["rg"]["available"] is True


def test_exclusion_config_reads_public_yaml_only(tmp_path, monkeypatch):
    facilities_dir = tmp_path / "facilities"
    facilities_dir.mkdir()
    (facilities_dir / "tcv.yaml").write_text(
        """
facility: tcv
name: TCV
machine: TCV
ssh_host: tcv
excludes:
  directories:
    - /public-only
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (facilities_dir / "tcv_private.yaml").write_text(
        """
excludes:
  directories:
    - /private-only
""".strip()
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(facility_module, "get_facilities_dir", lambda: facilities_dir)
    monkeypatch.setattr(
        discovery_config_module,
        "get_discovery_config",
        lambda: SimpleNamespace(exclusions=ExclusionConfig(directories=[".git"])),
    )
    facility_module.get_facility.cache_clear()
    facility_module.get_facility_with_infrastructure.cache_clear()

    config = discovery_config_module.get_exclusion_config_for_facility("tcv")

    assert ".git" in config.directories
    assert "/public-only" in config.directories
    assert "/private-only" not in config.directories

    config = discovery_config_module.get_exclusion_config_for_facility("tcv")

    assert ".git" in config.directories
    assert "/public-only" in config.directories
    assert "/private-only" not in config.directories
