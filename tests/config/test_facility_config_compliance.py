"""Facility config schema compliance tests.

Validates that all public facility YAML configs conform to the LinkML-derived
Pydantic schema. Tests are fully schema-driven: adding new fields, enums, or
model classes to the schema automatically extends test coverage.

Key design decisions:
- Auto-discovers all public YAML files via glob (new facilities get tested)
- Introspects Pydantic models at test time (schema changes are auto-incorporated)
- Detects unknown YAML keys that ConfiguredBaseModel's extra="ignore" silently drops
- Recursively validates nested structures against their typed model classes
- Validates enum values against schema-defined permissible values
"""

from __future__ import annotations

import enum
import types
from pathlib import Path
from typing import Any, Union, get_args, get_origin

import pytest
import yaml

from imas_codex.config.models import (
    ConfiguredBaseModel,
    FacilityConfig,
)

# ── Helpers ──────────────────────────────────────────────────────────────

FACILITIES_DIR = (
    Path(__file__).parent.parent.parent / "imas_codex" / "config" / "facilities"
)


def _public_config_files() -> list[Path]:
    """Discover all public facility config files (excluding private)."""
    return sorted(
        p for p in FACILITIES_DIR.glob("*.yaml") if not p.stem.endswith("_private")
    )


def _load_raw_yaml(path: Path) -> dict[str, Any]:
    """Load raw YAML without any merging or renaming."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _get_model_fields(model_cls: type[ConfiguredBaseModel]) -> dict[str, Any]:
    """Get all field names and their FieldInfo from a Pydantic model."""
    return dict(model_cls.model_fields)


def _resolve_inner_model(
    field_info: Any,
) -> type[ConfiguredBaseModel] | None:
    """Extract the Pydantic model class from a field's type annotation.

    Handles: Optional[X], list[X], X | None, and plain X.
    Returns None if the field's type is not a ConfiguredBaseModel subclass.
    """
    annotation = field_info.annotation
    return _extract_model_from_annotation(annotation)


def _extract_model_from_annotation(
    annotation: Any,
) -> type[ConfiguredBaseModel] | None:
    """Recursively extract ConfiguredBaseModel subclass from a type annotation."""
    if annotation is None:
        return None

    # Direct model class
    if isinstance(annotation, type) and issubclass(annotation, ConfiguredBaseModel):
        return annotation

    origin = get_origin(annotation)
    args = get_args(annotation)

    # Union / Optional (X | None)
    if origin is Union or origin is types.UnionType:
        for arg in args:
            if arg is type(None):
                continue
            result = _extract_model_from_annotation(arg)
            if result is not None:
                return result
        return None

    # list[X]
    if origin is list:
        for arg in args:
            result = _extract_model_from_annotation(arg)
            if result is not None:
                return result
        return None

    return None


def _resolve_enum_type(field_info: Any) -> type[enum.Enum] | None:
    """Extract the Enum class from a field's type annotation, if any."""
    annotation = field_info.annotation
    return _extract_enum_from_annotation(annotation)


def _extract_enum_from_annotation(annotation: Any) -> type[enum.Enum] | None:
    """Recursively extract Enum subclass from a type annotation."""
    if annotation is None:
        return None

    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return annotation

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Union or origin is types.UnionType:
        for arg in args:
            if arg is type(None):
                continue
            result = _extract_enum_from_annotation(arg)
            if result is not None:
                return result
        return None

    if origin is list:
        for arg in args:
            result = _extract_enum_from_annotation(arg)
            if result is not None:
                return result
        return None

    return None


def _is_list_field(field_info: Any) -> bool:
    """Check if a field is typed as a list."""
    annotation = field_info.annotation
    if get_origin(annotation) is list:
        return True
    # Optional[list[X]] or list[X] | None
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        for arg in get_args(annotation):
            if arg is type(None):
                continue
            if get_origin(arg) is list:
                return True
    return False


def _collect_unknown_keys(
    data: dict[str, Any],
    model_cls: type[ConfiguredBaseModel],
    path: str = "",
) -> list[str]:
    """Recursively find YAML keys not declared in the Pydantic model.

    Returns a list of dotted-path strings for each unknown key found.
    """
    violations = []
    fields = _get_model_fields(model_cls)
    known_keys = set(fields.keys()) | {"linkml_meta"}

    for key in data:
        # Skip YAML anchor keys (e.g., _wiki_defaults) — these are DRY
        # mechanisms, not real config fields.
        if key.startswith("_"):
            continue
        dotted = f"{path}.{key}" if path else key
        if key not in known_keys:
            violations.append(dotted)
            continue

        value = data[key]
        if value is None:
            continue

        field_info = fields.get(key)
        if field_info is None:
            continue

        inner_model = _resolve_inner_model(field_info)
        if inner_model is None:
            continue

        if _is_list_field(field_info) and isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    violations.extend(
                        _collect_unknown_keys(item, inner_model, f"{dotted}[{i}]")
                    )
        elif isinstance(value, dict):
            violations.extend(_collect_unknown_keys(value, inner_model, dotted))

    return violations


def _collect_enum_violations(
    data: dict[str, Any],
    model_cls: type[ConfiguredBaseModel],
    path: str = "",
) -> list[str]:
    """Recursively find enum field values not in the schema's permissible values."""
    violations = []
    fields = _get_model_fields(model_cls)

    for key, value in data.items():
        if value is None:
            continue

        field_info = fields.get(key)
        if field_info is None:
            continue

        dotted = f"{path}.{key}" if path else key

        enum_type = _resolve_enum_type(field_info)
        if enum_type is not None and isinstance(value, str):
            valid_values = {e.value for e in enum_type}
            if value not in valid_values:
                violations.append(
                    f"{dotted} = {value!r} (valid: {sorted(valid_values)})"
                )
            continue

        inner_model = _resolve_inner_model(field_info)
        if inner_model is None:
            continue

        if _is_list_field(field_info) and isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    violations.extend(
                        _collect_enum_violations(item, inner_model, f"{dotted}[{i}]")
                    )
        elif isinstance(value, dict):
            violations.extend(_collect_enum_violations(value, inner_model, dotted))

    return violations


def _collect_required_violations(
    data: dict[str, Any],
    model_cls: type[ConfiguredBaseModel],
    path: str = "",
) -> list[str]:
    """Recursively find missing required fields in YAML data."""
    violations = []
    fields = _get_model_fields(model_cls)

    # Check required fields at this level
    for field_name, field_info in fields.items():
        if field_info.is_required():
            dotted = f"{path}.{field_name}" if path else field_name
            if field_name not in data or data[field_name] is None:
                violations.append(dotted)

    # Recurse into nested models
    for key, value in data.items():
        if value is None:
            continue

        field_info = fields.get(key)
        if field_info is None:
            continue

        dotted = f"{path}.{key}" if path else key
        inner_model = _resolve_inner_model(field_info)
        if inner_model is None:
            continue

        if _is_list_field(field_info) and isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    violations.extend(
                        _collect_required_violations(
                            item, inner_model, f"{dotted}[{i}]"
                        )
                    )
        elif isinstance(value, dict):
            violations.extend(_collect_required_violations(value, inner_model, dotted))

    return violations


# ── Fixtures ─────────────────────────────────────────────────────────────

CONFIG_FILES = _public_config_files()
CONFIG_IDS = [p.stem for p in CONFIG_FILES]


@pytest.fixture(params=CONFIG_FILES, ids=CONFIG_IDS)
def config_path(request: pytest.FixtureRequest) -> Path:
    """Parametrized fixture yielding each public config file path."""
    return request.param


@pytest.fixture
def raw_yaml(config_path: Path) -> dict[str, Any]:
    """Load raw YAML data for a facility config file."""
    return _load_raw_yaml(config_path)


# ── Tests ────────────────────────────────────────────────────────────────


class TestPydanticValidation:
    """Verify all configs pass Pydantic model validation."""

    def test_valid_yaml_parse(self, config_path: Path) -> None:
        """YAML file parses without errors."""
        data = _load_raw_yaml(config_path)
        assert isinstance(data, dict), f"{config_path.name} did not parse as a dict"

    def test_pydantic_model_validates(self, raw_yaml: dict, config_path: Path) -> None:
        """Config passes FacilityConfig.model_validate() without errors.

        This catches type mismatches, malformed nested structures, and
        constraint violations — everything except unknown keys (which
        extra='ignore' silently drops).
        """
        # The loader renames 'facility' → 'id'; replicate that here
        data = dict(raw_yaml)
        if "facility" in data:
            data["id"] = data.pop("facility")

        FacilityConfig.model_validate(data)


class TestUnknownKeys:
    """Detect YAML keys that the schema silently ignores.

    ConfiguredBaseModel uses extra="ignore", so unknown keys are silently
    dropped during validation. This test catches misspelled or deprecated
    keys that would otherwise go unnoticed.
    """

    def test_no_unknown_keys(self, raw_yaml: dict, config_path: Path) -> None:
        """All YAML keys must be declared in the Pydantic model."""
        unknown = _collect_unknown_keys(raw_yaml, FacilityConfig)
        assert not unknown, (
            f"{config_path.name} has keys not in schema:\n  "
            + "\n  ".join(unknown)
            + "\n\nEither add these fields to facility_config.yaml "
            "or remove them from the config."
        )


class TestEnumValues:
    """Verify enum-typed fields contain only schema-defined values."""

    def test_enum_values_valid(self, raw_yaml: dict, config_path: Path) -> None:
        """All enum field values must be in the schema's permissible values."""
        violations = _collect_enum_violations(raw_yaml, FacilityConfig)
        assert not violations, (
            f"{config_path.name} has invalid enum values:\n  "
            + "\n  ".join(violations)
            + "\n\nUpdate the value or extend the enum in facility_config.yaml."
        )


class TestRequiredFields:
    """Verify required fields are present and non-null."""

    def test_required_fields_present(self, raw_yaml: dict, config_path: Path) -> None:
        """All required fields must be present and non-null."""
        # The loader renames 'facility' → 'id'; replicate that here.
        # Both 'facility' and 'id' are optional at the schema level
        # since only one needs to be present.
        data = dict(raw_yaml)
        if "facility" in data:
            data["id"] = data.pop("facility")

        violations = _collect_required_violations(data, FacilityConfig)
        assert not violations, (
            f"{config_path.name} is missing required fields:\n  "
            + "\n  ".join(violations)
            + "\n\nAdd these fields to the config or make them optional "
            "in facility_config.yaml."
        )


class TestSchemaCompleteness:
    """Verify the schema covers all model classes and enums.

    These tests don't check individual configs — they verify that the
    test infrastructure itself is complete and would catch issues.
    """

    def test_all_configs_discovered(self) -> None:
        """At least one config file must exist for tests to be meaningful."""
        assert len(CONFIG_FILES) > 0, (
            f"No public config files found in {FACILITIES_DIR}"
        )

    def test_facility_config_has_fields(self) -> None:
        """FacilityConfig model must have at least the core fields."""
        fields = set(_get_model_fields(FacilityConfig).keys())
        core_fields = {"name", "machine", "data_systems", "wiki_sites"}
        missing = core_fields - fields
        assert not missing, (
            f"FacilityConfig is missing core fields: {missing}. "
            "Schema may have changed incompatibly."
        )

    def test_all_model_classes_are_config_base(self) -> None:
        """All nested model types used in FacilityConfig are ConfiguredBaseModel."""
        fields = _get_model_fields(FacilityConfig)
        for field_name, field_info in fields.items():
            inner = _resolve_inner_model(field_info)
            if inner is not None:
                assert issubclass(inner, ConfiguredBaseModel), (
                    f"FacilityConfig.{field_name} references {inner.__name__} "
                    "which is not a ConfiguredBaseModel subclass"
                )
