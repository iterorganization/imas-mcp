"""Tests for mapping pipeline context formatter helpers.

These are pure functions that format data for prompt injection.
No Neo4j required.
"""

from imas_codex.ids.mapping import (
    _format_coordinate_context,
    _format_cross_facility_mappings,
    _format_identifier_schemas,
    _format_section_clusters,
    _format_version_context,
)


class TestFormatSectionClusters:
    def test_empty_list(self):
        assert _format_section_clusters([]) == "(no cluster data available)"

    def test_single_cluster(self):
        clusters = [
            {
                "label": "Temperature Profiles",
                "description": "Electron and ion temps",
                "paths": ["core_profiles/profiles_1d/electrons/temperature"],
            }
        ]
        result = _format_section_clusters(clusters)
        assert "Temperature Profiles" in result
        assert "Electron and ion temps" in result
        assert "core_profiles/profiles_1d/electrons/temperature" in result

    def test_many_paths_rendered(self):
        clusters = [
            {
                "label": "Big",
                "description": "",
                "paths": [f"a/b/c{i}" for i in range(15)],
            }
        ]
        result = _format_section_clusters(clusters)
        # All paths are rendered (no truncation)
        assert "a/b/c14" in result
        assert "**Big**" in result

    def test_no_description(self):
        clusters = [{"label": "X", "paths": []}]
        result = _format_section_clusters(clusters)
        assert "**X**" in result


class TestFormatIdentifierSchemas:
    def test_no_schemas(self):
        fields = [{"id": "a/b/c"}]
        assert _format_identifier_schemas(fields) == "(no identifier schemas)"

    def test_dict_schema(self):
        fields = [
            {
                "id": "equilibrium/time_slice/boundary/type",
                "identifier_schema": {
                    "schema_path": "boundary_type",
                    "documentation": "Type of boundary shape",
                    "options": [
                        {"name": "lcfs", "index": 0, "description": "Last closed"},
                        {"name": "limiter", "index": 1, "description": "Limiter"},
                    ],
                },
            }
        ]
        result = _format_identifier_schemas(fields)
        assert "boundary_type" in result
        assert "lcfs" in result
        assert "limiter" in result
        assert "equilibrium/time_slice/boundary/type" in result

    def test_object_schema(self):
        """Schema as Pydantic-like object with attributes."""
        from imas_codex.core.data_model import (
            IdentifierOption,
            IdentifierSchema,
        )

        schema = IdentifierSchema(
            schema_path="boundary_type",
            documentation="Type",
            options=[IdentifierOption(name="lcfs", index=0, description="")],
        )
        fields = [{"path": "x/y/z", "identifier_schema": schema}]
        result = _format_identifier_schemas(fields)
        assert "boundary_type" in result
        assert "x/y/z" in result

    def test_empty_fields(self):
        assert _format_identifier_schemas([]) == "(no identifier schemas)"


class TestFormatVersionContext:
    def test_empty_paths(self):
        assert _format_version_context({"paths": {}}) == "(no version change history)"

    def test_path_with_changes(self):
        ctx = {
            "paths": {
                "core_profiles/profiles_1d/electrons/pressure": {
                    "notable_changes": [
                        {
                            "version": "4.0.0",
                            "type": "units",
                            "summary": "New electron pressure path added",
                        }
                    ]
                }
            }
        }
        result = _format_version_context(ctx)
        assert "core_profiles/profiles_1d/electrons/pressure" in result
        assert "units" in result
        assert "v4.0.0" in result

    def test_path_without_changes(self):
        ctx = {"paths": {"a/b/c": {"notable_changes": []}}}
        result = _format_version_context(ctx)
        assert result == "(no notable version changes)"


class TestFormatCoordinateContext:
    def test_empty_fields(self):
        assert _format_coordinate_context([]) == "(no coordinate spec data)"

    def test_no_coordinates(self):
        fields = [{"id": "a/b", "coordinates": []}]
        assert _format_coordinate_context(fields) == "(no coordinate spec data)"

    def test_with_coordinates(self):
        fields = [
            {
                "id": "equilibrium/time_slice/profiles_1d/psi",
                "coordinates": ["rho_tor_norm", "time"],
            }
        ]
        result = _format_coordinate_context(fields)
        assert "equilibrium/time_slice/profiles_1d/psi" in result
        assert "rho_tor_norm" in result
        assert "time" in result


class TestFormatCrossFacilityMappings:
    def test_empty_list(self):
        assert _format_cross_facility_mappings([]) == ""

    def test_single_facility(self):
        rows = [
            {"facility": "tcv", "target_path": "equilibrium/time_slice/global_quantities/ip"},
            {"facility": "tcv", "target_path": "equilibrium/time_slice/profiles_1d/psi"},
        ]
        result = _format_cross_facility_mappings(rows)
        assert "**tcv**" in result
        assert "equilibrium/time_slice/global_quantities/ip" in result
        assert "equilibrium/time_slice/profiles_1d/psi" in result

    def test_multiple_facilities_sorted(self):
        rows = [
            {"facility": "tcv", "target_path": "pf_active/coil/current"},
            {"facility": "aug", "target_path": "pf_active/coil/voltage"},
        ]
        result = _format_cross_facility_mappings(rows)
        lines = result.strip().split("\n")
        assert lines[0].startswith("- **aug**")
        assert lines[1].startswith("- **tcv**")
