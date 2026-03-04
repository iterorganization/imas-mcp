"""Tests for auto-generated schema context and schema_for() function.

Tests the build-time generation (gen_schema_context.py) and runtime
schema_for() function (schema_context.py) that provides task-specific
schema slices to agents.
"""

from pathlib import Path

import pytest
import yaml

# =============================================================================
# Task Groups YAML validation
# =============================================================================


class TestTaskGroupsYAML:
    """Validate task_groups.yaml against LinkML schemas."""

    @pytest.fixture(scope="class")
    def task_groups(self):
        yaml_path = (
            Path(__file__).parent.parent.parent
            / "imas_codex"
            / "schemas"
            / "task_groups.yaml"
        )
        with open(yaml_path) as f:
            return yaml.safe_load(f)

    @pytest.fixture(scope="class")
    def all_schema_labels(self):
        """Get all node labels from both schemas."""
        from imas_codex.graph.schema import GraphSchema

        schemas_dir = Path(__file__).parent.parent.parent / "imas_codex" / "schemas"
        facility = GraphSchema(schemas_dir / "facility.yaml")
        dd = GraphSchema(schemas_dir / "imas_dd.yaml")
        return set(facility.node_labels + dd.node_labels)

    def test_task_groups_file_exists(self):
        path = (
            Path(__file__).parent.parent.parent
            / "imas_codex"
            / "schemas"
            / "task_groups.yaml"
        )
        assert path.exists()

    def test_task_groups_has_required_groups(self, task_groups):
        expected = {"signals", "wiki", "imas", "code", "facility", "trees"}
        assert set(task_groups.keys()) == expected

    def test_each_group_has_labels_and_description(self, task_groups):
        for name, group in task_groups.items():
            assert "labels" in group, f"Group '{name}' missing 'labels'"
            assert "description" in group, f"Group '{name}' missing 'description'"
            assert isinstance(group["labels"], list)
            assert len(group["labels"]) > 0

    def test_all_labels_exist_in_schemas(self, task_groups, all_schema_labels):
        """Every label referenced in task groups must exist in LinkML schemas."""
        for group_name, group in task_groups.items():
            for label in group["labels"]:
                assert label in all_schema_labels, (
                    f"Label '{label}' in task group '{group_name}' "
                    f"not found in LinkML schemas"
                )


# =============================================================================
# Schema context generation (gen_schema_context.py)
# =============================================================================


class TestGenSchemaContext:
    """Test the build-time schema context generator."""

    def test_generate_schema_context_produces_valid_python(self, tmp_path):
        from scripts.gen_schema_context import generate_schema_context

        output = tmp_path / "schema_context_data.py"
        generate_schema_context(output_path=output, force=True)

        assert output.exists()
        content = output.read_text()

        # Must be valid Python
        compile(content, str(output), "exec")

    def test_generated_module_has_required_symbols(self, tmp_path):
        import importlib.util

        from scripts.gen_schema_context import generate_schema_context

        output = tmp_path / "schema_context_data.py"
        generate_schema_context(output_path=output, force=True)

        spec = importlib.util.spec_from_file_location("schema_context_data", output)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        assert hasattr(mod, "NODE_LABEL_PROPS")
        assert hasattr(mod, "ENUM_VALUES")
        assert hasattr(mod, "RELATIONSHIPS")
        assert hasattr(mod, "VECTOR_INDEXES")
        assert hasattr(mod, "TASK_GROUPS")

    def test_node_label_props_contains_facility(self, tmp_path):
        import importlib.util

        from scripts.gen_schema_context import generate_schema_context

        output = tmp_path / "schema_context_data.py"
        generate_schema_context(output_path=output, force=True)

        spec = importlib.util.spec_from_file_location("schema_context_data", output)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        assert "Facility" in mod.NODE_LABEL_PROPS
        assert "FacilitySignal" in mod.NODE_LABEL_PROPS
        # DD labels too
        assert "IMASPath" in mod.NODE_LABEL_PROPS

    def test_relationships_are_tuples(self, tmp_path):
        import importlib.util

        from scripts.gen_schema_context import generate_schema_context

        output = tmp_path / "schema_context_data.py"
        generate_schema_context(output_path=output, force=True)

        spec = importlib.util.spec_from_file_location("schema_context_data", output)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        assert len(mod.RELATIONSHIPS) > 0
        for rel in mod.RELATIONSHIPS:
            assert len(rel) == 4  # (from, type, to, cardinality)

    def test_vector_indexes_match_schemas(self, tmp_path):
        import importlib.util

        from scripts.gen_schema_context import generate_schema_context

        output = tmp_path / "schema_context_data.py"
        generate_schema_context(output_path=output, force=True)

        spec = importlib.util.spec_from_file_location("schema_context_data", output)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        assert len(mod.VECTOR_INDEXES) > 0
        # Check some known indexes
        assert "wiki_chunk_embedding" in mod.VECTOR_INDEXES
        assert "imas_path_embedding" in mod.VECTOR_INDEXES

    def test_task_groups_loaded(self, tmp_path):
        import importlib.util

        from scripts.gen_schema_context import generate_schema_context

        output = tmp_path / "schema_context_data.py"
        generate_schema_context(output_path=output, force=True)

        spec = importlib.util.spec_from_file_location("schema_context_data", output)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        assert "signals" in mod.TASK_GROUPS
        assert "FacilitySignal" in mod.TASK_GROUPS["signals"]

    def test_enum_values_present(self, tmp_path):
        import importlib.util

        from scripts.gen_schema_context import generate_schema_context

        output = tmp_path / "schema_context_data.py"
        generate_schema_context(output_path=output, force=True)

        spec = importlib.util.spec_from_file_location("schema_context_data", output)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        assert len(mod.ENUM_VALUES) > 0
        # Check a known enum
        assert "PathStatus" in mod.ENUM_VALUES or "SourceFileStatus" in mod.ENUM_VALUES


# =============================================================================
# Runtime schema_for() function
# =============================================================================


class TestSchemaFor:
    """Test the runtime schema_for() function."""

    def test_schema_for_overview_returns_string(self):
        from imas_codex.graph.schema_context import schema_for

        result = schema_for(task="overview")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_schema_for_signals_task(self):
        from imas_codex.graph.schema_context import schema_for

        result = schema_for(task="signals")
        assert isinstance(result, str)
        assert "FacilitySignal" in result
        assert "DataAccess" in result
        # Should NOT include unrelated labels as section headers
        assert "## WikiPage" not in result
        assert "## IMASPath" not in result

    def test_schema_for_wiki_task(self):
        from imas_codex.graph.schema_context import schema_for

        result = schema_for(task="wiki")
        assert "WikiPage" in result
        assert "WikiChunk" in result

    def test_schema_for_imas_task(self):
        from imas_codex.graph.schema_context import schema_for

        result = schema_for(task="imas")
        assert "IMASPath" in result
        assert "DDVersion" in result

    def test_schema_for_specific_labels(self):
        from imas_codex.graph.schema_context import schema_for

        result = schema_for("Facility", "MDSplusTree")
        assert "Facility" in result
        assert "MDSplusTree" in result
        # Should not contain unrelated labels as section headers
        assert "## WikiChunk" not in result

    def test_schema_for_overview_is_compact(self):
        """Overview should contain all labels but be compact (no full property lists)."""
        from imas_codex.graph.schema_context import schema_for

        result = schema_for(task="overview")
        assert "Facility" in result
        assert "IMASPath" in result

    def test_schema_for_includes_vector_indexes(self):
        from imas_codex.graph.schema_context import schema_for

        result = schema_for(task="signals")
        # Should include relevant vector indexes
        assert "embedding" in result.lower() or "vector" in result.lower()

    def test_schema_for_includes_relationships(self):
        from imas_codex.graph.schema_context import schema_for

        result = schema_for(task="signals")
        # Should include relationships for the labels
        assert "DATA_ACCESS" in result or "BELONGS_TO_DIAGNOSTIC" in result

    def test_schema_for_includes_enums(self):
        from imas_codex.graph.schema_context import schema_for

        result = schema_for(task="facility")
        assert "discovered" in result or "PathStatus" in result

    def test_schema_for_unknown_task_raises(self):
        from imas_codex.graph.schema_context import schema_for

        with pytest.raises(ValueError, match="Unknown task"):
            schema_for(task="nonexistent")

    def test_schema_for_unknown_label_raises(self):
        from imas_codex.graph.schema_context import schema_for

        with pytest.raises(ValueError, match="Unknown label"):
            schema_for("NonexistentLabel")

    def test_schema_for_token_efficiency(self):
        """Task-specific schema should be much smaller than full schema."""
        from imas_codex.graph.schema_context import schema_for

        overview = schema_for(task="overview")
        signals = schema_for(task="signals")
        # Signals slice should be significantly smaller than overview
        assert len(signals) < len(overview)
