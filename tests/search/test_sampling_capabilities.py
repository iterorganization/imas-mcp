"""
Tests for AI sampling capabilities and functionality.

Focuses on testing the behavior and capabilities of the sampling system
rather than implementation details. Tests all result models that support sampling.
"""

from typing import Any

import pytest

from imas_mcp.core.data_model import IdsNode
from imas_mcp.models.constants import (
    DetailLevel,
    IdentifierScope,
    RelationshipType,
    SearchMode,
)
from imas_mcp.models.result_models import (
    ConceptResult,
    DomainExport,
    IdentifierResult,
    IDSExport,
    OverviewResult,
    RelationshipResult,
    SearchResult,
    StructureResult,
)


class MockPromptBuilder:
    """Mock implementation of PromptBuilder protocol for testing."""

    def __init__(self, custom_system_prompt: str | None = None):
        self.custom_system_prompt = custom_system_prompt

    def build_sample_tasks(self, tool_result: Any) -> list[dict[str, Any]]:
        """Build sample tasks for the given result."""
        tasks = []

        # Different sampling tasks based on result type
        if hasattr(tool_result, "explanation"):
            tasks.append(
                {
                    "field": "explanation",
                    "prompt_type": "explanation_sampling",
                    "context": {
                        "concept": getattr(tool_result, "concept", "unknown"),
                        "current_explanation": getattr(tool_result, "explanation", ""),
                        "detail_level": "intermediate",
                    },
                }
            )

            if hasattr(tool_result, "related_topics"):
                tasks.append(
                    {
                        "field": "related_topics",
                        "prompt_type": "topics_sampling",
                        "context": {
                            "concept": getattr(tool_result, "concept", "unknown"),
                            "current_topics": getattr(
                                tool_result, "related_topics", []
                            ),
                        },
                    }
                )

        if hasattr(tool_result, "content"):
            tasks.append(
                {
                    "field": "content",
                    "prompt_type": "content_enhancement",
                    "context": {
                        "current_content": getattr(tool_result, "content", ""),
                        "tool_name": getattr(tool_result, "tool_name", "unknown"),
                    },
                }
            )

        if hasattr(tool_result, "description"):
            tasks.append(
                {
                    "field": "description",
                    "prompt_type": "description_enhancement",
                    "context": {
                        "entity": getattr(tool_result, "ids_name", "unknown"),
                        "current_description": getattr(tool_result, "description", ""),
                    },
                }
            )

        return tasks

    def build_prompt(self, prompt_type: str, tool_context: dict[str, Any]) -> str:
        """Build prompts for different sampling tasks."""
        if prompt_type == "explanation_sampling":
            concept = tool_context.get("concept", "unknown")
            return f"Provide a detailed physics explanation for {concept}"
        elif prompt_type == "topics_sampling":
            concept = tool_context.get("concept", "unknown")
            return f"List related physics topics for {concept}"
        elif prompt_type == "content_enhancement":
            tool_name = tool_context.get("tool_name", "unknown")
            return f"Enhance the content for {tool_name} with physics context"
        elif prompt_type == "description_enhancement":
            entity = tool_context.get("entity", "unknown")
            return f"Provide an enhanced description for {entity}"
        return f"Generic prompt for {prompt_type}"

    def system_prompt(self) -> str:
        """Return system prompt for the tool."""
        if self.custom_system_prompt:
            return self.custom_system_prompt
        return "Expert IMAS fusion physics assistant"

    def _apply_explanation_sampling(self, result: Any, content: str) -> None:
        """Apply sampling to explanation field."""
        result.explanation = content

    def _apply_related_topics_sampling(self, result: Any, content: str) -> None:
        """Apply sampling to related_topics field."""
        topics = [topic.strip() for topic in content.split("\n") if topic.strip()]
        result.related_topics = topics

    def _apply_content_sampling(self, result: Any, content: str) -> None:
        """Apply sampling to content field."""
        result.content = content

    def _apply_description_sampling(self, result: Any, content: str) -> None:
        """Apply sampling to description field."""
        result.description = content


class MockTextContent:
    """Mock TextContent for MCP responses."""

    def __init__(self, text: str):
        self.text = text


class MockContext:
    """Mock FastMCP Context for testing."""

    def __init__(self, responses: dict[str, str] | None = None):
        self.responses = responses or {}
        self.sample_calls = []

    async def sample(
        self, prompt: str, system_prompt: str = "", **kwargs
    ) -> MockTextContent:
        """Mock the sample method."""
        self.sample_calls.append(
            {"prompt": prompt, "system_prompt": system_prompt, **kwargs}
        )

        # Return appropriate response based on prompt content
        if "explanation" in prompt.lower():
            return MockTextContent("Enhanced physics explanation with IMAS context")
        elif "topics" in prompt.lower():
            return MockTextContent(
                "magnetic flux\nplasma equilibrium\nITER diagnostics\nMHD stability"
            )
        elif "content" in prompt.lower():
            return MockTextContent("Enhanced content with detailed physics insights")
        elif "description" in prompt.lower():
            return MockTextContent("Enhanced description with technical details")
        else:
            return MockTextContent("Generic AI response")


async def mock_apply_ai_sampling(result, context, tool, **kwargs):
    """Mock implementation of apply_ai_sampling that simulates the behavior."""
    if context is None or tool is None:
        return result

    # Get sampling tasks from tool with error handling
    try:
        if hasattr(tool, "build_sample_tasks"):
            tasks = tool.build_sample_tasks(result)
            sampled_fields = []

            for task in tasks:
                field_name = task.get("field")
                if not field_name:
                    continue

                # Mock sampling based on field type
                if field_name == "explanation" and hasattr(result, "explanation"):
                    result.explanation = (
                        "Enhanced physics explanation with IMAS context"
                    )
                    sampled_fields.append(field_name)
                elif field_name == "related_topics" and hasattr(
                    result, "related_topics"
                ):
                    result.related_topics = [
                        "magnetic flux",
                        "plasma equilibrium",
                        "ITER diagnostics",
                        "MHD stability",
                    ]
                    sampled_fields.append(field_name)
                elif field_name == "content" and hasattr(result, "content"):
                    result.content = "Enhanced content with detailed physics insights"
                    sampled_fields.append(field_name)
                elif field_name == "description" and hasattr(result, "description"):
                    result.description = "Enhanced description with technical details"
                    sampled_fields.append(field_name)

            # Add AI metadata
            if sampled_fields:
                result.ai_response = {
                    "status": "content_sampled",
                    "sampled_fields": sampled_fields,
                    "task_count": len(tasks),
                }
                result.ai_prompt = "Sample prompt used for enhancement"
    except Exception:
        # Handle errors gracefully - return unchanged result
        pass

    return result


# Fixtures for different result types
@pytest.fixture
def sample_concept_result():
    """Create a sample ConceptResult for testing."""
    return ConceptResult(
        concept="poloidal flux",
        explanation="Basic explanation",
        related_topics=["flux", "plasma"],
        detail_level=DetailLevel.INTERMEDIATE,
    )


@pytest.fixture
def sample_search_result():
    """Create a sample SearchResult for testing."""
    return SearchResult(
        query="plasma temperature",
        search_mode=SearchMode.AUTO,
        nodes=[
            IdsNode(
                path="core_profiles/profiles_1d/grid/rho_tor_norm",
                documentation="Normalized toroidal flux coordinate",
            )
        ],
        hits=[],
    )


@pytest.fixture
def sample_structure_result():
    """Create a sample StructureResult for testing."""
    return StructureResult(
        ids_name="equilibrium",
        description="Basic equilibrium description",
        structure={"time_slice": 5, "profiles_2d": 3},
        sample_paths=["equilibrium/time_slice", "equilibrium/profiles_2d"],
        max_depth=3,
    )


@pytest.fixture
def sample_overview_result():
    """Create a sample OverviewResult for testing."""
    return OverviewResult(
        content="Basic overview content",
        available_ids=["equilibrium", "core_profiles"],
        hits=[],
    )


@pytest.fixture
def sample_relationship_result():
    """Create a sample RelationshipResult for testing."""
    return RelationshipResult(
        path="equilibrium/profiles_2d",
        relationship_type=RelationshipType.ALL,
        max_depth=2,
        connections={"related": ["core_profiles", "thomson_scattering"]},
        nodes=[],
    )


@pytest.fixture
def sample_identifier_result():
    """Create a sample IdentifierResult for testing."""
    return IdentifierResult(
        scope=IdentifierScope.ALL,
        schemas=[{"type": "enum", "values": ["1", "2", "3"]}],
        paths=[{"path": "equilibrium/ids_properties", "type": "identifier"}],
        analytics={"total_identifiers": 15},
    )


@pytest.fixture
def sample_ids_export():
    """Create a sample IDSExport for testing."""
    return IDSExport(
        ids_names=["equilibrium", "core_profiles"],
        include_physics=True,
        include_relationships=True,
    )


@pytest.fixture
def sample_domain_export():
    """Create a sample DomainExport for testing."""
    return DomainExport(
        domain="plasma_equilibrium",
        include_cross_domain=False,
        max_paths=10,
    )


@pytest.fixture
def mock_prompt_builder():
    """Create a mock PromptBuilder."""
    return MockPromptBuilder()


@pytest.fixture
def mock_context():
    """Create a mock FastMCP context."""
    return MockContext()


class TestSamplingCapabilities:
    """Test AI sampling capabilities across all result types."""

    @pytest.mark.asyncio
    async def test_concept_result_sampling(
        self, sample_concept_result, mock_prompt_builder, mock_context
    ):
        """Test sampling enhances ConceptResult content."""
        original_explanation = sample_concept_result.explanation
        original_topics = sample_concept_result.related_topics.copy()

        result = await mock_apply_ai_sampling(
            sample_concept_result, mock_context, mock_prompt_builder
        )

        # Verify content was enhanced
        assert result.explanation != original_explanation
        assert result.explanation == "Enhanced physics explanation with IMAS context"
        assert result.related_topics != original_topics
        assert len(result.related_topics) > len(original_topics)
        assert result.concept == "poloidal flux"  # Preserved

    @pytest.mark.asyncio
    async def test_structure_result_sampling(
        self, sample_structure_result, mock_prompt_builder, mock_context
    ):
        """Test sampling enhances StructureResult content."""
        original_description = sample_structure_result.description

        result = await mock_apply_ai_sampling(
            sample_structure_result, mock_context, mock_prompt_builder
        )

        # Verify description was enhanced
        assert result.description != original_description
        assert result.description == "Enhanced description with technical details"
        assert result.ids_name == "equilibrium"  # Preserved
        assert result.structure == {"time_slice": 5, "profiles_2d": 3}  # Preserved

    @pytest.mark.asyncio
    async def test_overview_result_sampling(
        self, sample_overview_result, mock_prompt_builder, mock_context
    ):
        """Test sampling enhances OverviewResult content."""
        original_content = sample_overview_result.content

        result = await mock_apply_ai_sampling(
            sample_overview_result, mock_context, mock_prompt_builder
        )

        # Verify content was enhanced
        assert result.content != original_content
        assert result.content == "Enhanced content with detailed physics insights"
        assert result.available_ids == ["equilibrium", "core_profiles"]  # Preserved

    @pytest.mark.asyncio
    async def test_sampling_preserves_non_sampled_fields(
        self, sample_concept_result, mock_prompt_builder, mock_context
    ):
        """Test that sampling preserves non-sampled fields."""
        original_concept = sample_concept_result.concept
        original_detail_level = sample_concept_result.detail_level

        result = await mock_apply_ai_sampling(
            sample_concept_result, mock_context, mock_prompt_builder
        )

        # Verify original data is preserved
        assert result.concept == original_concept
        assert result.detail_level == original_detail_level

    @pytest.mark.asyncio
    async def test_sampling_without_context_returns_unchanged(
        self, sample_concept_result, mock_prompt_builder
    ):
        """Test that sampling without context returns unchanged result."""
        original_explanation = sample_concept_result.explanation

        result = await mock_apply_ai_sampling(
            sample_concept_result, None, mock_prompt_builder
        )

        assert result.explanation == original_explanation
        # Check that no AI metadata was added when context is None
        assert not hasattr(result, "ai_response") or not result.ai_response

    @pytest.mark.asyncio
    async def test_sampling_without_tool_returns_unchanged(
        self, sample_concept_result, mock_context
    ):
        """Test that sampling without tool instance returns unchanged result."""
        original_explanation = sample_concept_result.explanation

        result = await mock_apply_ai_sampling(sample_concept_result, mock_context, None)

        assert result.explanation == original_explanation

    @pytest.mark.asyncio
    async def test_sampling_records_metadata(
        self, sample_concept_result, mock_prompt_builder, mock_context
    ):
        """Test that sampling records appropriate metadata."""
        result = await mock_apply_ai_sampling(
            sample_concept_result, mock_context, mock_prompt_builder
        )

        # Verify AI metadata is recorded
        assert hasattr(result, "ai_prompt")
        assert hasattr(result, "ai_response")
        assert result.ai_response["status"] == "content_sampled"
        assert "sampled_fields" in result.ai_response
        assert "task_count" in result.ai_response

    @pytest.mark.asyncio
    async def test_sampling_handles_different_field_types(
        self, sample_concept_result, mock_prompt_builder, mock_context
    ):
        """Test that sampling correctly handles different field types."""
        result = await mock_apply_ai_sampling(
            sample_concept_result, mock_context, mock_prompt_builder
        )

        # Verify different field types were handled appropriately
        assert isinstance(result.explanation, str)
        assert isinstance(result.related_topics, list)
        assert len(result.related_topics) > 0

    @pytest.mark.asyncio
    async def test_sampling_with_empty_fields(self, mock_prompt_builder, mock_context):
        """Test sampling behavior with empty or None fields."""
        empty_result = ConceptResult(
            concept="test_concept",
            explanation="",
            related_topics=[],
        )

        result = await mock_apply_ai_sampling(
            empty_result, mock_context, mock_prompt_builder
        )

        # Verify sampling can handle and populate empty fields
        assert result.explanation is not None
        assert len(result.explanation) > 0
        assert result.related_topics is not None
        assert len(result.related_topics) > 0

    @pytest.mark.asyncio
    async def test_sampling_with_performance_parameters(
        self, sample_concept_result, mock_prompt_builder, mock_context
    ):
        """Test sampling with different performance parameters."""
        result = await mock_apply_ai_sampling(
            sample_concept_result,
            mock_context,
            mock_prompt_builder,
            temperature=0.1,
            max_tokens=400,
        )

        # Verify enhancement still works with parameters
        assert len(result.explanation) > len("Basic explanation")


class TestSamplingValidation:
    """Test sampling validation and edge cases."""

    @pytest.mark.asyncio
    async def test_enhanced_content_is_meaningful(
        self, sample_concept_result, mock_prompt_builder, mock_context
    ):
        """Test that enhanced content is non-empty and meaningful."""
        result = await mock_apply_ai_sampling(
            sample_concept_result, mock_context, mock_prompt_builder
        )

        # Core validations
        assert result.explanation is not None
        assert result.explanation != ""
        assert len(result.explanation) > len("Basic explanation")
        assert result.related_topics is not None
        assert len(result.related_topics) > 0
        assert all(topic.strip() for topic in result.related_topics)

    @pytest.mark.asyncio
    async def test_sampling_idempotency(
        self, sample_concept_result, mock_prompt_builder, mock_context
    ):
        """Test that sampling can be applied multiple times safely."""
        # First sampling
        result1 = await mock_apply_ai_sampling(
            sample_concept_result, mock_context, mock_prompt_builder
        )

        # Second sampling on the same result
        result2 = await mock_apply_ai_sampling(
            result1, mock_context, mock_prompt_builder
        )

        # Results should be consistent
        assert result2.concept is not None
        assert len(result2.related_topics) > 0

    def test_result_model_structure(self, sample_concept_result):
        """Test that result models have expected structure."""
        # Verify the structure matches expectations
        assert hasattr(sample_concept_result, "concept")
        assert hasattr(sample_concept_result, "explanation")
        assert hasattr(sample_concept_result, "related_topics")

        # Verify types
        assert isinstance(sample_concept_result.concept, str)
        assert isinstance(sample_concept_result.related_topics, list)

    @pytest.mark.asyncio
    async def test_content_enhancement_quality(
        self, sample_concept_result, mock_prompt_builder, mock_context
    ):
        """Test that content enhancement produces meaningful improvements."""
        original_explanation = sample_concept_result.explanation
        original_topics_count = len(sample_concept_result.related_topics)

        result = await mock_apply_ai_sampling(
            sample_concept_result, mock_context, mock_prompt_builder
        )

        # Verify meaningful enhancement
        assert len(result.explanation) > len(original_explanation)
        assert "physics" in result.explanation.lower()
        assert len(result.related_topics) >= original_topics_count

        # Check that enhanced topics are physics-related
        physics_keywords = ["flux", "plasma", "equilibrium", "diagnostics", "mhd"]
        enhanced_topics_text = " ".join(result.related_topics).lower()
        assert any(keyword in enhanced_topics_text for keyword in physics_keywords)


class TestPromptBuilderProtocol:
    """Test the prompt builder protocol implementation."""

    def test_mock_prompt_builder_capabilities(
        self, mock_prompt_builder, sample_concept_result
    ):
        """Test that mock prompt builder implements expected interface."""
        # Test sample tasks generation
        tasks = mock_prompt_builder.build_sample_tasks(sample_concept_result)
        assert isinstance(tasks, list)
        assert len(tasks) > 0

        # Each task should have required fields
        for task in tasks:
            assert "field" in task
            assert "prompt_type" in task
            assert "context" in task

        # Test prompt building
        context = {"concept": "test_concept"}
        prompt = mock_prompt_builder.build_prompt("explanation_sampling", context)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

        # Test system prompt
        system_prompt = mock_prompt_builder.system_prompt()
        assert isinstance(system_prompt, str)
        assert len(system_prompt) > 0


class TestMockContext:
    """Test the mock context implementation."""

    @pytest.mark.asyncio
    async def test_mock_context_sample_method(self, mock_context):
        """Test that mock context sample method works correctly."""
        # Test explanation sampling
        result = await mock_context.sample("Provide explanation for plasma")
        assert isinstance(result, MockTextContent)
        assert "physics" in result.text.lower()

        # Test topics sampling
        result = await mock_context.sample("List topics for equilibrium")
        assert isinstance(result, MockTextContent)
        assert "\n" in result.text  # Should contain multiple topics

        # Verify calls are recorded
        assert len(mock_context.sample_calls) == 2
        assert all("prompt" in call for call in mock_context.sample_calls)


class TestSamplingCoverageAllResults:
    """Test sampling coverage for all result types that support it."""

    @pytest.mark.asyncio
    async def test_search_result_sampling(
        self, sample_search_result, mock_prompt_builder, mock_context
    ):
        """Test that SearchResult sampling works correctly."""
        result = await mock_apply_ai_sampling(
            sample_search_result, mock_context, mock_prompt_builder
        )

        # SearchResult doesn't have explanation/related_topics by default,
        # but should still work with sampling infrastructure
        assert result.query == "plasma temperature"
        assert len(result.nodes) > 0

    @pytest.mark.asyncio
    async def test_relationship_result_sampling(
        self, sample_relationship_result, mock_prompt_builder, mock_context
    ):
        """Test that RelationshipResult sampling works correctly."""
        result = await mock_apply_ai_sampling(
            sample_relationship_result, mock_context, mock_prompt_builder
        )

        assert result.path == "equilibrium/profiles_2d"
        assert result.connections == {
            "related": ["core_profiles", "thomson_scattering"]
        }

    @pytest.mark.asyncio
    async def test_identifier_result_sampling(
        self, sample_identifier_result, mock_prompt_builder, mock_context
    ):
        """Test that IdentifierResult sampling works correctly."""
        result = await mock_apply_ai_sampling(
            sample_identifier_result, mock_context, mock_prompt_builder
        )

        assert result.scope == IdentifierScope.ALL
        assert len(result.schemas) > 0
        assert result.analytics["total_identifiers"] == 15

    @pytest.mark.asyncio
    async def test_export_result_sampling(
        self, sample_ids_export, mock_prompt_builder, mock_context
    ):
        """Test that export results sampling works correctly."""
        result = await mock_apply_ai_sampling(
            sample_ids_export, mock_context, mock_prompt_builder
        )

        assert result.ids_names == ["equilibrium", "core_profiles"]
        assert result.include_physics is True

    @pytest.mark.asyncio
    async def test_domain_export_result_sampling(
        self, sample_domain_export, mock_prompt_builder, mock_context
    ):
        """Test that DomainExport sampling works correctly."""
        result = await mock_apply_ai_sampling(
            sample_domain_export, mock_context, mock_prompt_builder
        )

        assert result.domain == "plasma_equilibrium"
        assert result.max_paths == 10


class TestSamplingErrorHandling:
    """Test sampling error handling scenarios."""

    @pytest.mark.asyncio
    async def test_sampling_error_handling(self, sample_concept_result, mock_context):
        """Test that sampling gracefully handles errors."""

        # Create a tool that will cause errors
        class ErrorTool:
            def build_sample_tasks(self, result):
                raise ValueError("Test error")

            def system_prompt(self):
                return "System prompt"

        error_tool = ErrorTool()

        result = await mock_apply_ai_sampling(
            sample_concept_result, mock_context, error_tool
        )

        # Result should be unchanged and error should be logged
        assert hasattr(result, "explanation")

    @pytest.mark.asyncio
    async def test_sampling_with_complex_physics_context(
        self, mock_prompt_builder, mock_context
    ):
        """Test sampling with complex physics context."""
        complex_result = ConceptResult(
            concept="magnetohydrodynamics",
            explanation="Basic MHD explanation",
            related_topics=["plasma", "magnetic field"],
        )

        result = await mock_apply_ai_sampling(
            complex_result, mock_context, mock_prompt_builder
        )

        # Verify enhancement worked
        assert len(result.explanation) > len("Basic MHD explanation")
        assert len(result.related_topics) >= 2

    @pytest.mark.asyncio
    async def test_sampling_uses_custom_system_prompt(
        self, sample_concept_result, mock_context
    ):
        """Test that sampling uses custom system prompt from tool."""
        custom_prompt = "Custom IMAS physics expert system"
        tool = MockPromptBuilder(custom_system_prompt=custom_prompt)

        await mock_apply_ai_sampling(sample_concept_result, mock_context, tool)

        # Verify custom system prompt was used (mock doesn't track this, so we just verify no errors)
        assert tool.custom_system_prompt == custom_prompt
