"""Tests for error_handling.py - error handling decorator functionality."""

import pytest

from imas_codex.models.error_models import ToolError
from imas_codex.search.decorators.error_handling import (
    SearchError,
    ServiceError,
    ToolException,
    ValidationError,
    create_error_response,
    create_timeout_handler,
    generate_error_recovery_suggestions,
    get_fallback_response,
    handle_errors,
)


class TestToolExceptions:
    """Tests for tool exception classes."""

    def test_tool_exception_creation(self):
        """Test creating a ToolException."""
        exc = ToolException("Test error", query="test query", tool_name="test_tool")
        assert str(exc) == "Test error"
        assert exc.query == "test query"
        assert exc.tool_name == "test_tool"

    def test_validation_error_inheritance(self):
        """Test ValidationError inherits from ToolException."""
        exc = ValidationError("Invalid input")
        assert isinstance(exc, ToolException)
        assert isinstance(exc, Exception)

    def test_search_error_inheritance(self):
        """Test SearchError inherits from ToolException."""
        exc = SearchError("Search failed")
        assert isinstance(exc, ToolException)

    def test_service_error_inheritance(self):
        """Test ServiceError inherits from ToolException."""
        exc = ServiceError("Service unavailable")
        assert isinstance(exc, ToolException)


class TestCreateErrorResponse:
    """Tests for create_error_response function."""

    def test_create_from_string(self):
        """Test creating error response from string."""
        response = create_error_response("Something went wrong")
        assert isinstance(response, ToolError)
        assert response.error == "Something went wrong"

    def test_create_from_exception(self):
        """Test creating error response from exception."""
        exc = ValueError("Invalid value")
        response = create_error_response(exc)
        assert isinstance(response, ToolError)
        assert "Invalid value" in response.error
        assert response.context["error_type"] == "ValueError"

    def test_includes_query_in_context(self):
        """Test that query is included in context."""
        response = create_error_response("Error", query="test query")
        assert response.context["query"] == "test query"

    def test_includes_tool_name_in_context(self):
        """Test that tool name is included in context."""
        response = create_error_response("Error", tool_name="my_tool")
        assert response.context["tool_name"] == "my_tool"

    def test_includes_suggestions_by_default(self):
        """Test that suggestions are included by default."""
        response = create_error_response("Error")
        assert isinstance(response.suggestions, list)

    def test_can_disable_suggestions(self):
        """Test that suggestions can be disabled."""
        response = create_error_response("Error", include_suggestions=False)
        assert response.suggestions == []

    def test_includes_fallback_data(self):
        """Test that fallback data is included."""
        fallback = {"alternative": "data"}
        response = create_error_response("Error", fallback_data=fallback)
        assert response.fallback_data == fallback


class TestGenerateErrorRecoverySuggestions:
    """Tests for generate_error_recovery_suggestions function."""

    def test_validation_error_suggestions(self):
        """Test suggestions for validation errors."""
        suggestions = generate_error_recovery_suggestions(
            "Invalid input validation failed", "query", "tool"
        )
        assert len(suggestions) > 0
        assert any("parameter" in s.lower() for s in suggestions)

    def test_search_mode_error_suggestions(self):
        """Test suggestions for search mode errors."""
        suggestions = generate_error_recovery_suggestions(
            "Invalid search_mode parameter", "query", "tool"
        )
        assert any("search mode" in s.lower() for s in suggestions)

    def test_not_found_error_suggestions(self):
        """Test suggestions for not found errors."""
        suggestions = generate_error_recovery_suggestions(
            "No results found", "temperature", "search_imas_paths"
        )
        assert len(suggestions) > 0
        assert any(
            "broader" in s.lower() or "spelling" in s.lower() for s in suggestions
        )

    def test_timeout_error_suggestions(self):
        """Test suggestions for timeout errors."""
        suggestions = generate_error_recovery_suggestions(
            "Operation timeout exceeded", "complex query", "tool"
        )
        assert any("reduce" in s.lower() or "simpler" in s.lower() for s in suggestions)

    def test_service_error_suggestions(self):
        """Test suggestions for service errors."""
        suggestions = generate_error_recovery_suggestions(
            "Service connection failed", "query", "tool"
        )
        assert any("retry" in s.lower() or "service" in s.lower() for s in suggestions)

    def test_generic_error_suggestions(self):
        """Test suggestions for unknown errors."""
        suggestions = generate_error_recovery_suggestions(
            "Unknown error occurred", "query", "tool"
        )
        assert len(suggestions) > 0  # Should always have some suggestions


class TestGetFallbackResponse:
    """Tests for get_fallback_response function."""

    def test_search_suggestions_fallback(self):
        """Test search suggestions fallback response."""
        response = get_fallback_response(
            "search_suggestions", "temperature", "search_tool"
        )
        assert "message" in response
        assert "suggestions" in response
        assert len(response["suggestions"]) > 0

    def test_concept_suggestions_fallback(self):
        """Test concept suggestions fallback response."""
        response = get_fallback_response(
            "concept_suggestions", "plasma", "concept_tool"
        )
        assert "message" in response
        assert "suggestions" in response

    def test_unknown_fallback_type(self):
        """Test fallback response for unknown type."""
        response = get_fallback_response("unknown_type", "query", "tool")
        assert "message" in response
        assert "fallback_type" in response


class TestHandleErrorsDecorator:
    """Tests for handle_errors decorator."""

    @pytest.mark.asyncio
    async def test_successful_execution_returns_result(self):
        """Test that successful execution returns result."""

        @handle_errors()
        async def test_func():
            return {"result": "success"}

        result = await test_func()
        assert result == {"result": "success"}

    @pytest.mark.asyncio
    async def test_handles_validation_error(self):
        """Test handling ValidationError."""

        @handle_errors()
        async def test_func():
            raise ValidationError("Invalid input")

        result = await test_func()
        assert isinstance(result, ToolError)
        assert "Invalid input" in result.error

    @pytest.mark.asyncio
    async def test_handles_search_error(self):
        """Test handling SearchError."""

        @handle_errors()
        async def test_func():
            raise SearchError("Search failed")

        result = await test_func()
        assert isinstance(result, ToolError)
        assert "Search failed" in result.error

    @pytest.mark.asyncio
    async def test_handles_unexpected_error(self):
        """Test handling unexpected exceptions."""

        @handle_errors()
        async def test_func():
            raise RuntimeError("Unexpected error")

        result = await test_func()
        assert isinstance(result, ToolError)
        assert "Unexpected error" in result.error

    @pytest.mark.asyncio
    async def test_includes_fallback_data(self):
        """Test that fallback data is included when specified."""

        @handle_errors(fallback="search_suggestions")
        async def test_func():
            raise SearchError("Search failed")

        result = await test_func()
        assert result.fallback_data is not None
        assert "suggestions" in result.fallback_data


class TestCreateTimeoutHandler:
    """Tests for create_timeout_handler function."""

    @pytest.mark.asyncio
    async def test_fast_function_completes(self):
        """Test that fast function completes successfully."""

        @create_timeout_handler(timeout_seconds=5.0)
        async def fast_func():
            return "success"

        result = await fast_func()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_slow_function_times_out(self):
        """Test that slow function raises ServiceError."""
        import anyio

        @create_timeout_handler(timeout_seconds=0.1)
        async def slow_func():
            await anyio.sleep(1.0)
            return "too late"

        with pytest.raises(ServiceError) as exc_info:
            await slow_func()

        assert "timed out" in str(exc_info.value).lower()
