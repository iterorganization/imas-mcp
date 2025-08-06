"""Error response models for IMAS MCP tools."""

from typing import Any, Dict, List, Optional
from pydantic import Field

from imas_mcp.models.context_models import AIContext, QueryContext


class ToolError(AIContext, QueryContext):
    """Error response with suggestions, context, and fallback data."""

    error: str = Field(description="Error message")
    suggestions: List[str] = Field(
        default_factory=list, description="Suggested actions"
    )
    context: Dict[str, Any] = Field(default_factory=dict, description="Error context")
    fallback_data: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional fallback data when primary operation fails"
    )
