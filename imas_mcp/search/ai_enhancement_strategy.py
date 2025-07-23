"""
Selective AI Enhancement Strategy for IMAS MCP Tools.

This module provides backward compatibility and integration with the simplified
AI enhancement system. The main logic has been moved to ai_enhancer.py for
better organization and maintainability.
"""

import logging
from typing import Any, Optional

# Import the new simplified enhancement system
from .ai_enhancer import (
    EnhancementDecisionEngine,
    TOOL_ENHANCEMENT_CONFIG,
    EnhancementStrategy,
)

logger = logging.getLogger(__name__)

# Backward compatibility mapping for existing code
AI_ENHANCEMENT_STRATEGY = {
    tool: config["strategy"].value for tool, config in TOOL_ENHANCEMENT_CONFIG.items()
}


def should_apply_ai_enhancement(
    func_name: str, args: tuple, kwargs: dict, ctx: Optional[Any] = None
) -> bool:
    """
    Determine if AI enhancement should be applied based on tool strategy and context.

    This function provides backward compatibility with the old interface while
    using the new simplified decision engine.

    Args:
        func_name: Name of the function being called
        args: Function arguments
        kwargs: Function keyword arguments
        ctx: MCP context for AI enhancement

    Returns:
        Boolean indicating whether AI enhancement should be applied
    """
    return EnhancementDecisionEngine.should_enhance(func_name, args, kwargs, ctx)


def _evaluate_conditional_enhancement(
    func_name: str, args: tuple, kwargs: dict
) -> bool:
    """
    Evaluate conditional AI enhancement based on specific context.

    This function provides backward compatibility while delegating to the
    new simplified decision engine.

    Args:
        func_name: Name of the function being called
        args: Function arguments
        kwargs: Function keyword arguments

    Returns:
        Boolean indicating whether AI enhancement should be applied
    """
    # Get tool configuration
    config = TOOL_ENHANCEMENT_CONFIG.get(
        func_name, {"strategy": EnhancementStrategy.ALWAYS, "category": None}
    )

    if config["strategy"] != EnhancementStrategy.CONDITIONAL:
        return config["strategy"] == EnhancementStrategy.ALWAYS

    # Use the new decision engine for conditional evaluation
    return EnhancementDecisionEngine._evaluate_conditional(
        config["category"], func_name, args, kwargs
    )
