"""
Composite hints decorator that applies physics, query, and tool hints
based on a single hints mode.
"""

import functools
import logging
from collections.abc import Callable
from typing import Any, TypeVar

from ...models.constants import HintsMode, ResponseProfile
from ...models.result_models import SearchPathsResult
from .physics_hints import apply_physics_hints
from .query_hints import apply_query_hints
from .tool_hints import apply_tool_hints

F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


def _resolve_hints_mode(kwargs: dict[str, Any]) -> HintsMode:
    # Prefer explicit hints (if any). Otherwise map response_profile to hints mode.
    mode = kwargs.get("hints")
    if mode is None:
        profile = kwargs.get("response_profile")
        if isinstance(profile, str):
            profile_lower = profile.lower()
            if profile_lower == ResponseProfile.DETAILED.value:
                return HintsMode.ALL
            if profile_lower == ResponseProfile.STANDARD.value:
                return HintsMode.MINIMAL
            if profile_lower == ResponseProfile.MINIMAL.value:
                return HintsMode.NONE
        if isinstance(profile, ResponseProfile):
            if profile == ResponseProfile.DETAILED:
                return HintsMode.ALL
            if profile == ResponseProfile.STANDARD:
                return HintsMode.MINIMAL
            if profile == ResponseProfile.MINIMAL:
                return HintsMode.NONE
        # Default when profile not provided: MINIMAL for balanced response
        return HintsMode.MINIMAL
    if isinstance(mode, str):
        try:
            return HintsMode(mode)
        except ValueError:
            return HintsMode.MINIMAL
    if isinstance(mode, HintsMode):
        return mode
    return HintsMode.MINIMAL


def hints(tool_max: int = 4, query_max: int = 5) -> Callable[[F], F]:
    """Composite decorator to add all hint types according to hints mode.

    Physics hints are applied only when hints mode is ALL. In MINIMAL and NONE
    modes physics hints are skipped to keep latency and cost low.

    Args:
        tool_max: maximum number of tool hints
        query_max: maximum number of query hints
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            tool_instance = args[0] if args else None
            result = await func(*args, **kwargs)

            mode = _resolve_hints_mode(kwargs)
            if mode == HintsMode.NONE:
                return result

            # Apply physics hints first to enrich context for other hints
            physics_on = mode == HintsMode.ALL
            if physics_on and isinstance(result, SearchPathsResult):
                try:
                    if tool_instance and hasattr(tool_instance, "physics"):
                        result = await apply_physics_hints(
                            result, tool_instance.physics
                        )
                except Exception as e:
                    logger.debug(f"Physics hints skipped due to error: {e}")

            # Apply query/tool hints with caps (minimal mode => 1 each)
            q_cap = 1 if mode == HintsMode.MINIMAL else query_max
            t_cap = 1 if mode == HintsMode.MINIMAL else tool_max

            try:
                result = apply_query_hints(result, max_hints=q_cap)
            except Exception as e:
                logger.debug(f"Query hints skipped due to error: {e}")

            try:
                result = apply_tool_hints(result, max_hints=t_cap)
            except Exception as e:
                logger.debug(f"Tool hints skipped due to error: {e}")

            return result

        return wrapper  # type: ignore

    return decorator
