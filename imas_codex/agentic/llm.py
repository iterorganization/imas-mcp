"""LLM configuration for LlamaIndex agents.

For smolagents (recommended), use create_litellm_model() from agents.py.
For LlamaIndex agents (legacy wiki code), use get_llm() which returns OpenRouter.
"""

import os
from functools import lru_cache

from imas_codex.agentic.agents import PRESETS, get_model_id
from imas_codex.settings import get_agent_model

__all__ = [
    "PRESETS",
    "get_model_id",
    "get_llm",
]


@lru_cache(maxsize=8)
def get_llm(
    model: str | None = None,
    temperature: float = 0.3,
    context_window: int = 1_000_000,
    max_tokens: int = 4096,
):
    """
    Get a configured LLM for use with LlamaIndex agents.

    DEPRECATED: Use create_litellm_model() from agents.py for smolagents.
    This function is retained for legacy wiki code that uses LlamaIndex.

    Args:
        model: Model identifier or preset name (default: from config)
        temperature: Sampling temperature (0.0-1.0)
        context_window: Context window size for the model
        max_tokens: Maximum tokens in the response (default: 4096)

    Returns:
        Configured OpenRouter LLM instance

    Raises:
        ValueError: If OPENROUTER_API_KEY environment variable is not set
    """
    from llama_index.llms.openrouter import OpenRouter

    # Resolve preset names
    resolved_model = get_model_id(model or get_agent_model())

    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        try:
            from dotenv import load_dotenv

            load_dotenv()
            api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get(
                "OPENAI_API_KEY"
            )
        except ImportError:
            pass

    if not api_key:
        msg = (
            "OPENROUTER_API_KEY environment variable not set. "
            "Set it to your OpenRouter API key."
        )
        raise ValueError(msg)

    return OpenRouter(
        model=resolved_model,
        api_key=api_key,
        temperature=temperature,
        context_window=context_window,
        max_tokens=max_tokens,
    )
