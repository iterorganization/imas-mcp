"""
LLM configuration for LlamaIndex agents.

Provides factory functions for creating LLMs configured for OpenRouter.

Gemini 3 models have a "thinking mode" with dynamic thinking by default.
For agentic tasks, the model automatically adjusts reasoning depth.
"""

import os
from functools import lru_cache

from llama_index.llms.openrouter import OpenRouter

# Default model - Gemini 3 Flash Preview via OpenRouter
DEFAULT_MODEL = "google/gemini-3-flash-preview"


@lru_cache(maxsize=4)
def get_llm(
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    context_window: int = 1_000_000,
    max_tokens: int = 4096,
) -> OpenRouter:
    """
    Get a configured LLM for use with LlamaIndex agents.

    Uses OpenRouter as the API endpoint with the dedicated LlamaIndex
    OpenRouter integration.

    Args:
        model: Model identifier (default: google/gemini-3-flash-preview)
        temperature: Sampling temperature (0.0-1.0)
        context_window: Context window size for the model
        max_tokens: Maximum tokens in the response (default: 4096)

    Returns:
        Configured OpenRouter LLM instance

    Raises:
        ValueError: If OPENROUTER_API_KEY environment variable is not set
    """
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        # Try loading from .env file
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
        model=model,
        api_key=api_key,
        temperature=temperature,
        context_window=context_window,
        max_tokens=max_tokens,
    )


# Model presets for different use cases
MODELS = {
    "gemini-flash": "google/gemini-3-flash-preview",
    "gemini-2.5-flash": "google/gemini-2.5-flash",
    "gemini-pro": "google/gemini-2.5-pro-preview",
    "claude-sonnet": "anthropic/claude-sonnet-4",
    "gpt-4o": "openai/gpt-4o",
}


def get_model_id(preset: str) -> str:
    """Get full model ID from a preset name."""
    return MODELS.get(preset, preset)
