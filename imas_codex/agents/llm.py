"""
LLM configuration for LlamaIndex agents.

Provides factory functions for creating LLMs configured for OpenRouter.
"""

import os
from functools import lru_cache

from llama_index.llms.openai_like import OpenAILike

# Default model - Gemini 3 Flash Preview via OpenRouter
DEFAULT_MODEL = "google/gemini-3-flash-preview"


@lru_cache(maxsize=4)
def get_llm(
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    context_window: int = 1_000_000,
) -> OpenAILike:
    """
    Get a configured LLM for use with LlamaIndex agents.

    Uses OpenRouter as the API endpoint, configured via OPENAI_API_KEY
    environment variable (OpenRouter uses OpenAI-compatible API).

    Args:
        model: Model identifier (default: google/gemini-2.5-flash-preview-05-20)
        temperature: Sampling temperature (0.0-1.0)
        context_window: Context window size for the model

    Returns:
        Configured OpenAILike LLM instance

    Raises:
        ValueError: If OPENAI_API_KEY environment variable is not set
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        # Try loading from .env file
        try:
            from dotenv import load_dotenv

            load_dotenv()
            api_key = os.environ.get("OPENAI_API_KEY")
        except ImportError:
            pass

    if not api_key:
        msg = (
            "OPENAI_API_KEY environment variable not set. "
            "Set it to your OpenRouter API key."
        )
        raise ValueError(msg)

    return OpenAILike(
        model=model,
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
        temperature=temperature,
        is_chat_model=True,
        context_window=context_window,
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
