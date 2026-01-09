"""
LLM configuration for LlamaIndex agents.

Provides factory functions for creating LLMs configured for OpenRouter.
Model configuration is centralized in pyproject.toml under [tool.imas-codex.models].

Gemini 3 models have a "thinking mode" with dynamic thinking by default.
For agentic tasks, the model automatically adjusts reasoning depth.
"""

import os
from functools import lru_cache
from pathlib import Path

from llama_index.llms.openrouter import OpenRouter

# Default model fallback if config loading fails
DEFAULT_MODEL = "anthropic/claude-haiku-4.5"


def _load_model_config() -> dict[str, str]:
    """Load model configuration from pyproject.toml."""
    try:
        import tomllib

        pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
        if pyproject_path.exists():
            with open(pyproject_path, "rb") as f:
                config = tomllib.load(f)
            models = config.get("tool", {}).get("imas-codex", {}).get("models", {})
            return {
                "default": models.get("default", DEFAULT_MODEL),
                "discovery": models.get("discovery", DEFAULT_MODEL),
                "evaluation": models.get("evaluation", DEFAULT_MODEL),
                "enrichment": models.get("enrichment", DEFAULT_MODEL),
                "presets": models.get("presets", {}),
            }
    except Exception:
        pass
    return {
        "default": DEFAULT_MODEL,
        "discovery": DEFAULT_MODEL,
        "evaluation": DEFAULT_MODEL,
        "enrichment": DEFAULT_MODEL,
        "presets": {},
    }


# Load config at module import
_MODEL_CONFIG = _load_model_config()


def get_model_for_task(task: str) -> str:
    """Get the configured model for a specific task.

    Args:
        task: One of 'default', 'discovery', 'evaluation', 'enrichment'

    Returns:
        Full model identifier string
    """
    return _MODEL_CONFIG.get(task, _MODEL_CONFIG["default"])


# Model presets for convenience
MODELS = _MODEL_CONFIG.get("presets", {})
if not MODELS:
    MODELS = {
        "gemini-flash": "google/gemini-3-flash-preview",
        "gemini-pro": "google/gemini-3-pro-preview",
        "claude-haiku": "anthropic/claude-haiku-4.5",
        "claude-sonnet": "anthropic/claude-sonnet-4.5",
        "claude-opus": "anthropic/claude-opus-4.5",
    }


def get_model_id(preset: str) -> str:
    """Get full model ID from a preset name or return as-is."""
    return MODELS.get(preset, preset)


@lru_cache(maxsize=8)
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
        model: Model identifier or preset name (default: from config)
        temperature: Sampling temperature (0.0-1.0)
        context_window: Context window size for the model
        max_tokens: Maximum tokens in the response (default: 4096)

    Returns:
        Configured OpenRouter LLM instance

    Raises:
        ValueError: If OPENROUTER_API_KEY environment variable is not set
    """
    # Resolve preset names
    resolved_model = get_model_id(model)

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
        model=resolved_model,
        api_key=api_key,
        temperature=temperature,
        context_window=context_window,
        max_tokens=max_tokens,
    )
