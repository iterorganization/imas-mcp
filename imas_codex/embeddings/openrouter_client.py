"""OpenRouter API client for chat completions.

This module provides an OpenRouterClient class for chat completion
support for LLM tasks like cluster labeling.
"""

import logging
import os
import time

import requests

logger = logging.getLogger(__name__)


class OpenRouterError(Exception):
    """Base exception for OpenRouter API errors."""

    pass


class OpenRouterClient:
    """OpenRouter API client for chat completions."""

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """Initialize OpenRouter client.

        Args:
            model_name: Name of the LLM model to use
            api_key: OpenRouter API key (if None, will use OPENAI_API_KEY env var)
            base_url: Base URL for API (if None, will use OPENAI_BASE_URL env var)
            max_retries: Maximum number of retry attempts for failed requests
            retry_delay: Delay between retry attempts in seconds
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv(
            "OPENAI_BASE_URL", "https://openrouter.ai/api/v1"
        )
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        if not self.api_key:
            raise OpenRouterError(
                "OpenRouter API key required. Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

        if not self.base_url:
            raise OpenRouterError(
                "OpenRouter base URL required. Set OPENAI_BASE_URL environment variable or pass base_url parameter."
            )

        # Validate API key is not a placeholder
        if self.api_key.startswith("your_") and self.api_key.endswith("_here"):
            raise OpenRouterError("Invalid API key - appears to be placeholder text.")

        self.chat_url = f"{self.base_url.rstrip('/')}/chat/completions"

        logger.info(f"OpenRouter client initialized for model: {self.model_name}")

    def _get_headers(self) -> dict:
        """Get request headers for API calls."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "imas-codex",  # Optional: identify your application
            "X-Title": "IMAS Codex Server",
        }

    def make_chat_request(
        self,
        messages: list[dict],
        model: str | None = None,
        max_tokens: int = 100000,
        temperature: float = 0.3,
        timeout: int = 300,
    ) -> str:
        """Make a chat completion request to the API with retry logic.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model to use (defaults to self.model_name)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            timeout: Request timeout in seconds

        Returns:
            The assistant's response content string

        Raises:
            OpenRouterError: If the request fails after retries
        """
        data = {
            "model": model or self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    self.chat_url,
                    headers=self._get_headers(),
                    json=data,
                    timeout=(30, timeout),  # (connect_timeout, read_timeout)
                )

                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"]

                elif response.status_code == 429:  # Rate limited
                    if attempt < self.max_retries:
                        wait_time = self.retry_delay * (2**attempt)
                        logger.warning(
                            f"Rate limited, retrying in {wait_time:.1f} seconds..."
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        raise OpenRouterError(
                            f"Rate limit exceeded after {self.max_retries} retries"
                        )

                else:
                    error_msg = (
                        f"Chat request failed: {response.status_code} - {response.text}"
                    )
                    if attempt < self.max_retries:
                        logger.warning(f"{error_msg}, retrying...")
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        raise OpenRouterError(error_msg)

            except requests.RequestException as e:
                last_error = e
                if attempt < self.max_retries:
                    logger.warning(f"Request failed: {e}, retrying...")
                    time.sleep(self.retry_delay)
                else:
                    break

        raise OpenRouterError(
            f"Failed to make chat request after {self.max_retries} retries: {last_error}"
        )
