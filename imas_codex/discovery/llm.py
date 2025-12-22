"""
LLM client for the Discovery Engine agentic loop.

Uses OpenRouter API to communicate with Claude Opus 4.5 for
generating exploration scripts and analyzing results.

Supports both blocking and streaming modes for real-time
visibility into the agent's train of thought.
"""

import json
import os
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx
from dotenv import load_dotenv

# Type alias for stream callback
# Called with accumulated content as tokens arrive
StreamCallback = Callable[[str], Awaitable[None]]

# Load environment variables from .env file
load_dotenv(override=True)


@dataclass
class AgentResponse:
    """Parsed response from the LLM agent."""

    done: bool
    script: str | None = None
    findings: dict[str, Any] | None = None
    reasoning: str = ""
    raw: str = ""


@dataclass
class Message:
    """A chat message."""

    role: str  # "system", "user", "assistant"
    content: str


class AgentLLM:
    """
    LLM client for agentic exploration loops.

    Uses OpenRouter API to communicate with Claude or other models.
    Handles message formatting, API calls, and response parsing.
    """

    OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

    # Patterns for extracting script and completion blocks
    BASH_PATTERN = re.compile(r"```(?:bash|sh)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
    JSON_PATTERN = re.compile(r"```json\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
    DONE_PATTERN = re.compile(r'"done"\s*:\s*true', re.IGNORECASE)

    def __init__(
        self,
        model: str = "anthropic/claude-opus-4.5",
        api_key: str | None = None,
        timeout: float = 120.0,
        max_tokens: int = 4096,
    ):
        """
        Initialize the LLM client.

        Args:
            model: Model identifier for OpenRouter
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            timeout: Request timeout in seconds
            max_tokens: Maximum tokens in response
        """
        self.model = model
        self.api_key = api_key or os.environ.get(
            "OPENROUTER_API_KEY", os.environ.get("OPENAI_API_KEY")
        )
        self.timeout = timeout
        self.max_tokens = max_tokens

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY or OPENAI_API_KEY "
                "environment variable (or in .env file), or pass api_key parameter."
            )

    async def chat(
        self,
        messages: list[Message],
        max_tokens: int | None = None,
        on_stream: StreamCallback | None = None,
    ) -> AgentResponse:
        """
        Send messages to the LLM and get a parsed response.

        Args:
            messages: List of chat messages
            max_tokens: Override default max_tokens
            on_stream: Optional callback for streaming tokens.
                Called periodically with accumulated content as it arrives.
                Enables real-time visibility into the agent's thinking.

        Returns:
            AgentResponse with parsed script or findings
        """
        # Use streaming if callback provided
        if on_stream:
            return await self._chat_streaming(messages, max_tokens, on_stream)

        # Non-streaming path
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self.OPENROUTER_API_URL,
                headers=self._headers(),
                json=self._request_body(messages, max_tokens, stream=False),
            )

            response.raise_for_status()
            data = response.json()

        # Extract content from response
        content = data["choices"][0]["message"]["content"]
        return self.parse_response(content)

    async def _chat_streaming(
        self,
        messages: list[Message],
        max_tokens: int | None,
        on_stream: StreamCallback,
    ) -> AgentResponse:
        """
        Streaming version of chat that calls on_stream as tokens arrive.
        """
        content = ""
        last_emit_len = 0

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                self.OPENROUTER_API_URL,
                headers=self._headers(),
                json=self._request_body(messages, max_tokens, stream=True),
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue

                    data_str = line[6:]  # Remove "data: " prefix
                    if data_str == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        chunk = delta.get("content", "")
                        if chunk:
                            content += chunk
                            # Emit every ~100 chars or on newlines for responsiveness
                            if len(content) - last_emit_len > 100 or "\n" in chunk:
                                await on_stream(content)
                                last_emit_len = len(content)
                    except json.JSONDecodeError:
                        continue

        # Final emit with complete content
        if len(content) > last_emit_len:
            await on_stream(content)

        return self.parse_response(content)

    def _headers(self) -> dict[str, str]:
        """Return standard headers for OpenRouter API."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/iterorganization/imas-codex",
            "X-Title": "IMAS Codex Discovery Engine",
        }

    def _request_body(
        self,
        messages: list[Message],
        max_tokens: int | None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Build request body for OpenRouter API."""
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": 0.1,  # Low temperature for deterministic scripts
        }
        if stream:
            body["stream"] = True
        return body

    def chat_sync(
        self,
        messages: list[Message],
        max_tokens: int | None = None,
    ) -> AgentResponse:
        """
        Synchronous version of chat().

        Args:
            messages: List of chat messages
            max_tokens: Override default max_tokens

        Returns:
            AgentResponse with parsed script or findings
        """
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                self.OPENROUTER_API_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/iterorganization/imas-codex",
                    "X-Title": "IMAS Codex Discovery Engine",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": m.role, "content": m.content} for m in messages
                    ],
                    "max_tokens": max_tokens or self.max_tokens,
                    "temperature": 0.1,
                },
            )

            response.raise_for_status()
            data = response.json()

        content = data["choices"][0]["message"]["content"]
        return self.parse_response(content)

    def parse_response(self, content: str) -> AgentResponse:
        """
        Parse LLM response to extract script or completion signal.

        Args:
            content: Raw response content from LLM

        Returns:
            AgentResponse with extracted data
        """
        # Check for completion signal first
        json_matches = self.JSON_PATTERN.findall(content)
        for json_str in json_matches:
            try:
                data = json.loads(json_str)
                if data.get("done"):
                    return AgentResponse(
                        done=True,
                        findings=data.get("findings", {}),
                        reasoning=self._extract_reasoning(content),
                        raw=content,
                    )
            except json.JSONDecodeError:
                continue

        # Check for inline done signal (not in code block)
        if self.DONE_PATTERN.search(content):
            # Try to extract JSON from the content
            try:
                # Find JSON-like structure
                start = content.find("{")
                end = content.rfind("}") + 1
                if start >= 0 and end > start:
                    data = json.loads(content[start:end])
                    if data.get("done"):
                        return AgentResponse(
                            done=True,
                            findings=data.get("findings", {}),
                            reasoning=self._extract_reasoning(content),
                            raw=content,
                        )
            except json.JSONDecodeError:
                pass

        # Look for bash script
        bash_matches = self.BASH_PATTERN.findall(content)
        if bash_matches:
            # Use the last bash block (most refined)
            script = bash_matches[-1].strip()
            return AgentResponse(
                done=False,
                script=script,
                reasoning=self._extract_reasoning(content),
                raw=content,
            )

        # No script or completion found - treat as reasoning only
        return AgentResponse(
            done=False,
            script=None,
            reasoning=content,
            raw=content,
        )

    def _extract_reasoning(self, content: str) -> str:
        """Extract reasoning text (content before code blocks)."""
        # Remove code blocks
        text = self.BASH_PATTERN.sub("", content)
        text = self.JSON_PATTERN.sub("", text)
        return text.strip()


def system_message(content: str) -> Message:
    """Create a system message."""
    return Message(role="system", content=content)


def user_message(content: str) -> Message:
    """Create a user message."""
    return Message(role="user", content=content)


def assistant_message(content: str) -> Message:
    """Create an assistant message."""
    return Message(role="assistant", content=content)
