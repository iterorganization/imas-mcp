"""
LLM-based directory scoring with grounded evidence.

This module implements the scoring phase of graph-led discovery:
1. Query graph for scanned but unscored paths
2. Build batched prompts with directory context
3. Call LLM with structured output schema for reliable parsing
4. Apply grounded scoring function to compute final scores
5. Set expand_to for high-value paths

The scorer uses LiteLLM for model access (via OpenRouter) with Pydantic
models for structured output, then a deterministic function computes
the final score from evidence. This "grounded scoring" approach ensures
reproducibility.

Retry logic handles rate limiting (OpenRouter "Overloaded" errors).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

from imas_codex.agentic.agents import get_model_for_task
from imas_codex.agentic.prompt_loader import load_prompts
from imas_codex.discovery.models import (
    DirectoryEvidence,
    DirectoryScoringBatch,
    PathPurpose,
    ScoredBatch,
    ScoredDirectory,
    parse_path_purpose,
)

logger = logging.getLogger(__name__)

# Retry configuration for rate limiting
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # seconds, doubles each retry

# Dimension weights for grounded scoring
SCORE_WEIGHTS = {
    "code": 1.0,
    "data": 0.8,
    "imas": 1.2,
}

# Purposes that should be suppressed (lower scores)
SUPPRESSED_PURPOSES = {
    PathPurpose.system,
    PathPurpose.build_artifacts,
}


def grounded_score(
    score_code: float,
    score_data: float,
    score_imas: float,
    evidence: DirectoryEvidence,
    purpose: PathPurpose,
) -> float:
    """Compute combined score from dimension scores with evidence adjustments.

    Grounded scoring:
    1. Start with weighted average of dimension scores
    2. Boost for quality indicators (readme, makefile, git)
    3. Boost for IMAS indicators
    4. Suppress for low-quality purposes (system, build_artifacts)

    Args:
        score_code: Code interest dimension (0.0-1.0)
        score_data: Data interest dimension (0.0-1.0)
        score_imas: IMAS relevance dimension (0.0-1.0)
        evidence: Collected evidence from LLM
        purpose: Classified purpose

    Returns:
        Combined score (0.0-1.0)
    """
    # Start with weighted average
    total_weight = sum(SCORE_WEIGHTS.values())
    base_score = (
        score_code * SCORE_WEIGHTS["code"]
        + score_data * SCORE_WEIGHTS["data"]
        + score_imas * SCORE_WEIGHTS["imas"]
    ) / total_weight

    # Quality boost
    quality_boost = 0.0
    quality_lower = [q.lower() for q in evidence.quality_indicators]
    if any("readme" in q for q in quality_lower):
        quality_boost += 0.05
    if any("makefile" in q for q in quality_lower):
        quality_boost += 0.05
    if any("git" in q for q in quality_lower):
        quality_boost += 0.05

    # IMAS boost
    if len(evidence.imas_indicators) > 0:
        quality_boost += 0.10

    # Code diversity boost
    if len(evidence.code_indicators) > 3:
        quality_boost += 0.05

    # Purpose suppression
    purpose_multiplier = 1.0
    if purpose in SUPPRESSED_PURPOSES:
        purpose_multiplier = 0.3
    elif purpose == PathPurpose.test_files:
        purpose_multiplier = 0.6

    combined = (base_score + quality_boost) * purpose_multiplier

    # Clamp to [0, 1]
    return max(0.0, min(1.0, combined))


@dataclass
class DirectoryScorer:
    """Score directories using LLM with grounded evidence.

    Implements:
    1. Batch prompt construction from DirStats
    2. LLM evidence collection via LiteLLM/OpenRouter
    3. Deterministic grounded scoring from evidence
    4. Frontier expansion logic

    Args:
        model: Model name (None = use "score" task model from config)

    Example:
        scorer = DirectoryScorer()
        batch = scorer.score_batch(
            directories=[...],
            focus="equilibrium codes",
            threshold=0.7,
        )
    """

    model: str | None = None

    def __post_init__(self):
        """Initialize model from config if not provided."""
        if self.model is None:
            self.model = get_model_for_task("score")

    def score_batch(
        self,
        directories: list[dict[str, Any]],
        focus: str | None = None,
        threshold: float = 0.7,
    ) -> ScoredBatch:
        """Score a batch of directories using LLM with structured output.

        Args:
            directories: List of directory info dicts with:
              - path: str
              - total_files: int
              - total_dirs: int
              - has_readme: bool
              - has_makefile: bool
              - has_git: bool
              - file_type_counts: dict (optional)
              - patterns_detected: list (optional)
            focus: Natural language focus query (e.g., "equilibrium codes")
            threshold: Min score to expand (0.0-1.0)

        Returns:
            ScoredBatch with results and cost
        """
        import litellm

        if not directories:
            return ScoredBatch(
                scored_dirs=[],
                total_cost=0.0,
                model=self.model,
                tokens_used=0,
            )

        # Load prompt template
        system_prompt = self._build_system_prompt(focus)
        user_prompt = self._build_user_prompt(directories)

        # Get API key for OpenRouter
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable not set. "
                "Set it in .env or export it."
            )

        # Call LLM via LiteLLM (OpenRouter prefix) with structured output
        model_id = self.model
        if not model_id.startswith("openrouter/"):
            model_id = f"openrouter/{model_id}"

        # Retry loop for rate limiting / overloaded errors
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = litellm.completion(
                    model=model_id,
                    api_key=api_key,
                    max_tokens=4000,
                    response_format=DirectoryScoringBatch,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                break  # Success, exit retry loop
            except Exception as e:
                last_error = e
                error_msg = str(e).lower()
                # Retry on rate limiting, overloaded, or transient errors
                if any(
                    x in error_msg
                    for x in ["overloaded", "rate", "429", "503", "timeout"]
                ):
                    delay = RETRY_BASE_DELAY * (2**attempt)
                    logger.warning(
                        f"LLM request failed (attempt {attempt + 1}/{MAX_RETRIES}), "
                        f"retrying in {delay:.1f}s: {e}"
                    )
                    time.sleep(delay)
                else:
                    # Non-retryable error
                    raise
        else:
            # All retries exhausted
            raise last_error  # type: ignore[misc]

        # Parse response using structured output
        scored_dirs = self._parse_structured_response(response, directories, threshold)

        # Calculate cost from LiteLLM response
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        total_tokens = input_tokens + output_tokens

        # Cost calculation - LiteLLM may provide this, fallback to Claude Sonnet rates
        if (
            hasattr(response, "_hidden_params")
            and "response_cost" in response._hidden_params
        ):
            cost = response._hidden_params["response_cost"]
        else:
            # Fallback: Claude Sonnet 4.5 via OpenRouter: $3/$15 per 1M tokens
            cost = (input_tokens * 3 + output_tokens * 15) / 1_000_000

        return ScoredBatch(
            scored_dirs=scored_dirs,
            total_cost=cost,
            model=self.model,
            tokens_used=total_tokens,
        )

    def _build_system_prompt(self, focus: str | None = None) -> str:
        """Build system prompt for directory scoring.

        Loads prompt from prompts/discovery/scorer.md file.
        Raises error if prompt not found - no fallback.
        """
        import re

        prompts = load_prompts()
        prompt_def = prompts.get("discovery/scorer")
        if prompt_def is None:
            raise ValueError(
                "Required prompt 'discovery/scorer' not found. "
                "Ensure prompts/discovery/scorer.md exists."
            )

        prompt = prompt_def.content
        if focus:
            prompt = prompt.replace("{{ focus }}", focus)
            prompt = prompt.replace("{% if focus %}", "")
            prompt = prompt.replace("{% endif %}", "")
        else:
            # Remove focus section
            prompt = re.sub(
                r"{%\s*if focus\s*%}.*?{%\s*endif\s*%}",
                "",
                prompt,
                flags=re.DOTALL,
            )
        return prompt

    def _build_user_prompt(self, directories: list[dict[str, Any]]) -> str:
        """Build user prompt with directories to score.

        Includes child file/directory names for context - these are
        critical for the LLM to infer purpose from naming conventions.
        """
        import json as json_module

        lines = ["Score these directories:\n"]

        for i, d in enumerate(directories, 1):
            lines.append(f"\n## Directory {i}: {d['path']}")

            # Add DirStats
            lines.append(f"Total files: {d.get('total_files', 0)}")
            lines.append(f"Total dirs: {d.get('total_dirs', 0)}")

            file_types = d.get("file_type_counts")
            if file_types:
                if isinstance(file_types, str):
                    try:
                        file_types = json_module.loads(file_types)
                    except json_module.JSONDecodeError:
                        file_types = {}
                lines.append(f"File types: {file_types}")

            if d.get("has_readme"):
                lines.append("Has README")
            if d.get("has_makefile"):
                lines.append("Has Makefile")
            if d.get("has_git"):
                lines.append("Has .git")

            patterns = d.get("patterns_detected", [])
            if patterns:
                lines.append(f"Patterns: {', '.join(patterns)}")

            # Add child names for context (critical for LLM to infer purpose)
            child_names = d.get("child_names")
            if child_names:
                # Parse JSON if stored as string
                if isinstance(child_names, str):
                    try:
                        child_names = json_module.loads(child_names)
                    except json_module.JSONDecodeError:
                        child_names = []
                if child_names:
                    # Limit to first 30 names to avoid excessive tokens
                    names_to_show = child_names[:30]
                    lines.append(f"Contents: {', '.join(names_to_show)}")

        lines.append(
            "\n\nReturn results for each directory in order. "
            "The response format is enforced by the schema."
        )

        return "\n".join(lines)

    def _parse_structured_response(
        self,
        response,
        directories: list[dict[str, Any]],
        threshold: float,
    ) -> list[ScoredDirectory]:
        """Parse structured LLM response into ScoredDirectory objects.

        Uses LiteLLM's structured output - the response is already validated
        against the DirectoryScoringBatch Pydantic model.
        """
        try:
            # LiteLLM returns content as JSON string when using response_format
            content = response.choices[0].message.content
            batch = DirectoryScoringBatch.model_validate_json(content)
            results = batch.results
        except Exception as e:
            logger.warning(f"Failed to parse structured LLM response: {e}")
            # Fallback: return empty scores for all
            return [
                ScoredDirectory(
                    path=d["path"],
                    path_purpose=PathPurpose.unknown,
                    description="Parse error",
                    evidence=DirectoryEvidence(),
                    score_code=0.0,
                    score_data=0.0,
                    score_imas=0.0,
                    score=0.0,
                    should_expand=False,
                    skip_reason=f"LLM response parse failed: {e}",
                )
                for d in directories
            ]

        scored = []
        for i, result in enumerate(results[: len(directories)]):
            path = directories[i]["path"]

            # Clamp scores (should already be valid from schema)
            score_code = max(0.0, min(1.0, result.score_code))
            score_data = max(0.0, min(1.0, result.score_data))
            score_imas = max(0.0, min(1.0, result.score_imas))

            # Convert Pydantic enum to graph PathPurpose
            purpose = parse_path_purpose(result.path_purpose.value)

            # Build evidence from Pydantic model
            evidence = DirectoryEvidence(
                code_indicators=result.evidence.code_indicators,
                data_indicators=result.evidence.data_indicators,
                imas_indicators=result.evidence.imas_indicators,
                physics_indicators=result.evidence.physics_indicators,
                quality_indicators=result.evidence.quality_indicators,
            )

            # Compute grounded score
            combined = grounded_score(
                score_code,
                score_data,
                score_imas,
                evidence,
                purpose,
            )

            # Expansion decision
            should_expand = (
                combined >= threshold
                and result.should_expand
                and purpose not in SUPPRESSED_PURPOSES
            )

            scored_dir = ScoredDirectory(
                path=path,
                path_purpose=purpose,
                description=result.description,
                evidence=evidence,
                score_code=score_code,
                score_data=score_data,
                score_imas=score_imas,
                score=combined,
                should_expand=should_expand,
                keywords=result.keywords[:5] if result.keywords else [],
                physics_domain=result.physics_domain,
                expansion_reason=result.expansion_reason,
                skip_reason=result.skip_reason,
            )

            scored.append(scored_dir)

        return scored

    def _parse_response(
        self,
        response_text: str,
        directories: list[dict[str, Any]],
        threshold: float,
    ) -> list[ScoredDirectory]:
        """Parse unstructured LLM response (legacy fallback)."""
        import json

        try:
            # Extract JSON from response
            json_start = response_text.find("[")
            json_end = response_text.rfind("]") + 1
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON array found in response")

            json_str = response_text[json_start:json_end]
            results = json.loads(json_str)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            # Fallback: return empty scores for all
            return [
                ScoredDirectory(
                    path=d["path"],
                    path_purpose=PathPurpose.unknown,
                    description="Parse error",
                    evidence=DirectoryEvidence(),
                    score_code=0.0,
                    score_data=0.0,
                    score_imas=0.0,
                    score=0.0,
                    should_expand=False,
                    skip_reason=f"LLM response parse failed: {e}",
                )
                for d in directories
            ]

        scored = []
        for i, result in enumerate(results[: len(directories)]):
            path = directories[i]["path"]

            # Extract and clamp scores
            score_code = max(0.0, min(1.0, float(result.get("score_code", 0.0))))
            score_data = max(0.0, min(1.0, float(result.get("score_data", 0.0))))
            score_imas = max(0.0, min(1.0, float(result.get("score_imas", 0.0))))

            # Parse purpose
            purpose = parse_path_purpose(result.get("path_purpose", "unknown"))

            # Build evidence
            evidence_data = result.get("evidence", {})
            evidence = DirectoryEvidence(
                code_indicators=evidence_data.get("code_indicators", []),
                data_indicators=evidence_data.get("data_indicators", []),
                imas_indicators=evidence_data.get("imas_indicators", []),
                physics_indicators=evidence_data.get("physics_indicators", []),
                quality_indicators=evidence_data.get("quality_indicators", []),
            )

            # Compute grounded score
            combined = grounded_score(
                score_code,
                score_data,
                score_imas,
                evidence,
                purpose,
            )

            # Expansion decision
            should_expand = (
                combined >= threshold
                and result.get("should_expand", False)
                and purpose not in SUPPRESSED_PURPOSES
            )

            scored_dir = ScoredDirectory(
                path=path,
                path_purpose=purpose,
                description=result.get("description", ""),
                evidence=evidence,
                score_code=score_code,
                score_data=score_data,
                score_imas=score_imas,
                score=combined,
                should_expand=should_expand,
                keywords=result.get("keywords", [])[:5],  # Cap at 5
                physics_domain=result.get("physics_domain"),
                expansion_reason=result.get("expansion_reason"),
                skip_reason=result.get("skip_reason"),
            )

            scored.append(scored_dir)

        return scored
