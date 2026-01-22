"""
LLM-based directory scoring with grounded evidence.

This module implements the scoring phase of graph-led discovery:
1. Query graph for scanned but unscored paths
2. Build batched prompts with directory context
3. Call LLM to collect evidence and classify directories
4. Apply grounded scoring function to compute final scores
5. Set expand_to for high-value paths

The scorer uses LiteLLM for model access (via OpenRouter), then a
deterministic function computes the final score from evidence.
This "grounded scoring" approach ensures reproducibility.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

from imas_codex.agentic.agents import get_model_for_task
from imas_codex.agentic.prompt_loader import load_prompts
from imas_codex.discovery.models import (
    DirectoryEvidence,
    PathPurpose,
    ScoredBatch,
    ScoredDirectory,
    parse_path_purpose,
)

logger = logging.getLogger(__name__)


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
    PathPurpose.user_home,
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
        """Score a batch of directories.

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

        # Call LLM via LiteLLM (OpenRouter prefix)
        model_id = self.model
        if not model_id.startswith("openrouter/"):
            model_id = f"openrouter/{model_id}"

        response = litellm.completion(
            model=model_id,
            api_key=api_key,
            max_tokens=4000,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        # Parse response
        result_text = response.choices[0].message.content
        scored_dirs = self._parse_response(result_text, directories, threshold)

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
        """Build user prompt with directories to score."""
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
                        file_types = json.loads(file_types)
                    except json.JSONDecodeError:
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

        lines.append(
            "\n\nReturn JSON array with results for each directory (in order)."
        )

        return "\n".join(lines)

    def _parse_response(
        self,
        response_text: str,
        directories: list[dict[str, Any]],
        threshold: float,
    ) -> list[ScoredDirectory]:
        """Parse Claude's response into ScoredDirectory objects."""
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


def score_facility_paths(
    facility: str,
    limit: int = 100,
    batch_size: int = 25,
    focus: str | None = None,
    threshold: float = 0.7,
    model: str | None = None,
    budget: float | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Score scanned paths at a facility.

    Main entry point for the score phase. Queries graph for scanned
    but unscored paths, scores them in batches, and persists results.

    Args:
        facility: Facility ID
        limit: Maximum paths to score this run
        batch_size: Paths per LLM call
        focus: Natural language focus query
        threshold: Min score to expand
        model: LLM model name (None = use "score" task model from config)
        budget: Maximum spend in USD
        dry_run: If True, don't persist to graph

    Returns:
        Statistics dict with scored, expanded, cost, errors
    """
    from imas_codex.discovery.frontier import get_scorable_paths, mark_paths_scored

    # Get paths to score
    paths = get_scorable_paths(facility, limit=limit)
    if not paths:
        logger.info(f"No paths to score for {facility}")
        return {"scored": 0, "expanded": 0, "cost": 0.0, "errors": 0}

    scorer = DirectoryScorer(model=model)  # Uses config default if None
    total_scored = 0
    total_expanded = 0
    total_cost = 0.0
    total_errors = 0

    # Process in batches
    for i in range(0, len(paths), batch_size):
        batch = paths[i : i + batch_size]

        # Check budget
        if budget and total_cost >= budget:
            logger.info(f"Budget exhausted: ${total_cost:.2f} >= ${budget:.2f}")
            break

        try:
            result = scorer.score_batch(
                directories=batch,
                focus=focus,
                threshold=threshold,
            )

            total_cost += result.total_cost
            total_scored += len(result.scored_dirs)
            total_expanded += result.expanded_count

            if not dry_run:
                # Persist to graph
                score_data = [d.to_graph_dict() for d in result.scored_dirs]
                mark_paths_scored(facility, score_data)

            logger.info(
                f"Batch {i // batch_size + 1}: scored={len(result.scored_dirs)}, "
                f"expanded={result.expanded_count}, cost=${result.total_cost:.4f}"
            )

        except Exception as e:
            logger.exception(f"Error scoring batch {i // batch_size + 1}: {e}")
            total_errors += len(batch)

    return {
        "scored": total_scored,
        "expanded": total_expanded,
        "cost": total_cost,
        "errors": total_errors,
    }
