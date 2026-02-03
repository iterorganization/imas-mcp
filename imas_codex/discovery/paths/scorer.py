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
from imas_codex.discovery.paths.models import (
    DirectoryEvidence,
    DirectoryScoringBatch,
    ResourcePurpose,
    ScoredBatch,
    ScoredDirectory,
    parse_path_purpose,
)

logger = logging.getLogger(__name__)

# Suppress verbose litellm logging (rate limit messages, help links, etc.)
# These are set before import to avoid the default verbose output
os.environ.setdefault("LITELLM_LOG", "ERROR")
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("LiteLLM Proxy").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Retry configuration for rate limiting
MAX_RETRIES = 5
RETRY_BASE_DELAY = 5.0  # seconds, doubles each retry (5, 10, 20, 40, 80)

# Container expansion threshold (lower than default to explore containers)
CONTAINER_THRESHOLD = 0.1

# Per-purpose score names (for iteration)
PURPOSE_SCORE_NAMES = [
    "score_modeling_code",
    "score_analysis_code",
    "score_operations_code",
    "score_modeling_data",
    "score_experimental_data",
    "score_data_access",
    "score_workflow",
    "score_visualization",
    "score_documentation",
    "score_imas",
]

# Purposes that should be suppressed (lower scores)
# These get a 0.3 multiplier and should_expand=False
SUPPRESSED_PURPOSES = {
    ResourcePurpose.system,
    ResourcePurpose.build_artifact,
    ResourcePurpose.archive,
}

# Purposes that are containers (score = exploration potential)
CONTAINER_PURPOSES = {
    ResourcePurpose.container,
}


def grounded_score(
    scores: dict[str, float],
    evidence: DirectoryEvidence,
    purpose: ResourcePurpose,
) -> float:
    """Compute combined score from per-purpose scores with evidence adjustments.

    Uses MAX of per-purpose scores (not weighted average) so that paths excelling
    in a single dimension (e.g., pure data, pure docs) are not penalized.

    Grounded scoring with purpose-aware semantics:

    For CONTAINER purposes:
        Score = exploration potential (should we scan children?)

    For CODE/DATA purposes:
        Score = ingestion priority (should we process this?)

    For SKIP purposes (system, build_artifact, archive):
        Score = always low (0.3 multiplier)

    Args:
        scores: Dict of per-purpose scores (score_modeling_code, score_imas, etc.)
        evidence: Collected evidence from LLM
        purpose: Classified purpose

    Returns:
        Combined score (0.0-1.0)
    """
    # Use max of all per-purpose scores - paths may excel in only one dimension
    base_score = max(scores.values()) if scores else 0.0

    # Quality boost
    quality_boost = 0.0
    quality_lower = [q.lower() for q in evidence.quality_indicators]
    if any("readme" in q for q in quality_lower):
        quality_boost += 0.05
    if any("makefile" in q for q in quality_lower):
        quality_boost += 0.05
    if any("git" in q for q in quality_lower):
        quality_boost += 0.05

    # IMAS boost (from score_imas)
    if scores.get("score_imas", 0.0) > 0.3:
        quality_boost += 0.10

    # Code diversity boost
    if len(evidence.code_indicators) > 3:
        quality_boost += 0.05

    # Purpose-based multipliers
    # - Suppressed (system, build_artifact, archive): 0.3 - low value
    # - Test suite: 0.6 - some value but not primary
    # - Container: 1.0 - score already reflects exploration potential
    # - All others: 1.0 - full value
    purpose_multiplier = 1.0
    if purpose in SUPPRESSED_PURPOSES:
        purpose_multiplier = 0.3
    elif purpose == ResourcePurpose.test_suite:
        purpose_multiplier = 0.6

    combined = (base_score + quality_boost) * purpose_multiplier

    # No upper cap - allow natural values > 1.0 for paths with multiple
    # quality indicators. Use percentile normalization for ranking.
    # Scores > 1.0 indicate exceptional paths (e.g., git repo with README,
    # Makefile, IMAS indicators, and high dimension scores).
    return max(0.0, combined)


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
        facility: Facility ID for sampling calibration examples

    Example:
        scorer = DirectoryScorer(facility="tcv")
        batch = scorer.score_batch(
            directories=[...],
            focus="equilibrium codes",
            threshold=0.7,
        )
    """

    model: str | None = None
    facility: str | None = None

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
                # max_tokens=32000 supports batches of 50+ directories
                # Sonnet 4.5 has 200k context, output is ~250 tokens/dir
                # response_format ensures schema is enforced by LLM provider
                response = litellm.completion(
                    model=model_id,
                    api_key=api_key,
                    max_tokens=32000,
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
                    # Use debug level to reduce noise - rate limits are expected
                    logger.debug(
                        f"LLM rate limited (attempt {attempt + 1}/{MAX_RETRIES}), "
                        f"waiting {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    # Non-retryable error
                    raise
        else:
            # All retries exhausted
            raise last_error  # type: ignore[misc]

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

        # Calculate cost per path for tracking
        cost_per_path = cost / len(directories) if directories else 0.0

        # Parse response using structured output
        scored_dirs = self._parse_structured_response(
            response, directories, threshold, cost_per_path
        )

        return ScoredBatch(
            scored_dirs=scored_dirs,
            total_cost=cost,
            model=self.model,
            tokens_used=total_tokens,
        )

    def _build_system_prompt(self, focus: str | None = None) -> str:
        """Build system prompt for directory scoring.

        Uses render_prompt() for proper Jinja2 rendering with schema context.
        The scorer.md prompt is dynamic=true, so schema-derived values
        (path_purposes, score_dimensions) are injected automatically.

        Injects:
        - focus: optional natural language focus
        - example_paths: calibration examples from previously scored paths
        - dimension_examples: cross-facility examples per score dimension
        """
        from imas_codex.agentic.prompt_loader import render_prompt
        from imas_codex.discovery.paths.frontier import (
            sample_paths_by_dimension,
            sample_scored_paths,
        )

        # Build context for template rendering
        context: dict[str, Any] = {}

        # Add focus if provided
        if focus:
            context["focus"] = focus

        # Add example_paths for calibration (by combined score quartile)
        if self.facility:
            example_paths = sample_scored_paths(self.facility, per_quartile=3)
            has_examples = any(example_paths.get(q) for q in example_paths)
            if has_examples:
                context["example_paths"] = example_paths

        # Add dimension_examples for per-category calibration (cross-facility)
        dimension_examples = sample_paths_by_dimension(
            facility=self.facility,
            per_dimension=2,
            cross_facility=True,
        )
        has_dimension_examples = any(
            dimension_examples.get(d) for d in dimension_examples
        )
        if has_dimension_examples:
            context["dimension_examples"] = dimension_examples

        # Use render_prompt for proper Jinja2 rendering with schema context
        return render_prompt("discovery/scorer", context)

    def _build_user_prompt(self, directories: list[dict[str, Any]]) -> str:
        """Build user prompt with directories to score.

        Includes child file/directory names for context - these are
        critical for the LLM to infer purpose from naming conventions.
        """
        import json as json_module

        lines = [
            "Score these directories.",
            "(In Contents below, entries ending with / are subdirectories, "
            "others are files.)\n",
        ]

        for i, d in enumerate(directories, 1):
            # Full path is critical context - shown prominently
            lines.append(f"\n## Directory {i}")
            lines.append(f"Path: {d['path']}")

            # Add DirStats
            lines.append(
                f"Files: {d.get('total_files', 0)}, Dirs: {d.get('total_dirs', 0)}"
            )

            file_types = d.get("file_type_counts")
            if file_types:
                if isinstance(file_types, str):
                    try:
                        file_types = json_module.loads(file_types)
                    except json_module.JSONDecodeError:
                        file_types = {}
                lines.append(f"File types: {file_types}")

            # Quality indicators on one line
            quality = []
            if d.get("has_readme"):
                quality.append("README")
            if d.get("has_makefile"):
                quality.append("Makefile")
            if d.get("has_git"):
                quality.append(".git")
            if quality:
                lines.append(f"Quality: {', '.join(quality)}")

            patterns = d.get("patterns_detected", [])
            if patterns:
                lines.append(f"Patterns: {', '.join(patterns)}")

            # Numeric directory ratio warning (shot folders, run dirs)
            numeric_ratio = d.get("numeric_dir_ratio", 0)
            if numeric_ratio > 0.5:
                lines.append(
                    f"⚠️ DATA CONTAINER: {numeric_ratio:.0%} of subdirs are "
                    f"numeric (shot IDs/runs). Set should_expand=false."
                )

            # Prefer tree context over flat child_names (shows hierarchy)
            tree_context = d.get("tree_context")
            if tree_context:
                lines.append("Structure (tree -L 2):")
                lines.append(f"```\n{tree_context}\n```")
            else:
                # Fall back to flat child names
                child_names = d.get("child_names")
                if child_names:
                    if isinstance(child_names, str):
                        try:
                            child_names = json_module.loads(child_names)
                        except json_module.JSONDecodeError:
                            child_names = []
                    if child_names:
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
        cost_per_path: float = 0.0,
    ) -> list[ScoredDirectory]:
        """Parse structured LLM response into ScoredDirectory objects.

        Uses LiteLLM's structured output - the response is already validated
        against the DirectoryScoringBatch Pydantic model.

        Args:
            response: LiteLLM response object
            directories: Input directory info dicts
            threshold: Minimum score to expand
            cost_per_path: LLM cost per path (batch_cost / batch_size)

        Returns:
            List of ScoredDirectory objects with scores and cost_per_path set
        """
        import re

        try:
            # LiteLLM returns content as JSON string when using response_format
            content = response.choices[0].message.content
            # Sanitize: remove control characters (except newline/tab) and surrogates
            # LLM sometimes includes invalid Unicode from file paths
            content = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", content)
            content = content.encode("utf-8", errors="surrogateescape").decode(
                "utf-8", errors="replace"
            )
            batch = DirectoryScoringBatch.model_validate_json(content)
            results = batch.results
        except Exception as e:
            # CRITICAL: Validation error means LLM response was malformed.
            # DO NOT create ScoredDirectory objects that will be persisted.
            # Instead, raise the error so score_worker can revert paths to 'scanned'
            # status for retry in the next batch.
            logger.error(
                f"LLM validation error for batch of {len(directories)} paths: {e}. "
                "Paths will be reverted to scanned status."
            )
            raise ValueError(f"LLM response validation failed: {e}") from e

        scored = []
        for i, result in enumerate(results[: len(directories)]):
            path = directories[i]["path"]

            # Clamp per-purpose scores (should already be valid from schema)
            scores = {
                "score_modeling_code": max(
                    0.0, min(1.0, getattr(result, "score_modeling_code", 0.0))
                ),
                "score_analysis_code": max(
                    0.0, min(1.0, getattr(result, "score_analysis_code", 0.0))
                ),
                "score_operations_code": max(
                    0.0, min(1.0, getattr(result, "score_operations_code", 0.0))
                ),
                "score_modeling_data": max(
                    0.0, min(1.0, getattr(result, "score_modeling_data", 0.0))
                ),
                "score_experimental_data": max(
                    0.0, min(1.0, getattr(result, "score_experimental_data", 0.0))
                ),
                "score_data_access": max(
                    0.0, min(1.0, getattr(result, "score_data_access", 0.0))
                ),
                "score_workflow": max(
                    0.0, min(1.0, getattr(result, "score_workflow", 0.0))
                ),
                "score_visualization": max(
                    0.0, min(1.0, getattr(result, "score_visualization", 0.0))
                ),
                "score_documentation": max(
                    0.0, min(1.0, getattr(result, "score_documentation", 0.0))
                ),
                "score_imas": max(0.0, min(1.0, getattr(result, "score_imas", 0.0))),
            }

            # Convert Pydantic enum to graph ResourcePurpose
            purpose = parse_path_purpose(result.path_purpose.value)

            # Build evidence from Pydantic model
            evidence = DirectoryEvidence(
                code_indicators=result.evidence.code_indicators,
                data_indicators=result.evidence.data_indicators,
                doc_indicators=result.evidence.doc_indicators,
                imas_indicators=result.evidence.imas_indicators,
                physics_indicators=result.evidence.physics_indicators,
                quality_indicators=result.evidence.quality_indicators,
            )

            # Compute grounded score from per-purpose scores
            combined = grounded_score(scores, evidence, purpose)

            # Extract git metadata for penalty and expansion decisions
            has_git = directories[i].get("has_git", False)
            git_remote_url = directories[i].get("git_remote_url")

            # Determine if repo is accessible elsewhere (public or on internal servers)
            # Private repos on public hosts contain unique content we want to scan
            repo_accessible_elsewhere = False
            if has_git and git_remote_url:
                # Check if on public hosting service
                is_public_host = any(
                    host in git_remote_url.lower()
                    for host in ["github.com", "gitlab.com", "bitbucket.org"]
                )
                if is_public_host:
                    # Verify actual visibility via HTTP (confirms not private)
                    from imas_codex.discovery.paths.frontier import (
                        _is_repo_publicly_accessible,
                    )

                    repo_accessible_elsewhere = _is_repo_publicly_accessible(
                        git_remote_url, timeout=2.0
                    )
                else:
                    # Check if on facility's internal git servers
                    # TODO: Could load from facility config here
                    internal_servers = ["gitlab.iter.org", "git.iter.org"]
                    repo_accessible_elsewhere = any(
                        server in git_remote_url.lower() for server in internal_servers
                    )

            # Note: No score penalty for git repos. SoftwareRepo deduplication
            # handles clones/forks via root_commit or remote_url matching. Repos
            # are scored on their own merit; expansion is blocked for all git repos.

            # Expansion decision - containers use lower threshold
            effective_threshold = (
                CONTAINER_THRESHOLD if purpose in CONTAINER_PURPOSES else threshold
            )

            # Base expansion decision
            should_expand = (
                combined >= effective_threshold
                and result.should_expand
                and purpose not in SUPPRESSED_PURPOSES
            )

            # CRITICAL: Never expand git repos (code is available via git clone)
            # Even private repos don't need child expansion - files are at repo root
            if has_git:
                should_expand = False

            # CRITICAL: Never expand data containers (too many files)
            data_purposes = {
                ResourcePurpose.modeling_data,
                ResourcePurpose.experimental_data,
            }
            if purpose in data_purposes:
                should_expand = False

            # CRITICAL: High subdirectory count is a strong signal of data container
            # Even if LLM misclassifies as container, block expansion for dirs with
            # many subdirectories (typical of simulation run directories)
            total_dirs = directories[i].get("total_dirs", 0)
            if total_dirs > 100 and purpose == ResourcePurpose.container:
                should_expand = False
                # Log this override for debugging
                logger.debug(
                    f"Blocked expansion of {path}: {total_dirs} subdirs (likely data)"
                )

            # Enrichment decision - LLM decides, but override for known-large paths
            should_enrich = getattr(result, "should_enrich", True)
            enrich_skip_reason = getattr(result, "enrich_skip_reason", None)

            # Override: never enrich git repos with accessible remotes
            # (can get LOC from remote API)
            if has_git and repo_accessible_elsewhere:
                should_enrich = False
                enrich_skip_reason = (
                    enrich_skip_reason
                    or "git repo accessible elsewhere - code available via git"
                )
            elif has_git and git_remote_url:
                # Private repo with remote: still worth enriching
                # since we can't get metrics from the remote
                pass

            # Override: never enrich data containers (too many files)
            if purpose in data_purposes:
                should_enrich = False
                enrich_skip_reason = (
                    enrich_skip_reason or "data container - too many files"
                )

            # terminal_reason is NULL for LLM-scored paths - reason is derivable
            # from has_git, path_purpose, score. Only set for non-derivable cases
            # (access_denied, empty, parent_terminal, etc.)

            scored_dir = ScoredDirectory(
                path=path,
                path_purpose=purpose,
                description=result.description,
                evidence=evidence,
                score_modeling_code=scores["score_modeling_code"],
                score_analysis_code=scores["score_analysis_code"],
                score_operations_code=scores["score_operations_code"],
                score_modeling_data=scores["score_modeling_data"],
                score_experimental_data=scores["score_experimental_data"],
                score_data_access=scores["score_data_access"],
                score_workflow=scores["score_workflow"],
                score_visualization=scores["score_visualization"],
                score_documentation=scores["score_documentation"],
                score_imas=scores["score_imas"],
                score=combined,
                should_expand=should_expand,
                should_enrich=should_enrich,
                keywords=result.keywords[:5] if result.keywords else [],
                physics_domain=result.physics_domain,
                expansion_reason=result.expansion_reason,
                skip_reason=result.skip_reason,
                enrich_skip_reason=enrich_skip_reason,
                score_cost=cost_per_path,
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
            # Fallback: return empty scores for all (use container as neutral purpose)
            return [
                ScoredDirectory(
                    path=d["path"],
                    path_purpose=ResourcePurpose.container,
                    description="Parse error",
                    evidence=DirectoryEvidence(),
                    score=0.0,
                    should_expand=False,
                    skip_reason=f"LLM response parse failed: {e}",
                )
                for d in directories
            ]

        scored = []
        for i, result in enumerate(results[: len(directories)]):
            path = directories[i]["path"]

            # Extract and clamp per-purpose scores
            scores = {
                "score_modeling_code": max(
                    0.0, min(1.0, float(result.get("score_modeling_code", 0.0)))
                ),
                "score_analysis_code": max(
                    0.0, min(1.0, float(result.get("score_analysis_code", 0.0)))
                ),
                "score_operations_code": max(
                    0.0, min(1.0, float(result.get("score_operations_code", 0.0)))
                ),
                "score_modeling_data": max(
                    0.0, min(1.0, float(result.get("score_modeling_data", 0.0)))
                ),
                "score_experimental_data": max(
                    0.0, min(1.0, float(result.get("score_experimental_data", 0.0)))
                ),
                "score_data_access": max(
                    0.0, min(1.0, float(result.get("score_data_access", 0.0)))
                ),
                "score_workflow": max(
                    0.0, min(1.0, float(result.get("score_workflow", 0.0)))
                ),
                "score_visualization": max(
                    0.0, min(1.0, float(result.get("score_visualization", 0.0)))
                ),
                "score_documentation": max(
                    0.0, min(1.0, float(result.get("score_documentation", 0.0)))
                ),
                "score_imas": max(0.0, min(1.0, float(result.get("score_imas", 0.0)))),
            }

            # Parse purpose
            purpose = parse_path_purpose(result.get("path_purpose", "unknown"))

            # Build evidence
            evidence_data = result.get("evidence", {})
            evidence = DirectoryEvidence(
                code_indicators=evidence_data.get("code_indicators", []),
                data_indicators=evidence_data.get("data_indicators", []),
                doc_indicators=evidence_data.get("doc_indicators", []),
                imas_indicators=evidence_data.get("imas_indicators", []),
                physics_indicators=evidence_data.get("physics_indicators", []),
                quality_indicators=evidence_data.get("quality_indicators", []),
            )

            # Compute grounded score
            combined = grounded_score(scores, evidence, purpose)

            # Git metadata for expansion/enrichment decisions (no score penalty)
            has_git = directories[i].get("has_git", False)
            git_remote_url = directories[i].get("git_remote_url")
            # Note: No score penalty - SoftwareRepo dedup handles clone/fork detection

            # Expansion decision - containers use lower threshold
            effective_threshold = (
                CONTAINER_THRESHOLD if purpose in CONTAINER_PURPOSES else threshold
            )
            should_expand = (
                combined >= effective_threshold
                and result.get("should_expand", False)
                and purpose not in SUPPRESSED_PURPOSES
            )

            # Never expand git repos
            if has_git:
                should_expand = False

            # Enrichment override for git repos with remotes
            should_enrich = result.get("should_enrich", True)
            enrich_skip_reason = result.get("enrich_skip_reason")
            if has_git and git_remote_url:
                should_enrich = False
                enrich_skip_reason = (
                    enrich_skip_reason
                    or "git repo with remote - code available elsewhere"
                )

            scored_dir = ScoredDirectory(
                path=path,
                path_purpose=purpose,
                description=result.get("description", ""),
                evidence=evidence,
                score_modeling_code=scores["score_modeling_code"],
                score_analysis_code=scores["score_analysis_code"],
                score_operations_code=scores["score_operations_code"],
                score_modeling_data=scores["score_modeling_data"],
                score_experimental_data=scores["score_experimental_data"],
                score_data_access=scores["score_data_access"],
                score_workflow=scores["score_workflow"],
                score_visualization=scores["score_visualization"],
                score_documentation=scores["score_documentation"],
                score_imas=scores["score_imas"],
                score=combined,
                should_expand=should_expand,
                should_enrich=should_enrich,
                keywords=result.get("keywords", [])[:5],  # Cap at 5
                physics_domain=result.get("physics_domain"),
                expansion_reason=result.get("expansion_reason"),
                skip_reason=result.get("skip_reason"),
                enrich_skip_reason=enrich_skip_reason,
            )

            scored.append(scored_dir)

        return scored
