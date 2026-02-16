"""LLM-based directory scoring.

This module implements the scoring phase of graph-led discovery:
1. Query graph for scanned but unscored paths
2. Build batched prompts with directory context (parent/sibling scores,
   tree structure, file types, quality indicators)
3. Call LLM with structured output schema for reliable parsing
4. Trust the LLM's scores and expansion decisions directly

The LLM sees all relevant context in the prompt and makes calibrated
scoring and expansion decisions. The code applies only structural
overrides (git repos, data containers) that encode facts the LLM
cannot verify.

Retry logic handles rate limiting (OpenRouter "Overloaded" errors).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from imas_codex.discovery.base.llm import suppress_litellm_noise
from imas_codex.discovery.paths.models import (
    DirectoryEvidence,
    ResourcePurpose,
    ScoreBatch,
    ScoredBatch,
    ScoredDirectory,
    parse_path_purpose,
)
from imas_codex.settings import get_model

logger = logging.getLogger(__name__)

# Suppress litellm noise on import (print-based + logger-based)
suppress_litellm_noise()

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

# Structural expansion blocks — these encode facts the LLM cannot verify,
# not scoring judgments. Everything else is LLM-driven.
_DATA_PURPOSES = {
    ResourcePurpose.modeling_data,
    ResourcePurpose.experimental_data,
}


def grounded_score(
    scores: dict[str, float],
    input_data: dict[str, Any],
    purpose: ResourcePurpose,
) -> float:
    """Compute combined score from per-dimension LLM scores.

    Uses MAX of per-dimension scores so that paths excelling in a single
    dimension (e.g., pure data access, pure docs) are not penalized.

    The LLM has already seen all input context — file types, README,
    git indicators, tree structure, parent/sibling scores — and factored
    it into each dimension score. No post-hoc boosts, multipliers, or
    caps are applied. The prompt is the sole source of scoring logic.

    Args:
        scores: Dict of per-dimension scores (score_modeling_code, score_imas, etc.)
        input_data: Directory info dict (available for future use)
        purpose: Classified purpose (available for future use)

    Returns:
        Combined score (0.0-1.0)
    """
    if not scores:
        return 0.0
    return max(scores.values())


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
            self.model = get_model("language")

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
        from imas_codex.discovery.base.llm import call_llm_structured

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

        # Call LLM with shared retry+parse loop (retries on both API
        # errors and JSON/validation errors from truncated responses).
        # Model-aware token limits applied automatically.
        batch, cost, total_tokens = call_llm_structured(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_model=ScoreBatch,
        )

        # Calculate cost per path for tracking
        cost_per_path = cost / len(directories) if directories else 0.0

        # Map parsed results to ScoredDirectory objects
        scored_dirs = self._map_scored_directories(
            batch, directories, threshold, cost_per_path
        )

        return ScoredBatch(
            scored_dirs=scored_dirs,
            total_cost=cost,
            model=self.model,
            tokens_used=total_tokens,
        )

    async def async_score_batch(
        self,
        directories: list[dict[str, Any]],
        focus: str | None = None,
        threshold: float = 0.7,
    ) -> ScoredBatch:
        """Async version of score_batch using acall_llm_structured.

        Fully cancellable — no thread executors, uses litellm.acompletion().
        """
        from imas_codex.discovery.base.llm import acall_llm_structured

        if not directories:
            return ScoredBatch(
                scored_dirs=[],
                total_cost=0.0,
                model=self.model,
                tokens_used=0,
            )

        system_prompt = self._build_system_prompt(focus)
        user_prompt = self._build_user_prompt(directories)

        batch, cost, total_tokens = await acall_llm_structured(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_model=ScoreBatch,
        )

        cost_per_path = cost / len(directories) if directories else 0.0
        scored_dirs = self._map_scored_directories(
            batch, directories, threshold, cost_per_path
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
        - dimension_calibration: examples at 5 score levels per dimension
        """
        from imas_codex.agentic.prompt_loader import render_prompt
        from imas_codex.discovery.paths.frontier import (
            sample_dimension_calibration_examples,
        )

        # Build context for template rendering
        context: dict[str, Any] = {}

        # Add focus if provided
        if focus:
            context["focus"] = focus

        # Add dimension calibration examples (5 levels x 10 dimensions x 3 examples)
        # This provides comprehensive calibration for the LLM to understand
        # what scores have historically been assigned at each level
        dimension_calibration = sample_dimension_calibration_examples(
            facility=self.facility,
            per_level=3,
            tolerance=0.1,
        )
        # Only add if we have meaningful data
        has_calibration = any(
            any(examples for examples in dim_levels.values())
            for dim_levels in dimension_calibration.values()
        )
        if has_calibration:
            context["dimension_calibration"] = dimension_calibration

        # Use render_prompt for proper Jinja2 rendering with schema context
        return render_prompt("discovery/scorer", context)

    def _build_user_prompt(self, directories: list[dict[str, Any]]) -> str:
        """Build user prompt with directories to score.

        Includes child file/directory names for context - these are
        critical for the LLM to infer purpose from naming conventions.

        Also injects parent/sibling context from the graph to enable
        relative scoring decisions.
        """
        import json as json_module

        from imas_codex.discovery.paths.frontier import get_hierarchy_context

        # Query hierarchy context for all paths in the batch
        paths = [d["path"] for d in directories]
        hierarchy = {}
        if self.facility:
            try:
                hierarchy = get_hierarchy_context(self.facility, paths)
            except Exception:
                logger.debug("Failed to get hierarchy context", exc_info=True)

        lines = [
            "Score these directories.",
            "(In Contents below, entries ending with / are subdirectories, "
            "others are files.)\n",
        ]

        for i, d in enumerate(directories, 1):
            # Full path is critical context - shown prominently
            lines.append(f"\n## Directory {i}")
            lines.append(f"Path: {d['path']}")

            # Depth info
            depth = d.get("depth")
            if depth is not None:
                lines.append(f"Depth: {depth}")

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
            numeric_ratio = d.get("numeric_dir_ratio") or 0
            if numeric_ratio > 0.5:
                lines.append(
                    f"⚠️ DATA CONTAINER: {numeric_ratio:.0%} of subdirs are "
                    f"numeric (shot IDs/runs). Set should_expand=false."
                )

            # Parent/sibling context from graph
            ctx = hierarchy.get(d["path"])
            if ctx:
                parent = ctx.get("parent")
                if parent and parent.get("score") is not None:
                    lines.append(
                        f"Parent: {parent['path']} → "
                        f"{parent.get('purpose', '?')} "
                        f"(score: {parent['score']:.2f})"
                    )

                siblings = ctx.get("siblings", [])
                if siblings:
                    sib_strs = []
                    for s in siblings[:6]:
                        basename = s["path"].rstrip("/").split("/")[-1]
                        sib_strs.append(
                            f"{basename}={s.get('purpose', '?')}"
                            f"({s.get('score', 0):.1f})"
                        )
                    lines.append(f"Scored siblings: {', '.join(sib_strs)}")

            # Prefer tree context over flat child_names (shows hierarchy)
            tree_context = d.get("tree_context")
            if tree_context:
                lines.append("Structure (eza --tree):")
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

    def _map_scored_directories(
        self,
        batch: ScoreBatch,
        directories: list[dict[str, Any]],
        threshold: float,
        cost_per_path: float = 0.0,
    ) -> list[ScoredDirectory]:
        """Map parsed ScoreBatch to ScoredDirectory objects.

        JSON parsing and sanitization are handled by call_llm_structured().
        This method applies grounded scoring, expansion logic, and enrichment
        decisions to the already-validated Pydantic model.

        Args:
            batch: Parsed ScoreBatch from LLM response
            directories: Input directory info dicts
            threshold: Minimum score to expand
            cost_per_path: LLM cost per path (batch_cost / batch_size)

        Returns:
            List of ScoredDirectory objects with scores and cost_per_path set
        """
        import json as json_module

        results = batch.results

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

            # Build evidence from input data (not LLM response - schema simplified)
            file_types = directories[i].get("file_type_counts") or {}
            if isinstance(file_types, str):
                try:
                    file_types = json_module.loads(file_types)
                except (json_module.JSONDecodeError, TypeError):
                    file_types = {}
            code_exts = {"py", "f90", "f", "cpp", "c", "h", "jl", "m", "pro", "idl"}
            data_exts = {"nc", "h5", "hdf5", "csv", "dat", "mat"}
            evidence = DirectoryEvidence(
                code_indicators=[ext for ext in file_types if ext.lower() in code_exts],
                data_indicators=[ext for ext in file_types if ext.lower() in data_exts],
                doc_indicators=["README"] if directories[i].get("has_readme") else [],
                imas_indicators=[],  # Filled by enrichment worker
                physics_indicators=[],  # Filled by enrichment worker
                quality_indicators=[
                    name
                    for name, flag in [
                        ("has_readme", directories[i].get("has_readme")),
                        ("has_makefile", directories[i].get("has_makefile")),
                        ("has_git", directories[i].get("has_git")),
                    ]
                    if flag
                ],
            )

            # Compute grounded score from per-purpose scores and input data
            combined = grounded_score(scores, directories[i], purpose)

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

            # Expansion: trust the LLM's should_expand decision directly.
            # The prompt gives the LLM rich context (depth, parent/sibling
            # scores, tree structure, file types) to make calibrated decisions.
            # Only structural constraints override the LLM:
            should_expand = result.should_expand

            # STRUCTURAL: Never expand git repos (code available via git clone)
            if has_git:
                should_expand = False

            # STRUCTURAL: Never expand data containers (too many files)
            if purpose in _DATA_PURPOSES:
                should_expand = False

            # Enrichment decision - LLM decides, but override for known-large paths
            should_enrich = result.should_enrich
            enrich_skip_reason = result.enrich_skip_reason

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
            if purpose in _DATA_PURPOSES:
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

            # Build evidence from input data (not LLM response - schema simplified)
            file_types = directories[i].get("file_type_counts") or {}
            if isinstance(file_types, str):
                try:
                    file_types = json.loads(file_types)
                except (json.JSONDecodeError, TypeError):
                    file_types = {}
            code_exts = {"py", "f90", "f", "cpp", "c", "h", "jl", "m", "pro", "idl"}
            data_exts = {"nc", "h5", "hdf5", "csv", "dat", "mat"}
            evidence = DirectoryEvidence(
                code_indicators=[ext for ext in file_types if ext.lower() in code_exts],
                data_indicators=[ext for ext in file_types if ext.lower() in data_exts],
                doc_indicators=["README"] if directories[i].get("has_readme") else [],
                imas_indicators=[],  # Filled by enrichment worker
                physics_indicators=[],  # Filled by enrichment worker
                quality_indicators=[
                    name
                    for name, flag in [
                        ("has_readme", directories[i].get("has_readme")),
                        ("has_makefile", directories[i].get("has_makefile")),
                        ("has_git", directories[i].get("has_git")),
                    ]
                    if flag
                ],
            )

            # Compute grounded score from input data
            combined = grounded_score(scores, directories[i], purpose)

            # Git metadata for expansion/enrichment decisions (no score penalty)
            has_git = directories[i].get("has_git", False)
            git_remote_url = directories[i].get("git_remote_url")
            # Note: No score penalty - SoftwareRepo dedup handles clone/fork detection

            # Expansion: trust the LLM's should_expand decision directly.
            should_expand = result.get("should_expand", False)

            # STRUCTURAL: Never expand git repos (code available via git clone)
            if has_git:
                should_expand = False

            # STRUCTURAL: Never expand data containers (too many files)
            if purpose in _DATA_PURPOSES:
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
