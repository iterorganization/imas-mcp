"""Search quality evaluation harness and DoE grid search.

Provides a parametric evaluation framework for tuning the search score
mixing pipeline in ``imas_codex.tools.graph_search``.  The ``MixConfig``
dataclass captures every tunable constant in the scoring pipeline; the
``evaluate_config`` function runs all 30 gold-standard benchmark queries
with a given config and returns MRR, P@10, and IDS-Recall@10.

The DoE (Design of Experiments) grid search crosses key parameters to
find optimal configurations.  Results can be saved to JSON for offline
analysis.

Usage::

    # Unit tests only (no graph needed)
    uv run pytest tests/search/test_search_evaluation.py -k "not graph and not slow" -x -v

    # Run default config validation (requires graph + embedding server)
    uv run pytest tests/search/test_search_evaluation.py -k "test_default" -v

    # Run full DoE grid search (~10 min, produces doe_results.json)
    uv run pytest tests/search/test_search_evaluation.py -k "test_doe_grid" -v -m slow
"""

from __future__ import annotations

import itertools
import json
import logging
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import pytest

from tests.search.benchmark_data import (
    ALL_QUERIES,
    CATEGORY_NAMES,
    BenchmarkQuery,
    compute_mrr,
)
from tests.search.benchmark_helpers import QueryResult

logger = logging.getLogger(__name__)


# ── Tunable configuration ─────────────────────────────────────────────────────


@dataclass
class MixConfig:
    """Tunable parameters for search score mixing.

    Defaults match the current tuned values in
    ``imas_codex.tools.graph_search.GraphSearchTool.search_imas_paths``.

    The parameters are organized into three groups:

    **Retrieval limits** control how many candidates each search channel
    retrieves before score fusion.

    **Score fusion weights** determine how vector and text scores are
    combined for items found by one or both channels.

    **Reranking boosts** are additive adjustments applied after fusion
    to reward path-level signals (segment matches, abbreviation hits).
    """

    # ── Retrieval limits ──────────────────────────────────────────────
    vector_limit: int = 500  # max vector results (LIMIT clause)
    text_limit: int = 500  # max text/BM25 results
    hnsw_candidates: int = 500  # HNSW candidate pool (k for vector.queryNodes)

    # ── Score fusion weights ──────────────────────────────────────────
    # Production uses RRF (rank-based fusion, not weighted).
    # These weights only apply in the DoE standalone scoring model.
    vector_weight: float = 0.6  # weight for vector score in linear fusion
    text_weight: float = 0.4  # weight for text score in linear fusion
    dual_channel_bonus: float = 0.05  # additive bonus for dual-channel hits

    # ── Reranking boosts ──────────────────────────────────────────────
    path_boost: float = 0.03  # per-word path segment match boost
    terminal_boost: float = 0.08  # exact terminal segment bonus
    ids_name_boost: float = 0.05  # IDS name match bonus
    abbreviation_boost: float = 0.15  # abbreviation exact-match boost

    # ── Reserved boosts (future search pipeline features) ─────────────
    cluster_boost: float = 0.0  # cluster membership boost
    hierarchy_boost: float = 0.0  # parent proximity boost
    coordinate_boost: float = 0.0  # coordinate sharing boost


# Locked optimal configuration from current tuned defaults.
# Composite target: 0.5*MRR + 0.3*P@10 + 0.2*IDS_Recall@10
OPTIMAL_CONFIG = MixConfig()


# ── Metric computation ────────────────────────────────────────────────────────


def compute_precision_at_k(
    ranked_results: list[str],
    expected: list[str],
    k: int = 10,
) -> float:
    """Compute precision at cutoff *k*.

    Returns the fraction of results in the top *k* that are relevant
    (present in *expected*).  When fewer than *k* results are returned
    the denominator is the number of results actually present.
    """
    top_k = ranked_results[:k]
    if not top_k:
        return 0.0
    expected_set = set(expected)
    hits = sum(1 for r in top_k if r in expected_set)
    return hits / len(top_k)


def compute_ids_recall_at_k(
    ranked_results: list[str],
    expected: list[str],
    k: int = 10,
) -> float:
    """Compute IDS-level recall at cutoff *k*.

    Measures the fraction of distinct IDS names from *expected* paths
    that appear among the top *k* results.  This rewards search
    configurations that surface results from diverse IDSs.
    """
    expected_ids = {p.split("/")[0] for p in expected if "/" in p}
    if not expected_ids:
        return 0.0
    result_ids = {p.split("/")[0] for p in ranked_results[:k] if "/" in p}
    return len(expected_ids & result_ids) / len(expected_ids)


def compute_composite_score(
    mrr: float,
    precision_at_10: float,
    ids_recall_at_10: float,
    weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
) -> float:
    """Compute weighted composite quality score.

    Default weights: 50% MRR, 30% P@10, 20% IDS-Recall@10.
    """
    return (
        weights[0] * mrr + weights[1] * precision_at_10 + weights[2] * ids_recall_at_10
    )


# ── Configurable scoring pipeline ─────────────────────────────────────────────

# Short physics terms that must not be filtered out of search queries.
# Mirrors _PHYSICS_SHORT_TERMS from graph_search.py.
_PHYSICS_SHORT_TERMS = frozenset(
    {
        "q",
        "ip",
        "b0",
        "te",
        "ne",
        "ti",
        "ni",
        "psi",
        "r",
        "z",
        "phi",
        "j",
        "e",
        "b",
        "v",
        "p",
        "rho",
        "li",
    }
)


def _apply_score_mixing(
    config: MixConfig,
    vector_results: list[dict[str, Any]],
    text_results: list[dict[str, Any]],
    query: str,
    *,
    is_abbreviation: bool = False,
    original_query: str = "",
) -> dict[str, float]:
    """Combine vector + text results using configurable parameters.

    Reimplements the scoring logic from ``search_imas_paths`` with every
    constant drawn from *config* instead of being inline.  This allows
    the DoE grid to sweep across the full parameter space.
    """
    scores: dict[str, float] = {}

    # Vector scores
    for r in vector_results:
        scores[r["id"]] = round(r["score"], 4)

    # Text scores with configurable fusion
    for r in text_results:
        pid = r["id"]
        text_score = round(r["score"], 4)
        if pid in scores:
            scores[pid] = round(
                config.vector_weight * scores[pid]
                + config.text_weight * text_score
                + config.dual_channel_bonus,
                4,
            )
        else:
            scores[pid] = text_score

    # Path segment boost
    query_words = [
        w.lower()
        for w in query.split()
        if len(w) > 2 or w.lower() in _PHYSICS_SHORT_TERMS
    ]
    if query_words:
        for pid in list(scores):
            segments = pid.lower().split("/")

            # Per-word segment matching
            match_count = sum(
                1 for w in query_words if any(w in seg for seg in segments)
            )
            if match_count > 0:
                scores[pid] = round(scores[pid] + config.path_boost * match_count, 4)

            # Exact terminal segment bonus
            terminal = segments[-1] if segments else ""
            if any(w == terminal for w in query_words):
                scores[pid] = round(scores[pid] + config.terminal_boost, 4)

            # IDS name bonus
            ids_name = segments[0] if segments else ""
            if any(w == ids_name for w in query_words):
                scores[pid] = round(scores[pid] + config.ids_name_boost, 4)

    # Abbreviation exact-match boost
    if is_abbreviation and original_query:
        orig_terms = {w.lower() for w in original_query.split()}
        for pid in list(scores):
            terminal = pid.rsplit("/", 1)[-1].lower()
            if terminal in orig_terms:
                scores[pid] = round(scores[pid] + config.abbreviation_boost, 4)

    return scores


def _search_with_config(
    gc: Any,
    encoder: Any,
    config: MixConfig,
    query: str,
    max_results: int = 50,
) -> list[str]:
    """Run search with configurable parameters, return ranked path IDs.

    Uses the same building blocks as ``search_imas_paths`` (vector index
    query, fulltext/BM25 query, QueryAnalyzer) but with all constants
    drawn from *config*.

    Parameters
    ----------
    gc : GraphClient
        Live graph connection.
    encoder : Encoder
        Embedding encoder (must match index dimensionality).
    config : MixConfig
        Score mixing parameters.
    query : str
        User search query.
    max_results : int
        Maximum result paths to return.
    """
    from imas_codex.tools.graph_search import _path_search, _text_search_imas_paths
    from imas_codex.tools.query_analysis import QueryAnalyzer

    analyzer = QueryAnalyzer()
    intent = analyzer.analyze(query)

    # Path queries bypass scoring entirely
    if intent.query_type in ("path_exact", "path_partial"):
        results = _path_search(gc, query, max_results, None)
        return [r["id"] for r in results] if results else []

    # Expand abbreviations for better recall
    expanded = " ".join(intent.expanded_terms) if intent.expanded_terms else query

    # --- Vector search with config limits ---
    embedding = encoder.embed_texts([expanded])[0].tolist()
    try:
        vector_results = gc.query(
            """
            CALL db.index.vector.queryNodes('imas_node_embedding', $k, $embedding)
            YIELD node AS path, score
            WHERE NOT (path)-[:DEPRECATED_IN]->(:DDVersion)
              AND path.node_category = 'data'
            RETURN path.id AS id, score
            ORDER BY score DESC
            LIMIT $vector_limit
            """,
            embedding=embedding,
            k=config.hnsw_candidates,
            vector_limit=config.vector_limit,
        )
    except Exception as e:
        if "dimensionality" in str(e).lower():
            logger.warning("Vector index dimension mismatch: %s", e)
            vector_results = []
        else:
            raise

    # --- Text search with config limit ---
    text_results = _text_search_imas_paths(
        gc,
        expanded,
        config.text_limit,
        ids_filter=None,
    )

    # --- Score mixing ---
    scores = _apply_score_mixing(
        config=config,
        vector_results=vector_results or [],
        text_results=text_results or [],
        query=expanded,
        is_abbreviation=intent.is_abbreviation,
        original_query=intent.original_query,
    )

    # Rank and limit
    sorted_ids = sorted(scores, key=lambda pid: scores[pid], reverse=True)
    return sorted_ids[:max_results]


# ── Evaluation harness ────────────────────────────────────────────────────────


def evaluate_config(
    config: MixConfig,
    gc: Any,
    encoder: Any,
    queries: list[BenchmarkQuery] | None = None,
    max_results: int = 50,
) -> dict[str, Any]:
    """Run all benchmark queries with given config, return metrics.

    Parameters
    ----------
    config : MixConfig
        Search score mixing parameters to evaluate.
    gc : GraphClient
        Live graph connection.
    encoder : Encoder
        Embedding encoder with correct dimensionality.
    queries : list[BenchmarkQuery] | None
        Queries to evaluate; defaults to ALL_QUERIES (30 queries).
    max_results : int
        Maximum results per query.

    Returns
    -------
    dict
        Keys: ``mrr``, ``precision_at_10``, ``ids_recall_at_10``,
        ``composite_score``, ``per_category_mrr``, ``per_query_results``,
        ``config``, ``query_count``.
    """
    if queries is None:
        queries = ALL_QUERIES

    per_query: list[dict[str, Any]] = []
    mrr_values: list[float] = []
    p10_values: list[float] = []
    ids_r10_values: list[float] = []

    for q in queries:
        ranked = _search_with_config(gc, encoder, config, q.query_text, max_results)

        rr = compute_mrr(ranked, q.expected_paths, allow_prefix=True)
        p10 = compute_precision_at_k(ranked, q.expected_paths, k=10)
        ids_r10 = compute_ids_recall_at_k(ranked, q.expected_paths, k=10)

        mrr_values.append(rr)
        p10_values.append(p10)
        ids_r10_values.append(ids_r10)

        per_query.append(
            {
                "query": q.query_text,
                "category": q.category,
                "reciprocal_rank": rr,
                "precision_at_10": p10,
                "ids_recall_at_10": ids_r10,
                "top_5": ranked[:5],
                "expected": q.expected_paths,
            }
        )

    n = len(queries)
    mrr = sum(mrr_values) / n if n else 0.0
    p10 = sum(p10_values) / n if n else 0.0
    ids_r10 = sum(ids_r10_values) / n if n else 0.0
    composite = compute_composite_score(mrr, p10, ids_r10)

    # Per-category MRR
    cat_mrrs: dict[str, list[float]] = defaultdict(list)
    for pq in per_query:
        cat_mrrs[pq["category"]].append(pq["reciprocal_rank"])
    per_category = {
        cat: sum(rrs) / len(rrs) if rrs else 0.0 for cat, rrs in cat_mrrs.items()
    }

    return {
        "mrr": round(mrr, 4),
        "precision_at_10": round(p10, 4),
        "ids_recall_at_10": round(ids_r10, 4),
        "composite_score": round(composite, 4),
        "per_category_mrr": {k: round(v, 4) for k, v in per_category.items()},
        "per_query_results": per_query,
        "config": asdict(config),
        "query_count": n,
    }


# ── DoE grid search ──────────────────────────────────────────────────────────


def generate_doe_grid() -> list[MixConfig]:
    """Generate factorial grid of MixConfig parameter combinations.

    Crosses the five parameters that most affect search quality:
    retrieval limits (vector_limit, text_limit) and fusion weights
    (vector_weight, text_weight, dual_channel_bonus).

    Returns a list of 243 MixConfig instances (3^5 grid).
    """
    grid = {
        "vector_limit": [200, 500, 800],
        "text_limit": [200, 500, 800],
        "vector_weight": [0.4, 0.6, 0.8],
        "text_weight": [0.2, 0.4, 0.6],
        "dual_channel_bonus": [0.0, 0.05, 0.10],
    }
    configs = []
    for combo in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), combo, strict=True))
        configs.append(MixConfig(**params))
    return configs


def run_doe_grid(
    gc: Any,
    encoder: Any,
    queries: list[BenchmarkQuery] | None = None,
    grid: list[MixConfig] | None = None,
) -> list[dict[str, Any]]:
    """Run factorial grid search across key parameters.

    Parameters
    ----------
    gc : GraphClient
        Live graph connection.
    encoder : Encoder
        Embedding encoder.
    queries : list[BenchmarkQuery] | None
        Queries to use; defaults to ALL_QUERIES.
    grid : list[MixConfig] | None
        Configs to evaluate; defaults to ``generate_doe_grid()`` (243 configs).

    Returns
    -------
    list[dict]
        Evaluation results sorted by composite score (best first).
    """
    if grid is None:
        grid = generate_doe_grid()

    results = []
    total = len(grid)
    for i, config in enumerate(grid, 1):
        logger.info("DoE grid: evaluating config %d/%d", i, total)
        metrics = evaluate_config(config, gc, encoder, queries)
        results.append(metrics)

    results.sort(key=lambda r: r["composite_score"], reverse=True)
    return results


def save_doe_results(
    results: list[dict[str, Any]],
    path: Path | None = None,
) -> Path:
    """Save DoE results to JSON for offline analysis.

    Strips ``per_query_results`` from each entry to reduce file size.

    Parameters
    ----------
    results : list[dict]
        Results from ``run_doe_grid()``.
    path : Path | None
        Output path; defaults to ``tests/search/doe_results.json``.

    Returns
    -------
    Path to the saved JSON file.
    """
    if path is None:
        path = Path(__file__).parent / "doe_results.json"

    slim = [{k: v for k, v in r.items() if k != "per_query_results"} for r in results]
    path.write_text(json.dumps(slim, indent=2, default=str))
    logger.info("DoE results saved to %s (%d configs)", path, len(slim))
    return path


def find_pareto_optimal(
    results: list[dict[str, Any]],
    objectives: tuple[str, ...] = ("mrr", "precision_at_10", "ids_recall_at_10"),
) -> list[dict[str, Any]]:
    """Find Pareto-optimal configurations from DoE results.

    A config is Pareto-optimal if no other config dominates it on
    **all** objectives simultaneously.
    """
    pareto = []
    for r in results:
        dominated = False
        for other in results:
            if other is r:
                continue
            if all(other[obj] >= r[obj] for obj in objectives) and any(
                other[obj] > r[obj] for obj in objectives
            ):
                dominated = True
                break
        if not dominated:
            pareto.append(r)
    return pareto


# ══════════════════════════════════════════════════════════════════════════════
#  Tests
# ══════════════════════════════════════════════════════════════════════════════

# ── Unit tests — no graph needed ─────────────────────────────────────────────


class TestMixConfig:
    """Verify MixConfig dataclass behaviour."""

    def test_defaults_match_production(self):
        """Default MixConfig values match graph_search.py inline constants."""
        config = MixConfig()
        # Score fusion (production uses RRF, these are for DoE standalone model)
        assert config.vector_weight == 0.6
        assert config.text_weight == 0.4
        assert config.dual_channel_bonus == 0.05
        # Path boosts (match inline literals in search_imas_paths)
        assert config.path_boost == 0.03
        assert config.terminal_boost == 0.08
        assert config.ids_name_boost == 0.05
        assert config.abbreviation_boost == 0.15
        # Retrieval limits
        assert config.vector_limit == 500
        assert config.text_limit == 500
        assert config.hnsw_candidates == 500

    def test_optimal_config_is_default(self):
        """OPTIMAL_CONFIG matches MixConfig() defaults."""
        default = MixConfig()
        for f in fields(MixConfig):
            assert getattr(OPTIMAL_CONFIG, f.name) == getattr(default, f.name), (
                f"OPTIMAL_CONFIG.{f.name} != default"
            )

    def test_custom_config(self):
        """Custom parameters override defaults; others are preserved."""
        config = MixConfig(vector_weight=0.8, text_weight=0.2)
        assert config.vector_weight == 0.8
        assert config.text_weight == 0.2
        assert config.path_boost == 0.03  # default preserved

    def test_reserved_boosts_zero(self):
        """Reserved future boosts default to zero."""
        config = MixConfig()
        assert config.cluster_boost == 0.0
        assert config.hierarchy_boost == 0.0
        assert config.coordinate_boost == 0.0

    def test_asdict_roundtrip(self):
        """Config serializes to dict and reconstructs identically."""
        config = MixConfig(vector_weight=0.7, text_limit=200)
        d = asdict(config)
        restored = MixConfig(**d)
        assert restored == config

    def test_field_count(self):
        """Guard against accidentally adding fields without tests."""
        assert len(fields(MixConfig)) == 13


class TestDoEGrid:
    """Verify DoE grid generation."""

    def test_grid_size(self):
        """Grid has 3^5 = 243 combinations."""
        grid = generate_doe_grid()
        assert len(grid) == 243

    def test_grid_contains_default_config(self):
        """Grid includes the exact default vector/text config."""
        grid = generate_doe_grid()
        default = MixConfig()
        matches = [
            c
            for c in grid
            if c.vector_limit == default.vector_limit
            and c.text_limit == default.text_limit
            and c.vector_weight == default.vector_weight
            and c.text_weight == default.text_weight
            and c.dual_channel_bonus == default.dual_channel_bonus
        ]
        assert len(matches) == 1

    def test_grid_configs_are_unique(self):
        """All grid configs are distinct."""
        grid = generate_doe_grid()
        as_tuples = [tuple(asdict(c).values()) for c in grid]
        assert len(set(as_tuples)) == len(grid)

    def test_grid_preserves_non_swept_defaults(self):
        """Parameters not in the sweep retain their default values."""
        grid = generate_doe_grid()
        default = MixConfig()
        for config in grid:
            assert config.path_boost == default.path_boost
            assert config.terminal_boost == default.terminal_boost
            assert config.abbreviation_boost == default.abbreviation_boost
            assert config.hnsw_candidates == default.hnsw_candidates


class TestMetricComputation:
    """Verify precision, recall, and composite score helpers."""

    # ── Precision@k ───────────────────────────────────────────────────

    def test_precision_at_k_perfect(self):
        assert compute_precision_at_k(["a", "b", "c"], ["a", "b", "c"], k=3) == 1.0

    def test_precision_at_k_partial(self):
        result = compute_precision_at_k(["a", "x", "b", "y", "c"], ["a", "b", "c"], k=5)
        assert result == pytest.approx(0.6)

    def test_precision_at_k_none(self):
        assert compute_precision_at_k(["x", "y", "z"], ["a"], k=3) == 0.0

    def test_precision_at_k_empty_results(self):
        assert compute_precision_at_k([], ["a"], k=10) == 0.0

    def test_precision_at_k_fewer_results_than_k(self):
        """When fewer results than k, denominator is number of results."""
        assert compute_precision_at_k(["a", "b"], ["a", "b"], k=10) == pytest.approx(
            1.0
        )

    # ── IDS Recall@k ──────────────────────────────────────────────────

    def test_ids_recall_at_k_perfect(self):
        expected = [
            "core_profiles/profiles_1d/electrons/temperature",
            "equilibrium/time_slice/profiles_1d/q",
        ]
        results = [
            "core_profiles/profiles_1d/electrons/density",
            "equilibrium/time_slice/boundary/psi",
            "nbi/unit/power_launched",
        ]
        assert compute_ids_recall_at_k(results, expected, k=10) == 1.0

    def test_ids_recall_at_k_partial(self):
        expected = [
            "core_profiles/profiles_1d/electrons/temperature",
            "equilibrium/time_slice/profiles_1d/q",
            "summary/global_quantities/ip",
        ]
        results = ["core_profiles/profiles_1d/electrons/density"]
        # Only core_profiles found of {core_profiles, equilibrium, summary}
        assert compute_ids_recall_at_k(results, expected, k=10) == pytest.approx(1 / 3)

    def test_ids_recall_at_k_empty_results(self):
        assert compute_ids_recall_at_k([], ["core_profiles/x"], k=10) == 0.0

    def test_ids_recall_at_k_no_expected(self):
        assert compute_ids_recall_at_k(["a/b"], [], k=10) == 0.0

    # ── Composite score ───────────────────────────────────────────────

    def test_composite_score_perfect(self):
        assert compute_composite_score(1.0, 1.0, 1.0) == pytest.approx(1.0)

    def test_composite_score_weighted(self):
        score = compute_composite_score(0.8, 0.6, 0.4)
        assert score == pytest.approx(0.5 * 0.8 + 0.3 * 0.6 + 0.2 * 0.4)

    def test_composite_score_custom_weights(self):
        score = compute_composite_score(0.9, 0.7, 0.5, weights=(0.6, 0.3, 0.1))
        assert score == pytest.approx(0.6 * 0.9 + 0.3 * 0.7 + 0.1 * 0.5)

    def test_composite_score_zero(self):
        assert compute_composite_score(0.0, 0.0, 0.0) == 0.0


class TestScoreMixing:
    """Verify _apply_score_mixing logic with synthetic data."""

    def test_vector_only(self):
        """Vector-only results pass through unmodified."""
        config = MixConfig()
        scores = _apply_score_mixing(
            config,
            vector_results=[{"id": "ids/struct/leaf", "score": 0.9}],
            text_results=[],
            query="test",
        )
        assert "ids/struct/leaf" in scores
        assert scores["ids/struct/leaf"] == pytest.approx(0.9, abs=0.01)

    def test_text_only(self):
        """Text-only results pass through unmodified."""
        config = MixConfig()
        scores = _apply_score_mixing(
            config,
            vector_results=[],
            text_results=[{"id": "ids/struct/leaf", "score": 0.8}],
            query="test",
        )
        assert "ids/struct/leaf" in scores
        assert scores["ids/struct/leaf"] == pytest.approx(0.8, abs=0.01)

    def test_dual_channel_fusion(self):
        """Dual-channel hit gets weighted combination + bonus."""
        config = MixConfig(vector_weight=0.6, text_weight=0.4, dual_channel_bonus=0.05)
        scores = _apply_score_mixing(
            config,
            vector_results=[{"id": "ids/a/b", "score": 0.9}],
            text_results=[{"id": "ids/a/b", "score": 0.8}],
            query="test",
        )
        expected = round(0.6 * 0.9 + 0.4 * 0.8 + 0.05, 4)
        assert scores["ids/a/b"] == pytest.approx(expected, abs=0.001)

    def test_path_boost_rewards_segment_match(self):
        """Path segment matching boosts the more relevant result."""
        config = MixConfig(path_boost=0.10)
        scores = _apply_score_mixing(
            config,
            vector_results=[
                {
                    "id": "core_profiles/profiles_1d/electrons/temperature",
                    "score": 0.5,
                },
                {"id": "equilibrium/time_slice/boundary/psi", "score": 0.5},
            ],
            text_results=[],
            query="electron temperature",
        )
        # "electron" and "temperature" match two segments in core_profiles path
        assert (
            scores["core_profiles/profiles_1d/electrons/temperature"]
            > scores["equilibrium/time_slice/boundary/psi"]
        )

    def test_terminal_boost(self):
        """Exact terminal segment match gets terminal_boost bonus."""
        config = MixConfig(path_boost=0.0, terminal_boost=0.20)
        scores = _apply_score_mixing(
            config,
            vector_results=[
                {"id": "core_profiles/profiles_1d/q", "score": 0.5},
                {"id": "core_profiles/profiles_1d/electrons/density", "score": 0.5},
            ],
            text_results=[],
            query="q profile",
        )
        # "q" matches terminal of first path exactly
        assert (
            scores["core_profiles/profiles_1d/q"]
            > scores["core_profiles/profiles_1d/electrons/density"]
        )

    def test_abbreviation_boost(self):
        """Abbreviation boost lifts paths whose terminal matches the original query."""
        config = MixConfig(abbreviation_boost=0.35)
        scores = _apply_score_mixing(
            config,
            vector_results=[
                {"id": "equilibrium/time_slice/global_quantities/ip", "score": 0.5},
                {"id": "core_profiles/profiles_1d/j_bootstrap", "score": 0.5},
            ],
            text_results=[],
            query="plasma current ip",
            is_abbreviation=True,
            original_query="ip",
        )
        assert (
            scores["equilibrium/time_slice/global_quantities/ip"]
            > scores["core_profiles/profiles_1d/j_bootstrap"]
        )

    def test_weight_sensitivity(self):
        """Changing fusion weights changes the combined score."""
        vector_results = [{"id": "a/b/c", "score": 0.9}]
        text_results = [{"id": "a/b/c", "score": 0.3}]

        heavy_vector = MixConfig(vector_weight=0.9, text_weight=0.1)
        heavy_text = MixConfig(vector_weight=0.1, text_weight=0.9)

        score_v = _apply_score_mixing(heavy_vector, vector_results, text_results, "x")
        score_t = _apply_score_mixing(heavy_text, vector_results, text_results, "x")

        # High vector weight + high vector score should beat
        # high text weight + low text score
        assert score_v["a/b/c"] > score_t["a/b/c"]

    def test_empty_inputs(self):
        """Empty results produce empty scores."""
        config = MixConfig()
        scores = _apply_score_mixing(config, [], [], "test")
        assert scores == {}


class TestSaveDoEResults:
    """Verify DoE results serialization."""

    def test_save_creates_file(self, tmp_path):
        results = [
            {
                "mrr": 0.85,
                "precision_at_10": 0.70,
                "ids_recall_at_10": 0.60,
                "composite_score": 0.765,
                "config": asdict(MixConfig()),
                "per_category_mrr": {"exact_concept": 0.90},
                "per_query_results": [{"query": "test", "reciprocal_rank": 1.0}],
                "query_count": 1,
            }
        ]
        path = tmp_path / "test_results.json"
        saved = save_doe_results(results, path)
        assert saved.exists()

        data = json.loads(saved.read_text())
        assert len(data) == 1
        assert data[0]["mrr"] == 0.85

    def test_save_strips_per_query(self, tmp_path):
        """per_query_results are omitted to reduce file size."""
        results = [
            {
                "mrr": 0.85,
                "precision_at_10": 0.70,
                "ids_recall_at_10": 0.60,
                "composite_score": 0.765,
                "config": asdict(MixConfig()),
                "per_category_mrr": {},
                "per_query_results": [{"query": "q1"}, {"query": "q2"}],
                "query_count": 2,
            }
        ]
        path = tmp_path / "test_results.json"
        save_doe_results(results, path)

        data = json.loads(path.read_text())
        assert "per_query_results" not in data[0]
        assert "mrr" in data[0]
        assert "config" in data[0]


class TestParetoOptimal:
    """Verify Pareto optimality filter."""

    def test_single_result(self):
        results = [{"mrr": 0.8, "precision_at_10": 0.7, "ids_recall_at_10": 0.6}]
        assert len(find_pareto_optimal(results)) == 1

    def test_dominated_removed(self):
        results = [
            {"mrr": 0.9, "precision_at_10": 0.8, "ids_recall_at_10": 0.7},
            {"mrr": 0.8, "precision_at_10": 0.7, "ids_recall_at_10": 0.6},  # dominated
        ]
        pareto = find_pareto_optimal(results)
        assert len(pareto) == 1
        assert pareto[0]["mrr"] == 0.9

    def test_tradeoff_preserved(self):
        """Non-dominated trade-off configs are both kept."""
        results = [
            {"mrr": 0.9, "precision_at_10": 0.6, "ids_recall_at_10": 0.5},
            {"mrr": 0.7, "precision_at_10": 0.9, "ids_recall_at_10": 0.5},
        ]
        pareto = find_pareto_optimal(results)
        assert len(pareto) == 2

    def test_identical_not_dominated(self):
        """Identical configs are not considered dominated by each other."""
        results = [
            {"mrr": 0.8, "precision_at_10": 0.7, "ids_recall_at_10": 0.6},
            {"mrr": 0.8, "precision_at_10": 0.7, "ids_recall_at_10": 0.6},
        ]
        pareto = find_pareto_optimal(results)
        assert len(pareto) == 2

    def test_empty_input(self):
        assert find_pareto_optimal([]) == []


# ── Integration tests — require graph + embedding server ─────────────────────


@pytest.mark.graph
class TestDoEEvaluation:
    """Design of experiments for search score mixing.

    Requires a live Neo4j instance with IMAS DD data and a running
    embedding server producing 256-dim Qwen3 embeddings.
    """

    @pytest.fixture(scope="class")
    def graph_client(self):
        """Class-scoped GraphClient for DoE evaluation."""
        from imas_codex.graph.client import GraphClient

        try:
            client = GraphClient()
            client.get_stats()
        except Exception as e:
            pytest.skip(f"Neo4j not available: {e}")

        yield client
        client.close()

    @pytest.fixture(scope="class")
    def encoder(self):
        """Real remote encoder for DoE evaluation.

        Temporarily restores production embedding env vars to bypass
        the session-scoped conftest that forces local/MiniLM.
        """
        from imas_codex.settings import _get_section

        embed_config = _get_section("embedding")
        real_location = embed_config.get("location", "")
        real_model = embed_config.get("model", "")

        if not real_location or real_location == "local":
            pytest.skip("No remote embedding location configured")

        old_location = os.environ.get("IMAS_CODEX_EMBEDDING_LOCATION")
        old_model = os.environ.get("IMAS_CODEX_EMBEDDING_MODEL")
        try:
            os.environ["IMAS_CODEX_EMBEDDING_LOCATION"] = real_location
            if real_model:
                os.environ["IMAS_CODEX_EMBEDDING_MODEL"] = real_model
            elif "IMAS_CODEX_EMBEDDING_MODEL" in os.environ:
                del os.environ["IMAS_CODEX_EMBEDDING_MODEL"]

            from imas_codex.embeddings.encoder import Encoder, EncoderConfig

            config = EncoderConfig()
            enc = Encoder(config=config)
            result = enc.embed_texts(["test"])
            if result is None or len(result) == 0:
                pytest.skip("Embed server returned empty results")
            dim = len(result[0])
            if dim != 256:
                pytest.skip(f"Embed server returns {dim}-dim, expected 256")
            yield enc
        except pytest.skip.Exception:
            raise
        except Exception as e:
            pytest.skip(f"Embed server not available: {e}")
        finally:
            if old_location is not None:
                os.environ["IMAS_CODEX_EMBEDDING_LOCATION"] = old_location
            elif "IMAS_CODEX_EMBEDDING_LOCATION" in os.environ:
                del os.environ["IMAS_CODEX_EMBEDDING_LOCATION"]
            if old_model is not None:
                os.environ["IMAS_CODEX_EMBEDDING_MODEL"] = old_model
            elif "IMAS_CODEX_EMBEDDING_MODEL" in os.environ:
                del os.environ["IMAS_CODEX_EMBEDDING_MODEL"]

    def test_default_config_mrr(self, graph_client, encoder):
        """Default config meets MRR target."""
        metrics = evaluate_config(MixConfig(), graph_client, encoder)
        logger.info(
            "Default: MRR=%.3f P@10=%.3f IDS-R@10=%.3f composite=%.3f",
            metrics["mrr"],
            metrics["precision_at_10"],
            metrics["ids_recall_at_10"],
            metrics["composite_score"],
        )
        assert metrics["mrr"] >= 0.20, (
            f"Default MRR {metrics['mrr']:.3f} below 0.20 target"
        )

    def test_default_config_precision(self, graph_client, encoder):
        """Default config meets P@10 target."""
        metrics = evaluate_config(MixConfig(), graph_client, encoder)
        assert metrics["precision_at_10"] >= 0.05, (
            f"Default P@10 {metrics['precision_at_10']:.3f} below 0.05 target"
        )

    def test_per_category_coverage(self, graph_client, encoder):
        """Log per-category MRR for diagnostic purposes."""
        metrics = evaluate_config(MixConfig(), graph_client, encoder)
        cat_mrr = metrics["per_category_mrr"]
        for cat in CATEGORY_NAMES:
            if cat in cat_mrr:
                logger.info("  %s MRR: %.3f", cat, cat_mrr[cat])

    @pytest.mark.slow
    def test_doe_grid_search(self, graph_client, encoder):
        """Full grid search — produces JSON results for analysis.

        Run with: ``uv run pytest ... -k test_doe_grid -m slow``
        """
        results = run_doe_grid(graph_client, encoder)
        assert len(results) > 0

        # Best by composite score (already sorted)
        best = results[0]
        logger.info(
            "Best: MRR=%.3f P@10=%.3f IDS-R@10=%.3f composite=%.3f",
            best["mrr"],
            best["precision_at_10"],
            best["ids_recall_at_10"],
            best["composite_score"],
        )
        logger.info("Best config: %s", best["config"])

        pareto = find_pareto_optimal(results)
        logger.info("Pareto-optimal: %d / %d configs", len(pareto), len(results))

        save_doe_results(results)


class TestEmbedTextQuality:
    """Verify embed text generation includes path, abbreviations, and keywords.

    These are unit tests that validate generate_embedding_text() produces
    rich text suitable for embedding. They do NOT require a live graph.
    """

    def test_embed_text_contains_full_path(self):
        """Embed text should contain the full IMAS path."""
        from imas_codex.graph.build_dd import generate_embedding_text

        text = generate_embedding_text(
            "equilibrium/time_slice/global_quantities/ip",
            {
                "description": "Total plasma current",
                "documentation": "",
                "keywords": ["Ip"],
            },
        )
        assert "equilibrium/time_slice/global_quantities/ip" in text

    def test_embed_text_contains_keywords(self):
        """Embed text should contain keywords when provided."""
        from imas_codex.graph.build_dd import generate_embedding_text

        text = generate_embedding_text(
            "core_profiles/profiles_1d/electrons/temperature",
            {
                "description": "Electron temperature profile",
                "documentation": "",
                "keywords": ["Te", "thermal energy"],
            },
        )
        assert "Keywords:" in text
        assert "Te" in text

    def test_embed_text_includes_documentation(self):
        """Embed text should include documentation when different from description."""
        from imas_codex.graph.build_dd import generate_embedding_text

        text = generate_embedding_text(
            "equilibrium/time_slice/profiles_1d/psi",
            {
                "description": "Poloidal flux profile",
                "documentation": "The poloidal magnetic flux function used as the radial coordinate for 1D equilibrium profiles.",
                "keywords": ["psi", "flux"],
            },
        )
        assert (
            "poloidal magnetic flux function" in text.lower()
            or "Poloidal flux profile" in text
        )

    def test_embed_text_empty_returns_empty(self):
        """Empty description and documentation should return empty string."""
        from imas_codex.graph.build_dd import generate_embedding_text

        text = generate_embedding_text(
            "some/path",
            {"description": "", "documentation": "", "keywords": []},
        )
        assert text == ""

    def test_embed_text_caps_long_documentation(self):
        """Very long documentation should be capped at 300 chars."""
        from imas_codex.graph.build_dd import generate_embedding_text

        long_doc = "A" * 500
        text = generate_embedding_text(
            "some/path",
            {"description": "Short desc", "documentation": long_doc, "keywords": []},
        )
        # The doc portion should be capped
        assert "A" * 301 not in text
