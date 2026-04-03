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

    # ── Graph-native boosts ───────────────────────────────────────────
    cluster_boost: float = 0.02  # cluster membership boost
    hierarchy_boost: float = 0.02  # parent proximity boost
    coordinate_boost: float = 0.01  # coordinate sharing boost


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

    # --- Graph-native boost proxies ---
    # In production, these query the graph. In DoE evaluation, apply
    # a fixed additive boost to simulate their effect on ranking.
    # The actual boost magnitude is what we're tuning.
    if config.cluster_boost > 0:
        for pid in list(scores):
            # Proxy: paths with more segments (deeper in hierarchy) tend
            # to be cluster members. Apply boost proportionally.
            depth = pid.count("/")
            if depth >= 3:  # typical cluster member depth
                scores[pid] = round(scores[pid] + config.cluster_boost, 4)

    if config.hierarchy_boost > 0:
        for pid in list(scores):
            # Proxy: parent proximity — deeper paths near query terms
            segments = pid.lower().split("/")
            for w in query_words:
                if any(w in seg for seg in segments[:-1]):  # parent match
                    scores[pid] = round(scores[pid] + config.hierarchy_boost, 4)
                    break

    if config.coordinate_boost > 0:
        for pid in list(scores):
            # Proxy: coordinate-sharing paths tend to have known suffixes
            terminal = pid.rsplit("/", 1)[-1] if "/" in pid else pid
            if terminal in {"r", "z", "phi", "rho_tor_norm", "psi", "time"}:
                scores[pid] = round(scores[pid] + config.coordinate_boost, 4)

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
            CALL () {
              SEARCH path:IMASNode
              USING VECTOR INDEX imas_node_embedding
              WHERE path.node_category = 'data'
              WITH path, vector.similarity.cosine(path.embedding, $embedding) AS score
              ORDER BY score DESC
              LIMIT $k
            }
            WHERE NOT (path)-[:DEPRECATED_IN]->(:DDVersion)
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


def generate_doe_grid(two_phase: bool = True) -> list[MixConfig]:
    """Generate Design of Experiments configurations.

    When *two_phase* is ``True`` (default), generates a manageable grid:

    - Phase A: 243 configs sweeping fusion weights (as before)
    - Phase B: 27 configs sweeping graph boosts with optimal fusion defaults

    Total: up to 270 configs (vs 6561 for full factorial), with duplicates
    removed so configs that appear in both phases are not repeated.
    """
    configs: list[MixConfig] = []
    seen: set[tuple] = set()

    # Phase A: Fusion parameter sweep (original grid)
    fusion_grid = {
        "vector_limit": [200, 500, 800],
        "text_limit": [200, 500, 800],
        "vector_weight": [0.4, 0.6, 0.8],
        "text_weight": [0.2, 0.4, 0.6],
        "dual_channel_bonus": [0.0, 0.05, 0.10],
    }
    keys_a = list(fusion_grid.keys())
    for combo in itertools.product(*(fusion_grid[k] for k in keys_a)):
        kwargs = dict(zip(keys_a, combo, strict=True))
        config = MixConfig(**kwargs)
        key = tuple(sorted(asdict(config).items()))
        if key not in seen:
            seen.add(key)
            configs.append(config)

    if two_phase:
        # Phase B: Graph boost sweep with default fusion params
        boost_grid = {
            "cluster_boost": [0.0, 0.02, 0.05],
            "hierarchy_boost": [0.0, 0.02, 0.05],
            "coordinate_boost": [0.0, 0.01, 0.03],
        }
        keys_b = list(boost_grid.keys())
        for combo in itertools.product(*(boost_grid[k] for k in keys_b)):
            kwargs = dict(zip(keys_b, combo, strict=True))
            config = MixConfig(**kwargs)  # Uses default fusion params
            key = tuple(sorted(asdict(config).items()))
            if key not in seen:
                seen.add(key)
                configs.append(config)

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

    def test_graph_native_boost_defaults(self):
        """Graph-native boosts have non-zero production defaults."""
        config = MixConfig()
        assert config.cluster_boost == 0.02
        assert config.hierarchy_boost == 0.02
        assert config.coordinate_boost == 0.01

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
        """Grid has at least 243 fusion configs plus boost phase configs."""
        grid = generate_doe_grid()
        assert len(grid) >= 243  # At least the original fusion grid
        assert len(grid) <= 270  # Plus boost grid minus overlaps

    def test_grid_size_single_phase(self):
        """Single-phase grid (two_phase=False) has exactly 3^5 = 243 configs."""
        grid = generate_doe_grid(two_phase=False)
        assert len(grid) == 243

    def test_grid_contains_default_config(self):
        """Grid includes the exact full default config (fusion + boost defaults)."""
        grid = generate_doe_grid()
        default = MixConfig()
        exact_matches = [c for c in grid if c == default]
        assert len(exact_matches) == 1

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

    def test_grid_includes_boost_configs(self):
        """Grid includes configs with non-zero graph-native boosts."""
        grid = generate_doe_grid()
        boost_configs = [
            c
            for c in grid
            if c.cluster_boost > 0 or c.hierarchy_boost > 0 or c.coordinate_boost > 0
        ]
        assert len(boost_configs) > 0, "Grid should include non-zero boost configs"


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


@pytest.mark.graph
@pytest.mark.slow
class TestDimensionComparison:
    """Compare vector search quality across Matryoshka dimensions.

    Strategy: Re-embed a sample of graph nodes at each dimension,
    embed benchmark queries at each dimension, compute cosine similarity
    directly (bypassing the vector index), and report MRR.

    This requires:
    - A live Neo4j instance with IMAS DD data
    - A running embedding server that supports Matryoshka dimensions
    """

    SAMPLE_SIZE = 100
    DIMENSIONS = [256, 512, 1024]

    @pytest.fixture(scope="class")
    def graph_client(self):
        """Class-scoped GraphClient."""
        from imas_codex.graph.client import GraphClient

        try:
            client = GraphClient()
            client.get_stats()
        except Exception as e:
            pytest.skip(f"Neo4j not available: {e}")
        yield client
        client.close()

    @pytest.fixture(scope="class")
    def sample_nodes(self, graph_client):
        """Sample representative nodes from the graph for re-embedding.

        Selects nodes that appear in benchmark expected paths plus
        random high-quality nodes to ensure coverage.
        """
        # Collect expected paths from benchmarks
        expected_ids = set()
        for q in ALL_QUERIES:
            expected_ids.update(q.expected_paths)

        # Fetch nodes with their embed text
        expected_nodes = graph_client.query(
            """
            UNWIND $ids AS pid
            MATCH (n:IMASNode {id: pid})
            WHERE n.description IS NOT NULL
            RETURN n.id AS id, n.description AS description
            """,
            ids=list(expected_ids),
        )

        # Fill remaining sample with random enriched nodes
        remaining = self.SAMPLE_SIZE - len(expected_nodes or [])
        if remaining > 0:
            random_nodes = graph_client.query(
                """
                MATCH (n:IMASNode)
                WHERE n.description IS NOT NULL
                  AND n.node_category = 'data'
                  AND n.embedding IS NOT NULL
                WITH n, rand() AS r
                ORDER BY r
                LIMIT $limit
                RETURN n.id AS id, n.description AS description
                """,
                limit=remaining,
            )
            all_nodes = (expected_nodes or []) + (random_nodes or [])
        else:
            all_nodes = expected_nodes or []

        if len(all_nodes) < 10:
            pytest.skip(f"Only {len(all_nodes)} sample nodes available")

        return all_nodes[: self.SAMPLE_SIZE]

    @pytest.fixture(scope="class")
    def dimension_embeddings(self, sample_nodes):
        """Embed sample nodes and benchmark queries at each Matryoshka dimension.

        Returns dict[int, dict] with keys:
            - 'node_embeddings': dict[str, list[float]] for sample nodes
            - 'query_embeddings': dict[str, list[float]] for benchmark queries
        """
        from imas_codex.embeddings.encoder import Encoder, EncoderConfig
        from imas_codex.settings import _get_section

        embed_config = _get_section("embedding")
        real_location = embed_config.get("location", "")
        real_model = embed_config.get("model", "")

        if not real_location or real_location == "local":
            pytest.skip("No remote embedding location configured")

        old_location = os.environ.get("IMAS_CODEX_EMBEDDING_LOCATION")
        old_model = os.environ.get("IMAS_CODEX_EMBEDDING_MODEL")

        results = {}
        try:
            os.environ["IMAS_CODEX_EMBEDDING_LOCATION"] = real_location
            if real_model:
                os.environ["IMAS_CODEX_EMBEDDING_MODEL"] = real_model

            node_texts = [f"{n['id']} {n['description']}" for n in sample_nodes]
            query_texts = [q.query_text for q in ALL_QUERIES]

            for dim in self.DIMENSIONS:
                os.environ["IMAS_CODEX_EMBEDDING_DIMENSION"] = str(dim)
                config = EncoderConfig()
                enc = Encoder(config=config)

                # Embed nodes (document mode)
                node_vecs = enc.embed_texts(node_texts, prompt_name="passage")
                if node_vecs is None or len(node_vecs) == 0:
                    pytest.skip(f"Embed server returned empty for dim {dim}")

                actual_dim = len(node_vecs[0])
                if actual_dim != dim:
                    logger.warning(
                        "Requested dim %d but got %d — using actual",
                        dim,
                        actual_dim,
                    )

                node_embs = {
                    sample_nodes[i]["id"]: node_vecs[i].tolist()
                    for i in range(len(sample_nodes))
                }

                # Embed queries (query mode)
                query_vecs = enc.embed_texts(query_texts, prompt_name="query")
                query_embs = {
                    ALL_QUERIES[i].query_text: query_vecs[i].tolist()
                    for i in range(len(ALL_QUERIES))
                }

                results[dim] = {
                    "node_embeddings": node_embs,
                    "query_embeddings": query_embs,
                    "actual_dim": actual_dim,
                }

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
            if "IMAS_CODEX_EMBEDDING_DIMENSION" in os.environ:
                del os.environ["IMAS_CODEX_EMBEDDING_DIMENSION"]

        return results

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import math

        dot = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _vector_search(
        self,
        query_embedding: list[float],
        node_embeddings: dict[str, list[float]],
        limit: int = 50,
    ) -> list[str]:
        """Rank nodes by cosine similarity to query embedding."""
        similarities = [
            (nid, self._cosine_similarity(query_embedding, emb))
            for nid, emb in node_embeddings.items()
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [nid for nid, _ in similarities[:limit]]

    @pytest.mark.parametrize("dim", [256, 512, 1024])
    def test_vector_mrr_by_dimension(self, dimension_embeddings, dim):
        """Measure raw vector MRR at a specific Matryoshka dimension."""
        if dim not in dimension_embeddings:
            pytest.skip(f"Dimension {dim} embeddings not available")

        dim_data = dimension_embeddings[dim]
        node_embs = dim_data["node_embeddings"]
        query_embs = dim_data["query_embeddings"]

        rrs = []
        for q in ALL_QUERIES:
            if q.query_text not in query_embs:
                continue
            ranked = self._vector_search(query_embs[q.query_text], node_embs)
            expected = set(q.expected_paths)
            rr = 0.0
            for rank, pid in enumerate(ranked, start=1):
                if pid in expected:
                    rr = 1.0 / rank
                    break
            rrs.append(rr)

        mrr = sum(rrs) / len(rrs) if rrs else 0.0
        actual_dim = dim_data.get("actual_dim", dim)
        logger.info(
            "Vector MRR at dim %d (actual %d): %.3f (%d queries)",
            dim,
            actual_dim,
            mrr,
            len(rrs),
        )

        # No hard assertion — this is diagnostic
        # Decision criteria from plan:
        # Upgrade to 512 if vector MRR improves by >=0.10 (from ~0.15 to >=0.25)
        # Upgrade to 1024 only if additional gain over 512 is >=0.05

    def test_dimension_comparison_summary(self, dimension_embeddings):
        """Compare MRR across all dimensions and log recommendation."""
        mrr_by_dim = {}
        for dim in self.DIMENSIONS:
            if dim not in dimension_embeddings:
                continue
            dim_data = dimension_embeddings[dim]
            node_embs = dim_data["node_embeddings"]
            query_embs = dim_data["query_embeddings"]

            rrs = []
            for q in ALL_QUERIES:
                if q.query_text not in query_embs:
                    continue
                ranked = self._vector_search(query_embs[q.query_text], node_embs)
                expected = set(q.expected_paths)
                rr = 0.0
                for rank, pid in enumerate(ranked, start=1):
                    if pid in expected:
                        rr = 1.0 / rank
                        break
                rrs.append(rr)

            mrr = sum(rrs) / len(rrs) if rrs else 0.0
            mrr_by_dim[dim] = mrr

        logger.info("=== Dimension Comparison Summary ===")
        for dim, mrr in sorted(mrr_by_dim.items()):
            logger.info("  dim %4d: MRR = %.3f", dim, mrr)

        if 256 in mrr_by_dim and 512 in mrr_by_dim:
            delta_512 = mrr_by_dim[512] - mrr_by_dim[256]
            logger.info("  Δ(512 vs 256): %.3f", delta_512)
            if delta_512 >= 0.10:
                logger.info("  → RECOMMEND upgrade to dim 512")
            else:
                logger.info("  → Keep dim 256 (512 gain < 0.10)")

        if 512 in mrr_by_dim and 1024 in mrr_by_dim:
            delta_1024 = mrr_by_dim[1024] - mrr_by_dim[512]
            logger.info("  Δ(1024 vs 512): %.3f", delta_1024)
            if delta_1024 >= 0.05:
                logger.info("  → RECOMMEND upgrade to dim 1024")
            else:
                logger.info("  → Keep dim 512 (1024 gain < 0.05)")


# ── Cross-domain, cross-lingual dimension evaluation ──────────────────────────


# Source documents for multilingual evaluation.
# Each entry: (query_lang, query_text, expected_doc_lang, doc_keyword)
# doc_keyword is a substring that should appear in the correct result.
CROSS_LINGUAL_QUERIES: list[dict[str, str]] = [
    # English queries → English documents
    {
        "query_lang": "en",
        "query": "electron temperature profile",
        "doc_lang": "en",
        "domain": "imas_dd",
    },
    {
        "query_lang": "en",
        "query": "plasma current measurement",
        "doc_lang": "en",
        "domain": "imas_dd",
    },
    {
        "query_lang": "en",
        "query": "equilibrium reconstruction code",
        "doc_lang": "en",
        "domain": "code",
    },
    {
        "query_lang": "en",
        "query": "safety factor q profile",
        "doc_lang": "en",
        "domain": "imas_dd",
    },
    {
        "query_lang": "en",
        "query": "Thomson scattering calibration",
        "doc_lang": "en",
        "domain": "wiki",
    },
    {
        "query_lang": "en",
        "query": "COCOS sign conventions",
        "doc_lang": "en",
        "domain": "wiki",
    },
    {
        "query_lang": "en",
        "query": "magnetic field measurement",
        "doc_lang": "en",
        "domain": "signal",
    },
    {
        "query_lang": "en",
        "query": "poloidal flux function",
        "doc_lang": "en",
        "domain": "imas_dd",
    },
    # English queries → Japanese documents (cross-lingual retrieval)
    {
        "query_lang": "en",
        "query": "plasma diagnostics system",
        "doc_lang": "ja",
        "domain": "wiki",
    },
    {
        "query_lang": "en",
        "query": "experiment database",
        "doc_lang": "ja",
        "domain": "wiki",
    },
    {
        "query_lang": "en",
        "query": "data acquisition system",
        "doc_lang": "ja",
        "domain": "wiki",
    },
    {
        "query_lang": "en",
        "query": "vacuum vessel structure",
        "doc_lang": "ja",
        "domain": "wiki",
    },
    # Japanese queries → Japanese documents
    {
        "query_lang": "ja",
        "query": "プラズマ電流測定",
        "doc_lang": "ja",
        "domain": "wiki",
    },
    {
        "query_lang": "ja",
        "query": "電子温度プロファイル",
        "doc_lang": "ja",
        "domain": "wiki",
    },
    {
        "query_lang": "ja",
        "query": "磁気計測システム",
        "doc_lang": "ja",
        "domain": "wiki",
    },
    {
        "query_lang": "ja",
        "query": "実験データベース",
        "doc_lang": "ja",
        "domain": "wiki",
    },
    {
        "query_lang": "ja",
        "query": "トムソン散乱計測",
        "doc_lang": "ja",
        "domain": "wiki",
    },
    # Japanese queries → English documents (cross-lingual retrieval)
    {
        "query_lang": "ja",
        "query": "プラズマ電流",
        "doc_lang": "en",
        "domain": "imas_dd",
    },
    {
        "query_lang": "ja",
        "query": "電子密度プロファイル",
        "doc_lang": "en",
        "domain": "imas_dd",
    },
    {"query_lang": "ja", "query": "安全係数", "doc_lang": "en", "domain": "imas_dd"},
    {"query_lang": "ja", "query": "磁場測定", "doc_lang": "en", "domain": "signal"},
    # French queries → English documents (cross-lingual retrieval)
    {
        "query_lang": "fr",
        "query": "température électronique du plasma",
        "doc_lang": "en",
        "domain": "imas_dd",
    },
    {
        "query_lang": "fr",
        "query": "profil de densité électronique",
        "doc_lang": "en",
        "domain": "imas_dd",
    },
    {
        "query_lang": "fr",
        "query": "courant plasma mesuré",
        "doc_lang": "en",
        "domain": "imas_dd",
    },
    {
        "query_lang": "fr",
        "query": "reconstruction d'équilibre",
        "doc_lang": "en",
        "domain": "code",
    },
    {
        "query_lang": "fr",
        "query": "facteur de sécurité",
        "doc_lang": "en",
        "domain": "imas_dd",
    },
    # French queries → French documents (in-language)
    {
        "query_lang": "fr",
        "query": "système de diagnostic plasma",
        "doc_lang": "fr",
        "domain": "wiki",
    },
    {
        "query_lang": "fr",
        "query": "acquisition de données",
        "doc_lang": "fr",
        "domain": "wiki",
    },
    # English queries → French documents (cross-lingual)
    {
        "query_lang": "en",
        "query": "plasma diagnostic system",
        "doc_lang": "fr",
        "domain": "wiki",
    },
    {
        "query_lang": "en",
        "query": "data acquisition procedure",
        "doc_lang": "fr",
        "domain": "wiki",
    },
    # Abbreviation queries (stress test — same across languages)
    {"query_lang": "en", "query": "Ip", "doc_lang": "en", "domain": "abbreviation"},
    {"query_lang": "en", "query": "Te", "doc_lang": "en", "domain": "abbreviation"},
    {
        "query_lang": "en",
        "query": "ne profile",
        "doc_lang": "en",
        "domain": "abbreviation",
    },
    {"query_lang": "en", "query": "q95", "doc_lang": "en", "domain": "abbreviation"},
    {"query_lang": "en", "query": "Zeff", "doc_lang": "en", "domain": "abbreviation"},
]

# All language pair combinations for the cross-product matrix
LANGUAGE_PAIRS = [
    ("en", "en"),
    ("en", "ja"),
    ("en", "fr"),
    ("ja", "en"),
    ("ja", "ja"),
    ("fr", "en"),
    ("fr", "fr"),
]

# Content domain categories for per-domain MRR breakdown
CONTENT_DOMAINS = ["imas_dd", "wiki", "code", "signal", "abbreviation"]


@pytest.mark.graph
@pytest.mark.slow
class TestMultiDomainDimensionEval:
    """Cross-domain, cross-lingual dimension evaluation.

    Extends TestDimensionComparison with:
    - Multi-domain corpus (IMAS DD, wiki chunks, code, signals)
    - Multi-language coverage (English, Japanese, French)
    - Cross-lingual query evaluation (EN→JA, JA→EN, FR→EN, etc.)
    - Per-domain and per-language MRR breakdown
    - Language pair cross-product matrix

    Requires live Neo4j + embedding server.
    """

    DIMENSIONS = [256, 512, 1024]

    @pytest.fixture(scope="class")
    def graph_client(self):
        """Class-scoped GraphClient."""
        from imas_codex.graph.client import GraphClient

        try:
            client = GraphClient()
            client.get_stats()
        except Exception as e:
            pytest.skip(f"Neo4j not available: {e}")
        yield client
        client.close()

    @pytest.fixture(scope="class")
    def multi_domain_corpus(self, graph_client):
        """Sample representative content across all domains and languages.

        Returns dict with keys per domain, each a list of
        {id, text, lang, domain} dicts.
        """
        corpus: dict[str, list[dict[str, str]]] = defaultdict(list)

        # IMAS DD nodes (English, from enriched descriptions)
        imas_nodes = graph_client.query("""
            MATCH (n:IMASNode)
            WHERE n.description IS NOT NULL
              AND n.node_category = 'data'
              AND n.embedding IS NOT NULL
            WITH n, rand() AS r ORDER BY r LIMIT 50
            RETURN n.id AS id,
                   (n.id + '. ' + n.description) AS text
        """)
        for n in imas_nodes or []:
            corpus["imas_dd"].append(
                {"id": n["id"], "text": n["text"], "lang": "en", "domain": "imas_dd"}
            )

        # English wiki chunks
        en_wiki = graph_client.query("""
            MATCH (p:WikiPage)-[:HAS_CHUNK]->(c:WikiChunk)
            WHERE c.text IS NOT NULL AND c.embedding IS NOT NULL
              AND (p.content_language IS NULL OR p.content_language = 'en')
            WITH c, rand() AS r ORDER BY r LIMIT 30
            RETURN c.id AS id, substring(c.text, 0, 500) AS text
        """)
        for w in en_wiki or []:
            corpus["wiki"].append(
                {"id": w["id"], "text": w["text"], "lang": "en", "domain": "wiki"}
            )

        # Japanese wiki chunks (JT-60SA)
        ja_wiki = graph_client.query("""
            MATCH (p:WikiPage {content_language: 'ja'})-[:HAS_CHUNK]->(c:WikiChunk)
            WHERE c.text IS NOT NULL AND c.embedding IS NOT NULL
            WITH c, rand() AS r ORDER BY r LIMIT 30
            RETURN c.id AS id, substring(c.text, 0, 500) AS text
        """)
        for w in ja_wiki or []:
            corpus["wiki"].append(
                {"id": w["id"], "text": w["text"], "lang": "ja", "domain": "wiki"}
            )

        # French-containing wiki chunks (TCV/ITER — detected by character patterns)
        fr_wiki = graph_client.query("""
            MATCH (c:WikiChunk)
            WHERE c.text IS NOT NULL AND c.embedding IS NOT NULL
              AND c.text =~ '.*[àâéèêëîïôùûüç].*'
            WITH c, rand() AS r ORDER BY r LIMIT 15
            RETURN c.id AS id, substring(c.text, 0, 500) AS text
        """)
        for w in fr_wiki or []:
            corpus["wiki"].append(
                {"id": w["id"], "text": w["text"], "lang": "fr", "domain": "wiki"}
            )

        # Code chunks (English — code is almost always English)
        code_chunks = graph_client.query("""
            MATCH (cc:CodeChunk)
            WHERE cc.text IS NOT NULL AND cc.embedding IS NOT NULL
            WITH cc, rand() AS r ORDER BY r LIMIT 20
            RETURN cc.id AS id, substring(cc.text, 0, 500) AS text
        """)
        for c in code_chunks or []:
            corpus["code"].append(
                {"id": c["id"], "text": c["text"], "lang": "en", "domain": "code"}
            )

        # Signal descriptions (English)
        signals = graph_client.query("""
            MATCH (s:FacilitySignal)
            WHERE s.description IS NOT NULL AND s.embedding IS NOT NULL
            WITH s, rand() AS r ORDER BY r LIMIT 20
            RETURN s.id AS id, s.description AS text
        """)
        for s in signals or []:
            corpus["signal"].append(
                {"id": s["id"], "text": s["text"], "lang": "en", "domain": "signal"}
            )

        total = sum(len(v) for v in corpus.values())
        if total < 20:
            pytest.skip(f"Only {total} corpus docs available, need ≥20")

        lang_counts = defaultdict(int)
        for docs in corpus.values():
            for d in docs:
                lang_counts[d["lang"]] += 1
        logger.info(
            "Multi-domain corpus: %d docs — %s",
            total,
            ", ".join(f"{lang}={cnt}" for lang, cnt in sorted(lang_counts.items())),
        )

        return dict(corpus)

    @pytest.fixture(scope="class")
    def multilingual_embeddings(self, multi_domain_corpus):
        """Embed corpus and queries at each dimension.

        Returns dict[int, dict] with:
            - 'doc_embeddings': dict[doc_id, list[float]]
            - 'doc_metadata': dict[doc_id, {lang, domain}]
            - 'query_embeddings': dict[query_text, list[float]]
            - 'actual_dim': int
        """
        from imas_codex.embeddings.encoder import Encoder, EncoderConfig
        from imas_codex.settings import _get_section

        embed_config = _get_section("embedding")
        real_location = embed_config.get("location", "")
        real_model = embed_config.get("model", "")

        if not real_location or real_location == "local":
            pytest.skip("No remote embedding location configured")

        old_env = {
            k: os.environ.get(k)
            for k in [
                "IMAS_CODEX_EMBEDDING_LOCATION",
                "IMAS_CODEX_EMBEDDING_MODEL",
                "IMAS_CODEX_EMBEDDING_DIMENSION",
            ]
        }

        # Flatten corpus
        all_docs: list[dict[str, str]] = []
        for docs in multi_domain_corpus.values():
            all_docs.extend(docs)
        doc_texts = [d["text"] for d in all_docs]
        doc_ids = [d["id"] for d in all_docs]
        doc_metadata = {
            d["id"]: {"lang": d["lang"], "domain": d["domain"]} for d in all_docs
        }

        # Collect all query texts
        query_texts = [q["query"] for q in CROSS_LINGUAL_QUERIES]

        results = {}
        try:
            os.environ["IMAS_CODEX_EMBEDDING_LOCATION"] = real_location
            if real_model:
                os.environ["IMAS_CODEX_EMBEDDING_MODEL"] = real_model

            for dim in self.DIMENSIONS:
                os.environ["IMAS_CODEX_EMBEDDING_DIMENSION"] = str(dim)
                config = EncoderConfig()
                enc = Encoder(config=config)

                # Embed documents (passage mode)
                doc_vecs = enc.embed_texts(doc_texts, prompt_name="passage")
                if doc_vecs is None or len(doc_vecs) == 0:
                    pytest.skip(f"Embed server returned empty for dim {dim}")

                actual_dim = len(doc_vecs[0])
                if actual_dim != dim:
                    logger.warning(
                        "Requested dim %d but got %d — server may not support Matryoshka",
                        dim,
                        actual_dim,
                    )

                doc_embs = {
                    doc_ids[i]: doc_vecs[i].tolist() for i in range(len(doc_ids))
                }

                # Embed queries (query mode)
                query_vecs = enc.embed_texts(query_texts, prompt_name="query")
                query_embs = {
                    query_texts[i]: query_vecs[i].tolist()
                    for i in range(len(query_texts))
                }

                results[dim] = {
                    "doc_embeddings": doc_embs,
                    "doc_metadata": doc_metadata,
                    "query_embeddings": query_embs,
                    "actual_dim": actual_dim,
                }

        except pytest.skip.Exception:
            raise
        except Exception as e:
            pytest.skip(f"Embed server not available: {e}")
        finally:
            for key, val in old_env.items():
                if val is not None:
                    os.environ[key] = val
                elif key in os.environ:
                    del os.environ[key]

        return results

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two vectors."""
        import math

        dot = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _rank_docs(
        self,
        query_emb: list[float],
        doc_embeddings: dict[str, list[float]],
        doc_metadata: dict[str, dict[str, str]],
        *,
        filter_lang: str | None = None,
        filter_domain: str | None = None,
        limit: int = 50,
    ) -> list[tuple[str, float]]:
        """Rank documents by cosine similarity, optionally filtering."""
        candidates = []
        for doc_id, emb in doc_embeddings.items():
            meta = doc_metadata.get(doc_id, {})
            if filter_lang and meta.get("lang") != filter_lang:
                continue
            if filter_domain and meta.get("domain") != filter_domain:
                continue
            sim = self._cosine_similarity(query_emb, emb)
            candidates.append((doc_id, sim))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:limit]

    @pytest.mark.parametrize("dim", [256, 512, 1024])
    def test_per_domain_mrr(self, multilingual_embeddings, dim):
        """MRR breakdown by content domain at each dimension."""
        if dim not in multilingual_embeddings:
            pytest.skip(f"Dimension {dim} not available")

        data = multilingual_embeddings[dim]
        doc_embs = data["doc_embeddings"]
        doc_meta = data["doc_metadata"]
        query_embs = data["query_embeddings"]

        domain_rrs: dict[str, list[float]] = defaultdict(list)

        for q_info in CROSS_LINGUAL_QUERIES:
            q_text = q_info["query"]
            domain = q_info["domain"]
            if q_text not in query_embs:
                continue

            ranked = self._rank_docs(
                query_embs[q_text],
                doc_embs,
                doc_meta,
                filter_domain=domain if domain != "abbreviation" else None,
            )
            # For this eval, RR = 1/rank of first result with matching language
            target_lang = q_info["doc_lang"]
            rr = 0.0
            for rank, (doc_id, _score) in enumerate(ranked, start=1):
                if doc_meta.get(doc_id, {}).get("lang") == target_lang:
                    rr = 1.0 / rank
                    break
            domain_rrs[domain].append(rr)

        logger.info("=== Per-Domain MRR at dim %d ===", dim)
        for domain in CONTENT_DOMAINS:
            rrs = domain_rrs.get(domain, [])
            mrr = sum(rrs) / len(rrs) if rrs else 0.0
            logger.info("  %-14s: MRR = %.3f  (%d queries)", domain, mrr, len(rrs))

    @pytest.mark.parametrize("dim", [256, 512, 1024])
    def test_cross_lingual_mrr_matrix(self, multilingual_embeddings, dim):
        """Cross-lingual MRR matrix: query_lang × doc_lang at each dimension.

        Evaluates the cross-product of all language pairs to measure
        how well the model handles cross-lingual retrieval at each
        embedding dimension.
        """
        if dim not in multilingual_embeddings:
            pytest.skip(f"Dimension {dim} not available")

        data = multilingual_embeddings[dim]
        doc_embs = data["doc_embeddings"]
        doc_meta = data["doc_metadata"]
        query_embs = data["query_embeddings"]

        pair_rrs: dict[tuple[str, str], list[float]] = defaultdict(list)

        for q_info in CROSS_LINGUAL_QUERIES:
            q_text = q_info["query"]
            q_lang = q_info["query_lang"]
            doc_lang = q_info["doc_lang"]
            if q_text not in query_embs:
                continue

            ranked = self._rank_docs(query_embs[q_text], doc_embs, doc_meta)
            rr = 0.0
            for rank, (doc_id, _score) in enumerate(ranked, start=1):
                if doc_meta.get(doc_id, {}).get("lang") == doc_lang:
                    rr = 1.0 / rank
                    break
            pair_rrs[(q_lang, doc_lang)].append(rr)

        logger.info("=== Cross-Lingual MRR Matrix at dim %d ===", dim)
        header = "query\\doc  " + "  ".join(f"{lang:>6}" for lang in ["en", "ja", "fr"])
        logger.info("  %s", header)
        for q_lang in ["en", "ja", "fr"]:
            row_parts = []
            for d_lang in ["en", "ja", "fr"]:
                rrs = pair_rrs.get((q_lang, d_lang), [])
                mrr = sum(rrs) / len(rrs) if rrs else float("nan")
                row_parts.append(f"{mrr:6.3f}" if rrs else "   N/A")
            logger.info("  %-9s %s", q_lang, "  ".join(row_parts))

    def test_multilingual_dimension_summary(self, multilingual_embeddings):
        """Cross-domain + cross-lingual dimension comparison with evidence.

        Reports overall MRR, per-domain MRR deltas, and cross-lingual
        MRR deltas across dimensions. Provides an evidence-based
        recommendation for the target embedding dimension.
        """
        overall_mrr: dict[int, float] = {}
        domain_mrr: dict[int, dict[str, float]] = {}
        xlingual_mrr: dict[int, dict[str, float]] = {}

        for dim in self.DIMENSIONS:
            if dim not in multilingual_embeddings:
                continue

            data = multilingual_embeddings[dim]
            doc_embs = data["doc_embeddings"]
            doc_meta = data["doc_metadata"]
            query_embs = data["query_embeddings"]

            all_rrs: list[float] = []
            domain_rrs: dict[str, list[float]] = defaultdict(list)
            in_lang_rrs: list[float] = []
            cross_lang_rrs: list[float] = []

            for q_info in CROSS_LINGUAL_QUERIES:
                q_text = q_info["query"]
                q_lang = q_info["query_lang"]
                doc_lang = q_info["doc_lang"]
                domain = q_info["domain"]
                if q_text not in query_embs:
                    continue

                ranked = self._rank_docs(query_embs[q_text], doc_embs, doc_meta)
                rr = 0.0
                for rank, (doc_id, _score) in enumerate(ranked, start=1):
                    if doc_meta.get(doc_id, {}).get("lang") == doc_lang:
                        rr = 1.0 / rank
                        break

                all_rrs.append(rr)
                domain_rrs[domain].append(rr)
                if q_lang == doc_lang:
                    in_lang_rrs.append(rr)
                else:
                    cross_lang_rrs.append(rr)

            overall_mrr[dim] = sum(all_rrs) / len(all_rrs) if all_rrs else 0.0
            domain_mrr[dim] = {
                d: sum(rrs) / len(rrs) if rrs else 0.0 for d, rrs in domain_rrs.items()
            }
            xlingual_mrr[dim] = {
                "in_language": sum(in_lang_rrs) / len(in_lang_rrs)
                if in_lang_rrs
                else 0.0,
                "cross_lingual": sum(cross_lang_rrs) / len(cross_lang_rrs)
                if cross_lang_rrs
                else 0.0,
            }

        # Log comprehensive results
        logger.info("=" * 70)
        logger.info("MULTI-DOMAIN DIMENSION EVALUATION SUMMARY")
        logger.info("=" * 70)

        logger.info("\n  Overall MRR:")
        for dim in sorted(overall_mrr):
            logger.info("    dim %4d: %.3f", dim, overall_mrr[dim])

        logger.info("\n  Per-Domain MRR:")
        for domain in CONTENT_DOMAINS:
            parts = []
            for dim in sorted(domain_mrr):
                val = domain_mrr[dim].get(domain, 0.0)
                parts.append(f"{dim}={val:.3f}")
            logger.info("    %-14s: %s", domain, "  ".join(parts))

        logger.info("\n  Cross-Lingual MRR:")
        for dim in sorted(xlingual_mrr):
            xl = xlingual_mrr[dim]
            logger.info(
                "    dim %4d: in-lang=%.3f  cross-lang=%.3f",
                dim,
                xl["in_language"],
                xl["cross_lingual"],
            )

        # Decision logic
        logger.info("\n  === DIMENSION DECISION ===")
        dims_sorted = sorted(overall_mrr.keys())
        if len(dims_sorted) < 2:
            logger.info("  Insufficient dimensions to compare")
            return

        base_dim = dims_sorted[0]
        recommendation = base_dim

        for dim in dims_sorted[1:]:
            delta = overall_mrr[dim] - overall_mrr[base_dim]
            logger.info(
                "  Δ(%d vs %d) overall: %.3f",
                dim,
                base_dim,
                delta,
            )

            # Check cross-lingual improvement
            xl_base = xlingual_mrr.get(base_dim, {}).get("cross_lingual", 0.0)
            xl_dim = xlingual_mrr.get(dim, {}).get("cross_lingual", 0.0)
            xl_delta = xl_dim - xl_base
            logger.info(
                "  Δ(%d vs %d) cross-lingual: %.3f",
                dim,
                base_dim,
                xl_delta,
            )

            # Decision criteria:
            # - Overall MRR gain ≥ 0.05 across domains → upgrade
            # - Cross-lingual MRR gain ≥ 0.03 → upgrade (multilingual is
            #   disproportionately helped by higher dims)
            # - Any domain shows MRR gain ≥ 0.08 → upgrade
            domain_max_delta = 0.0
            for domain in CONTENT_DOMAINS:
                d_base = domain_mrr.get(base_dim, {}).get(domain, 0.0)
                d_dim = domain_mrr.get(dim, {}).get(domain, 0.0)
                d_delta = d_dim - d_base
                if d_delta > domain_max_delta:
                    domain_max_delta = d_delta

            if delta >= 0.05 or xl_delta >= 0.03 or domain_max_delta >= 0.08:
                recommendation = dim
                reason = []
                if delta >= 0.05:
                    reason.append(f"overall Δ={delta:.3f}≥0.05")
                if xl_delta >= 0.03:
                    reason.append(f"cross-lingual Δ={xl_delta:.3f}≥0.03")
                if domain_max_delta >= 0.08:
                    reason.append(f"domain max Δ={domain_max_delta:.3f}≥0.08")
                logger.info(
                    "  → EVIDENCE supports dim %d: %s",
                    dim,
                    ", ".join(reason),
                )
            else:
                logger.info(
                    "  → Insufficient evidence for dim %d (Δ=%.3f<0.05, "
                    "xl_Δ=%.3f<0.03, domain_max_Δ=%.3f<0.08)",
                    dim,
                    delta,
                    xl_delta,
                    domain_max_delta,
                )

        logger.info("\n  RECOMMENDATION: dimension = %d", recommendation)
        logger.info(
            "  (quantization makes %d-dim storage equivalent to %d-dim unquantized)",
            recommendation,
            recommendation // 4,
        )
        logger.info("=" * 70)

    def test_dimension_validates_actual_output(self, multilingual_embeddings):
        """Verify the embedding server respects the dimension request.

        After Phase 0 bug fix, the actual dimension should match the
        requested dimension. If not, the bug fix is incomplete.
        """
        for dim in self.DIMENSIONS:
            if dim not in multilingual_embeddings:
                continue
            actual = multilingual_embeddings[dim]["actual_dim"]
            assert actual == dim, (
                f"Bug not fixed: requested dim {dim} but got {actual}. "
                f"Check that RemoteEmbeddingClient sends dimension in request body."
            )


class TestEmbedTextQuality:
    """Verify embed text generation produces dimension-appropriate format.

    At dim < 512, embed text is kept concise (path + description only).
    At dim ≥ 512, doc excerpts and keywords are included for richer context.
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

    def test_embed_text_at_dim_256(self, monkeypatch):
        """At dim 256, text is path + description only."""
        monkeypatch.setenv("IMAS_CODEX_EMBEDDING_DIMENSION", "256")
        from imas_codex.graph.build_dd import generate_embedding_text

        text = generate_embedding_text(
            "core_profiles/profiles_1d/electrons/temperature",
            {
                "description": "Electron temperature (Te) profile",
                "documentation": "Long documentation text here.",
                "keywords": ["Te", "thermal energy"],
            },
        )
        assert (
            text
            == "core_profiles/profiles_1d/electrons/temperature. Electron temperature (Te) profile"
        )
        # Keywords and doc are excluded at dim 256
        assert "Long documentation" not in text
        assert "thermal energy" not in text

    def test_embed_text_at_dim_512_includes_doc(self, monkeypatch):
        """At dim ≥ 512, doc excerpts are included."""
        monkeypatch.setenv("IMAS_CODEX_EMBEDDING_DIMENSION", "512")
        from imas_codex.graph.build_dd import generate_embedding_text

        text = generate_embedding_text(
            "core_profiles/profiles_1d/electrons/temperature",
            {
                "description": "Electron temperature (Te) profile",
                "documentation": "The electron temperature measured by Thomson scattering.",
                "keywords": ["Te", "thermal energy"],
            },
        )
        assert "core_profiles/profiles_1d/electrons/temperature" in text
        assert "Electron temperature (Te) profile" in text
        assert "Thomson scattering" in text
        assert "Te, thermal energy" in text

    def test_embed_text_at_dim_1024_includes_doc_and_keywords(self, monkeypatch):
        """At dim 1024, full doc excerpt + keywords are included."""
        monkeypatch.setenv("IMAS_CODEX_EMBEDDING_DIMENSION", "1024")
        from imas_codex.graph.build_dd import generate_embedding_text

        text = generate_embedding_text(
            "equilibrium/time_slice/profiles_1d/psi",
            {
                "description": "Poloidal flux profile",
                "documentation": "The poloidal magnetic flux function used as the radial coordinate.",
                "keywords": ["psi", "flux", "poloidal"],
            },
        )
        assert "equilibrium/time_slice/profiles_1d/psi" in text
        assert "Poloidal flux profile" in text
        assert "poloidal magnetic flux function" in text
        assert "psi, flux, poloidal" in text

    def test_embed_text_no_dup_when_doc_matches_desc(self, monkeypatch):
        """At high dim, don't duplicate when doc == desc."""
        monkeypatch.setenv("IMAS_CODEX_EMBEDDING_DIMENSION", "1024")
        from imas_codex.graph.build_dd import generate_embedding_text

        text = generate_embedding_text(
            "some/path",
            {
                "description": "Same text",
                "documentation": "Same text",
                "keywords": [],
            },
        )
        # doc == desc, so only appears once
        assert text == "some/path. Same text"

    def test_embed_text_uses_description_over_documentation(self):
        """Prefer description; doc is only used as fallback."""
        from imas_codex.graph.build_dd import generate_embedding_text

        text = generate_embedding_text(
            "equilibrium/time_slice/profiles_1d/psi",
            {
                "description": "Poloidal flux profile",
                "documentation": "The poloidal magnetic flux function used as the radial coordinate.",
                "keywords": ["psi", "flux"],
            },
        )
        assert "Poloidal flux profile" in text

    def test_embed_text_empty_returns_empty(self):
        """Empty description and documentation should return empty string."""
        from imas_codex.graph.build_dd import generate_embedding_text

        text = generate_embedding_text(
            "some/path",
            {"description": "", "documentation": "", "keywords": []},
        )
        assert text == ""

    def test_embed_text_fallback_to_documentation(self):
        """When no description, use documentation as fallback."""
        from imas_codex.graph.build_dd import generate_embedding_text

        text = generate_embedding_text(
            "some/path",
            {"description": "", "documentation": "Fallback doc text", "keywords": []},
        )
        assert text == "some/path. Fallback doc text"
