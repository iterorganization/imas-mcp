"""MRR calculation and assertion helpers for search benchmarks.

Provides utilities for computing Mean Reciprocal Rank (MRR) and
generating detailed failure reports when search quality regresses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tests.search.benchmark_data import BenchmarkQuery


@dataclass
class QueryResult:
    """Result of running a single benchmark query."""

    query: BenchmarkQuery
    returned_paths: list[str]
    reciprocal_rank: float = 0.0

    def __post_init__(self):
        self.reciprocal_rank = self._compute_rr()

    def _compute_rr(self) -> float:
        """Compute reciprocal rank: 1/rank of first relevant result."""
        for rank, path in enumerate(self.returned_paths, start=1):
            if path in self.expected_set:
                return 1.0 / rank
        return 0.0

    @property
    def expected_set(self) -> set[str]:
        return set(self.query.expected_paths)

    @property
    def hit(self) -> bool:
        """Whether any expected path appears in results."""
        return self.reciprocal_rank > 0.0

    @property
    def hit_at_1(self) -> bool:
        """Whether the top result is an expected path."""
        return self.reciprocal_rank == 1.0


@dataclass
class BenchmarkResults:
    """Aggregated results for a search method benchmark."""

    method_name: str
    query_results: list[QueryResult] = field(default_factory=list)

    @property
    def mrr(self) -> float:
        """Mean Reciprocal Rank across all queries."""
        if not self.query_results:
            return 0.0
        return sum(qr.reciprocal_rank for qr in self.query_results) / len(
            self.query_results
        )

    @property
    def precision_at_1(self) -> float:
        """Fraction of queries with correct answer at rank 1."""
        if not self.query_results:
            return 0.0
        return sum(1 for qr in self.query_results if qr.hit_at_1) / len(
            self.query_results
        )

    @property
    def recall_at_5(self) -> float:
        """Fraction of queries with at least one hit in top 5."""
        if not self.query_results:
            return 0.0
        hits = sum(
            1
            for qr in self.query_results
            if any(p in qr.expected_set for p in qr.returned_paths[:5])
        )
        return hits / len(self.query_results)

    @property
    def failures(self) -> list[QueryResult]:
        """Queries with no expected path in any returned result."""
        return [qr for qr in self.query_results if not qr.hit]

    def per_category_mrr(self) -> dict[str, float]:
        """MRR broken down by query category."""
        from collections import defaultdict

        by_cat: dict[str, list[float]] = defaultdict(list)
        for qr in self.query_results:
            by_cat[qr.query.category].append(qr.reciprocal_rank)
        return {cat: sum(rrs) / len(rrs) if rrs else 0.0 for cat, rrs in by_cat.items()}

    def summary(self) -> str:
        """Human-readable summary for test output."""
        lines = [
            f"=== {self.method_name} Benchmark ===",
            f"  MRR:    {self.mrr:.3f}",
            f"  P@1:    {self.precision_at_1:.3f}",
            f"  R@5:    {self.recall_at_5:.3f}",
            f"  Queries: {len(self.query_results)}",
            f"  Hits:    {len(self.query_results) - len(self.failures)}",
            f"  Misses:  {len(self.failures)}",
        ]
        cat_mrr = self.per_category_mrr()
        if cat_mrr:
            lines.append("  Per-category MRR:")
            for cat, mrr in sorted(cat_mrr.items()):
                lines.append(f"    {cat}: {mrr:.3f}")
        return "\n".join(lines)


def assert_mrr_above(results: BenchmarkResults, threshold: float) -> None:
    """Assert MRR meets threshold with detailed failure diagnostics.

    Raises AssertionError with per-query breakdown showing which queries
    failed and what was returned instead of the expected paths.
    """
    if results.mrr >= threshold:
        return

    lines = [
        f"{results.method_name} MRR = {results.mrr:.3f} < {threshold:.3f}",
        "",
        results.summary(),
        "",
        "Failed queries:",
    ]

    for qr in results.failures[:10]:
        lines.append(f"  Query: '{qr.query.query_text}' [{qr.query.category}]")
        lines.append(f"    Expected: {qr.query.expected_paths[:3]}")
        lines.append(f"    Got top 5: {qr.returned_paths[:5]}")
        if qr.query.notes:
            lines.append(f"    Notes: {qr.query.notes}")

    raise AssertionError("\n".join(lines))


def assert_precision_at_1_above(results: BenchmarkResults, threshold: float) -> None:
    """Assert P@1 meets threshold."""
    if results.precision_at_1 >= threshold:
        return

    misses_at_1 = [qr for qr in results.query_results if not qr.hit_at_1]
    lines = [
        f"{results.method_name} P@1 = {results.precision_at_1:.3f} < {threshold:.3f}",
        "",
        f"Queries missing at rank 1 ({len(misses_at_1)}):",
    ]
    for qr in misses_at_1[:10]:
        lines.append(
            f"  '{qr.query.query_text}': got '{qr.returned_paths[0] if qr.returned_paths else '(empty)'}', expected one of {qr.query.expected_paths[:2]}"
        )

    raise AssertionError("\n".join(lines))


def run_benchmark(
    method_name: str,
    queries: list[BenchmarkQuery],
    search_fn,
    limit: int = 50,
) -> BenchmarkResults:
    """Run a benchmark by calling search_fn(query_text, limit) for each query.

    Parameters
    ----------
    method_name:
        Human-readable name for this search method.
    queries:
        Benchmark queries to evaluate.
    search_fn:
        Callable(query_text: str, limit: int) -> list[str]
        Must return a list of IMAS path IDs in ranked order.
    limit:
        Max results to request from the search function.

    Returns
    -------
    BenchmarkResults with per-query reciprocal ranks and aggregate MRR.
    """
    results = BenchmarkResults(method_name=method_name)
    for q in queries:
        returned = search_fn(q.query_text, limit)
        results.query_results.append(QueryResult(query=q, returned_paths=returned))
    return results
