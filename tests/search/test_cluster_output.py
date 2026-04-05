"""Tests for cluster label output and See Also enrichment in search results."""

from __future__ import annotations

from imas_codex.models.constants import SearchMode
from imas_codex.search.search_strategy import SearchHit


class TestSearchHitClusterLabels:
    """Verify SearchHit accepts and stores cluster labels."""

    def test_cluster_labels_default_empty(self):
        hit = SearchHit(
            path="core_profiles/profiles_1d/electrons/temperature",
            ids_name="core_profiles",
            documentation="Electron temperature",
            score=0.9,
            rank=1,
            search_mode=SearchMode.HYBRID,
        )
        assert hit.cluster_labels == []

    def test_cluster_labels_populated(self):
        hit = SearchHit(
            path="core_profiles/profiles_1d/electrons/temperature",
            ids_name="core_profiles",
            documentation="Electron temperature",
            score=0.9,
            rank=1,
            search_mode=SearchMode.HYBRID,
            cluster_labels=["electron temperature", "kinetic profiles"],
        )
        assert hit.cluster_labels == ["electron temperature", "kinetic profiles"]

    def test_see_also_default_empty(self):
        hit = SearchHit(
            path="core_profiles/profiles_1d/electrons/temperature",
            ids_name="core_profiles",
            documentation="Electron temperature",
            score=0.9,
            rank=1,
            search_mode=SearchMode.HYBRID,
        )
        assert hit.see_also == []

    def test_see_also_populated(self):
        hit = SearchHit(
            path="core_profiles/profiles_1d/electrons/temperature",
            ids_name="core_profiles",
            documentation="Electron temperature",
            score=0.9,
            rank=1,
            search_mode=SearchMode.HYBRID,
            see_also=["ece/channel/t_e", "summary/local/itb/t_e"],
        )
        assert len(hit.see_also) == 2


class TestClusterLabelFormatting:
    """Verify cluster labels appear in formatted output."""

    def test_cluster_labels_in_report(self):
        from imas_codex.llm.search_formatters import format_search_dd_report
        from imas_codex.models.result_models import SearchPathsResult

        hit = SearchHit(
            path="core_profiles/profiles_1d/electrons/temperature",
            ids_name="core_profiles",
            documentation="Electron temperature",
            score=0.9,
            rank=1,
            search_mode=SearchMode.HYBRID,
            cluster_labels=["electron temperature", "kinetic profiles"],
        )
        result = SearchPathsResult(
            hits=[hit],
            summary={
                "query": "electron temp",
                "search_mode": "hybrid",
                "hits_returned": 1,
                "ids_coverage": ["core_profiles"],
            },
            query="electron temp",
            search_mode=SearchMode.HYBRID,
            physics_domains=["core_profiles"],
        )
        report = format_search_dd_report(result)
        assert "Clusters:" in report
        assert "electron temperature" in report

    def test_see_also_in_report(self):
        from imas_codex.llm.search_formatters import format_search_dd_report
        from imas_codex.models.result_models import SearchPathsResult

        hit = SearchHit(
            path="core_profiles/profiles_1d/electrons/temperature",
            ids_name="core_profiles",
            documentation="Electron temperature",
            score=0.9,
            rank=1,
            search_mode=SearchMode.HYBRID,
            see_also=[
                "ece/channel/t_e",
                "edge_profiles/ggd/electrons/temperature",
                "summary/local/itb/t_e",
                "summary/line_average/t_e",
            ],
        )
        result = SearchPathsResult(
            hits=[hit],
            summary={
                "query": "electron temp",
                "search_mode": "hybrid",
                "hits_returned": 1,
                "ids_coverage": ["core_profiles"],
            },
            query="electron temp",
            search_mode=SearchMode.HYBRID,
            physics_domains=["core_profiles"],
        )
        report = format_search_dd_report(result)
        assert "See also:" in report
        assert "+1 more" in report

    def test_no_cluster_labels_no_output(self):
        from imas_codex.llm.search_formatters import format_search_dd_report
        from imas_codex.models.result_models import SearchPathsResult

        hit = SearchHit(
            path="core_profiles/profiles_1d/electrons/temperature",
            ids_name="core_profiles",
            documentation="Electron temperature",
            score=0.9,
            rank=1,
            search_mode=SearchMode.HYBRID,
        )
        result = SearchPathsResult(
            hits=[hit],
            summary={
                "query": "electron temp",
                "search_mode": "hybrid",
                "hits_returned": 1,
                "ids_coverage": ["core_profiles"],
            },
            query="electron temp",
            search_mode=SearchMode.HYBRID,
            physics_domains=["core_profiles"],
        )
        report = format_search_dd_report(result)
        assert "Clusters:" not in report
        assert "See also:" not in report
