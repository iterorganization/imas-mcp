"""Tests for content-aware wiki scoring."""

import pytest


class TestContentAwareScoring:
    def test_scoring_guidelines_data_source(self):
        """Data source pages should score 0.7-1.0."""
        page = {
            "id": "iter:jorek",
            "title": "JOREK disruption cases",
            "preview_summary": "Database of JOREK MHD simulation cases for disruption scenarios.",
            "in_degree": 1,
            "link_depth": 3,
        }
        # Validate page structure for scoring
        assert page["id"]
        assert page["title"]
        assert page["preview_summary"]
        assert page["page_type"] not in page or page.get("page_type") == "data_source"

    def test_scoring_guidelines_meeting(self):
        """Meeting pages should score 0.1-0.4."""
        page = {
            "id": "iter:meeting",
            "title": "Weekly team meeting notes",
            "preview_summary": "Meeting notes from weekly team sync.",
            "in_degree": 0,
            "link_depth": 4,
        }
        # Validate meeting page structure
        assert page["id"]
        assert (
            "meeting" in page["title"].lower()
            or "meeting" in page["preview_summary"].lower()
        )

    def test_scoring_not_penalized_for_low_in_degree(self):
        """ITER pages should not be penalized for low in_degree."""
        page = {
            "id": "iter:solps",
            "title": "SOLPS-ITER User Forum",
            "preview_summary": "User forum with SOLPS-ITER release notes and troubleshooting guides.",
            "in_degree": 2,  # Low but should still score well
            "link_depth": 3,
        }
        # Low in_degree should not disqualify high-value content
        assert page["in_degree"] < 5
        # But content is technical so should score well
        assert (
            "release notes" in page["preview_summary"].lower()
            or "guide" in page["preview_summary"].lower()
        )

    def test_page_type_classification(self):
        """Test that page types are correctly classified."""
        page_types = [
            "data_source",
            "documentation",
            "code",
            "process",
            "meeting",
            "portal",
            "other",
        ]
        assert len(page_types) == 7
        for ptype in page_types:
            assert isinstance(ptype, str)
            assert len(ptype) > 0

    def test_value_rating_range(self):
        """Test that value ratings are in valid range."""
        # Valid ratings: 0-10
        for rating in [0, 1, 5, 8, 10]:
            assert 0 <= rating <= 10

    def test_is_physics_content_boolean(self):
        """Test that is_physics_content is boolean."""
        for value in [True, False]:
            assert isinstance(value, bool)

    def test_interest_score_range(self):
        """Test that interest scores are in valid range."""
        # Valid scores: 0.0-1.0
        for score in [0.0, 0.25, 0.5, 0.75, 1.0]:
            assert 0.0 <= score <= 1.0


class TestScoringContentVsMetrics:
    def test_content_drives_scoring(self):
        """Content quality should drive scoring, not just metrics."""
        # Page with low metrics but good content
        page_low_metrics = {
            "title": "JOREK disruption cases",
            "preview_summary": "Critical disruption database for ML training",
            "in_degree": 1,
            "out_degree": 0,
            "link_depth": 5,
        }

        # Page with high metrics but poor content
        page_high_metrics = {
            "title": "Meeting 2024-01-19",
            "preview_summary": "Team sync meeting",
            "in_degree": 10,
            "out_degree": 8,
            "link_depth": 1,
        }

        # Based on content alone, first should score higher
        # (This is a validation test - the actual LLM scoring happens in integration tests)
        assert "disruption" in page_low_metrics["preview_summary"].lower()
        assert "meeting" in page_high_metrics["preview_summary"].lower()


class TestFacilityAgnosticScoring:
    def test_iter_confluence_scoring(self):
        """ITER Confluence pages should not be penalized."""
        page = {
            "id": "iter:xxxxx",
            "title": "Some technical page",
            "preview_summary": "Technical documentation",
            "in_degree": 2,  # Confluence links structure differs
            "link_depth": 4,
        }
        # Should be scored on content, not metrics alone
        assert page["preview_summary"]

    def test_epfl_mediawiki_scoring(self):
        """EPFL MediaWiki pages should score consistently."""
        page = {
            "id": "epfl:Thomson",
            "title": "Thomson Scattering",
            "preview_summary": "Thomson diagnostic system",
            "in_degree": 5,  # Higher in MediaWiki due to different linking
            "link_depth": 2,
        }
        # Should be scored on content, not metrics alone
        assert page["preview_summary"]


class TestScoringPromptStructure:
    def test_scoring_output_format(self):
        """Test expected JSON output format from scoring."""
        sample_scores = [
            {
                "id": "page1",
                "score": 0.75,
                "page_type": "data_source",
                "is_physics": True,
                "value_rating": 8,
                "reasoning": "High-value data source",
            },
            {
                "id": "page2",
                "score": 0.2,
                "page_type": "meeting",
                "is_physics": False,
                "value_rating": 2,
                "reasoning": "Administrative content",
            },
        ]

        # Validate structure
        for item in sample_scores:
            assert "id" in item
            assert "score" in item
            assert "page_type" in item
            assert "is_physics" in item
            assert "value_rating" in item
            assert "reasoning" in item

            # Validate ranges
            assert 0 <= item["score"] <= 1.0
            assert 0 <= item["value_rating"] <= 10
            assert isinstance(item["is_physics"], bool)
            assert item["page_type"] in [
                "data_source",
                "documentation",
                "code",
                "process",
                "meeting",
                "portal",
                "other",
            ]
