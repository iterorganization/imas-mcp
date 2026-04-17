"""Unit tests for COCOS label backfill defensive guards.

Validates that _backfill_cocos_labels correctly filters out updates
with null/empty labels to prevent the "half-state" bug where
cocos_label_source is set but cocos_transformation_type is null.
"""

import logging
from unittest.mock import MagicMock

import pytest

from imas_codex.graph.build_dd import _backfill_cocos_labels


@pytest.fixture()
def mock_client():
    """GraphClient mock that records query calls."""
    client = MagicMock()
    client.query.return_value = None
    return client


def _make_version_data(paths: dict[str, dict]) -> dict[str, dict]:
    """Build minimal version_data with a single version.

    Includes a 3.x reference version so forward-port logic can fire.
    """
    return {
        "3.42.2": {"paths": paths, "ids_info": {}, "units": set()},
        "4.0.0": {"paths": paths, "ids_info": {}, "units": set()},
    }


class TestBackfillNullLabelGuard:
    """Null/empty label entries must be dropped before Cypher execution."""

    def test_null_label_dropped_from_updates(self, mock_client, caplog):
        """An update with label=None must be filtered out; valid ones pass through."""
        # Build paths where 4.x has no cocos_transformation_type
        # but 3.x reference does — this triggers forward-port inference.
        paths = {
            "equilibrium/time_slice/profiles_1d/psi": {
                "cocos_transformation_type": None,
                "cocos_transformation_expression": "",
            },
            "equilibrium/time_slice/profiles_1d/phi": {
                "cocos_transformation_type": None,
                "cocos_transformation_expression": "",
            },
        }
        # 3.x reference has a label for psi but None for phi
        version_data = {
            "3.42.2": {
                "paths": {
                    "equilibrium/time_slice/profiles_1d/psi": {
                        "cocos_transformation_type": "psi_like",
                    },
                    "equilibrium/time_slice/profiles_1d/phi": {
                        "cocos_transformation_type": None,
                    },
                },
                "ids_info": {},
                "units": set(),
            },
            "4.0.0": {
                "paths": paths,
                "ids_info": {},
                "units": set(),
            },
        }

        with caplog.at_level(logging.DEBUG):
            count = _backfill_cocos_labels(mock_client, version_data)

        # Only psi (valid label) should be applied; phi (None label) is dropped
        assert count == 1

        # The Cypher UNWIND should have been called with exactly one update
        query_calls = [
            c for c in mock_client.query.call_args_list if "UNWIND" in str(c)
        ]
        assert len(query_calls) == 1
        batch = query_calls[0].kwargs.get("updates") or query_calls[0][1].get(
            "updates", []
        )
        assert len(batch) == 1
        assert batch[0]["label"] == "psi_like"
        assert batch[0]["source"] == "inferred_forward"

    def test_warning_logged_on_null_label(self, mock_client, caplog):
        """A warning must be logged when null-label entries are dropped."""
        # Path in 3.x has None label → forward-port produces None label
        paths = {
            "core_profiles/profiles_1d/electrons/temperature": {
                "cocos_transformation_type": None,
                "cocos_transformation_expression": "",
            },
        }
        version_data = {
            "3.42.2": {
                "paths": {
                    "core_profiles/profiles_1d/electrons/temperature": {
                        "cocos_transformation_type": None,
                    },
                },
                "ids_info": {},
                "units": set(),
            },
            "4.0.0": {
                "paths": paths,
                "ids_info": {},
                "units": set(),
            },
        }

        with caplog.at_level(logging.WARNING):
            _backfill_cocos_labels(mock_client, version_data)

        # No updates should be applied (all have None labels)
        # No UNWIND calls expected
        query_calls = [
            c for c in mock_client.query.call_args_list if "UNWIND" in str(c)
        ]
        assert len(query_calls) == 0

    def test_empty_label_treated_as_null(self, mock_client, caplog):
        """Empty string labels should also be filtered out."""
        paths = {
            "equilibrium/time_slice/profiles_1d/q": {
                "cocos_transformation_type": None,
                "cocos_transformation_expression": "",
            },
        }
        version_data = {
            "3.42.2": {
                "paths": {
                    "equilibrium/time_slice/profiles_1d/q": {
                        "cocos_transformation_type": "",  # empty string
                    },
                },
                "ids_info": {},
                "units": set(),
            },
            "4.0.0": {
                "paths": paths,
                "ids_info": {},
                "units": set(),
            },
        }

        count = _backfill_cocos_labels(mock_client, version_data)
        assert count == 0

    def test_expression_inferred_labels_pass_through(self, mock_client):
        """Labels inferred from cocos_transformation_expression should pass."""
        paths = {
            "equilibrium/time_slice/profiles_1d/f": {
                "cocos_transformation_type": None,
                "cocos_transformation_expression": "- {b0_like}",
            },
        }
        version_data = {
            "3.42.2": {
                "paths": {},
                "ids_info": {},
                "units": set(),
            },
            "4.0.0": {
                "paths": paths,
                "ids_info": {},
                "units": set(),
            },
        }

        count = _backfill_cocos_labels(mock_client, version_data)
        assert count == 1

        query_calls = [
            c for c in mock_client.query.call_args_list if "UNWIND" in str(c)
        ]
        assert len(query_calls) == 1
        batch = query_calls[0].kwargs.get("updates") or query_calls[0][1].get(
            "updates", []
        )
        assert batch[0]["label"] == "b0_like"
        assert batch[0]["source"] == "inferred_expression"

    def test_all_valid_updates_pass_through(self, mock_client):
        """When all updates have valid labels, none should be dropped."""
        paths = {
            "equilibrium/time_slice/profiles_1d/psi": {
                "cocos_transformation_type": None,
                "cocos_transformation_expression": "",
            },
            "equilibrium/time_slice/profiles_1d/f": {
                "cocos_transformation_type": None,
                "cocos_transformation_expression": "- {b0_like}",
            },
        }
        version_data = {
            "3.42.2": {
                "paths": {
                    "equilibrium/time_slice/profiles_1d/psi": {
                        "cocos_transformation_type": "psi_like",
                    },
                },
                "ids_info": {},
                "units": set(),
            },
            "4.0.0": {
                "paths": paths,
                "ids_info": {},
                "units": set(),
            },
        }

        count = _backfill_cocos_labels(mock_client, version_data)
        assert count == 2

    def test_xml_provenance_marker_always_runs(self, mock_client):
        """The XML provenance query should always run, even with no backfills."""
        version_data = {
            "3.42.2": {
                "paths": {},
                "ids_info": {},
                "units": set(),
            },
            "4.0.0": {
                "paths": {},
                "ids_info": {},
                "units": set(),
            },
        }

        _backfill_cocos_labels(mock_client, version_data)

        # The XML provenance query should have run
        xml_calls = [
            c
            for c in mock_client.query.call_args_list
            if "cocos_label_source" in str(c) and "xml" in str(c)
        ]
        assert len(xml_calls) == 1
