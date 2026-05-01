"""Regression test: fan-out trigger dims must match the actual review schema.

Plan 39 §5.1 specifies that ``refine_trigger_comment_dims`` allow-lists the
``reviewer_comments_per_dim_name`` dims used to extract a fan-out trigger
excerpt.  A previous default of ``("clarity", "disambiguation")`` silently
disabled the trigger because those dims do not exist on the review rubric
(actual dims: grammar / semantic / convention / completeness).  This test
locks the defaults to the real schema so the bug cannot recur.
"""

from imas_codex.standard_names.fanout.config import FanoutSettings, load_settings
from imas_codex.standard_names.models import StandardNameQualityScoreNameOnly

REVIEW_NAME_DIMS = frozenset(StandardNameQualityScoreNameOnly.model_fields.keys())


def test_default_trigger_dims_subset_of_review_dims():
    s = FanoutSettings()
    assert set(s.refine_trigger_comment_dims).issubset(REVIEW_NAME_DIMS), (
        "refine_trigger_comment_dims default must be a subset of the actual "
        f"name-review rubric dims {sorted(REVIEW_NAME_DIMS)}"
    )


def test_loaded_settings_trigger_dims_subset_of_review_dims():
    s = load_settings()
    assert set(s.refine_trigger_comment_dims).issubset(REVIEW_NAME_DIMS), (
        "Loaded refine_trigger_comment_dims must be a subset of the actual "
        f"name-review rubric dims {sorted(REVIEW_NAME_DIMS)}; check "
        "[tool.imas-codex.sn.fanout] refine-trigger-comment-dims in pyproject.toml"
    )


def test_default_trigger_dims_nonempty():
    s = FanoutSettings()
    assert s.refine_trigger_comment_dims, (
        "refine_trigger_comment_dims must be non-empty — an empty allow-list "
        "would unconditionally disable fan-out"
    )
