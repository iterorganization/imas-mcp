"""Tests for source path encoding, parsing, splitting, and merging."""

from __future__ import annotations

from imas_codex.standard_names.source_paths import (
    DD_PREFIX,
    SourceURI,
    encode_dd_source,
    encode_source_path,
    ids_prefix_for_source_paths,
    merge_source_paths,
    merge_source_types,
    normalize_source_path,
    parse_source_path,
    parse_source_uri,
    split_source_paths,
    strip_dd_prefix,
)


class TestEncodeSourcePath:
    def test_dd_path_stored_with_dd_prefix(self):
        result = encode_source_path("dd", "equilibrium/time_slice/profiles_1d/psi")
        assert result == "dd:equilibrium/time_slice/profiles_1d/psi"

    def test_signal_stored_with_prefix(self):
        result = encode_source_path("signals", "tcv:ip/measured")
        assert result == "tcv:ip/measured"

    def test_dd_path_has_dd_colon(self):
        result = encode_source_path(
            "dd", "core_profiles/profiles_1d/electrons/temperature"
        )
        assert result.startswith("dd:")

    def test_signal_id_preserves_facility_colon(self):
        result = encode_source_path("signals", "jet:te/core")
        assert result == "jet:te/core"


class TestParseSourcePath:
    def test_dd_path_no_colon(self):
        source_type, source_id = parse_source_path(
            "equilibrium/time_slice/profiles_1d/psi"
        )
        assert source_type == "dd"
        assert source_id == "equilibrium/time_slice/profiles_1d/psi"

    def test_signal_path_with_colon(self):
        source_type, source_id = parse_source_path("tcv:ip/measured")
        assert source_type == "signals"
        assert source_id == "tcv:ip/measured"

    def test_dd_path_full_round_trip(self):
        original = "magnetics/flux_loop/flux/data"
        encoded = encode_source_path("dd", original)
        parsed_type, parsed_id = parse_source_path(encoded)
        assert parsed_type == "dd"
        assert parsed_id == original

    def test_signal_path_full_round_trip(self):
        original = "jet:te/core"
        encoded = encode_source_path("signals", original)
        parsed_type, parsed_id = parse_source_path(encoded)
        assert parsed_type == "signals"
        assert parsed_id == original

    def test_multi_colon_classified_as_signals(self):
        # e.g. "facility:diag:path" — still signals because it contains a colon
        source_type, _ = parse_source_path("tcv:thomson:te_core")
        assert source_type == "signals"


class TestSplitSourcePaths:
    def test_mixed_sources(self):
        paths = [
            "equilibrium/time_slice/profiles_1d/psi",
            "tcv:ip/measured",
            "core_profiles/profiles_1d/electrons/temperature",
            "jet:te/core",
        ]
        result = split_source_paths(paths)
        assert set(result["dd"]) == {
            "equilibrium/time_slice/profiles_1d/psi",
            "core_profiles/profiles_1d/electrons/temperature",
        }
        assert set(result["signals"]) == {"tcv:ip/measured", "jet:te/core"}

    def test_empty_list(self):
        assert split_source_paths([]) == {}

    def test_dd_only(self):
        paths = ["a/b/c", "x/y/z"]
        result = split_source_paths(paths)
        assert "dd" in result
        assert "signals" not in result

    def test_signals_only(self):
        paths = ["tcv:ip/measured", "jet:b0/measured"]
        result = split_source_paths(paths)
        assert "signals" in result
        assert "dd" not in result

    def test_each_path_appears_once(self):
        paths = ["a/b", "tcv:x/y"]
        result = split_source_paths(paths)
        assert result["dd"] == ["a/b"]
        assert result["signals"] == ["tcv:x/y"]


class TestMergeSourcePaths:
    def test_dedup_union(self):
        result = merge_source_paths(["a/b", "c/d"], ["c/d", "e/f"])
        assert result == ["a/b", "c/d", "e/f"]

    def test_deterministic_sort(self):
        r1 = merge_source_paths(["z/path", "a/path"], [])
        r2 = merge_source_paths(["a/path", "z/path"], [])
        assert r1 == r2

    def test_empty_inputs(self):
        assert merge_source_paths([], []) == []

    def test_left_only(self):
        result = merge_source_paths(["a/b", "c/d"], [])
        assert result == ["a/b", "c/d"]

    def test_right_only(self):
        result = merge_source_paths([], ["x/y"])
        assert result == ["x/y"]

    def test_result_is_sorted(self):
        result = merge_source_paths(["z/z", "a/a"], ["m/m"])
        assert result == sorted(result)

    def test_no_duplicates_in_result(self):
        result = merge_source_paths(["a/b", "a/b"], ["a/b"])
        assert result == ["a/b"]


class TestMergeSourceTypes:
    def test_dedup_union(self):
        result = merge_source_types(["dd"], ["dd", "signals"])
        assert result == ["dd", "signals"]

    def test_single_type(self):
        assert merge_source_types(["dd"], []) == ["dd"]

    def test_empty_both(self):
        assert merge_source_types([], []) == []

    def test_result_is_sorted(self):
        result = merge_source_types(["signals"], ["dd"])
        assert result == sorted(result)

    def test_no_duplicates(self):
        result = merge_source_types(["dd", "dd"], ["dd"])
        assert result == ["dd"]


# =============================================================================
# New dd: prefix tests
# =============================================================================


class TestDDPrefix:
    def test_constant_value(self):
        assert DD_PREFIX == "dd:"


class TestEncodeDDSource:
    def test_adds_prefix(self):
        assert encode_dd_source("eq/ts/p1d/psi") == "dd:eq/ts/p1d/psi"

    def test_idempotent(self):
        assert encode_dd_source("dd:eq/ts/p1d/psi") == "dd:eq/ts/p1d/psi"


class TestNormalizeSourcePath:
    def test_bare_dd_gets_prefix(self):
        assert normalize_source_path("eq/ts") == "dd:eq/ts"

    def test_already_prefixed_unchanged(self):
        assert normalize_source_path("dd:eq/ts") == "dd:eq/ts"

    def test_facility_unchanged(self):
        assert normalize_source_path("tcv:ip") == "tcv:ip"


class TestStripDDPrefix:
    def test_strips_prefix(self):
        assert strip_dd_prefix("dd:eq/ts/p1d/psi") == "eq/ts/p1d/psi"

    def test_bare_path_unchanged(self):
        assert strip_dd_prefix("eq/ts/p1d/psi") == "eq/ts/p1d/psi"

    def test_facility_path_unchanged(self):
        assert strip_dd_prefix("tcv:ip") == "tcv:ip"


class TestIDSPrefixForSourcePaths:
    def test_returns_dd_prefix(self):
        assert ids_prefix_for_source_paths("equilibrium") == "dd:equilibrium/"

    def test_core_profiles(self):
        assert ids_prefix_for_source_paths("core_profiles") == "dd:core_profiles/"


class TestParseSourceURI:
    def test_dd_prefixed(self):
        uri = parse_source_uri("dd:eq/ts/p1d/psi")
        assert uri.source_type == "dd"
        assert uri.source_id == "eq/ts/p1d/psi"
        assert uri.uri == "dd:eq/ts/p1d/psi"

    def test_facility(self):
        uri = parse_source_uri("tcv:ip/measured")
        assert uri.source_type == "tcv"
        assert uri.source_id == "ip/measured"

    def test_bare_legacy(self):
        uri = parse_source_uri("eq/ts/p1d/psi")
        assert uri.source_type == "dd"
        assert uri.source_id == "eq/ts/p1d/psi"

    def test_frozen(self):
        uri = parse_source_uri("dd:eq/ts")
        assert isinstance(uri, SourceURI)


class TestNormalizationBoundary:
    """Test the critical boundary: source_paths stores dd: URIs,
    IMASNode.id stores bare paths."""

    def test_encode_then_strip_round_trips(self):
        bare = "equilibrium/time_slice/profiles_1d/psi"
        encoded = encode_source_path("dd", bare)
        assert encoded == f"dd:{bare}"
        assert strip_dd_prefix(encoded) == bare

    def test_split_dd_prefixed_paths(self):
        paths = ["dd:eq/ts/p1d/psi", "tcv:ip/measured"]
        result = split_source_paths(paths)
        assert result["dd"] == ["eq/ts/p1d/psi"]
        assert result["signals"] == ["tcv:ip/measured"]
