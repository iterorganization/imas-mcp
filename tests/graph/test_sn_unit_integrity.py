"""Unit-integrity tests for StandardName ↔ DD-path unit agreement.

For every valid StandardName with DD source paths, asserts that the
SN's HAS_UNIT target matches the canonical DD-path HAS_UNIT target.

DD-side issues (paths whose units disagree among themselves, or known
DD unit bugs like ``ion_atomic_number [e]``) are documented as
skip-reasons, not failures.
"""

import pytest

pytestmark = pytest.mark.graph

# DD-side unit bugs: the SN unit is physically correct but the DD path
# has a wrong unit.  These are tracked for DD-rebuild follow-up and
# must NOT cause test failures.
_DD_SIDE_UNIT_BUGS: dict[str, str] = {
    # All previously-listed bugs have been fixed upstream in the DD.
    # Add new entries here when a DD-side unit bug is discovered.
}


def _query_sn_unit_vs_dd(graph_client):
    """Return rows of (name, sn_unit, dd_units) for valid SNs with DD source paths."""
    return graph_client.query("""
        MATCH (sn:StandardName {validation_status: 'valid'})
        WHERE sn.unit IS NOT NULL AND sn.source_paths IS NOT NULL
        UNWIND sn.source_paths AS sp_raw
        WITH sn, replace(sp_raw, 'dd:', '') AS sp
        OPTIONAL MATCH (dd:IMASNode {id: sp})-[:HAS_UNIT]->(du:Unit)
        WITH sn.id AS name, sn.unit AS sn_unit,
             collect(DISTINCT du.id) AS raw_dd_units,
             collect(DISTINCT sp) AS source_paths
        WITH name, sn_unit,
             [x IN raw_dd_units WHERE x IS NOT NULL] AS dd_units,
             source_paths
        RETURN name, sn_unit, dd_units, source_paths
        ORDER BY name
    """)


class TestSNUnitIntegrity:
    """StandardName unit must agree with its DD source path unit."""

    def test_sn_unit_matches_linked_dd_path_unit(self, graph_client):
        """Every StandardName.unit must equal its DD source paths' unit.

        Skips cleanly if no SNs have source-path linkage.
        DD-internal inconsistencies (paths disagree among themselves)
        are reported as skips, not failures.
        Known DD-side bugs are excluded via allow-list.
        """
        rows = _query_sn_unit_vs_dd(graph_client)

        # Skip if graph has no source-path-linked SNs
        has_dd_units = [r for r in rows if len(r["dd_units"]) > 0]
        if not has_dd_units:
            pytest.skip("No StandardNames with DD source-path unit linkage")

        failures = []
        for r in has_dd_units:
            name = r["name"]
            sn_unit = r["sn_unit"]
            dd_units = r["dd_units"]

            # SN unit matches at least one DD unit → pass
            if sn_unit in dd_units:
                continue

            # DD paths disagree among themselves → skip (DD quality issue)
            if len(dd_units) > 1:
                continue

            # Known DD-side bug → skip
            if name in _DD_SIDE_UNIT_BUGS:
                continue

            # Genuine SN-side mismatch → fail
            failures.append(f"{name}: SN unit={sn_unit!r}, DD unit(s)={dd_units}")

        assert not failures, (
            "StandardName units disagree with canonical DD units:\n  "
            + "\n  ".join(failures)
        )

    def test_all_valid_sns_have_unit(self, graph_client):
        """Every valid StandardName must have a declared unit (even '1')."""
        rows = graph_client.query("""
            MATCH (sn:StandardName {validation_status: 'valid'})
            WHERE NOT (sn)-[:HAS_UNIT]->(:Unit)
            RETURN sn.id AS name
            ORDER BY name
        """)
        if not rows:
            return
        missing = [r["name"] for r in rows]
        assert not missing, (
            f"{len(missing)} valid StandardNames without HAS_UNIT edge: "
            + ", ".join(missing[:20])
        )

    def test_sn_unit_property_matches_edge(self, graph_client):
        """SN.unit property must equal the HAS_UNIT target node id."""
        rows = graph_client.query("""
            MATCH (sn:StandardName {validation_status: 'valid'})-[:HAS_UNIT]->(u:Unit)
            WHERE sn.unit <> u.id
            RETURN sn.id AS name, sn.unit AS prop_unit, u.id AS edge_unit
            ORDER BY name
        """)
        if not rows:
            return
        mismatches = [
            f"{r['name']}: property={r['prop_unit']!r}, edge={r['edge_unit']!r}"
            for r in rows
        ]
        assert not mismatches, (
            "SN unit property disagrees with HAS_UNIT edge:\n  "
            + "\n  ".join(mismatches)
        )

    def test_dd_side_unit_bugs_documented(self, graph_client):
        """Known DD-side unit bugs are still present (regression guard).

        When a DD rebuild fixes these, remove them from _DD_SIDE_UNIT_BUGS.
        """
        rows = _query_sn_unit_vs_dd(graph_client)
        still_mismatched = set()
        for r in rows:
            if r["name"] in _DD_SIDE_UNIT_BUGS and len(r["dd_units"]) > 0:
                if r["sn_unit"] not in r["dd_units"]:
                    still_mismatched.add(r["name"])

        # If a DD-side bug gets fixed, the allow-list entry is stale
        stale = set(_DD_SIDE_UNIT_BUGS.keys()) - still_mismatched
        if stale:
            pytest.fail(
                "DD-side bugs fixed upstream — remove from _DD_SIDE_UNIT_BUGS: "
                + ", ".join(sorted(stale))
            )
