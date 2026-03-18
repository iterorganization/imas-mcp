#!/usr/bin/env python
"""Query TCV SignalSources to understand skip behavior."""

from imas_codex.graph.client import GraphClient

with GraphClient() as gc:
    # Get discovered SignalSources on TCV
    result = gc.query("""
        MATCH (sg:SignalSource {facility_id: 'tcv', status: 'discovered'})
        RETURN sg.id AS id, sg.group_key AS group_key, sg.member_count AS member_count,
               sg.representative_id AS representative_id
        ORDER BY sg.member_count DESC
        LIMIT 30
    """)
    print(f"Total discovered SignalSources on TCV: {len(result)}")
    for r in result:
        print(f"  Group: {r['group_key']}, members: {r['member_count']}")
        print(f"    rep: {r['representative_id']}")

    # Now check how many have skipped representatives
    result2 = gc.query("""
        MATCH (sg:SignalSource {facility_id: 'tcv', status: 'discovered'})
        MATCH (rep:FacilitySignal {id: sg.representative_id})
        WHERE rep.status = 'skipped'
        RETURN sg.id AS id, sg.group_key AS group_key, rep.skip_reason AS skip_reason,
               rep.accessor AS rep_accessor
        LIMIT 30
    """)
    print(f"\nSignalSources with skipped representatives: {len(result2)}")
    for r in result2:
        print(f"  Group: {r['group_key']}")
        print(f"    rep: {r['rep_accessor']}, skip: {r['skip_reason']}")

    # Check what ALL members look like for one of these groups
    if result2:
        sample_group_id = result2[0]["id"]
        members = gc.query(
            """
            MATCH (s:FacilitySignal)-[:MEMBER_OF]->(sg:SignalSource {id: $id})
            RETURN s.accessor AS accessor, s.status AS status, s.skip_reason AS skip_reason
            ORDER BY s.accessor
            LIMIT 10
        """,
            id=sample_group_id,
        )
        print(f"\nSample members of {sample_group_id}:")
        for m in members:
            print(f"  {m['accessor']}: {m['status']}, skip={m['skip_reason']}")
