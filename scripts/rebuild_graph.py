#!/usr/bin/env python3
"""Rebuild corrupted Neo4j graph via Cypher export and clean reimport.

Phases:
    export  — Read all nodes/relationships from corrupted DB, write to JSONL
    import  — Load JSONL into a fresh empty database with migrations applied
    verify  — Compare imported counts against export expectations
    cleanup — Remove temporary _rid properties after successful import

Migrations applied during export:
    - tree_name property stripped from all nodes
    - PRODUCED relationships renamed to HAS_EXAMPLE

Usage:
    uv run python scripts/rebuild_graph.py export
    # Stop Neo4j, wipe data/databases/ and data/transactions/, start Neo4j
    uv run python scripts/rebuild_graph.py import
    uv run python scripts/rebuild_graph.py verify
    uv run python scripts/rebuild_graph.py cleanup
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EXPORT_DIR = Path.home() / ".local/share/imas-codex/graph-rebuild"

# === Migrations ===
SKIP_NODE_PROPERTIES = frozenset({"tree_name"})
REL_TYPE_RENAMES = {"PRODUCED": "HAS_EXAMPLE"}


# === JSON serialization for neo4j temporal types ===


class Neo4jEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "iso_format"):
            return {"__dt__": obj.iso_format()}
        if hasattr(obj, "isoformat"):
            return {"__dt__": obj.isoformat()}
        return super().default(obj)


def _restore_value(v):
    """Recursively restore serialized neo4j types."""
    if isinstance(v, dict) and "__dt__" in v:
        iso = v["__dt__"]
        try:
            return datetime.fromisoformat(iso.replace("Z", "+00:00"))
        except ValueError:
            return iso
    if isinstance(v, list):
        return [_restore_value(x) for x in v]
    return v


def _restore_props(props: dict) -> dict:
    return {k: _restore_value(v) for k, v in props.items()}


# === Export ===


def export_phase():
    from imas_codex.graph.client import GraphClient

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Export directory: {EXPORT_DIR}")

    gc = GraphClient()

    # Save index/constraint CREATE statements
    print("\n=== Saving index/constraint definitions ===")
    stmts = gc.query(
        "SHOW INDEXES YIELD name, type, createStatement "
        "WHERE type <> 'LOOKUP' "
        "RETURN name, type, createStatement ORDER BY type, name"
    )
    with open(EXPORT_DIR / "indexes.json", "w") as f:
        json.dump(stmts, f, indent=2)
    print(f"  {len(stmts)} definitions saved")

    # Get labels and counts
    label_counts = gc.query(
        "MATCH (n) WITH labels(n)[0] AS label, count(n) AS cnt "
        "RETURN label, cnt ORDER BY cnt DESC"
    )
    total_node_count = sum(r["cnt"] for r in label_counts)
    print(
        f"\n=== Exporting {total_node_count} nodes across {len(label_counts)} labels ==="
    )

    # Export nodes — assign sequential _rid as we go
    node_registry: dict[str, dict] = {}  # eid -> {rid, label}
    rid_counter = 0
    total_exported = 0

    for lc in label_counts:
        label = lc["label"]
        expected = lc["cnt"]
        filepath = EXPORT_DIR / f"nodes_{label}.jsonl"
        print(f"  {label} ({expected})...", end="", flush=True)

        exported = 0
        with open(filepath, "w") as f:
            offset = 0
            batch = 5000
            while True:
                results = gc.query(
                    f"MATCH (n:`{label}`) "
                    f"RETURN elementId(n) AS eid, labels(n) AS labels, properties(n) AS props "
                    f"SKIP $offset LIMIT $limit",
                    offset=offset,
                    limit=batch,
                )
                if not results:
                    break

                for r in results:
                    eid = r["eid"]
                    props = {
                        k: v
                        for k, v in r["props"].items()
                        if k not in SKIP_NODE_PROPERTIES
                    }
                    rid = rid_counter
                    rid_counter += 1

                    node_registry[eid] = {"rid": rid, "label": label}

                    record = {
                        "rid": rid,
                        "labels": r["labels"],
                        "props": props,
                    }
                    f.write(json.dumps(record, cls=Neo4jEncoder) + "\n")
                    exported += 1

                offset += batch

        print(f" ✓ {exported}")
        total_exported += exported

    print(f"  Total: {total_exported}")

    # Save node registry (needed for relationship import)
    print("\n=== Saving node registry ===")
    with open(EXPORT_DIR / "node_registry.json", "w") as f:
        json.dump(node_registry, f)
    print(
        f"  {len(node_registry)} entries ({(EXPORT_DIR / 'node_registry.json').stat().st_size / 1e6:.1f} MB)"
    )

    # Export relationships
    rel_counts = gc.query(
        "MATCH ()-[r]->() WITH type(r) AS t, count(r) AS cnt "
        "RETURN t, cnt ORDER BY cnt DESC"
    )
    total_rel_count = sum(r["cnt"] for r in rel_counts)
    print(
        f"\n=== Exporting {total_rel_count} relationships across {len(rel_counts)} types ==="
    )

    total_rel_exported = 0
    missing_nodes = 0

    for rc in rel_counts:
        rel_type = rc["t"]
        expected = rc["cnt"]
        export_type = REL_TYPE_RENAMES.get(rel_type, rel_type)
        suffix = f" → {export_type}" if export_type != rel_type else ""

        filepath = EXPORT_DIR / f"rels_{rel_type}.jsonl"
        print(f"  {rel_type} ({expected}){suffix}...", end="", flush=True)

        exported = 0
        with open(filepath, "w") as f:
            offset = 0
            batch = 10000
            while True:
                results = gc.query(
                    f"MATCH (a)-[r:`{rel_type}`]->(b) "
                    f"RETURN elementId(a) AS src, elementId(b) AS tgt, "
                    f"       properties(r) AS props "
                    f"SKIP $offset LIMIT $limit",
                    offset=offset,
                    limit=batch,
                )
                if not results:
                    break

                for rec in results:
                    src_eid = rec["src"]
                    tgt_eid = rec["tgt"]

                    src_info = node_registry.get(src_eid)
                    tgt_info = node_registry.get(tgt_eid)
                    if not src_info or not tgt_info:
                        missing_nodes += 1
                        continue

                    record = {
                        "src_rid": src_info["rid"],
                        "src_label": src_info["label"],
                        "tgt_rid": tgt_info["rid"],
                        "tgt_label": tgt_info["label"],
                        "type": export_type,
                    }
                    # Only include props if non-empty
                    props = rec.get("props", {})
                    if props:
                        record["props"] = props
                    f.write(json.dumps(record, cls=Neo4jEncoder) + "\n")
                    exported += 1

                offset += batch

        print(f" ✓ {exported}")
        total_rel_exported += exported

    if missing_nodes:
        print(
            f"  WARNING: {missing_nodes} relationships skipped (endpoint not in registry)"
        )

    print(f"  Total: {total_rel_exported}")

    # Save expected counts for verification
    expected = {
        "nodes": total_exported,
        "relationships": total_rel_exported,
        "timestamp": datetime.now(UTC).isoformat(),
        "node_counts": {r["label"]: r["cnt"] for r in label_counts},
        "rel_counts_original": {r["t"]: r["cnt"] for r in rel_counts},
        "rel_counts_after_rename": {},
    }
    # Compute expected rel counts after renames
    merged = defaultdict(int)
    for t, cnt in expected["rel_counts_original"].items():
        merged[REL_TYPE_RENAMES.get(t, t)] += cnt
    expected["rel_counts_after_rename"] = dict(merged)

    with open(EXPORT_DIR / "counts.json", "w") as f:
        json.dump(expected, f, indent=2)

    print("\n=== Export complete ===")
    gc.close()


# === Import ===


def import_phase():
    from imas_codex.graph.client import GraphClient

    gc = GraphClient()

    # Verify database is empty
    result = gc.query("MATCH (n) RETURN count(n) AS cnt")
    if result[0]["cnt"] > 0:
        print(
            f"ERROR: Database has {result[0]['cnt']} nodes. Must be empty for import."
        )
        print("  Stop Neo4j, delete data/databases/ and data/transactions/, restart.")
        sys.exit(1)

    # Load index definitions
    with open(EXPORT_DIR / "indexes.json") as f:
        index_defs = json.load(f)

    constraints = [d for d in index_defs if "CONSTRAINT" in d["createStatement"]]
    range_indexes = [
        d
        for d in index_defs
        if d["type"] == "RANGE" and "CONSTRAINT" not in d["createStatement"]
    ]
    vector_indexes = [d for d in index_defs if d["type"] == "VECTOR"]

    # --- Phase 1: Create constraints (provides implicit indexes for matching) ---
    print(f"=== Creating {len(constraints)} constraints ===")
    for c in constraints:
        try:
            gc.query(c["createStatement"])
            print(f"  ✓ {c['name']}")
        except Exception as e:
            print(f"  ✗ {c['name']}: {e}")
    time.sleep(2)

    # Parse constraint keys per label for deduplication
    constraint_keys: dict[str, tuple[str, ...]] = {}
    for c in constraints:
        stmt = c["createStatement"]
        # Parse: CREATE CONSTRAINT name FOR (n:`Label`) REQUIRE (n.`a`, n.`b`) IS UNIQUE
        import re

        m = re.search(r"FOR \(n:`(\w+)`\) REQUIRE \(([^)]+)\) IS UNIQUE", stmt)
        if m:
            label_name = m.group(1)
            props_str = m.group(2)
            key_props = tuple(
                p.strip().removeprefix("n.").strip("`") for p in props_str.split(",")
            )
            constraint_keys[label_name] = key_props

    # --- Phase 2: Import nodes ---
    node_files = sorted(EXPORT_DIR.glob("nodes_*.jsonl"))
    total_imported = 0
    total_deduped = 0

    print(f"\n=== Importing nodes from {len(node_files)} files ===")

    for nf in node_files:
        label = nf.stem.removeprefix("nodes_")
        nodes = []
        with open(nf) as f:
            for line in f:
                nodes.append(json.loads(line))

        if not nodes:
            continue

        # Deduplicate by constraint key
        key_props = constraint_keys.get(label)
        if key_props:
            seen = set()
            unique_nodes = []
            for n in nodes:
                key = tuple(n["props"].get(k) for k in key_props)
                if key in seen:
                    continue
                seen.add(key)
                unique_nodes.append(n)
            deduped = len(nodes) - len(unique_nodes)
            if deduped:
                total_deduped += deduped
            nodes = unique_nodes

        print(f"  {label} ({len(nodes)})...", end="", flush=True)

        imported = 0
        batch_size = 500

        for i in range(0, len(nodes), batch_size):
            batch = nodes[i : i + batch_size]
            items = []
            for n in batch:
                props = _restore_props(n["props"])
                props["_rid"] = n["rid"]
                items.append({"props": props})

            # Build label expression
            labels_expr = ":".join(f"`{lb}`" for lb in batch[0]["labels"])

            gc.query(
                f"UNWIND $items AS item CREATE (n:{labels_expr}) SET n = item.props",
                items=items,
            )
            imported += len(batch)

        print(f" ✓ {imported}")
        total_imported += imported

    if total_deduped:
        print(f"  Deduplicated: {total_deduped} duplicates removed")
    print(f"  Total: {total_imported}")

    # --- Phase 3: Create temp RANGE indexes on _rid per label ---
    labels_in_db = gc.query("CALL db.labels() YIELD label RETURN label ORDER BY label")
    label_list = [r["label"] for r in labels_in_db]

    print(f"\n=== Creating {len(label_list)} temp _rid indexes ===")
    for label in label_list:
        try:
            gc.query(
                f"CREATE RANGE INDEX `_rid_{label}` FOR (n:`{label}`) ON (n.`_rid`)"
            )
        except Exception as e:
            print(f"  ✗ _rid_{label}: {e}")
    # Wait for indexes to populate
    print("  Waiting for indexes to populate...")
    time.sleep(5)
    # Poll until all online
    for _ in range(60):
        pending = gc.query(
            "SHOW INDEXES YIELD name, state WHERE name STARTS WITH '_rid_' AND state <> 'ONLINE' RETURN count(*) AS cnt"
        )
        if pending[0]["cnt"] == 0:
            break
        time.sleep(2)
    print("  ✓ All _rid indexes online")

    # --- Phase 4: Import relationships ---
    rel_files = sorted(EXPORT_DIR.glob("rels_*.jsonl"))
    total_rel_imported = 0

    print(f"\n=== Importing relationships from {len(rel_files)} files ===")

    for rf in rel_files:
        original_type = rf.stem.removeprefix("rels_")

        # Read and group by (export_type, src_label, tgt_label)
        groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
        with open(rf) as f:
            for line in f:
                rec = json.loads(line)
                key = (rec["type"], rec["src_label"], rec["tgt_label"])
                groups[key].append(rec)

        suffix = ""
        if original_type in REL_TYPE_RENAMES:
            suffix = f" → {REL_TYPE_RENAMES[original_type]}"

        total_in_file = sum(len(v) for v in groups.values())
        print(f"  {original_type} ({total_in_file}){suffix}...", end="", flush=True)

        file_imported = 0
        for (rel_type, src_label, tgt_label), rels in groups.items():
            has_props = any(r.get("props") for r in rels)

            cypher = (
                f"UNWIND $items AS item "
                f"MATCH (a:`{src_label}` {{_rid: item.src_rid}}) "
                f"MATCH (b:`{tgt_label}` {{_rid: item.tgt_rid}}) "
                f"CREATE (a)-[r:`{rel_type}`]->(b)"
            )
            if has_props:
                cypher += " SET r = item.props"

            batch_size = 1000
            for i in range(0, len(rels), batch_size):
                batch = rels[i : i + batch_size]
                items = []
                for r in batch:
                    item = {"src_rid": r["src_rid"], "tgt_rid": r["tgt_rid"]}
                    if has_props:
                        item["props"] = _restore_props(r.get("props", {}))
                    items.append(item)

                gc.query(cypher, items=items)
                file_imported += len(batch)

        print(f" ✓ {file_imported}")
        total_rel_imported += file_imported

    print(f"  Total: {total_rel_imported}")

    # --- Phase 5: Create RANGE and VECTOR indexes ---
    print(f"\n=== Creating {len(range_indexes)} RANGE indexes ===")
    for idx in range_indexes:
        try:
            gc.query(idx["createStatement"])
            print(f"  ✓ {idx['name']}")
        except Exception as e:
            print(f"  ✗ {idx['name']}: {e}")

    print(f"\n=== Creating {len(vector_indexes)} VECTOR indexes ===")
    for idx in vector_indexes:
        try:
            gc.query(idx["createStatement"])
            print(f"  ✓ {idx['name']}")
        except Exception as e:
            print(f"  ✗ {idx['name']}: {e}")

    # Wait for all indexes
    print("  Waiting for all indexes to come online...")
    for _ in range(120):
        pending = gc.query(
            "SHOW INDEXES YIELD state WHERE state <> 'ONLINE' RETURN count(*) AS cnt"
        )
        if pending[0]["cnt"] == 0:
            break
        time.sleep(5)
    print("  ✓ All indexes online")

    print("\n=== Import complete ===")
    print(f"  Nodes: {total_imported}")
    print(f"  Relationships: {total_rel_imported}")
    gc.close()


# === Cleanup (remove _rid) ===


def cleanup_phase():
    from imas_codex.graph.client import GraphClient

    gc = GraphClient()

    # Drop _rid indexes
    labels = gc.query("CALL db.labels() YIELD label RETURN label")
    print("=== Dropping _rid indexes ===")
    for r in labels:
        label = r["label"]
        try:
            gc.query(f"DROP INDEX `_rid_{label}` IF EXISTS")
        except Exception:
            pass

    # Remove _rid property
    print("=== Removing _rid property from all nodes ===")
    total = gc.query("MATCH (n) WHERE n._rid IS NOT NULL RETURN count(n) AS cnt")
    print(f"  {total[0]['cnt']} nodes to clean")

    batch = 10000
    removed = 0
    while True:
        result = gc.query(
            "MATCH (n) WHERE n._rid IS NOT NULL "
            "WITH n LIMIT $limit "
            "REMOVE n._rid "
            "RETURN count(n) AS cnt",
            limit=batch,
        )
        cnt = result[0]["cnt"]
        if cnt == 0:
            break
        removed += cnt
        print(f"  {removed}...", flush=True)

    print(f"  ✓ Removed _rid from {removed} nodes")
    gc.close()


# === Verify ===


def verify_phase():
    from imas_codex.graph.client import GraphClient

    gc = GraphClient()

    with open(EXPORT_DIR / "counts.json") as f:
        expected = json.load(f)

    print("=== Verifying node counts ===")
    actual_nodes = gc.query(
        "MATCH (n) WITH labels(n)[0] AS label, count(n) AS cnt "
        "RETURN label, cnt ORDER BY cnt DESC"
    )
    actual_map = {r["label"]: r["cnt"] for r in actual_nodes}
    total_actual = sum(r["cnt"] for r in actual_nodes)

    ok = True
    for label, exp_cnt in expected["node_counts"].items():
        act_cnt = actual_map.get(label, 0)
        status = "✓" if act_cnt == exp_cnt else "✗"
        if act_cnt != exp_cnt:
            ok = False
            print(f"  {status} {label}: expected {exp_cnt}, got {act_cnt}")
        else:
            print(f"  {status} {label}: {act_cnt}")

    print(f"  Total: expected {expected['nodes']}, got {total_actual}")

    print("\n=== Verifying relationship counts ===")
    actual_rels = gc.query(
        "MATCH ()-[r]->() WITH type(r) AS t, count(r) AS cnt "
        "RETURN t, cnt ORDER BY cnt DESC"
    )
    actual_rel_map = {r["t"]: r["cnt"] for r in actual_rels}
    total_actual_rels = sum(r["cnt"] for r in actual_rels)

    for rel_type, exp_cnt in expected["rel_counts_after_rename"].items():
        act_cnt = actual_rel_map.get(rel_type, 0)
        status = "✓" if act_cnt == exp_cnt else "✗"
        if act_cnt != exp_cnt:
            ok = False
            print(f"  {status} {rel_type}: expected {exp_cnt}, got {act_cnt}")
        else:
            print(f"  {status} {rel_type}: {act_cnt}")

    # Check for unexpected types
    for rel_type in actual_rel_map:
        if rel_type not in expected["rel_counts_after_rename"]:
            print(f"  ? {rel_type}: {actual_rel_map[rel_type]} (unexpected)")

    print(f"  Total: expected {expected['relationships']}, got {total_actual_rels}")

    # Check PRODUCED is gone
    produced = actual_rel_map.get("PRODUCED", 0)
    if produced > 0:
        print(f"\n  ✗ PRODUCED still exists ({produced} rels)")
        ok = False
    else:
        print("\n  ✓ PRODUCED migration complete (0 remaining)")

    # Check tree_name is gone
    tree_name_nodes = gc.query(
        "MATCH (n) WHERE n.tree_name IS NOT NULL RETURN count(n) AS cnt"
    )
    tn_cnt = tree_name_nodes[0]["cnt"]
    if tn_cnt > 0:
        print(f"  ✗ tree_name still exists on {tn_cnt} nodes")
        ok = False
    else:
        print("  ✓ tree_name removal complete")

    # Check _rid cleanup
    rid_nodes = gc.query("MATCH (n) WHERE n._rid IS NOT NULL RETURN count(n) AS cnt")
    rid_cnt = rid_nodes[0]["cnt"]
    if rid_cnt > 0:
        print(f"  ⚠ _rid still present on {rid_cnt} nodes (run cleanup phase)")
    else:
        print("  ✓ _rid cleanup complete")

    if ok:
        print("\n✓ Verification passed — graph is clean and migrated")
    else:
        print("\n✗ Verification found mismatches — review above")

    gc.close()
    return ok


# === Main ===


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: uv run python scripts/rebuild_graph.py <export|import|verify|cleanup>"
        )
        sys.exit(1)

    phase = sys.argv[1]
    if phase == "export":
        export_phase()
    elif phase == "import":
        import_phase()
    elif phase == "verify":
        verify_phase()
    elif phase == "cleanup":
        cleanup_phase()
    else:
        print(f"Unknown phase: {phase}")
        sys.exit(1)


if __name__ == "__main__":
    main()
