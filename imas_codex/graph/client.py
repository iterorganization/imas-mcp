"""Neo4j client for the facility knowledge graph.

This module provides a high-level client for interacting with the Neo4j
knowledge graph, including schema initialization and CRUD operations.

Connection settings resolve via named graph profiles — see
:mod:`imas_codex.graph.profiles` for the full resolution chain.

Example:
    >>> from imas_codex.graph import GraphClient
    >>> with GraphClient() as client:
    ...     client.initialize_schema()
    ...     client.create_node("Facility", "tcv", {"name": "EPFL/TCV"})

    >>> # Connect to a specific graph profile
    >>> with GraphClient.from_profile("tcv") as client:
    ...     print(client.get_stats())
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Self

from neo4j import Driver, GraphDatabase, Session

from imas_codex.graph.profiles import resolve_graph
from imas_codex.graph.schema import GraphSchema, get_schema
from imas_codex.settings import (
    get_embedding_dimension,
    get_graph_password,
    get_graph_uri,
    get_graph_username,
)

# Suppress noisy Neo4j warnings about unknown property keys
# These are harmless (e.g., retry_count doesn't exist until first failure)
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


# =============================================================================
# Vector Index Configuration (Schema-Derived)
# =============================================================================


def _get_all_vector_indexes() -> list[tuple[str, str, str]]:
    """Derive all expected vector indexes from LinkML schemas.

    Returns list of (index_name, node_label, property_name) tuples.
    Combines facility.yaml and imas_dd.yaml schemas, deduplicating.
    """
    from pathlib import Path

    from imas_codex.graph.schema import GraphSchema

    schemas_dir = Path(__file__).parent.parent / "schemas"

    # Collect from both schemas
    indexes = {}
    for schema_file in ["facility.yaml", "imas_dd.yaml"]:
        gs = GraphSchema(schemas_dir / schema_file)
        for idx in gs.vector_indexes:
            # Deduplicate by index name (same index may appear in both)
            indexes[idx[0]] = idx

    return list(indexes.values())


# Cache the result (computed once at module load)
EXPECTED_VECTOR_INDEXES: list[tuple[str, str, str]] = _get_all_vector_indexes()


# =============================================================================
# Relationship Types (Schema-Derived)
# =============================================================================


def _get_all_relationship_types() -> set[str]:
    """Derive all expected relationship types from LinkML schemas.

    Returns a set of SCREAMING_SNAKE_CASE relationship type names.
    Combines facility.yaml and imas_dd.yaml schemas.
    """
    from pathlib import Path

    from imas_codex.graph.schema import GraphSchema

    schemas_dir = Path(__file__).parent.parent / "schemas"

    # Collect from both schemas
    rel_types = set()
    for schema_file in ["facility.yaml", "imas_dd.yaml"]:
        gs = GraphSchema(schemas_dir / schema_file)
        rel_types.update(gs.relationship_types)

    return rel_types


# Cache the result (computed once at module load)
EXPECTED_RELATIONSHIP_TYPES: set[str] = _get_all_relationship_types()


@dataclass
class GraphClient:
    """Client for Neo4j knowledge graph operations.

    The graph structure is derived from the LinkML schema (schemas/facility.yaml)
    via GraphSchema, which provides node labels, relationship types, and constraints.

    Connection settings are resolved via named graph profiles:
        1. ``IMAS_CODEX_GRAPH`` env selects the profile (default: ``"iter"``)
        2. Profiles map to ports by convention (iter→7687, tcv→7688, …)
        3. ``NEO4J_URI`` / ``NEO4J_PASSWORD`` env vars override any profile

    Construct with no arguments to use the active profile, or pass a
    profile name explicitly::

        GraphClient()                        # active profile
        GraphClient.from_profile("tcv")      # specific profile

    Attributes:
        uri: Neo4j Bolt URI
        username: Neo4j username (default: neo4j)
        password: Neo4j password
        profile_name: Name of the resolved graph profile
    """

    uri: str = field(default_factory=get_graph_uri)
    username: str = field(default_factory=get_graph_username)
    password: str = field(default_factory=get_graph_password)
    profile_name: str = field(default="")
    _driver: Driver | None = field(default=None, init=False, repr=False)
    _schema: GraphSchema | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the Neo4j driver."""
        if not self.profile_name:
            from imas_codex.graph.profiles import get_active_graph_name

            self.profile_name = get_active_graph_name()

        # Lightweight conflict warning: if both a tunnel and local Neo4j
        # might be on the same port, warn the user.
        try:
            from imas_codex.graph.profiles import is_port_bound_by_tunnel

            # Extract port from URI for the check
            if ":" in self.uri:
                port_str = self.uri.rsplit(":", 1)[-1].split("/")[0]
                if port_str.isdigit():
                    port = int(port_str)
                    if is_port_bound_by_tunnel(port):
                        logger.warning(
                            "Port %d appears bound by an SSH tunnel AND "
                            "may conflict with a local Neo4j instance. "
                            "Consider setting IMAS_CODEX_TUNNEL_BOLT_{HOST} "
                            "in .env to use a different tunnel port.",
                            port,
                        )
        except Exception:
            pass  # Best-effort check, never block initialization

        self._driver = GraphDatabase.driver(
            self.uri, auth=(self.username, self.password)
        )
        self._schema = get_schema()

    @classmethod
    def from_profile(cls, name: str) -> "GraphClient":
        """Create a GraphClient connected to a specific named graph profile.

        Args:
            name: Profile name (e.g. ``"tcv"``, ``"jt60sa"``).

        Returns:
            GraphClient connected to the resolved profile.
        """
        profile = resolve_graph(name)
        return cls(
            uri=profile.uri,
            username=profile.username,
            password=profile.password,
            profile_name=profile.name,
        )

    @property
    def schema(self) -> GraphSchema:
        """Get the graph schema."""
        return self._schema  # type: ignore[return-value]

    def close(self) -> None:
        """Close the Neo4j driver."""
        if self._driver:
            self._driver.close()
            self._driver = None

    def __enter__(self) -> Self:
        """Enter context manager."""
        return self

    def __exit__(self, *_: Any) -> None:
        """Exit context manager."""
        self.close()

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Get a Neo4j session as a context manager."""
        if not self._driver:
            msg = "GraphClient is closed"
            raise RuntimeError(msg)
        sess = self._driver.session()
        try:
            yield sess
        finally:
            sess.close()

    # =========================================================================
    # Schema Management
    # =========================================================================

    def initialize_schema(self) -> None:
        """Create constraints and indexes for the schema.

        Constraints and indexes are derived from the LinkML schema:
        - Unique constraints on identifier fields
        - Indexes for common query patterns
        - Vector indexes for semantic search

        Handles conflicts where a RANGE index exists with the same name
        as a constraint (e.g., from DD build) by dropping the index first.
        """
        # Get constraints from schema (derived from identifier fields)
        constraints = self.schema.constraint_statements()

        # Get indexes from schema (common lookup patterns)
        indexes = self.schema.index_statements()

        with self.session() as sess:
            # Get existing indexes and constraints to detect conflicts
            existing_indexes = {
                r["name"]
                for r in sess.run(
                    "SHOW INDEXES YIELD name, type WHERE type <> 'LOOKUP' RETURN name, type"
                )
                if r["type"] != "UNIQUENESS"  # Constraint-backed indexes are fine
            }
            existing_constraints = {
                r["name"] for r in sess.run("SHOW CONSTRAINTS YIELD name RETURN name")
            }

            for stmt in constraints:
                # Extract constraint name from "CREATE CONSTRAINT <name> IF NOT EXISTS ..."
                parts = stmt.split()
                if (
                    len(parts) >= 3
                    and parts[0] == "CREATE"
                    and parts[1] == "CONSTRAINT"
                ):
                    constraint_name = parts[2]
                    # Drop conflicting RANGE index if it exists
                    if (
                        constraint_name in existing_indexes
                        and constraint_name not in existing_constraints
                    ):
                        sess.run(f"DROP INDEX {constraint_name}")
                        logger.debug(f"Dropped conflicting index: {constraint_name}")
                sess.run(stmt)

            for stmt in indexes:
                sess.run(stmt)

        # Create vector indexes for semantic search
        self.ensure_vector_indexes()

    def ensure_vector_indexes(self) -> None:
        """Create vector indexes for semantic search if they don't exist.

        Creates all vector indexes derived from LinkML schemas (classes with
        embedding or centroid slots). Index configuration comes from:
        - GraphSchema.vector_indexes property (schema-derived)
        - vector_index_name annotation for custom index names

        Requires Neo4j 5.x+ with vector index support.
        """
        dim = get_embedding_dimension()

        with self.session() as sess:
            # Get existing indexes
            result = sess.run(
                "SHOW INDEXES YIELD name WHERE name IN $names RETURN name",
                names=[idx[0] for idx in EXPECTED_VECTOR_INDEXES],
            )
            existing = {record["name"] for record in result}

            for index_name, label, prop in EXPECTED_VECTOR_INDEXES:
                if index_name in existing:
                    continue  # Index already exists

                try:
                    sess.run(f"""
                        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                        FOR (n:{label}) ON n.{prop}
                        OPTIONS {{
                            indexConfig: {{
                                `vector.dimensions`: {dim},
                                `vector.similarity_function`: 'cosine'
                            }}
                        }}
                    """)
                    logger.debug(f"Created vector index: {index_name}")
                except Exception as e:
                    # Vector indexes may not be available in all Neo4j editions
                    logger.warning(
                        f"Failed to create vector index {index_name} "
                        f"(may require Neo4j 5.x+): {e}"
                    )

    def drop_all(self) -> int:
        """Delete all nodes and relationships.

        Returns:
            Number of nodes deleted.
        """
        with self.session() as sess:
            result = sess.run("MATCH (n) DETACH DELETE n RETURN count(n) as count")
            record = result.single()
            return record["count"] if record else 0

    def get_stats(self) -> dict[str, int]:
        """Get basic node and relationship counts.

        Returns:
            Dictionary with total node and relationship counts.
        """
        with self.session() as sess:
            node_result = sess.run("MATCH (n) RETURN count(n) as count")
            node_record = node_result.single()
            node_count = node_record["count"] if node_record else 0

            rel_result = sess.run("MATCH ()-[r]->() RETURN count(r) as count")
            rel_record = rel_result.single()
            rel_count = rel_record["count"] if rel_record else 0

            return {"nodes": node_count, "relationships": rel_count}

    def get_label_counts(self) -> dict[str, int]:
        """Get node counts per label.

        Returns:
            Dictionary mapping label names to counts.
        """
        with self.session() as sess:
            result = sess.run(
                "MATCH (n) RETURN labels(n)[0] as label, count(*) as count"
            )
            return {record["label"]: record["count"] for record in result}

    # =========================================================================
    # Generic Node Operations
    # =========================================================================

    def create_node(
        self,
        label: str,
        node_id: Any,
        props: dict[str, Any],
        id_field: str = "id",
    ) -> None:
        """Create or update a node.

        Args:
            label: Node label (class name from schema)
            node_id: Value of the identifier field
            props: Node properties
            id_field: Name of the identifier field (default: "id")
        """
        with self.session() as sess:
            sess.run(
                f"MERGE (n:{label} {{{id_field}: $id}}) SET n += $props",
                id=node_id,
                props=props,
            )

    def create_nodes(
        self,
        label: str,
        items: list[dict[str, Any]],
        id_field: str = "id",
        batch_size: int = 50,
        create_relationships: bool = True,
    ) -> dict[str, int]:
        """Create or update multiple nodes using UNWIND for efficiency.

        Uses batched UNWIND queries for optimal Neo4j performance.
        Automatically creates relationships based on schema-defined slots
        with class ranges (e.g., facility_id -> Facility creates AT_FACILITY edge).

        Args:
            label: Node label (class name from schema)
            items: List of property dicts, each must contain id_field
            id_field: Name of the identifier field (default: "id")
            batch_size: Number of nodes per UNWIND batch (default: 50)
            create_relationships: If True, create edges for all schema-defined
                relationship fields found in items. Default True.

        Returns:
            Dict with counts: {"processed": N, "relationships": {rel_type: count}}

        Example:
            >>> client.create_nodes("FacilityPath", [
            ...     {"id": "tcv:/home/codes", "path": "/home/codes", "facility_id": "tcv"},
            ...     {"id": "tcv:/home/anasrv", "path": "/home/anasrv", "facility_id": "tcv"},
            ... ])
            {"processed": 2, "relationships": {"AT_FACILITY": 2}}
        """
        if not items:
            return {"processed": 0, "relationships": {}}

        processed = 0
        rel_counts: dict[str, int] = {}

        # Node creation query (always runs first)
        node_query = f"""
            UNWIND $batch AS item
            MERGE (n:{label} {{{id_field}: item.{id_field}}})
            SET n += item
        """

        # Get schema-defined relationships for this class
        relationships = (
            self.schema.get_relationships_from(label) if create_relationships else []
        )

        with self.session() as sess:
            for i in range(0, len(items), batch_size):
                batch = items[i : i + batch_size]

                # Create nodes
                sess.run(node_query, batch=batch)
                processed += len(batch)

                # Create relationships based on schema
                for rel in relationships:
                    # Check if any items in batch have this relationship field
                    rel_batch = [
                        item for item in batch if item.get(rel.slot_name) is not None
                    ]
                    if not rel_batch:
                        continue

                    # Build relationship query
                    rel_query = f"""
                        UNWIND $batch AS item
                        MATCH (n:{label} {{{id_field}: item.{id_field}}})
                        MATCH (t:{rel.to_class} {{id: item.{rel.slot_name}}})
                        MERGE (n)-[:{rel.cypher_type}]->(t)
                    """
                    sess.run(rel_query, batch=rel_batch)
                    rel_counts[rel.cypher_type] = rel_counts.get(
                        rel.cypher_type, 0
                    ) + len(rel_batch)

        return {"processed": processed, "relationships": rel_counts}

    def create_relationship(
        self,
        from_label: str,
        from_id: Any,
        to_label: str,
        to_id: Any,
        rel_type: str,
        from_id_field: str = "id",
        to_id_field: str = "id",
        props: dict[str, Any] | None = None,
    ) -> None:
        """Create a relationship between two nodes.

        Args:
            from_label: Source node label (class name from schema)
            from_id: Source node identifier value
            to_label: Target node label (class name from schema)
            to_id: Target node identifier value
            rel_type: Relationship type (SCREAMING_SNAKE_CASE)
            from_id_field: Source node identifier field name
            to_id_field: Target node identifier field name
            props: Optional relationship properties
        """
        props_clause = " SET r += $props" if props else ""
        query = (
            f"MATCH (a:{from_label} {{{from_id_field}: $from_id}}), "
            f"(b:{to_label} {{{to_id_field}: $to_id}}) "
            f"MERGE (a)-[r:{rel_type}]->(b){props_clause}"
        )
        with self.session() as sess:
            sess.run(query, from_id=from_id, to_id=to_id, props=props or {})

    # =========================================================================
    # High-Level Creation Methods
    # =========================================================================

    def ensure_facility(self, facility_id: str) -> None:
        """Ensure a Facility node exists, creating a minimal one if needed.

        This should be called at the start of any discovery pipeline to
        guarantee that AT_FACILITY relationships won't silently fail due
        to a missing target node.

        Uses MERGE so it's safe to call multiple times (idempotent).

        Args:
            facility_id: Facility identifier (e.g., "tcv", "iter")
        """
        with self.session() as sess:
            sess.run(
                "MERGE (f:Facility {id: $id}) "
                "ON CREATE SET f.name = $id, f.created_at = datetime()",
                id=facility_id,
            )

    def create_facility(
        self,
        facility_id: str,
        name: str,
        ssh_host: str | None = None,
        machine: str | None = None,
        **extra: Any,
    ) -> None:
        """Create or update a Facility node."""
        props = {"id": facility_id, "name": name}
        if ssh_host:
            props["ssh_host"] = ssh_host
        if machine:
            props["machine"] = machine
        props.update({k: v for k, v in extra.items() if v is not None})
        self.create_node("Facility", facility_id, props)

    def create_mdsplus_server(
        self,
        hostname: str,
        facility_id: str,
        role: str | None = None,
        **extra: Any,
    ) -> None:
        """Create MDSplusServer node and link to Facility."""
        self.ensure_facility(facility_id)
        props = {"hostname": hostname, "facility_id": facility_id}
        if role:
            props["role"] = role
        props.update({k: v for k, v in extra.items() if v is not None})
        self.create_node("MDSplusServer", hostname, props, id_field="hostname")
        self.create_relationship(
            "MDSplusServer",
            hostname,
            "Facility",
            facility_id,
            "AT_FACILITY",
            from_id_field="hostname",
        )

    def create_diagnostic(
        self,
        facility_id: str,
        name: str,
        category: str | None = None,
        description: str | None = None,
        **extra: Any,
    ) -> None:
        """Create Diagnostic node and link to Facility."""
        self.ensure_facility(facility_id)
        props = {"name": name, "facility_id": facility_id}
        if category:
            props["category"] = category
        if description:
            props["description"] = description
        props.update({k: v for k, v in extra.items() if v is not None})
        self.create_node("Diagnostic", name, props, id_field="name")
        self.create_relationship(
            "Diagnostic",
            name,
            "Facility",
            facility_id,
            "AT_FACILITY",
            from_id_field="name",
        )

    def create_imas_path(
        self,
        path: str,
        ids: str,
        description: str | None = None,
        units: str | None = None,
    ) -> None:
        """Create IMASPath node."""
        props = {"path": path, "ids": ids}
        if description:
            props["description"] = description
        if units:
            props["units"] = units
        self.create_node("IMASPath", path, props, id_field="path")

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_facility(self, facility_id: str) -> dict[str, Any] | None:
        """Get a facility by ID."""
        with self.session() as sess:
            result = sess.run(
                "MATCH (n:Facility {id: $id}) RETURN n",
                id=facility_id,
            )
            record = result.single()
            return dict(record["n"]) if record else None

    def get_facilities(self) -> list[dict[str, Any]]:
        """Get all facilities."""
        with self.session() as sess:
            result = sess.run("MATCH (n:Facility) RETURN n ORDER BY n.id")
            return [dict(record["n"]) for record in result]

    # =========================================================================
    # Cross-Facility Query Methods
    # =========================================================================

    def get_nodes_by_facility(
        self, label: str, facility_id: str
    ) -> list[dict[str, Any]]:
        """Get all nodes of a type for a specific facility.

        Args:
            label: Node label (e.g., "Diagnostic", "Tool", "TreeNode")
            facility_id: Facility identifier

        Returns:
            List of node properties as dicts.
        """
        with self.session() as sess:
            result = sess.run(
                f"MATCH (n:{label} {{facility_id: $facility_id}}) RETURN n",
                facility_id=facility_id,
            )
            return [dict(record["n"]) for record in result]

    def get_nodes_across_facilities(
        self, label: str, **filters: Any
    ) -> list[dict[str, Any]]:
        """Get nodes of a type across all facilities with optional filters.

        Args:
            label: Node label (e.g., "Diagnostic", "Tool")
            **filters: Property filters (e.g., category="magnetics")

        Returns:
            List of dicts with node properties and facility_id.
        """
        where_clause = ""
        if filters:
            conditions = [f"n.{k} = ${k}" for k in filters]
            where_clause = "WHERE " + " AND ".join(conditions)

        query = f"""
            MATCH (n:{label})-[:AT_FACILITY]->(f:Facility)
            {where_clause}
            RETURN n, f.id as facility_id
            ORDER BY f.id, n.name
        """
        with self.session() as sess:
            result = sess.run(query, **filters)
            return [
                {**dict(record["n"]), "facility_id": record["facility_id"]}
                for record in result
            ]

    def compare_facilities(
        self, label: str, facility_ids: list[str]
    ) -> dict[str, list[dict[str, Any]]]:
        """Compare nodes of a type across multiple facilities.

        Args:
            label: Node label (e.g., "Tool", "Diagnostic")
            facility_ids: List of facility IDs to compare

        Returns:
            Dict mapping facility_id to list of nodes.
        """
        result: dict[str, list[dict[str, Any]]] = {fid: [] for fid in facility_ids}
        with self.session() as sess:
            query_result = sess.run(
                f"""
                MATCH (n:{label})-[:AT_FACILITY]->(f:Facility)
                WHERE f.id IN $facility_ids
                RETURN n, f.id as facility_id
                ORDER BY f.id
                """,
                facility_ids=facility_ids,
            )
            for record in query_result:
                fid = record["facility_id"]
                if fid in result:
                    result[fid].append(dict(record["n"]))
        return result

    def find_shared_imas_mappings(
        self, facility_ids: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Find IMAS paths that are mapped at multiple facilities.

        Args:
            facility_ids: Optional list of facilities to check.
                         If None, checks all facilities.

        Returns:
            List of dicts with IMAS path and facilities that map to it.
        """
        where_clause = ""
        if facility_ids:
            where_clause = "WHERE m.facility_id IN $facility_ids"

        query = f"""
            MATCH (m:IMASMapping)-[:TARGET_PATH]->(imas:IMASPath)
            {where_clause}
            WITH imas, collect(DISTINCT m.facility_id) as facilities
            WHERE size(facilities) > 1
            RETURN imas.path as path, imas.ids as ids, facilities
            ORDER BY size(facilities) DESC, imas.path
        """
        params = {"facility_ids": facility_ids} if facility_ids else {}
        with self.session() as sess:
            result = sess.run(query, **params)
            return [dict(record) for record in result]

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        """Execute a Cypher query and return results as dicts."""
        with self.session() as sess:
            result = sess.run(cypher, **params)
            return [dict(record) for record in result]
