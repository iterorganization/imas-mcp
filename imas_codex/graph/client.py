"""Neo4j client for the facility knowledge graph.

This module provides a high-level client for interacting with the Neo4j
knowledge graph, including schema initialization and CRUD operations.

The graph structure is derived from the LinkML schema (schemas/facility.yaml)
via the GraphSchema class, which is the single source of truth.

Example:
    >>> from imas_codex.graph import GraphClient
    >>> with GraphClient() as client:
    ...     client.initialize_schema()
    ...     client.create_node("Facility", "epfl", {"name": "EPFL/TCV"})
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from neo4j import Driver, GraphDatabase, Session

# Lazy import for schema - only needed for admin operations, not queries
# This avoids ~25s linkml_runtime import overhead for query-only usage
if TYPE_CHECKING:
    from imas_codex.graph.schema import GraphSchema

# Suppress noisy Neo4j warnings about unknown property keys
# These are harmless (e.g., retry_count doesn't exist until first failure)
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)


@dataclass
class GraphClient:
    """Client for Neo4j knowledge graph operations.

    The graph structure is derived from the LinkML schema (schemas/facility.yaml)
    via GraphSchema, which provides node labels, relationship types, and constraints.

    Attributes:
        uri: Neo4j Bolt URI (default: bolt://localhost:7687)
        username: Neo4j username (default: neo4j)
        password: Neo4j password (default: imas-codex)
    """

    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "imas-codex"
    _driver: Driver | None = field(default=None, init=False, repr=False)
    _schema: GraphSchema | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the Neo4j driver."""
        self._driver = GraphDatabase.driver(
            self.uri, auth=(self.username, self.password)
        )
        # Schema loaded lazily on first access to avoid linkml_runtime import

    @property
    def schema(self) -> GraphSchema:
        """Get the graph schema (lazy loaded).

        The schema is only loaded when first accessed, avoiding the ~25s
        linkml_runtime import overhead for query-only operations.
        """
        if self._schema is None:
            from imas_codex.graph.schema import get_schema

            self._schema = get_schema()
        return self._schema

    def close(self) -> None:
        """Close the Neo4j driver."""
        if self._driver:
            self._driver.close()
            self._driver = None

    def __enter__(self) -> GraphClient:
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
        """
        # Get constraints from schema (derived from identifier fields)
        constraints = self.schema.constraint_statements()

        # Get indexes from schema (common lookup patterns)
        indexes = self.schema.index_statements()

        with self.session() as sess:
            for stmt in constraints + indexes:
                sess.run(stmt)

        # Create vector indexes for semantic search
        self.ensure_vector_indexes()

    def ensure_vector_indexes(self) -> None:
        """Create vector indexes for semantic search if they don't exist.

        Creates a vector index on CodeChunk.embedding for similarity search.
        Requires Neo4j 5.x+ with vector index support.
        """
        # Check if vector index exists
        with self.session() as sess:
            result = sess.run(
                "SHOW INDEXES YIELD name WHERE name = 'code_chunk_embedding' RETURN name"
            )
            if result.single():
                return  # Index already exists

            # Create vector index for CodeChunk embeddings
            # Using 384 dimensions for all-MiniLM-L6-v2 model
            try:
                sess.run("""
                    CREATE VECTOR INDEX code_chunk_embedding IF NOT EXISTS
                    FOR (c:CodeChunk) ON c.embedding
                    OPTIONS {
                        indexConfig: {
                            `vector.dimensions`: 384,
                            `vector.similarity_function`: 'cosine'
                        }
                    }
                """)
            except Exception as e:
                # Vector indexes may not be available in all Neo4j editions
                import logging

                logging.getLogger(__name__).warning(
                    f"Failed to create vector index (may require Neo4j 5.x+): {e}"
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
        facility_id_field: str | None = "facility_id",
    ) -> dict[str, int]:
        """Create or update multiple nodes using UNWIND for efficiency.

        Uses batched UNWIND queries for optimal Neo4j performance.
        Optionally creates FACILITY_ID relationships in the same transaction.

        Args:
            label: Node label (class name from schema)
            items: List of property dicts, each must contain id_field
            id_field: Name of the identifier field (default: "id")
            batch_size: Number of nodes per UNWIND batch (default: 50)
            facility_id_field: If set, create FACILITY_ID relationships
                for items containing this field. Set to None to skip.

        Returns:
            Dict with counts: {"processed": N}

        Example:
            >>> client.create_nodes("FacilityPath", [
            ...     {"id": "epfl:/home/codes", "path": "/home/codes", "facility_id": "epfl"},
            ...     {"id": "epfl:/home/anasrv", "path": "/home/anasrv", "facility_id": "epfl"},
            ... ])
            {"processed": 2}
        """
        if not items:
            return {"processed": 0}

        processed = 0

        # Build query with optional facility relationship
        if facility_id_field and label != "Facility":
            # Combined node + relationship creation
            query = f"""
                UNWIND $batch AS item
                MERGE (n:{label} {{{id_field}: item.{id_field}}})
                SET n += item
                WITH n, item
                WHERE item.{facility_id_field} IS NOT NULL
                MATCH (f:Facility {{id: item.{facility_id_field}}})
                MERGE (n)-[:FACILITY_ID]->(f)
            """
        else:
            # Node creation only
            query = f"""
                UNWIND $batch AS item
                MERGE (n:{label} {{{id_field}: item.{id_field}}})
                SET n += item
            """

        with self.session() as sess:
            for i in range(0, len(items), batch_size):
                batch = items[i : i + batch_size]
                sess.run(query, batch=batch)
                processed += len(batch)

        return {"processed": processed}

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
            "FACILITY_ID",
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
            "FACILITY_ID",
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
            MATCH (n:{label})-[:FACILITY_ID]->(f:Facility)
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
                MATCH (n:{label})-[:FACILITY_ID]->(f:Facility)
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
