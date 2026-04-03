"""Neo4j client for the facility knowledge graph.

This module provides a high-level client for interacting with the Neo4j
knowledge graph, including schema initialization and CRUD operations.

Connection settings resolve via the active graph profile — see
:mod:`imas_codex.graph.profiles` for the full resolution chain.

Example:
    >>> from imas_codex.graph import GraphClient
    >>> with GraphClient() as client:
    ...     client.initialize_schema()
    ...     client.create_node("Facility", "tcv", {"name": "EPFL/TCV"})

    >>> # Connect via the active graph profile
    >>> with GraphClient.from_profile() as client:
    ...     print(client.get_stats())
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Self

from neo4j import Driver, GraphDatabase, Session

from imas_codex.graph.profiles import resolve_neo4j
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
logging.getLogger("neo4j.pool").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


# =============================================================================
# Vector Index Configuration (From Pre-Built Schema Context Data)
# =============================================================================

# Use pre-built schema context data (generated during uv sync) for indexes.
# This avoids loading LinkML schemas at import time, which fails on Windows
# due to symlink issues and is slower than using pre-computed data.
try:
    from imas_codex.graph.schema_context_data import (
        FULLTEXT_INDEXES as _FULLTEXT_INDEXES_DATA,
        VECTOR_INDEXES as _VECTOR_INDEXES_DATA,
    )

    # Convert schema_context_data format: {name: (label, prop)} -> [(name, label, prop)]
    EXPECTED_VECTOR_INDEXES: list[tuple[str, str, str]] = [
        (name, label, prop) for name, (label, prop) in _VECTOR_INDEXES_DATA.items()
    ]
    EXPECTED_FULLTEXT_INDEXES: list[tuple[str, str, list[str]]] = [
        (name, label, props) for name, (label, props) in _FULLTEXT_INDEXES_DATA.items()
    ]
except (ImportError, SyntaxError) as e:
    # During build phase, schema_context_data.py may not exist or be corrupted.
    # Provide empty defaults - ensureindex methods will be no-ops.
    logger.debug(
        f"schema_context_data.py not available ({e}) - using empty index lists. "
        "Run 'uv run build-models --force' to regenerate."
    )
    EXPECTED_VECTOR_INDEXES = []
    EXPECTED_FULLTEXT_INDEXES = []


# =============================================================================
# Relationship Types (From Pre-Built Schema Context Data)
# =============================================================================

try:
    from imas_codex.graph.schema_context_data import RELATIONSHIPS

    # RELATIONSHIPS format: [(from_label, rel_type, to_label, cardinality), ...]
    EXPECTED_RELATIONSHIP_TYPES: set[str] = {rel[1] for rel in RELATIONSHIPS}
except (ImportError, SyntaxError):
    # Fallback during build or when file is corrupted
    EXPECTED_RELATIONSHIP_TYPES = set()


@dataclass
class GraphClient:
    """Client for Neo4j knowledge graph operations.

    The graph structure is derived from the LinkML schema (schemas/facility.yaml)
    via GraphSchema, which provides node labels, relationship types, and constraints.

    Connection settings are resolved via the active graph profile:
        1. ``neo4j/`` symlink points to the active graph in ``.neo4j/<hash>/``
        2. ``IMAS_CODEX_GRAPH_LOCATION`` env selects where Neo4j runs
        3. Locations map to ports by convention (iter→7687, tcv→7688, …)
        4. ``NEO4J_URI`` / ``NEO4J_PASSWORD`` env vars override any profile

    Construct with no arguments to use the active profile::

        GraphClient()
        GraphClient.from_profile()  # explicit profile resolution

    Attributes:
        uri: Neo4j Bolt URI
        username: Neo4j username (default: neo4j)
        password: Neo4j password
        graph_name: Name of the resolved graph (data identity)
    """

    uri: str = field(default_factory=get_graph_uri)
    username: str = field(default_factory=get_graph_username)
    password: str = field(default_factory=get_graph_password)
    graph_name: str = field(default="")
    _driver: Driver | None = field(default=None, init=False, repr=False)
    _schema: GraphSchema | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the Neo4j driver."""
        if not self.graph_name:
            from imas_codex.graph.profiles import get_active_graph_name

            self.graph_name = get_active_graph_name()

        self._driver = GraphDatabase.driver(
            self.uri,
            auth=(self.username, self.password),
            **self._driver_kwargs(self.uri),
        )
        self._schema = get_schema()

    @classmethod
    def from_profile(cls) -> "GraphClient":
        """Create a GraphClient connected via the active graph profile.

        Returns:
            GraphClient connected to the resolved profile.
        """
        profile = resolve_neo4j()
        return cls(
            uri=profile.uri,
            username=profile.username,
            password=profile.password,
            graph_name=profile.name,
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

    @staticmethod
    def _pool_size_for_uri(uri: str) -> int:
        """Choose a conservative pool size for tunneled connections."""
        from imas_codex.remote.tunnel import TUNNEL_OFFSET

        pool_size = 100
        try:
            port_str = uri.rsplit(":", 1)[-1].rstrip("/")
            port = int(port_str)
            if port >= TUNNEL_OFFSET:
                pool_size = 5
        except (ValueError, IndexError):
            pass
        return pool_size

    @classmethod
    def _driver_kwargs(cls, uri: str) -> dict[str, Any]:
        """Build Neo4j driver kwargs with bounded pooled connection reuse."""
        return {
            "max_connection_pool_size": cls._pool_size_for_uri(uri),
            "connection_acquisition_timeout": 30,
            "max_connection_lifetime": 300,
        }

    def _recreate_driver(self, uri: str | None = None) -> None:
        """Close and recreate the Neo4j driver, optionally with a new URI."""
        target_uri = uri or self.uri
        if self._driver:
            try:
                self._driver.close()
            except Exception:
                pass
        self._driver = GraphDatabase.driver(
            target_uri,
            auth=(self.username, self.password),
            **self._driver_kwargs(target_uri),
        )
        self.uri = target_uri

    def __enter__(self) -> Self:
        """Enter context manager."""
        return self

    def __exit__(self, *_: Any) -> None:
        """Exit context manager."""
        self.close()

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Get a Neo4j session as a context manager.

        On connection failure (dead tunnel), attempts to re-establish the
        tunnel and recreate the driver before raising.
        """
        if not self._driver:
            msg = "GraphClient is closed"
            raise RuntimeError(msg)
        try:
            sess = self._driver.session()
        except Exception as exc:
            if self._is_connection_error(exc) and self._try_reconnect(exc):
                sess = self._driver.session()  # type: ignore[union-attr]
            else:
                raise
        try:
            yield sess
        finally:
            sess.close()

    def _try_reconnect(self, exc: Exception | None = None) -> bool:
        """Attempt driver recreation, then tunnel reconnection if needed.

        Returns True if the driver was successfully recreated.
        """
        from imas_codex.graph.profiles import invalidate_uri_cache, reconnect_tunnel

        detail = f": {exc}" if exc else ""
        logger.warning("Graph connection failed%s, recreating driver...", detail)
        try:
            self._recreate_driver()
            return True
        except Exception:
            logger.warning(
                "Driver recreation failed, attempting tunnel reconnection..."
            )

        if not reconnect_tunnel():
            logger.error("Tunnel reconnection failed")
            return False

        # Cache is invalidated by reconnect_tunnel; re-resolve URI
        invalidate_uri_cache()
        new_uri = get_graph_uri()
        logger.info("Recreating Neo4j driver with URI: %s", new_uri)
        try:
            self._recreate_driver(new_uri)
            return True
        except Exception:
            logger.exception("Neo4j driver recreation failed after tunnel reconnect")
            return False

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

        Processes both facility and DD schemas to cover all node types.

        Resilient to tunnel drops: if a connection error occurs mid-way,
        attempts tunnel reconnection and retries the remaining statements.
        """
        try:
            self._initialize_schema_impl()
        except Exception as e:
            if not self._is_connection_error(e):
                raise
            logger.warning("Schema init failed with connection error, reconnecting...")
            if self._try_reconnect():
                self._initialize_schema_impl()
            else:
                raise

    def _initialize_schema_impl(self) -> None:
        """Inner implementation of schema initialization."""
        # Get constraints and indexes from both schemas
        dd_schema = GraphSchema(schema_path="imas_codex/schemas/imas_dd.yaml")
        constraints = self.schema.constraint_statements()
        constraints.extend(dd_schema.constraint_statements())
        indexes = self.schema.index_statements()
        indexes.extend(dd_schema.index_statements())

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

        # Create fulltext indexes for BM25 text search
        self.ensure_fulltext_indexes()

    def ensure_vector_indexes(self) -> None:
        """Create or recreate vector indexes with quantization.

        Creates all vector indexes derived from LinkML schemas (classes with
        embedding or centroid slots). Index configuration comes from:
        - GraphSchema.vector_indexes property (schema-derived)
        - vector_index_name annotation for custom index names

        Detects dimension mismatches (e.g. after upgrading from 256 to 1024)
        and drops stale indexes before recreation.

        Requires Neo4j 5.x+ with vector index support.
        """
        dim = get_embedding_dimension()

        with self.session() as sess:
            # Check for dimension mismatches on existing vector indexes
            try:
                mismatch_result = sess.run(
                    "SHOW INDEXES YIELD name, type, options "
                    "WHERE type = 'VECTOR' "
                    "RETURN name, options.indexConfig.`vector.dimensions` AS dim"
                )
                for idx in mismatch_result:
                    if idx["dim"] != dim:
                        logger.info(
                            "Dropping vector index %s (dim %s != configured %d)",
                            idx["name"],
                            idx["dim"],
                            dim,
                        )
                        sess.run(f"DROP INDEX `{idx['name']}`")
            except Exception as e:
                logger.warning("Could not check vector index dimensions: %s", e)

            # Get existing indexes (after any drops above)
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
                                `vector.similarity_function`: 'cosine',
                                `vector.quantization.enabled`: true
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

    def ensure_fulltext_indexes(self) -> None:
        """Create fulltext indexes for BM25 text search if they don't exist.

        Creates all fulltext indexes derived from LinkML schema class annotations.
        """
        with self.session() as sess:
            result = sess.run(
                "SHOW INDEXES YIELD name WHERE name IN $names RETURN name",
                names=[idx[0] for idx in EXPECTED_FULLTEXT_INDEXES],
            )
            existing = {record["name"] for record in result}

            for index_name, label, props in EXPECTED_FULLTEXT_INDEXES:
                if index_name in existing:
                    continue

                props_str = ", ".join(f"n.{p}" for p in props)
                try:
                    sess.run(
                        f"CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS "
                        f"FOR (n:{label}) ON EACH [{props_str}] "
                        "OPTIONS { indexConfig: { "
                        "`fulltext.analyzer`: 'standard-no-stop-words' } }"
                    )
                    logger.debug(f"Created fulltext index: {index_name}")
                except Exception as e:
                    logger.warning(f"Failed to create fulltext index {index_name}: {e}")

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
    ) -> None:
        """Create or update a node.

        Args:
            label: Node label (class name from schema)
            node_id: Value of the id field
            props: Node properties
        """
        with self.session() as sess:
            sess.run(
                f"MERGE (n:{label} {{id: $id}}) SET n += $props",
                id=node_id,
                props=props,
            )

    def create_nodes(
        self,
        label: str,
        items: list[dict[str, Any]],
        batch_size: int = 50,
        create_relationships: bool = True,
    ) -> dict[str, int]:
        """Create or update multiple nodes using UNWIND for efficiency.

        Uses batched UNWIND queries for optimal Neo4j performance.
        Automatically creates relationships based on schema-defined slots
        with class ranges (e.g., facility_id -> Facility creates AT_FACILITY edge).

        Args:
            label: Node label (class name from schema)
            items: List of property dicts, each must contain 'id'
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

        # Auto-embed items with description but no embedding
        if label in self.schema.description_embeddable_labels:
            needs_embedding = [
                (i, item)
                for i, item in enumerate(items)
                if item.get("description") and not item.get("embedding")
            ]
            if needs_embedding:
                try:
                    from imas_codex.embeddings import get_encoder

                    encoder = get_encoder()
                    texts = [item["description"] for _, item in needs_embedding]
                    vectors = encoder.embed_texts(texts)
                    from datetime import UTC, datetime

                    now = datetime.now(UTC).isoformat()
                    for (idx, _), vec in zip(needs_embedding, vectors, strict=True):
                        items[idx]["embedding"] = vec.tolist()
                        items[idx]["embedded_at"] = now
                except Exception:
                    logger.warning("Auto-embedding unavailable for %s, skipping", label)

        processed = 0
        rel_counts: dict[str, int] = {}

        # Node creation query (always runs first)
        node_query = f"""
            UNWIND $batch AS item
            MERGE (n:{label} {{id: item.id}})
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
                        MATCH (n:{label} {{id: item.id}})
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
        props: dict[str, Any] | None = None,
    ) -> None:
        """Create a relationship between two nodes.

        Args:
            from_label: Source node label (class name from schema)
            from_id: Source node identifier value
            to_label: Target node label (class name from schema)
            to_id: Target node identifier value
            rel_type: Relationship type (SCREAMING_SNAKE_CASE)
            props: Optional relationship properties
        """
        props_clause = " SET r += $props" if props else ""
        query = (
            f"MATCH (a:{from_label} {{id: $from_id}}), "
            f"(b:{to_label} {{id: $to_id}}) "
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
        server_id = f"{facility_id}:{hostname}"
        props = {"id": server_id, "hostname": hostname, "facility_id": facility_id}
        if role:
            props["role"] = role
        props.update({k: v for k, v in extra.items() if v is not None})
        self.create_node("MDSplusServer", server_id, props)
        self.create_relationship(
            "MDSplusServer",
            server_id,
            "Facility",
            facility_id,
            "AT_FACILITY",
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
        diag_id = f"{facility_id}:{name}"
        props = {"id": diag_id, "name": name, "facility_id": facility_id}
        if category:
            props["category"] = category
        if description:
            props["description"] = description
        props.update({k: v for k, v in extra.items() if v is not None})
        self.create_node("Diagnostic", diag_id, props)
        self.create_relationship(
            "Diagnostic",
            diag_id,
            "Facility",
            facility_id,
            "AT_FACILITY",
        )

    def create_imas_path(
        self,
        path: str,
        ids: str,
        description: str | None = None,
        units: str | None = None,
    ) -> None:
        """Create IMASNode node."""
        props = {"id": path, "path": path, "ids": ids}
        if description:
            props["description"] = description
        if units:
            props["units"] = units
        self.create_node("IMASNode", path, props)

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
            label: Node label (e.g., "Diagnostic", "Tool", "SignalNode")
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
            MATCH (m:IMASMapping)-[:TARGET_PATH]->(imas:IMASNode)
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
        """Execute a Cypher query and return results as dicts.

        Retries once on connection failure after attempting tunnel
        re-establishment.  For database critical errors (OOM, txlog
        corruption), waits for the database to recover (systemd
        ``Restart=on-failure`` or manual restart) before retrying.
        """
        try:
            with self.session() as sess:
                result = sess.run(cypher, **params)
                return [dict(record) for record in result]
        except Exception as e:
            if not self._is_connection_error(e):
                raise
            is_db_critical = self._is_database_critical_error(e)
            if is_db_critical:
                logger.warning(
                    "Neo4j database in critical error state, "
                    "waiting for auto-restart before retrying: %s",
                    e,
                )
                self._wait_for_database_recovery()
            else:
                logger.warning("Query failed with connection error: %s", e)
            if self._try_reconnect():
                with self.session() as sess:
                    result = sess.run(cypher, **params)
                    return [dict(record) for record in result]
            raise

    @staticmethod
    def _is_database_critical_error(exc: Exception) -> bool:
        """Check if the error is specifically a Neo4j database critical error."""
        from neo4j.exceptions import DatabaseError

        if isinstance(exc, DatabaseError):
            msg = str(exc).lower()
            return "critical error" in msg or "needs to be restarted" in msg
        return False

    def _wait_for_database_recovery(self, timeout: int = 60) -> None:
        """Wait for Neo4j to recover from a critical error.

        Systemd services with ``Restart=on-failure`` will auto-restart
        Neo4j.  We wait up to ``timeout`` seconds for the HTTP endpoint
        to come back before proceeding with driver recreation.
        """
        import time
        import urllib.request

        from imas_codex.graph.profiles import resolve_neo4j
        from imas_codex.remote.tunnel import TUNNEL_OFFSET

        try:
            profile = resolve_neo4j(auto_tunnel=False)
        except Exception:
            time.sleep(10)
            return

        # Determine effective HTTP port (may be tunneled)
        http_port = profile.http_port
        host = "localhost"
        if profile.host and f":{profile.bolt_port + TUNNEL_OFFSET}" in self.uri:
            http_port = profile.http_port + TUNNEL_OFFSET
        elif profile.host:
            host = profile.host

        url = f"http://{host}:{http_port}/"
        deadline = time.time() + timeout

        # Close existing driver so Neo4j can fully shut down
        if self._driver:
            try:
                self._driver.close()
            except Exception:
                pass
            self._driver = None

        logger.info("Waiting up to %ds for Neo4j recovery at %s", timeout, url)
        while time.time() < deadline:
            time.sleep(5)
            try:
                urllib.request.urlopen(url, timeout=3)
                logger.info("Neo4j recovered")
                return
            except Exception:
                pass
        logger.warning("Neo4j did not recover within %ds", timeout)

    @staticmethod
    def _is_connection_error(exc: Exception) -> bool:
        """Check if an exception indicates a dead connection/tunnel or
        a database-level critical error that requires restart.

        Neo4j raises ``DatabaseError`` with code
        ``Neo.DatabaseError.Transaction.TransactionStartFailed`` when the
        database enters an unrecoverable state (OOM, corrupted txlog,
        GPFS lock issues).  Treating this as a connection-level error
        allows the retry path to reconnect after the database restarts.
        """
        from neo4j.exceptions import DatabaseError, ServiceUnavailable

        msg = str(exc).lower()
        if isinstance(exc, ServiceUnavailable | ConnectionError | OSError):
            return True
        if any(
            token in msg
            for token in (
                "defunct connection",
                "failed to read from defunct connection",
                "socketdeadlineexceedederror",
                "connection reset",
                "connection refused",
                "timed out",
            )
        ):
            return True
        if isinstance(exc, DatabaseError):
            if "critical error" in msg or "needs to be restarted" in msg:
                return True
        # Check nested cause
        cause = getattr(exc, "__cause__", None)
        if isinstance(cause, Exception) and cause is not exc:
            return GraphClient._is_connection_error(cause)
        return False
