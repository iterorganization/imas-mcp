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

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from neo4j import Driver, GraphDatabase, Session

from imas_codex.graph.schema import GraphSchema, get_schema


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
        """Initialize the Neo4j driver and schema."""
        self._driver = GraphDatabase.driver(
            self.uri, auth=(self.username, self.password)
        )
        self._schema = get_schema()

    @property
    def schema(self) -> GraphSchema:
        """Get the graph schema."""
        if self._schema is None:
            self._schema = get_schema()
        return self._schema

    def close(self) -> None:
        """Close the Neo4j driver."""
        if self._driver:
            self._driver.close()
            self._driver = None

    def __enter__(self) -> "GraphClient":
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
        """
        # Get constraints from schema (derived from identifier fields)
        constraints = self.schema.constraint_statements()

        # Get indexes from schema (common lookup patterns)
        indexes = self.schema.index_statements()

        with self.session() as sess:
            for stmt in constraints + indexes:
                sess.run(stmt)

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
                "CALL db.labels() YIELD label "
                "RETURN label, size([(n) WHERE label IN labels(n) | n]) as count"
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

    def create_tool(
        self,
        facility_id: str,
        name: str,
        available: bool,
        path: str | None = None,
        version: str | None = None,
        category: str | None = None,
        **extra: Any,
    ) -> None:
        """Create Tool node and link to Facility."""
        tool_id = f"{facility_id}:{name}"
        props = {
            "id": tool_id,
            "facility_id": facility_id,
            "name": name,
            "available": available,
        }
        if path:
            props["path"] = path
        if version:
            props["version"] = version
        if category:
            props["category"] = category
        props.update({k: v for k, v in extra.items() if v is not None})
        self.create_node("Tool", tool_id, props)
        self.create_relationship(
            "Tool",
            tool_id,
            "Facility",
            facility_id,
            "FACILITY_ID",
        )

    def create_python_environment(
        self,
        facility_id: str,
        version: str,
        path: str,
        is_default: bool = False,
        packages: list[str] | None = None,
        **extra: Any,
    ) -> None:
        """Create PythonEnvironment node and link to Facility."""
        env_id = f"{facility_id}:python:{version}"
        props = {
            "id": env_id,
            "facility_id": facility_id,
            "version": version,
            "path": path,
            "is_default": is_default,
        }
        if packages:
            props["packages"] = packages
        props.update({k: v for k, v in extra.items() if v is not None})
        self.create_node("PythonEnvironment", env_id, props)
        self.create_relationship(
            "PythonEnvironment",
            env_id,
            "Facility",
            facility_id,
            "FACILITY_ID",
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

    def get_tools(self, facility_id: str) -> list[dict[str, Any]]:
        """Get all tools for a facility."""
        with self.session() as sess:
            result = sess.run(
                "MATCH (t:Tool {facility_id: $facility_id}) RETURN t ORDER BY t.name",
                facility_id=facility_id,
            )
            return [dict(record["t"]) for record in result]

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        """Execute a Cypher query and return results as dicts."""
        with self.session() as sess:
            result = sess.run(cypher, **params)
            return [dict(record) for record in result]
