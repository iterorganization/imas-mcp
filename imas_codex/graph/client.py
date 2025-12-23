"""Neo4j client for the facility knowledge graph.

This module provides a high-level client for interacting with the Neo4j
knowledge graph, including schema initialization and CRUD operations.

Example:
    >>> from imas_codex.graph import GraphClient
    >>> with GraphClient() as client:
    ...     client.initialize_schema()
    ...     client.create_node(NodeLabel.FACILITY, "epfl", {"name": "EPFL/TCV"})
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from neo4j import Driver, GraphDatabase, Session

from imas_codex.graph.cypher import NodeLabel, RelationType


@dataclass
class GraphClient:
    """Client for Neo4j knowledge graph operations.

    Attributes:
        uri: Neo4j Bolt URI (default: bolt://localhost:7687)
        username: Neo4j username (default: neo4j)
        password: Neo4j password (default: imas-codex)
    """

    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "imas-codex"
    _driver: Driver | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the Neo4j driver."""
        self._driver = GraphDatabase.driver(
            self.uri, auth=(self.username, self.password)
        )

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

        Sets up unique constraints on node identifiers and indexes for
        common query patterns.
        """
        constraints = [
            # Core
            f"CREATE CONSTRAINT facility_id IF NOT EXISTS FOR (n:{NodeLabel.FACILITY.value}) REQUIRE n.id IS UNIQUE",
            # Data Infrastructure
            f"CREATE CONSTRAINT server_hostname IF NOT EXISTS FOR (n:{NodeLabel.MDSPLUS_SERVER.value}) REQUIRE n.hostname IS UNIQUE",
            f"CREATE CONSTRAINT tree_name IF NOT EXISTS FOR (n:{NodeLabel.MDSPLUS_TREE.value}) REQUIRE (n.name, n.facility_id) IS UNIQUE",
            f"CREATE CONSTRAINT treenode_path IF NOT EXISTS FOR (n:{NodeLabel.TREE_NODE.value}) REQUIRE (n.path, n.facility_id) IS UNIQUE",
            f"CREATE CONSTRAINT tdi_name IF NOT EXISTS FOR (n:{NodeLabel.TDI_FUNCTION.value}) REQUIRE (n.name, n.facility_id) IS UNIQUE",
            f"CREATE CONSTRAINT data_location IF NOT EXISTS FOR (n:{NodeLabel.DATA_LOCATION.value}) REQUIRE (n.path, n.facility_id) IS UNIQUE",
            # Environment
            f"CREATE CONSTRAINT python_env IF NOT EXISTS FOR (n:{NodeLabel.PYTHON_ENVIRONMENT.value}) REQUIRE n.id IS UNIQUE",
            f"CREATE CONSTRAINT os IF NOT EXISTS FOR (n:{NodeLabel.OPERATING_SYSTEM.value}) REQUIRE n.facility_id IS UNIQUE",
            f"CREATE CONSTRAINT compiler IF NOT EXISTS FOR (n:{NodeLabel.COMPILER.value}) REQUIRE n.id IS UNIQUE",
            f"CREATE CONSTRAINT module_sys IF NOT EXISTS FOR (n:{NodeLabel.MODULE_SYSTEM.value}) REQUIRE n.facility_id IS UNIQUE",
            # Tools
            f"CREATE CONSTRAINT tool IF NOT EXISTS FOR (n:{NodeLabel.TOOL.value}) REQUIRE n.id IS UNIQUE",
            # Semantics
            f"CREATE CONSTRAINT diagnostic IF NOT EXISTS FOR (n:{NodeLabel.DIAGNOSTIC.value}) REQUIRE (n.name, n.facility_id) IS UNIQUE",
            f"CREATE CONSTRAINT analysis_code IF NOT EXISTS FOR (n:{NodeLabel.ANALYSIS_CODE.value}) REQUIRE (n.name, n.facility_id) IS UNIQUE",
            # IMAS
            f"CREATE CONSTRAINT imas_path IF NOT EXISTS FOR (n:{NodeLabel.IMAS_PATH.value}) REQUIRE n.path IS UNIQUE",
            f"CREATE CONSTRAINT imas_mapping IF NOT EXISTS FOR (n:{NodeLabel.IMAS_MAPPING.value}) REQUIRE n.id IS UNIQUE",
        ]

        indexes = [
            # Common lookups
            f"CREATE INDEX facility_ssh IF NOT EXISTS FOR (n:{NodeLabel.FACILITY.value}) ON (n.ssh_host)",
            f"CREATE INDEX server_role IF NOT EXISTS FOR (n:{NodeLabel.MDSPLUS_SERVER.value}) ON (n.role)",
            f"CREATE INDEX treenode_type IF NOT EXISTS FOR (n:{NodeLabel.TREE_NODE.value}) ON (n.node_type)",
            f"CREATE INDEX diagnostic_category IF NOT EXISTS FOR (n:{NodeLabel.DIAGNOSTIC.value}) ON (n.category)",
            f"CREATE INDEX code_type IF NOT EXISTS FOR (n:{NodeLabel.ANALYSIS_CODE.value}) ON (n.code_type)",
            f"CREATE INDEX tool_category IF NOT EXISTS FOR (n:{NodeLabel.TOOL.value}) ON (n.category)",
            f"CREATE INDEX tool_available IF NOT EXISTS FOR (n:{NodeLabel.TOOL.value}) ON (n.available)",
        ]

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
        label: NodeLabel,
        node_id: Any,
        props: dict[str, Any],
        id_field: str = "id",
    ) -> None:
        """Create or update a node.

        Args:
            label: Node label
            node_id: Value of the identifier field
            props: Node properties
            id_field: Name of the identifier field (default: "id")
        """
        with self.session() as sess:
            sess.run(
                f"MERGE (n:{label.value} {{{id_field}: $id}}) SET n += $props",
                id=node_id,
                props=props,
            )

    def create_relationship(
        self,
        from_label: NodeLabel,
        from_id: Any,
        to_label: NodeLabel,
        to_id: Any,
        rel_type: RelationType,
        from_id_field: str = "id",
        to_id_field: str = "id",
        props: dict[str, Any] | None = None,
    ) -> None:
        """Create a relationship between two nodes.

        Args:
            from_label: Source node label
            from_id: Source node identifier value
            to_label: Target node label
            to_id: Target node identifier value
            rel_type: Relationship type
            from_id_field: Source node identifier field name
            to_id_field: Target node identifier field name
            props: Optional relationship properties
        """
        props_clause = " SET r += $props" if props else ""
        query = (
            f"MATCH (a:{from_label.value} {{{from_id_field}: $from_id}}), "
            f"(b:{to_label.value} {{{to_id_field}: $to_id}}) "
            f"MERGE (a)-[r:{rel_type.value}]->(b){props_clause}"
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
        self.create_node(NodeLabel.FACILITY, facility_id, props)

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
        self.create_node(NodeLabel.MDSPLUS_SERVER, hostname, props, id_field="hostname")
        self.create_relationship(
            NodeLabel.MDSPLUS_SERVER,
            hostname,
            NodeLabel.FACILITY,
            facility_id,
            RelationType.HOSTED_BY,
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
        self.create_node(NodeLabel.TOOL, tool_id, props)
        self.create_relationship(
            NodeLabel.TOOL,
            tool_id,
            NodeLabel.FACILITY,
            facility_id,
            RelationType.BELONGS_TO,
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
        self.create_node(NodeLabel.PYTHON_ENVIRONMENT, env_id, props)
        self.create_relationship(
            NodeLabel.PYTHON_ENVIRONMENT,
            env_id,
            NodeLabel.FACILITY,
            facility_id,
            RelationType.BELONGS_TO,
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
        self.create_node(NodeLabel.DIAGNOSTIC, name, props, id_field="name")
        self.create_relationship(
            NodeLabel.DIAGNOSTIC,
            name,
            NodeLabel.FACILITY,
            facility_id,
            RelationType.BELONGS_TO,
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
        self.create_node(NodeLabel.IMAS_PATH, path, props, id_field="path")

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_facility(self, facility_id: str) -> dict[str, Any] | None:
        """Get a facility by ID."""
        with self.session() as sess:
            result = sess.run(
                f"MATCH (n:{NodeLabel.FACILITY.value} {{id: $id}}) RETURN n",
                id=facility_id,
            )
            record = result.single()
            return dict(record["n"]) if record else None

    def get_facilities(self) -> list[dict[str, Any]]:
        """Get all facilities."""
        with self.session() as sess:
            result = sess.run(
                f"MATCH (n:{NodeLabel.FACILITY.value}) RETURN n ORDER BY n.id"
            )
            return [dict(record["n"]) for record in result]

    def get_tools(self, facility_id: str) -> list[dict[str, Any]]:
        """Get all tools for a facility."""
        with self.session() as sess:
            result = sess.run(
                f"MATCH (t:{NodeLabel.TOOL.value} {{facility_id: $facility_id}}) "
                "RETURN t ORDER BY t.name",
                facility_id=facility_id,
            )
            return [dict(record["t"]) for record in result]

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        """Execute a Cypher query and return results as dicts."""
        with self.session() as sess:
            result = sess.run(cypher, **params)
            return [dict(record) for record in result]
