"""Tests for Phase 3: Embedded Neo4j in Docker.

Validates the Docker configuration for graph-native MCP server:
- Dockerfile multi-stage build structure
- Entrypoint script lifecycle management
- Docker-compose configuration
- Health check integration
"""

import re
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "Dockerfile"
ENTRYPOINT = ROOT / "docker-entrypoint.sh"
COMPOSE = ROOT / "docker-compose.yml"


class TestDockerfileStructure:
    """Validate Dockerfile multi-stage build for embedded Neo4j."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.content = DOCKERFILE.read_text()

    def test_has_neo4j_source_stage(self):
        """Neo4j binaries extracted from official image."""
        assert re.search(r"FROM\s+neo4j:\S+\s+AS\s+neo4j-src", self.content), (
            "Missing neo4j-src build stage"
        )

    def test_has_graph_loader_stage(self):
        """Separate stage loads graph dump into Neo4j data directory."""
        assert re.search(r"FROM\s+neo4j:\S+\s+AS\s+graph-loader", self.content), (
            "Missing graph-loader build stage"
        )

    def test_ghcr_token_required(self):
        """Build secret GHCR_TOKEN is required (no fallback)."""
        assert "mount=type=secret,id=GHCR_TOKEN" in self.content
        assert "GHCR_TOKEN build secret is required" in self.content

    def test_build_fails_without_token(self):
        """Script exits with error if GHCR_TOKEN is empty."""
        assert "exit 1" in self.content

    def test_oras_pull_imas_only(self):
        """Pulls IMAS-only graph package, not full graph."""
        assert "imas-codex-graph-imas" in self.content

    def test_graph_tag_arg_defaults_to_latest(self):
        """GRAPH_TAG build arg defaults to 'latest'."""
        match = re.search(r'ARG\s+GRAPH_TAG="?(\w+)"?', self.content)
        assert match, "Missing GRAPH_TAG ARG"
        assert match.group(1) == "latest"

    def test_neo4j_admin_load(self):
        """Uses neo4j-admin to load dump into data directory."""
        assert "neo4j-admin database load" in self.content

    def test_copies_jre_from_neo4j(self):
        """JRE is copied from neo4j-src stage, not installed separately."""
        assert re.search(r"COPY\s+--from=neo4j-src\s+/opt/java/openjdk", self.content)

    def test_copies_neo4j_binaries(self):
        """Neo4j binaries copied from neo4j-src stage."""
        assert re.search(r"COPY\s+--from=neo4j-src\s+/var/lib/neo4j", self.content)

    def test_copies_graph_data(self):
        """Pre-loaded graph data copied from graph-loader stage."""
        assert re.search(r"COPY\s+--from=graph-loader\s+/data", self.content)

    def test_neo4j_listens_localhost_only(self):
        """Neo4j bolt and HTTP bound to localhost (not exposed externally)."""
        assert "127.0.0.1:7687" in self.content
        assert "127.0.0.1:7474" in self.content

    def test_auth_disabled(self):
        """Auth disabled for embedded use (internal only)."""
        assert "auth_enabled=false" in self.content

    def test_entrypoint_is_script(self):
        """Entrypoint runs the shell supervisor script."""
        assert re.search(
            r'ENTRYPOINT\s+\["/usr/local/bin/docker-entrypoint.sh"\]',
            self.content,
        )

    def test_healthcheck_checks_both_services(self):
        """HEALTHCHECK verifies both Neo4j and MCP server."""
        # Look for health check that checks both ports
        assert "7474" in self.content  # Neo4j HTTP
        assert "8000" in self.content  # MCP server
        assert "HEALTHCHECK" in self.content

    def test_only_mcp_port_exposed(self):
        """Only port 8000 (MCP) is exposed; Neo4j ports are internal."""
        expose_lines = re.findall(r"EXPOSE\s+(\d+)", self.content)
        assert "8000" in expose_lines
        assert "7687" not in expose_lines, "Bolt port should not be exposed"
        assert "7474" not in expose_lines, "HTTP port should not be exposed"

    def test_neo4j_env_vars_set(self):
        """Runtime environment includes Neo4j connection variables."""
        assert "NEO4J_URI" in self.content
        assert "NEO4J_HOME" in self.content

    def test_no_file_based_build_steps(self):
        """File-based build steps removed (schemas, path-map, embeddings, clusters)."""
        # Check for actual RUN commands, not comments
        run_blocks = re.findall(
            r"^RUN\s.*?(?=\n(?:FROM|ARG|ENV|COPY|WORKDIR|EXPOSE|ENTRYPOINT|CMD|LABEL|HEALTHCHECK|#|$)|\Z)",
            self.content,
            re.MULTILINE | re.DOTALL,
        )
        run_text = "\n".join(run_blocks)
        assert "uv run --no-dev build-schemas" not in run_text
        assert "uv run --no-dev build-path-map" not in run_text
        assert "uv run --no-dev build-embeddings" not in run_text
        assert "clusters build" not in run_text

    def test_build_models_remains(self):
        """build-models step is preserved (needed for generated Python code)."""
        assert "build-models" in self.content


class TestEntrypointScript:
    """Validate the docker-entrypoint.sh process supervisor."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.content = ENTRYPOINT.read_text()

    def test_file_exists_and_executable(self):
        assert ENTRYPOINT.exists()
        assert ENTRYPOINT.stat().st_mode & 0o111, "Script must be executable"

    def test_valid_bash_syntax(self):
        """Script passes bash syntax check."""
        result = subprocess.run(
            ["bash", "-n", str(ENTRYPOINT)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Bash syntax error: {result.stderr}"

    def test_starts_neo4j_in_background(self):
        """Neo4j started as background process."""
        assert "neo4j" in self.content
        assert "console" in self.content
        assert "&" in self.content

    def test_waits_for_neo4j_ready(self):
        """Health check loop waits for Neo4j to respond."""
        assert "curl" in self.content
        assert "7474" in self.content

    def test_has_timeout(self):
        """Health check loop has a finite timeout."""
        # Should have a max iteration count
        assert re.search(r"seq\s+\d+\s+\d+", self.content)

    def test_fails_if_neo4j_not_ready(self):
        """Script exits with error if Neo4j fails to start."""
        assert "exit 1" in self.content
        assert "failed to start" in self.content.lower()

    def test_starts_mcp_server(self):
        """Launches imas-codex MCP server after Neo4j is ready."""
        assert "imas-codex" in self.content

    def test_trap_handler(self):
        """Signal trap for graceful shutdown."""
        assert "trap" in self.content
        assert "SIGTERM" in self.content
        assert "SIGINT" in self.content

    def test_cleanup_stops_both_processes(self):
        """Cleanup function sends TERM to both Neo4j and MCP."""
        assert "NEO4J_PID" in self.content
        assert "MCP_PID" in self.content
        assert "kill" in self.content

    def test_neo4j_crash_detected(self):
        """Script detects if Neo4j process exits during health check."""
        assert "kill -0" in self.content

    def test_set_errexit(self):
        """Script uses set -e for safety."""
        assert "set -e" in self.content


class TestDockerCompose:
    """Validate docker-compose.yml for graph-native architecture."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.config = yaml.safe_load(COMPOSE.read_text())

    def test_imas_codex_service_exists(self):
        assert "imas-codex" in self.config["services"]

    def test_imas_codex_no_neo4j_dependency(self):
        """imas-codex should not depend on standalone neo4j service."""
        svc = self.config["services"]["imas-codex"]
        depends = svc.get("depends_on", {})
        if isinstance(depends, list):
            assert "neo4j" not in depends
        elif isinstance(depends, dict):
            assert "neo4j" not in depends

    def test_ghcr_token_secret(self):
        """Build uses GHCR_TOKEN secret (not openai_api_key)."""
        svc = self.config["services"]["imas-codex"]
        build_secrets = svc.get("build", {}).get("secrets", [])
        assert "ghcr_token" in build_secrets

    def test_graph_tag_build_arg(self):
        """GRAPH_TAG build arg is configurable."""
        svc = self.config["services"]["imas-codex"]
        build_args = svc.get("build", {}).get("args", [])
        graph_tag_args = [a for a in build_args if "GRAPH_TAG" in str(a)]
        assert graph_tag_args, "Missing GRAPH_TAG build arg"

    def test_standalone_neo4j_has_graph_profile(self):
        """Standalone neo4j service is behind 'graph' profile (not default)."""
        neo4j_svc = self.config["services"].get("neo4j", {})
        profiles = neo4j_svc.get("profiles", [])
        assert "graph" in profiles, "Standalone neo4j must be in 'graph' profile"

    def test_healthcheck_uses_curl(self):
        """Health check uses curl (available in container)."""
        svc = self.config["services"]["imas-codex"]
        healthcheck = svc.get("healthcheck", {})
        test_cmd = healthcheck.get("test", [])
        assert any("curl" in str(t) for t in test_cmd)

    def test_secrets_section(self):
        """Top-level secrets section references GHCR_TOKEN env var."""
        secrets = self.config.get("secrets", {})
        assert "ghcr_token" in secrets
        ghcr = secrets["ghcr_token"]
        assert ghcr.get("environment") == "GHCR_TOKEN"
