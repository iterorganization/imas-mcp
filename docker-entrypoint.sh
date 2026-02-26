#!/bin/bash
# Entrypoint for the IMAS Codex MCP server with embedded Neo4j.
#
# Starts Neo4j in the background, waits for it to become available,
# then starts the MCP server. On shutdown (SIGTERM/SIGINT), stops
# both processes gracefully.

set -e

NEO4J_HOME="${NEO4J_HOME:-/opt/neo4j}"
NEO4J_DATA="${NEO4J_HOME}/data"
NEO4J_LOG="${NEO4J_HOME}/logs/neo4j.log"

# Ensure log directory exists
mkdir -p "${NEO4J_HOME}/logs"

cleanup() {
    echo "Shutting down..."
    # Stop MCP server
    if [ -n "$MCP_PID" ] && kill -0 "$MCP_PID" 2>/dev/null; then
        kill -TERM "$MCP_PID" 2>/dev/null || true
        wait "$MCP_PID" 2>/dev/null || true
    fi
    # Stop Neo4j
    if [ -n "$NEO4J_PID" ] && kill -0 "$NEO4J_PID" 2>/dev/null; then
        kill -TERM "$NEO4J_PID" 2>/dev/null || true
        wait "$NEO4J_PID" 2>/dev/null || true
    fi
    exit 0
}

trap cleanup SIGTERM SIGINT

# Start Neo4j in background
echo "Starting Neo4j..."
"${NEO4J_HOME}/bin/neo4j" console > "$NEO4J_LOG" 2>&1 &
NEO4J_PID=$!

# Wait for Neo4j to become ready
echo "Waiting for Neo4j..."
for i in $(seq 1 60); do
    if curl -sf http://localhost:7474/ > /dev/null 2>&1; then
        echo "Neo4j ready (${i}s)"
        break
    fi
    if ! kill -0 "$NEO4J_PID" 2>/dev/null; then
        echo "Neo4j process exited unexpectedly. Log:"
        cat "$NEO4J_LOG" 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

# Verify Neo4j is actually responding
if ! curl -sf http://localhost:7474/ > /dev/null 2>&1; then
    echo "Neo4j failed to start within 60s. Log:"
    tail -50 "$NEO4J_LOG" 2>/dev/null || true
    exit 1
fi

# Start MCP server
echo "Starting IMAS Codex MCP server..."
exec imas-codex "$@" &
MCP_PID=$!

# Wait for either process to exit
wait -n "$NEO4J_PID" "$MCP_PID" 2>/dev/null || true
cleanup
