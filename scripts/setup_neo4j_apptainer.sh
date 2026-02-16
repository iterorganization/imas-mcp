#!/bin/bash
# Set up Neo4j as an Apptainer container with systemd service
#
# Usage: ./setup_neo4j_apptainer.sh [--system|--user]
#
# Options:
#   --system  Install as system service (requires sudo, recommended for WSL)
#   --user    Install as user service (requires working systemd user session)
#
# Environment variables:
#   NEO4J_VERSION   - Neo4j version (default: 2025.11-community)
#   NEO4J_PASSWORD  - Neo4j password (default: imas-codex)
#   NEO4J_DATA_DIR  - Data directory (default: ~/.local/share/imas-codex/neo4j)
#   NEO4J_SIF_DIR   - Container image directory (default: ~/apptainer)
#
# This script:
# 1. Creates data directories
# 2. Pulls the Neo4j container image
# 3. Creates neo4j.conf
# 4. Sets the initial password
# 5. Creates and enables a systemd service

set -euo pipefail

# Configuration
NEO4J_VERSION="${NEO4J_VERSION:-2025.11-community}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-imas-codex}"
NEO4J_DATA_DIR="${NEO4J_DATA_DIR:-$HOME/.local/share/imas-codex/neo4j}"
NEO4J_SIF_DIR="${NEO4J_SIF_DIR:-$HOME/apptainer}"
SERVICE_TYPE="${1:---system}"  # Default to system service (works better on WSL)

# Derived paths
NEO4J_SIF="${NEO4J_SIF_DIR}/neo4j_${NEO4J_VERSION}.sif"
SERVICE_NAME="imas-codex-neo4j.service"

echo "=== Neo4j Apptainer Setup ==="
echo "Neo4j Version: $NEO4J_VERSION"
echo "Data Directory: $NEO4J_DATA_DIR"
echo "Container Image: $NEO4J_SIF"
echo "Service Type: $SERVICE_TYPE"
echo ""

# Check for apptainer
if ! command -v apptainer &> /dev/null; then
    echo "Error: Apptainer not found. Please run install_apptainer.sh first."
    exit 1
fi

# Step 1: Create directories
echo "Creating directories..."
mkdir -p "${NEO4J_DATA_DIR}"/{data,logs,conf,import}
mkdir -p "${NEO4J_SIF_DIR}"

# Step 2: Pull Neo4j image (if not exists)
if [[ -f "$NEO4J_SIF" ]]; then
    echo "Neo4j image already exists: $NEO4J_SIF"
else
    echo "Pulling Neo4j ${NEO4J_VERSION} image (this may take a few minutes)..."
    apptainer pull "$NEO4J_SIF" "docker://neo4j:${NEO4J_VERSION}"
fi

# Step 3: Create neo4j.conf
echo "Creating neo4j.conf..."
cat > "${NEO4J_DATA_DIR}/conf/neo4j.conf" << 'EOF'
# Neo4j configuration for IMAS Codex
# Memory settings sized for shared HPC login node
# heap (4g) + pagecache (4g) + JVM overhead fits within 12G systemd limit

# Initial heap size
server.memory.heap.initial_size=1g

# Maximum heap size
server.memory.heap.max_size=4g

# Page cache size - covers full graph data in memory
server.memory.pagecache.size=4g

# Network settings
server.default_listen_address=0.0.0.0
server.bolt.listen_address=:7687
server.http.listen_address=:7474

# Database location
server.directories.data=/data

# Logging
server.directories.logs=/logs

# Import directory
server.directories.import=/import
db.recovery.fail_on_missing_files=false
EOF

# Step 4: Set initial password (only works on fresh data directory)
if [[ ! -d "${NEO4J_DATA_DIR}/data/databases" ]]; then
    echo "Setting initial Neo4j password..."
    apptainer exec \
        --bind "${NEO4J_DATA_DIR}/data:/data" \
        --writable-tmpfs \
        "$NEO4J_SIF" \
        neo4j-admin dbms set-initial-password "$NEO4J_PASSWORD"
else
    echo "Database already exists, skipping password setup."
    echo "If you need to reset the password, remove ${NEO4J_DATA_DIR}/data/* and re-run."
fi

# Step 5: Create systemd service
echo "Creating systemd service..."

# Expand paths for service file
DATA_DIR_EXPANDED=$(realpath "$NEO4J_DATA_DIR")
SIF_EXPANDED=$(realpath "$NEO4J_SIF")

SERVICE_CONTENT="[Unit]
Description=Neo4j Graph Database (IMAS Codex)
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/apptainer exec \\
    --bind ${DATA_DIR_EXPANDED}/data:/data \\
    --bind ${DATA_DIR_EXPANDED}/logs:/logs \\
    --bind ${DATA_DIR_EXPANDED}/import:/import \\
    --bind ${DATA_DIR_EXPANDED}/conf:/var/lib/neo4j/conf \\
    --writable-tmpfs \\
    --env NEO4J_AUTH=neo4j/${NEO4J_PASSWORD} \\
    --env NEO4J_server_jvm_additional=\"-Dfile.encoding=UTF-8 --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/java.lang=ALL-UNNAMED\" \\
    ${SIF_EXPANDED} \\
    neo4j console
Restart=on-failure
RestartSec=5
CPUQuota=400%
MemoryMax=12G
MemoryHigh=10G
Nice=10

[Install]"

if [[ "$SERVICE_TYPE" == "--system" ]]; then
    # System service (requires sudo, works on WSL)
    SERVICE_CONTENT="${SERVICE_CONTENT}
WantedBy=multi-user.target"
    
    # Add User/Group for system service
    SERVICE_CONTENT=$(echo "$SERVICE_CONTENT" | sed "s/\[Service\]/[Service]\nUser=$(whoami)\nGroup=$(id -gn)/")
    
    SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}"
    
    echo "Installing system service (requires sudo)..."
    echo "$SERVICE_CONTENT" | sudo tee "$SERVICE_FILE" > /dev/null
    
    sudo systemctl daemon-reload
    sudo systemctl enable "$SERVICE_NAME"
    
    echo ""
    echo "=== Setup Complete ==="
    echo ""
    echo "Start the service:"
    echo "  sudo systemctl start $SERVICE_NAME"
    echo ""
    echo "Check status:"
    echo "  sudo systemctl status $SERVICE_NAME"
    echo ""
    echo "View logs:"
    echo "  cat ${NEO4J_DATA_DIR}/logs/neo4j.log | tail -50"
    echo ""
    echo "Neo4j Browser: http://localhost:7474"
    echo "Bolt URI: bolt://localhost:7687"
    echo "Credentials: neo4j / ${NEO4J_PASSWORD}"
    
else
    # User service
    SERVICE_CONTENT="${SERVICE_CONTENT}
WantedBy=default.target"
    
    SERVICE_DIR="$HOME/.config/systemd/user"
    SERVICE_FILE="${SERVICE_DIR}/${SERVICE_NAME}"
    
    mkdir -p "$SERVICE_DIR"
    echo "$SERVICE_CONTENT" > "$SERVICE_FILE"
    
    echo ""
    echo "=== Setup Complete ==="
    echo ""
    echo "NOTE: User services require a working D-Bus session."
    echo "On WSL, you may need to run: sudo loginctl enable-linger $USER"
    echo ""
    echo "Enable and start the service:"
    echo "  systemctl --user daemon-reload"
    echo "  systemctl --user enable $SERVICE_NAME"
    echo "  systemctl --user start $SERVICE_NAME"
    echo ""
    echo "Check status:"
    echo "  systemctl --user status $SERVICE_NAME"
    echo ""
    echo "Neo4j Browser: http://localhost:7474"
    echo "Bolt URI: bolt://localhost:7687"
    echo "Credentials: neo4j / ${NEO4J_PASSWORD}"
fi
