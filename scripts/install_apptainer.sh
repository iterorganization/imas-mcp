#!/bin/bash
# Install Apptainer on Ubuntu/Debian systems
# Requires sudo privileges
#
# Usage: ./install_apptainer.sh
#
# This script:
# 1. Downloads the latest Apptainer .deb package from GitHub
# 2. Installs it via apt
# 3. Verifies the installation

set -euo pipefail

APPTAINER_VERSION="${APPTAINER_VERSION:-1.4.5}"
TEMP_DIR="${TEMP_DIR:-/tmp}"

echo "=== Apptainer Installation Script ==="

# Check if already installed
if command -v apptainer &> /dev/null; then
    INSTALLED_VERSION=$(apptainer --version | awk '{print $3}')
    echo "Apptainer already installed: $INSTALLED_VERSION"
    if [[ "$INSTALLED_VERSION" == "$APPTAINER_VERSION" ]]; then
        echo "Already at target version $APPTAINER_VERSION, nothing to do."
        exit 0
    fi
    echo "Upgrading from $INSTALLED_VERSION to $APPTAINER_VERSION..."
fi

# Check for sudo
if ! sudo -n true 2>/dev/null; then
    echo "This script requires sudo privileges."
    echo "Please run: sudo $0"
    exit 1
fi

# Detect architecture
ARCH=$(dpkg --print-architecture)
if [[ "$ARCH" != "amd64" ]]; then
    echo "Error: Only amd64 architecture is supported (detected: $ARCH)"
    exit 1
fi

# Download URL
DEB_URL="https://github.com/apptainer/apptainer/releases/download/v${APPTAINER_VERSION}/apptainer_${APPTAINER_VERSION}_amd64.deb"
DEB_FILE="${TEMP_DIR}/apptainer_${APPTAINER_VERSION}_amd64.deb"

echo "Downloading Apptainer v${APPTAINER_VERSION}..."
curl -fsSL -o "$DEB_FILE" "$DEB_URL"

echo "Installing Apptainer..."
sudo apt-get update -qq
sudo apt-get install -y "$DEB_FILE"

# Cleanup
rm -f "$DEB_FILE"

# Verify installation
echo ""
echo "=== Installation Complete ==="
apptainer --version
echo ""
echo "Apptainer installed successfully!"
