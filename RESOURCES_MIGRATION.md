# Index Files Now Use Resources Directory

## Overview

IMAS MCP index files are now stored exclusively in the `imas_mcp/resources/` directory within the package. This ensures that index files are properly packaged and distributed with the wheel.

## What Changed

- Index files are now stored in `imas_mcp/resources/` (within the package)
- Index files are automatically included in the wheel build
- The build hook creates index files directly in the resources directory

## For Users

- No action required - the change is transparent
- Index files are automatically available after package installation
- The package works with pre-built index files included in the distribution

## For Developers

- Index files are generated directly in `imas_mcp/resources/` during development
- The resources directory is properly gitignored (except for `.gitkeep` files)
- The DataDictionaryIndex class uses the resources directory by default

This change ensures that users installing the package via pip have immediate access to pre-built index files without needing the IMAS Data Dictionary dependency.
