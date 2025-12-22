<!-- AUTO-GENERATED from filesystem.yaml - DO NOT EDIT -->

## Expected Output Structure

When finishing filesystem exploration, provide YAML matching this structure:

### FilesystemArtifact

Filesystem exploration results. Contains discovered paths, directory structure, and organization patterns.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `facility` | string | yes | Facility identifier (e.g., epfl, jet, iter) |
| `explored_at` | datetime | yes | Timestamp when this artifact was created |
| `root_paths` | list[string] | yes | Root paths that were explored |
| `tree` | FilesystemNode | no | Root filesystem node (tree structure) |
| `statistics` | ScanStatistics | no | Exploration statistics |
| `important_paths` | list[ImportantPath] | no | Notable paths discovered during exploration |
| `notes` | list[string] | no | Freeform observations about filesystem organization |

### FilesystemNode

A node in the filesystem tree. Can represent files, directories, or symlinks. Forms a recursive tree structure.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `path` | string | yes | Absolute path to this node |
| `name` | string | no | Basename of the path |
| `node_type` | NodeType | yes | Type of filesystem node |
| `size` | integer | no | Size in bytes (for files) |
| `modified` | datetime | no | Last modification timestamp |
| `permissions` | string | no | Unix permission string (e.g., rwxr-xr-x) |
| `children` | list[FilesystemNode] | no | Child nodes (for directories) |
| `symlink_target` | string | no | Target path for symlinks |

### ScanStatistics

Statistics about a filesystem exploration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `total_files` | integer | no | Total number of files discovered |
| `total_directories` | integer | no | Total number of directories discovered |
| `total_symlinks` | integer | no | Total number of symlinks discovered |
| `total_size_bytes` | integer | no | Total size of all files in bytes |
| `scan_duration_seconds` | float | no | How long the exploration took |
| `excluded_paths` | integer | no | Number of paths excluded by filters |

### ImportantPath

A notable path with description of its purpose

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `path` | string | yes | Absolute path |
| `purpose` | string | yes | What this path contains or is used for |
| `path_type` | PathType | no | Type of path (data, code, docs, config) |

### Example

```yaml
root_paths:
  - "example_value"
tree:
  path: "example_value"
  name: "example_value"
  node_type: "example_value"
  size: 0
  modified: "2025-01-01T00:00:00Z"
  permissions: "example_value"
  children:
    -
      # ... (recursive reference to FilesystemNode)
  symlink_target: "example_value"
statistics:
  total_files: 0
  total_directories: 0
  total_symlinks: 0
  total_size_bytes: 0
  scan_duration_seconds: "example_value"
  excluded_paths: 0
important_paths:
  -
    path: "example_value"
    purpose: "example_value"
    path_type: "example_value"
notes:
  - "example_value"
```
