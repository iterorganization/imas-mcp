<!-- AUTO-GENERATED from tools.yaml - DO NOT EDIT -->

## Expected Output Structure

When finishing tools exploration, provide YAML matching this structure:

### ToolsArtifact

Collection of available CLI tools on the remote facility. Organized by category for easy reference.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `facility` | string | yes | Facility identifier (e.g., epfl, jet, iter) |
| `explored_at` | datetime | yes | Timestamp when this artifact was created |
| `tools` | list[ToolInfo] | yes | List of discovered tools |
| `notes` | list[string] | no | Freeform observations about tool availability |

### ToolInfo

Information about a single CLI tool

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | Tool name (e.g., rg, grep, h5dump) |
| `available` | boolean | yes | Whether the tool is available and executable |
| `path` | string | no | Absolute path to the tool binary (if available) |
| `version` | string | no | Tool version string (if available) |
| `category` | ToolCategory | no | Tool category for organization |

### Example

```yaml
tools:
  -
    name: "example_value"
    available: true
    path: "example_value"
    version: "example_value"
    category: "example_value"
notes:
  - "example_value"
```
