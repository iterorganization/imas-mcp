<!-- AUTO-GENERATED from environment.yaml - DO NOT EDIT -->

## Expected Output Structure

When finishing environment exploration, provide YAML matching this structure:

### EnvironmentArtifact

Complete environment snapshot from facility exploration. Includes Python, OS, compilers, and module system details.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `facility` | string | yes | Facility identifier (e.g., epfl, jet, iter) |
| `explored_at` | datetime | yes | Timestamp when this artifact was created |
| `python` | PythonInfo | yes | Python installation details |
| `os` | OSInfo | yes | Operating system information |
| `compilers` | list[CompilerInfo] | no | Available compilers on the system |
| `module_system` | ModuleSystemInfo | no | Environment module system (Lmod, Environment Modules) |
| `notes` | list[string] | no | Freeform observations about the environment |

### PythonInfo

Python installation details

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `version` | string | yes | Python version string (e.g., 3.9.21) |
| `path` | string | yes | Absolute path to Python binary |
| `packages` | list[string] | no | Installed Python packages (format name==version) |

### OSInfo

Operating system information

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | OS distribution name (e.g., RHEL, Ubuntu, CentOS) |
| `version` | string | yes | OS version (e.g., 9.6, 22.04) |
| `kernel` | string | no | Kernel version string |

### CompilerInfo

Compiler installation details

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | Compiler name (e.g., gcc, gfortran, icx, ifx) |
| `version` | string | no | Compiler version string |
| `path` | string | no | Absolute path to compiler binary |

### ModuleSystemInfo

Environment module system information

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `available` | boolean | yes | Whether a module system is available |
| `type` | string | no | Module system type (lmod, environment-modules, none) |
| `loaded_modules` | list[string] | no | Currently loaded modules |

### Example

```yaml
python:
  version: "example_value"
  path: "example_value"
  packages:
    - "example_value"
os:
  name: "example_value"
  version: "example_value"
  kernel: "example_value"
compilers:
  -
    name: "example_value"
    version: "example_value"
    path: "example_value"
module_system:
  available: true
  type: "example_value"
  loaded_modules:
    - "example_value"
notes:
  - "example_value"
```
