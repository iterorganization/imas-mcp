<!-- AUTO-GENERATED from data.yaml - DO NOT EDIT -->

## Expected Output Structure

When finishing data exploration, provide YAML matching this structure:

### DataArtifact

Data access and organization patterns for a facility. Includes MDSplus, HDF5, NetCDF, and other data systems.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `facility` | string | yes | Facility identifier (e.g., epfl, jet, iter) |
| `explored_at` | datetime | yes | Timestamp when this artifact was created |
| `mdsplus` | MDSplusInfo | no | MDSplus configuration and trees |
| `hdf5` | HDF5Info | no | HDF5 data patterns |
| `netcdf` | NetCDFInfo | no | NetCDF data patterns |
| `data_formats` | list[DataFormatInfo] | no | Other data formats found |
| `shot_ranges` | list[ShotRange] | no | Known shot number ranges |
| `notes` | list[string] | no | Freeform observations about data organization |

### MDSplusInfo

MDSplus data system configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `available` | boolean | yes | Whether MDSplus is accessible |
| `server` | string | no | MDSplus server hostname |
| `trees` | list[string] | no | Available MDSplus tree names |
| `default_tree` | string | no | Default tree for this facility |
| `python_bindings` | boolean | no | Whether Python MDSplus bindings work |
| `example_signals` | list[string] | no | Example signal paths that work |

### HDF5Info

HDF5 data patterns and locations

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `available` | boolean | yes | Whether HDF5 files are present |
| `locations` | list[string] | no | Paths where HDF5 files are found |
| `naming_patterns` | list[string] | no | File naming conventions observed |
| `h5dump_available` | boolean | no | Whether h5dump tool is available |

### NetCDFInfo

NetCDF data patterns and locations

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `available` | boolean | yes | Whether NetCDF files are present |
| `locations` | list[string] | no | Paths where NetCDF files are found |
| `naming_patterns` | list[string] | no | File naming conventions observed |
| `ncdump_available` | boolean | no | Whether ncdump tool is available |

### DataFormatInfo

Information about a data format

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | Format name (e.g., CSV, JSON, MAT, IDL) |
| `extension` | string | no | File extension |
| `locations` | list[string] | no | Paths where this format is found |
| `description` | string | no | What this format is used for |

### ShotRange

A range of shot numbers

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | no | Name or description of this range |
| `start` | integer | no | First shot number |
| `end` | integer | no | Last shot number (or current) |
| `description` | string | no | What this shot range represents |

### Example

```yaml
mdsplus:
  available: true
  server: "example_value"
  trees:
    - "example_value"
  default_tree: "example_value"
  python_bindings: true
  example_signals:
    - "example_value"
hdf5:
  available: true
  locations:
    - "example_value"
  naming_patterns:
    - "example_value"
  h5dump_available: true
netcdf:
  available: true
  locations:
    - "example_value"
  naming_patterns:
    - "example_value"
  ncdump_available: true
data_formats:
  -
    name: "example_value"
    extension: "example_value"
    locations:
      - "example_value"
    description: "example_value"
shot_ranges:
  -
    name: "example_value"
    start: 0
    end: 0
    description: "example_value"
notes:
  - "example_value"
```
