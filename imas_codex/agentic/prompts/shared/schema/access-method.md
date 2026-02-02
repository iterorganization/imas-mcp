## AccessMethod Schema

Graph nodes for **self-contained** data access patterns. Each node must contain
everything needed to load data - no external config required.

| Field | Required | Description |
|-------|----------|-------------|
| `id` | ✓ | `{facility}:{method_type}:{variant}` e.g., `tcv:mdsplus:tree_tdi` |
| `facility_id` | ✓ | Parent facility ID |
| `name` | | Human-readable name |
| `method_type` | ✓ | System type: `mdsplus`, `imas`, `hdf5`, `rest`, `uda`, `cli`, `matlab`, `idl` |
| `library` | ✓ | Import target: `MDSplus`, `imas`, `pyuda`, `jet.data.sal` |
| `access_type` | ✓ | `local` (on-machine), `remote` (network), `ssh` (tunnel) |

### Environment Setup (Self-Contained)

| Field | Description |
|-------|-------------|
| `setup_commands` | Shell commands to run before Python (e.g., `["module load python/3.9"]`) |
| `environment_variables` | Env vars to set (e.g., `{"MDSPLUS_SERVER": "..."}`) |

### Code Templates (with placeholders)

| Template | Placeholders | Purpose |
|----------|--------------|---------|
| `imports_template` | | Import statements |
| `connection_template` | `{server}`, `{data_source}`, `{shot}` | Open connection |
| `data_template` | `{accessor}`, `{shot}` | Retrieve data |
| `time_template` | `{accessor}` | Retrieve time axis |
| `cleanup_template` | | Close connection |

### Data Source & Validation

| Field | Description |
|-------|-------------|
| `data_source` | Default tree/database/path (e.g., `tcv_shot`, `ppf`) |
| `discovery_shot` | Known-good shot for testing |
| `verified_date` | Date last validated (YYYY-MM-DD) |
| `full_example` | Complete working code example |
| `documentation_url` | Wiki or external doc URL |
| `documentation_local` | Path to docs on facility |
