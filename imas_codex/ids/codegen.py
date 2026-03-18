"""Code generation for IMAS mapping extraction and assembly scripts.

Generates self-contained Python scripts from DataAccess templates and
signal mapping metadata:

  generate_extraction_script:   Remote data extraction script (10.1)
  validate_assembly_code:       Static validation of generated assembly code (10.5)
  generate_assembly_code:       Default assembly code snippets (10.3)
  cache_extraction_result:      Cache extracted data locally (10.7)
  load_cached_extraction:       Load cached extraction data (10.7)
"""

from __future__ import annotations

import ast
import json
import logging
import textwrap
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 10.7 Extraction Cache
# ---------------------------------------------------------------------------

CACHE_DIR = Path.home() / ".cache" / "imas-codex" / "extractions"


def cache_extraction_result(
    facility: str,
    shot: int,
    results: dict[str, Any],
) -> Path:
    """Cache extraction results for a (facility, shot) pair.

    Stores results as a JSON file at:
    ~/.cache/imas-codex/extractions/{facility}/{shot}/results.json

    Args:
        facility: Facility identifier.
        shot: Shot number.
        results: Extraction results dict (signal_id → data).

    Returns:
        Path to the cached results file.
    """
    cache_path = CACHE_DIR / facility / str(shot)
    cache_path.mkdir(parents=True, exist_ok=True)
    results_file = cache_path / "results.json"
    results_file.write_text(json.dumps(results, default=str))
    logger.debug("Cached extraction for %s/%d at %s", facility, shot, results_file)
    return results_file


def load_cached_extraction(
    facility: str,
    shot: int,
) -> dict[str, Any] | None:
    """Load cached extraction results for a (facility, shot) pair.

    Args:
        facility: Facility identifier.
        shot: Shot number.

    Returns:
        Cached results dict, or None if not cached.
    """
    results_file = CACHE_DIR / facility / str(shot) / "results.json"
    if results_file.exists():
        logger.debug("Loading cached extraction from %s", results_file)
        return json.loads(results_file.read_text())
    return None


# ---------------------------------------------------------------------------
# 10.1 Data Extraction Script Generation
# ---------------------------------------------------------------------------


def generate_extraction_script(
    facility: str,
    ids_name: str,
    section_signals: dict[str, list[dict]],
    data_access: dict[str, Any],
    *,
    max_points: int | None = 10000,
) -> str:
    """Generate a data extraction script from DataAccess templates.

    Produces a self-contained Python script that extracts signal data
    from a facility data system via MDSplus, IMAS, HDF5, etc.
    The script reads its configuration (shot number, signal list) from
    stdin as JSON.

    Args:
        facility: Facility identifier (e.g., "tcv").
        ids_name: IDS name (e.g., "pf_active").
        section_signals: Mapping of section_path to list of signal dicts.
            Each signal dict has: id, accessor, data_source_name.
        data_access: DataAccess node properties dict with template fields:
            imports_template, connection_template, data_template,
            time_template, cleanup_template.
        max_points: Maximum time points per signal (None = no decimation).

    Returns:
        Complete Python script string for remote execution.
    """
    imports_template = data_access.get("imports_template", "")
    connection_template = data_access.get("connection_template", "")
    data_template = data_access.get("data_template", "")
    time_template = data_access.get("time_template", "")
    cleanup_template = data_access.get("cleanup_template", "")

    # Build signal list grouped by data source for connection reuse
    all_signals: list[dict] = []
    for section_path, signals in section_signals.items():
        for sig in signals:
            all_signals.append({**sig, "section_path": section_path})

    # Group by data_source_name
    by_source: dict[str, list[dict]] = {}
    for sig in all_signals:
        ds = sig.get("data_source_name", "default")
        by_source.setdefault(ds, []).append(sig)

    # Build decimation snippet
    decimation_code = ""
    if max_points:
        decimation_code = textwrap.dedent(f"""\
            # Decimate if too many points
            if hasattr(data, '__len__') and len(data) > {max_points}:
                step = max(1, len(data) // {max_points})
                data = data[::step]
                if time is not None and hasattr(time, '__len__'):
                    time = time[::step]
        """)

    # Build per-source extraction blocks
    source_blocks: list[str] = []
    for data_source, signals in by_source.items():
        conn_code = connection_template.replace("{data_source}", data_source)
        cleanup_code = cleanup_template.replace("{data_source}", data_source)

        signal_entries = []
        for sig in signals:
            accessor = sig.get("accessor", "")
            sig_id = sig.get("id", "")
            data_code = data_template.replace("{accessor}", accessor)
            time_code = (
                time_template.replace("{accessor}", accessor) if time_template else ""
            )
            signal_entries.append(
                f"        {{\n"
                f'            "id": {sig_id!r},\n'
                f'            "data_code": {data_code!r},\n'
                f'            "time_code": {time_code!r},\n'
                f"        }},"
            )

        signals_list = "\n".join(signal_entries)

        block = textwrap.dedent(f"""\
    # --- Data source: {data_source} ---
    _signals_{data_source} = [
{signals_list}
    ]

    try:
        {conn_code}

        for _sig in _signals_{data_source}:
            try:
                data = eval(_sig["data_code"])
                time = eval(_sig["time_code"]) if _sig["time_code"] else None
{textwrap.indent(decimation_code, "                ")}
                results[_sig["id"]] = {{
                    "success": True,
                    "data": data.tolist() if hasattr(data, 'tolist') else data,
                    "time": time.tolist() if time is not None and hasattr(time, 'tolist') else None,
                    "shape": list(data.shape) if hasattr(data, 'shape') else [1],
                    "dtype": str(data.dtype) if hasattr(data, 'dtype') else type(data).__name__,
                }}
            except Exception as _e:
                results[_sig["id"]] = {{"success": False, "error": str(_e)[:200]}}

        {cleanup_code}
    except Exception as _conn_e:
        for _sig in _signals_{data_source}:
            results[_sig["id"]] = {{"success": False, "error": f"Connection error: {{_conn_e}}"[:200]}}
""")
        source_blocks.append(block)

    extraction_body = "\n".join(source_blocks)

    # Use eval() for template-substituted code — the plan's pattern uses
    # direct code injection, but eval() of pre-substituted strings is safer
    # because the accessor values come from graph data, not user input.
    # In production, consider generating proper function calls instead.
    script = textwrap.dedent(f"""\
#!/usr/bin/env python3
\"\"\"Auto-generated data extraction for {facility}:{ids_name}.

Generated by imas-codex mapping pipeline.
Reads configuration from stdin (JSON with "shot" field).
Outputs extraction results as JSON on stdout.
\"\"\"
import json
import sys

try:
    import msgpack
    import numpy as np
    USE_MSGPACK = True
except ImportError:
    USE_MSGPACK = False

{imports_template}

config = json.load(sys.stdin)
shot = config["shot"]
results = {{}}

{extraction_body}

# Output results
if USE_MSGPACK:
    # Binary output — more efficient for large arrays
    sys.stdout.buffer.write(
        msgpack.packb({{"results": results}}, use_bin_type=True)
    )
else:
    json.dump({{"results": results}}, sys.stdout)
""")

    return script


# ---------------------------------------------------------------------------
# 10.5 Tier 2: Static code validation
# ---------------------------------------------------------------------------


def validate_assembly_code(code: str, function_name: str) -> list[str]:
    """Validate generated assembly code without execution.

    Performs static analysis:
    1. Syntax check (ast.parse)
    2. Function discovery and signature validation
    3. Forbidden import detection (security)

    Args:
        code: Python code string to validate.
        function_name: Expected function name in the code.

    Returns:
        List of error messages (empty = valid).
    """
    errors: list[str] = []

    # 1. Parse check
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"Syntax error: {e}"]

    # 2. Find the assembly function
    funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    target = [f for f in funcs if f.name == function_name]
    if not target:
        errors.append(f"Function '{function_name}' not found in generated code")
        return errors

    func = target[0]

    # 3. Check function signature: must accept (ids, signals, mappings)
    args = [a.arg for a in func.args.args]
    expected = {"ids", "signals", "mappings"}
    if not expected.issubset(set(args)):
        errors.append(f"Function must accept {sorted(expected)}, got {args}")

    # 4. Check for forbidden operations (security)
    allowed_modules = {
        "numpy",
        "np",
        "math",
        "imas",
        "json",
        "collections",
        "functools",
        "itertools",
        "typing",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in allowed_modules:
                    errors.append(f"Forbidden import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] not in allowed_modules:
                errors.append(f"Forbidden import: from {node.module}")

    return errors


# ---------------------------------------------------------------------------
# 10.3 Assembly Code Generation
# ---------------------------------------------------------------------------


def generate_assembly_code(
    section_path: str,
    ids_name: str,
    mappings: list[dict[str, Any]],
    pattern: str = "array_per_node",
) -> tuple[str, str]:
    """Generate a default assembly code snippet for a section.

    Produces executable Python code that maps extracted signal data
    into an imas-python IDS struct-array. This serves as a fallback
    when the LLM does not generate assembly_code in the AssemblyConfig.

    Args:
        section_path: IMAS section path (e.g., "pf_active/coil").
        ids_name: IDS name (e.g., "pf_active").
        mappings: List of mapping dicts with source_id, target_id,
            source_property, transform_expression, source_units,
            target_units, cocos_label.
        pattern: Assembly pattern (from AssemblyPattern).

    Returns:
        Tuple of (code_string, function_name).
    """
    # Derive function name from section path
    func_name = f"assemble_{section_path.replace('/', '_')}"
    safe_section = section_path.replace("'", "\\'")

    # Determine array attribute from section path relative to IDS
    # e.g., pf_active/coil → coil, magnetics/flux_loop → flux_loop
    if "/" in section_path:
        array_attr = section_path.split("/", 1)[1].replace("/", ".")
    else:
        array_attr = section_path

    if pattern == "array_per_node":
        code = textwrap.dedent(f'''\
def {func_name}(ids, signals, mappings):
    """Assemble {safe_section} from mapped signals."""
    import numpy as np

    sources = sorted(set(m["source_id"] for m in mappings))
    getattr(ids, "{array_attr.split(".")[0]}").resize(len(sources))

    for i, source_id in enumerate(sources):
        entry = getattr(ids, "{array_attr}")[i]
        sig_data = signals.get(source_id, {{}})
        source_mappings = [m for m in mappings if m["source_id"] == source_id]

        for m in source_mappings:
            rel_path = m["target_id"].removeprefix("{safe_section}/")
            value = sig_data.get(m.get("source_property", "value"))
            if value is not None:
                if m.get("transform_expression") and m["transform_expression"] != "value":
                    value = execute_transform(value, m["transform_expression"])
                set_nested(entry, rel_path.replace("/", "."), value)
''')
    elif pattern == "concatenate":
        code = textwrap.dedent(f'''\
def {func_name}(ids, signals, mappings):
    """Assemble {safe_section} by concatenating sources."""
    import numpy as np

    by_target = {{}}
    for m in mappings:
        by_target.setdefault(m["target_id"], []).append(m)

    for target_id, target_mappings in by_target.items():
        arrays = []
        for m in target_mappings:
            sig_data = signals.get(m["source_id"], {{}})
            value = sig_data.get(m.get("source_property", "value"))
            if value is not None:
                if m.get("transform_expression") and m["transform_expression"] != "value":
                    value = execute_transform(value, m["transform_expression"])
                arrays.append(np.asarray(value))
        if arrays:
            combined = np.concatenate(arrays)
            rel_path = target_id.removeprefix("{safe_section}/")
            set_nested(ids, rel_path.replace("/", "."), combined)
''')
    else:
        # Generic fallback for other patterns
        code = textwrap.dedent(f'''\
def {func_name}(ids, signals, mappings):
    """Assemble {safe_section} ({pattern})."""
    import numpy as np

    for m in mappings:
        sig_data = signals.get(m["source_id"], {{}})
        value = sig_data.get(m.get("source_property", "value"))
        if value is not None:
            if m.get("transform_expression") and m["transform_expression"] != "value":
                value = execute_transform(value, m["transform_expression"])
            rel_path = m["target_id"].removeprefix("{safe_section}/")
            set_nested(ids, rel_path.replace("/", "."), value)
''')

    return code, func_name
