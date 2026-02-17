#!/usr/bin/env python3
"""Extract TDI signal metadata via source parsing and runtime probing.

Two-phase extraction:
1. **Source parsing**: Parse .fun files to extract case() quantities,
   build_path() references, make_with_units(), and function signatures.
2. **Runtime probing**: Execute TDI expressions via MDSplus to extract
   actual MDSplus paths (from TreePath results), units (via units_of()),
   and data shape/dtype.

This replaces the simpler extract_tdi_functions.py which only did static
case() parsing and missed direct accessor functions entirely.

Designed to run remotely on TCV (no external dependencies beyond MDSplus).

Usage:
    python extract_tdi_signals.py /usr/local/CRPP/tdi/tcv [--shot 85000] [--probe]

Output: JSON with function metadata and signal catalog.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CaseBlock:
    """A parsed case() block from a dispatch function."""

    quantity: str
    build_paths: list[str] = field(default_factory=list)
    units: list[str] = field(default_factory=list)
    tdi_deps: list[str] = field(default_factory=list)
    source_snippet: str = ""  # The case block code


@dataclass
class SignalProbe:
    """Runtime probe result for a TDI signal."""

    accessor: str
    success: bool = False
    result_type: str = ""
    mds_path: str = ""  # From TreePath result
    units: str = ""  # From units_of()
    time_units: str = ""  # From units_of(dim_of())
    shape: list[int] | str = field(default_factory=list)
    dtype: str = ""
    error: str = ""


@dataclass
class TDISignal:
    """A signal accessible via a TDI function.

    Combines source-parsed metadata with optional runtime probe data.
    """

    function_name: str
    quantity: str  # Empty for direct accessors
    accessor: str  # Full TDI expression: tcv_get('IP') or li_1()

    # From source parsing
    build_paths: list[str] = field(default_factory=list)
    source_units: list[str] = field(default_factory=list)
    tdi_deps: list[str] = field(default_factory=list)
    source_snippet: str = ""
    section_comment: str = ""  # Physics domain hint from source

    # From runtime probing (optional)
    probe: SignalProbe | None = None


@dataclass
class TDIFunctionInfo:
    """Metadata for a TDI function file."""

    name: str
    path: str
    signature: str = ""
    description: str = ""
    parameters: list[str] = field(default_factory=list)
    function_type: str = ""  # dispatch, direct, utility, inventory
    source_code: str = ""
    mdsplus_trees: list[str] = field(default_factory=list)
    tdi_dependencies: list[str] = field(default_factory=list)
    case_quantities: list[str] = field(default_factory=list)
    signals: list[TDISignal] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Source parsing helpers
# ---------------------------------------------------------------------------

# TDI/MDSplus builtins to ignore when finding function dependencies
_BUILTINS = frozenset(
    {
        "if_error",
        "make_signal",
        "make_with_units",
        "make_param",
        "make_dim",
        "data",
        "dim_of",
        "units_of",
        "help_of",
        "validation_of",
        "size",
        "shape",
        "spread",
        "replicate",
        "replicate_s",
        "return",
        "switch",
        "case",
        "upcase",
        "index",
        "extract",
        "present",
        "not",
        "and",
        "or",
        "xor",
        "maxval",
        "minval",
        "count",
        "sum",
        "mean",
        "sqrt",
        "abs",
        "nint",
        "cvt",
        "pack",
        "firstloc",
        "intersection",
        "word",
        "atand",
        "atan2",
        "floor",
        "ceil",
        "build_path",
        "getnci",
        "node_exists",
        "using",
        "execute",
        "ndesc",
        "dscptr",
        "kind",
        "fs_float",
        "fix_roprand",
        "fix_roprand_mask",
        "default",
        "break",
        "write",
        "sort",
        "build_signal",
        "build_with_units",
        "array",
        "zero",
        "set_range",
        "conv_date",
        "for",
        "while",
        "if",
        "else",
    }
)

# Source selector case values (not signal quantities)
_SOURCE_SELECTORS = frozenset(
    {
        "FBTE",
        "FBTE.M",
        "LIUQE",
        "LIUQE.M",
        "LIUQE2",
        "LIUQE.M2",
        "LIUQE.M3",
        "LIUQE3",
        "FLAT",
        "FLAT.M",
        "RAMP",
        "RAMP.M",
        "RUNS",
        "RUNS.M",
        "MAGNETICS",
        "PCS",
    }
)


def _extract_section_comments(content: str) -> dict[int, str]:
    """Extract section comments like /* CASE --- PLASMA CURRENT ---*/

    Returns mapping of line_number -> section_name.
    """
    sections = {}
    for i, line in enumerate(content.split("\n")):
        m = re.match(r"/\*\s*CASE\s*[-]+\s*(.+?)\s*[-]*/", line)
        if m:
            sections[i] = m.group(1).strip()
    return sections


def _find_section_for_line(sections: dict[int, str], target_line: int) -> str:
    """Find which section a given line number belongs to."""
    best = ""
    for line_no, section in sorted(sections.items()):
        if line_no <= target_line:
            best = section
    return best


def _extract_case_blocks(content: str) -> list[CaseBlock]:
    """Extract all case() blocks with their content from a dispatch function."""
    blocks = []

    # Find case("NAME") { ... break; } patterns
    pattern = r'case\s*\(\s*["\']([^"\']+)["\']\s*\)\s*\{(.*?)break;\s*\}'
    for m in re.finditer(pattern, content, re.DOTALL | re.IGNORECASE):
        qty = m.group(1).upper()
        block_code = m.group(2).strip()

        if qty in _SOURCE_SELECTORS:
            continue

        cb = CaseBlock(
            quantity=qty,
            source_snippet=block_code[:500],  # Truncate for transport
        )

        # Extract build_path() references
        cb.build_paths = re.findall(
            r'build_path\(\s*["\']([^"\']+)["\']\s*\)', block_code
        )

        # Extract make_with_units
        cb.units = re.findall(
            r'make_with_units\([^,]+,\s*["\']([^"\']+)["\']\s*\)', block_code
        )

        # Extract TDI function dependencies
        calls = re.findall(r"\b([a-z_][a-z0-9_]*)\s*\(", block_code, re.IGNORECASE)
        cb.tdi_deps = sorted(
            {
                c.lower()
                for c in calls
                if c.lower() not in _BUILTINS and c.lower() not in ("case", "default")
            }
        )

        blocks.append(cb)

    return blocks


def _find_tdi_functions_in_dir(tdi_dir: str) -> set[str]:
    """Get the set of all .fun file names (without extension) in directory."""
    names = set()
    try:
        for f in os.listdir(tdi_dir):
            if f.endswith(".fun"):
                names.add(f[:-4])
    except OSError:
        pass
    return names


def _classify_function_type(
    name: str,
    content: str,
    case_blocks: list[CaseBlock],
    sig_params: list[str],
) -> str:
    """Classify function type based on content analysis."""
    # Internal helpers
    if name.startswith("_"):
        return "internal"

    # Dispatch function (case-based)
    if len(case_blocks) > 2:
        return "dispatch"

    # Check if it returns string arrays (inventory function)
    if re.search(r"return\s*\(\s*\[.*?\\\\", content, re.DOTALL):
        return "inventory"

    # Check if it has no meaningful parameters (direct accessor)
    meaningful_params = [
        p for p in sig_params if p != "_do" and not p.startswith("_opt")
    ]
    if len(meaningful_params) <= 1:
        return "direct"

    # Has parameters but not dispatch → parametric
    if len(sig_params) > 1:
        return "parametric"

    return "utility"


# ---------------------------------------------------------------------------
# Main parsing
# ---------------------------------------------------------------------------


def parse_tdi_function(
    filepath: str, known_functions: set[str]
) -> TDIFunctionInfo | None:
    """Parse a single .fun file and extract comprehensive metadata."""
    try:
        with open(filepath, encoding="utf-8", errors="replace") as f:
            content = f.read()
    except OSError:
        return None

    name = Path(filepath).stem
    info = TDIFunctionInfo(name=name, path=filepath, source_code=content)

    # Extract header comment/description
    header = re.search(r"/\*([^*]|\*(?!/))*\*/", content, re.DOTALL)
    if header:
        info.description = header.group(0).strip("/* \n\t")[:500]

    # Extract signature and parameters
    sig = re.search(
        r"(?:public\s+)?fun\s+(\w+)\s*\((.*?)\)", content, re.IGNORECASE | re.DOTALL
    )
    if sig:
        info.signature = f"{sig.group(1)}({sig.group(2).strip()})"
        info.parameters = re.findall(
            r"(?:in|optional\s+in|out)\s+(\w+)", sig.group(2), re.IGNORECASE
        )

    # Extract case blocks (for dispatch functions)
    case_blocks = _extract_case_blocks(content)
    info.case_quantities = [cb.quantity for cb in case_blocks]

    # Classify function type
    info.function_type = _classify_function_type(
        name, content, case_blocks, info.parameters
    )

    # Extract global build_path references (for non-dispatch functions)
    all_bp = re.findall(r'build_path\(\s*["\']([^"\']+)["\']\s*\)', content)
    trees = set()
    for p in all_bp:
        tm = re.match(r"\\\\?(\w+)::", p)
        if tm:
            trees.add(tm.group(1).lower())
    info.mdsplus_trees = sorted(trees)

    # Extract TDI function dependencies
    func_calls = re.findall(r"\b([a-z_][a-z0-9_]*)\s*\(", content, re.IGNORECASE)
    deps = set()
    for fc in func_calls:
        fc_lower = fc.lower()
        if (
            fc_lower not in _BUILTINS
            and fc_lower != name.lower()
            and fc_lower in known_functions
        ):
            deps.add(fc_lower)
    info.tdi_dependencies = sorted(deps)

    # Extract section comments for physics domain hints
    sections = _extract_section_comments(content)

    # Build signals
    if info.function_type == "dispatch":
        # Each case quantity is a signal
        for cb in case_blocks:
            # Find section for this case
            # Search for the case line in content
            case_pattern = f'case("{cb.quantity}")'
            case_pos = content.lower().find(case_pattern.lower())
            line_no = content[:case_pos].count("\n") if case_pos >= 0 else 0
            section = _find_section_for_line(sections, line_no)

            accessor = f"{name}('{cb.quantity}')"
            signal = TDISignal(
                function_name=name,
                quantity=cb.quantity,
                accessor=accessor,
                build_paths=cb.build_paths,
                source_units=cb.units,
                tdi_deps=cb.tdi_deps,
                source_snippet=cb.source_snippet,
                section_comment=section,
            )
            info.signals.append(signal)

    elif info.function_type == "direct":
        # Function itself is the signal (no quantity argument)
        accessor = f"{name}()"
        all_units = re.findall(
            r'make_with_units\([^,]+,\s*["\']([^"\']+)["\']\s*\)', content
        )
        signal = TDISignal(
            function_name=name,
            quantity="",
            accessor=accessor,
            build_paths=sorted(set(all_bp)),
            source_units=sorted(set(all_units)),
            tdi_deps=sorted(deps - {name}),
        )
        info.signals.append(signal)

    elif info.function_type == "inventory":
        # Returns string arrays of MDSplus paths - special handling
        accessor = f"{name}()"
        signal = TDISignal(
            function_name=name,
            quantity="",
            accessor=accessor,
            build_paths=sorted(set(all_bp)),
        )
        info.signals.append(signal)

    return info


# ---------------------------------------------------------------------------
# Runtime probing
# ---------------------------------------------------------------------------


def probe_signals(
    signals: list[TDISignal],
    shot: int,
    tree_name: str = "tcv_shot",
) -> None:
    """Execute TDI signals via MDSplus and collect runtime metadata.

    Modifies signals in-place, adding probe results.
    Requires MDSplus to be available.
    """
    try:
        import MDSplus
    except ImportError:
        print("MDSplus not available, skipping runtime probing", file=sys.stderr)
        return

    try:
        tree = MDSplus.Tree(tree_name, shot, "readonly")
    except Exception as e:
        print(f"Cannot open tree {tree_name} shot {shot}: {e}", file=sys.stderr)
        return

    for signal in signals:
        probe = SignalProbe(accessor=signal.accessor)
        try:
            result = tree.tdiExecute(signal.accessor)
            probe.result_type = type(result).__name__
            probe.success = True

            # If result is a TreePath, extract MDSplus path
            if hasattr(result, "path"):
                probe.mds_path = str(result.path)

            # Extract units via units_of()
            try:
                units = tree.tdiExecute(f"units_of({signal.accessor})").data()
                u = str(units).strip()
                if u and u != " ":
                    probe.units = u
            except Exception:
                pass

            # Extract time units
            try:
                tu = tree.tdiExecute(f"units_of(dim_of({signal.accessor}))").data()
                t = str(tu).strip()
                if t and t != " ":
                    probe.time_units = t
            except Exception:
                pass

            # Get shape/dtype
            try:
                data = result.data()
                probe.shape = list(data.shape) if hasattr(data, "shape") else []
                probe.dtype = str(data.dtype)
            except Exception:
                pass

        except Exception as e:
            probe.error = str(e)[:200]

        signal.probe = probe

    try:
        tree.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def extract_all(
    tdi_dir: str,
    shot: int | None = None,
    do_probe: bool = False,
) -> dict:
    """Extract comprehensive TDI signal metadata.

    Args:
        tdi_dir: Path to TDI function directory
        shot: Shot number for runtime probing
        do_probe: Whether to execute TDI expressions for runtime metadata

    Returns:
        Dict with function_count, signal_count, functions, and signals.
    """
    known_funs = _find_tdi_functions_in_dir(tdi_dir)

    functions: list[TDIFunctionInfo] = []
    all_signals: list[TDISignal] = []

    for fname in sorted(os.listdir(tdi_dir)):
        if not fname.endswith(".fun"):
            continue
        fpath = os.path.join(tdi_dir, fname)
        info = parse_tdi_function(fpath, known_funs)
        if info:
            functions.append(info)
            all_signals.extend(info.signals)

    # Runtime probing
    if do_probe and shot is not None and all_signals:
        probe_signals(all_signals, shot)

    # Serialize
    func_dicts = []
    for f in functions:
        d = {
            "name": f.name,
            "path": f.path,
            "signature": f.signature,
            "description": f.description,
            "parameters": f.parameters,
            "function_type": f.function_type,
            "mdsplus_trees": f.mdsplus_trees,
            "tdi_dependencies": f.tdi_dependencies,
            "case_quantities": f.case_quantities,
            "signal_count": len(f.signals),
        }
        func_dicts.append(d)

    signal_dicts = []
    for s in all_signals:
        d = {
            "function_name": s.function_name,
            "quantity": s.quantity,
            "accessor": s.accessor,
            "build_paths": s.build_paths,
            "source_units": s.source_units,
            "tdi_deps": s.tdi_deps,
            "section_comment": s.section_comment,
        }
        if s.probe:
            d["probe"] = {
                "success": s.probe.success,
                "result_type": s.probe.result_type,
                "mds_path": s.probe.mds_path,
                "units": s.probe.units,
                "time_units": s.probe.time_units,
                "shape": s.probe.shape,
                "dtype": s.probe.dtype,
                "error": s.probe.error,
            }
        signal_dicts.append(d)

    # Summary
    by_type = {}
    for f in functions:
        by_type[f.function_type] = by_type.get(f.function_type, 0) + 1

    probed = sum(1 for s in all_signals if s.probe and s.probe.success)
    with_paths = sum(1 for s in all_signals if s.build_paths)
    with_units = sum(1 for s in all_signals if s.source_units)
    with_mds_path = sum(1 for s in all_signals if s.probe and s.probe.mds_path)
    with_probe_units = sum(1 for s in all_signals if s.probe and s.probe.units)

    return {
        "summary": {
            "function_count": len(functions),
            "signal_count": len(all_signals),
            "functions_by_type": by_type,
            "signals_with_build_paths": with_paths,
            "signals_with_source_units": with_units,
            "signals_probed_ok": probed if do_probe else None,
            "signals_with_mds_path": with_mds_path if do_probe else None,
            "signals_with_probe_units": with_probe_units if do_probe else None,
        },
        "functions": func_dicts,
        "signals": signal_dicts,
    }


def main():
    """CLI entry point."""
    import argparse

    # Check if input is via stdin (JSON)
    if not sys.stdin.isatty():
        try:
            input_data = json.load(sys.stdin)
            tdi_dir = input_data.get("tdi_path", "")
            shot = input_data.get("shot")
            do_probe = input_data.get("probe", False)
        except json.JSONDecodeError:
            print("Invalid JSON on stdin", file=sys.stderr)
            sys.exit(1)
    else:
        parser = argparse.ArgumentParser(description="Extract TDI signal metadata")
        parser.add_argument("tdi_dir", help="Path to TDI function directory")
        parser.add_argument("--shot", type=int, default=None, help="Shot for probing")
        parser.add_argument(
            "--probe", action="store_true", help="Enable runtime probing"
        )
        args = parser.parse_args()
        tdi_dir = args.tdi_dir
        shot = args.shot
        do_probe = args.probe

    if not os.path.isdir(tdi_dir):
        print(f"Not a directory: {tdi_dir}", file=sys.stderr)
        sys.exit(1)

    result = extract_all(tdi_dir, shot=shot, do_probe=do_probe)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
