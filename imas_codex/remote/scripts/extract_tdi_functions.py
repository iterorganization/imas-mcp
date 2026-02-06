#!/usr/bin/env python3
"""Extract TDI function metadata from .fun files.

This script parses TDI function files to extract:
- Function signatures and documentation
- Supported quantities (from case statements)
- MDSplus paths (from build_path calls)
- Dependencies on other TDI functions

Designed to run remotely on TCV (no external dependencies).

Usage:
    python extract_tdi_functions.py /usr/local/CRPP/tdi/tcv

Output: JSON array of function metadata.
"""

from __future__ import annotations

import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class TDIQuantity:
    """A quantity accessible via a TDI function."""

    name: str
    mdsplus_paths: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)  # Other TDI calls
    description: str = ""


@dataclass
class TDIFunctionMeta:
    """Metadata extracted from a TDI .fun file."""

    name: str
    path: str
    description: str = ""
    signature: str = ""
    source_code: str = ""  # Full .fun file content for LLM context
    parameters: list[str] = field(default_factory=list)
    quantities: list[TDIQuantity] = field(default_factory=list)
    mdsplus_trees: set[str] = field(default_factory=set)
    tdi_dependencies: set[str] = field(default_factory=set)
    has_shot_conditional: bool = False
    shot_conditionals: list[str] = field(default_factory=list)


def parse_tdi_function(filepath: str) -> TDIFunctionMeta | None:
    """Parse a .fun file and extract metadata."""
    try:
        with open(filepath, encoding="utf-8", errors="replace") as f:
            content = f.read()
    except OSError:
        return None

    name = Path(filepath).stem
    meta = TDIFunctionMeta(name=name, path=filepath, source_code=content)

    # Extract description from header comment
    header_match = re.search(r"/\*([^*]|\*(?!/))*\*/", content, re.DOTALL)
    if header_match:
        meta.description = header_match.group(0).strip("/* \n\t")

    # Extract function signature
    sig_match = re.search(
        r"(?:public\s+)?fun\s+(\w+)\s*\((.*?)\)", content, re.IGNORECASE | re.DOTALL
    )
    if sig_match:
        meta.signature = f"{sig_match.group(1)}({sig_match.group(2).strip()})"
        # Parse parameters
        params_str = sig_match.group(2)
        params = re.findall(
            r"(?:in|optional\s+in|out)\s+(\w+)", params_str, re.IGNORECASE
        )
        meta.parameters = params

    # Extract case statement quantities
    # Pattern: case("QUANTITY_NAME") or case ('QUANTITY_NAME')
    case_matches = re.findall(
        r'case\s*\(\s*["\']([^"\']+)["\']\s*\)', content, re.IGNORECASE
    )
    seen_quantities = set()
    for q in case_matches:
        q_upper = q.upper()
        if q_upper not in seen_quantities:
            seen_quantities.add(q_upper)
            meta.quantities.append(TDIQuantity(name=q_upper))

    # Extract MDSplus paths from build_path calls
    # Pattern: build_path("\\TREE::PATH")
    path_matches = re.findall(
        r'build_path\s*\(\s*["\']([^"\']+)["\']', content, re.IGNORECASE
    )
    for path in path_matches:
        # Extract tree name from path like \\RESULTS::...
        tree_match = re.match(r"\\\\?(\w+)::", path)
        if tree_match:
            meta.mdsplus_trees.add(tree_match.group(1).lower())

    # Also find getnci references (alternative path format)
    getnci_matches = re.findall(
        r'getnci\s*\(\s*["\']([^"\']+)["\']', content, re.IGNORECASE
    )
    for path in getnci_matches:
        tree_match = re.match(r"\\\\?(\w+)::", path)
        if tree_match:
            meta.mdsplus_trees.add(tree_match.group(1).lower())

    # Find TDI function dependencies (calls to other TDI functions)
    # Pattern: function_name(...) where function_name is not a builtin
    builtins = {
        "if_error",
        "make_signal",
        "make_with_units",
        "data",
        "dim_of",
        "size",
        "shape",
        "spread",
        "replicate",
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
        "build_path",
        "getnci",
        "make_dim",
        "using",
        "execute",
        "node_exists",
        "fs_float",
        "default",
        "break",
        "write",
    }
    func_calls = re.findall(r"\b([a-z_][a-z0-9_]*)\s*\(", content, re.IGNORECASE)
    for func in func_calls:
        func_lower = func.lower()
        if (
            func_lower not in builtins
            and func_lower != name.lower()
            and func_lower.startswith(
                (
                    "tcv_",
                    "fir_",
                    "get_",
                    "pol_",
                    "li_",
                    "time_",
                    "fbte_",
                    "ddtliu",
                    "ddtsm",
                )
            )
        ):
            meta.tdi_dependencies.add(func_lower)

    # Check for shot-conditional logic
    shot_patterns = [
        r"\$SHOT\s*[<>=!]+\s*\d+",  # $SHOT < 69874
        r"\$SHOT\s*==\s*-1",  # Model shot check
        r"\$SHOT\s*>=\s*900000",  # Fictitious shot range
    ]
    for pattern in shot_patterns:
        matches = re.findall(pattern, content)
        if matches:
            meta.has_shot_conditional = True
            meta.shot_conditionals.extend(matches)

    return meta


def extract_all_functions(tdi_dir: str) -> list[dict]:
    """Extract metadata from all .fun files in directory."""
    results = []

    fun_files = sorted(Path(tdi_dir).glob("*.fun"))
    for fun_file in fun_files:
        meta = parse_tdi_function(str(fun_file))
        if meta:
            # Convert to dict, handling sets
            d = asdict(meta)
            d["mdsplus_trees"] = sorted(d["mdsplus_trees"])
            d["tdi_dependencies"] = sorted(d["tdi_dependencies"])
            d["quantities"] = [asdict(q) for q in meta.quantities]
            results.append(d)

    return results


def main():
    """Main entry point - handles both CLI args and JSON stdin."""

    tdi_dir = None

    # Check for CLI argument first
    if len(sys.argv) >= 2:
        tdi_dir = sys.argv[1]
    else:
        # Try to read from stdin (JSON format)
        try:
            import json

            input_data = json.load(sys.stdin)
            tdi_dir = input_data.get("tdi_path")
        except Exception:
            pass

    if not tdi_dir:
        print("Usage: extract_tdi_functions.py <tdi_directory>", file=sys.stderr)
        print('Or pass JSON: {"tdi_path": "/path/to/tdi"}', file=sys.stderr)
        sys.exit(1)

        print(f"Error: {tdi_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    results = extract_all_functions(tdi_dir)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
