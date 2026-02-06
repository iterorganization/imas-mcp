#!/usr/bin/env python3
"""High-throughput batched signal check script.

Optimized for checking thousands of signals by:
1. Grouping signals by (tree_name, shot) to minimize tree opens
2. Processing all signals for a tree/shot combination in one batch
3. Using efficient TDI execution with minimal overhead per signal

Requirements:
- Python 3.8+ (stdlib only except MDSplus)
- MDSplus Python bindings

Usage:
    echo '{"signals": [...]}' | python3 check_signals_batch.py

Input (JSON on stdin):
    {
        "signals": [
            {"id": "tcv:results:/ip", "accessor": "\\ip", "tree_name": "results", "shot": 80000},
            {"id": "tcv:results:/ne", "accessor": "\\ne", "tree_name": "results", "shot": 80000},
            ...
        ],
        "timeout_per_group": 30
    }

Output (JSON on stdout):
    {
        "results": [
            {"id": "tcv:results:/ip", "success": true, "shape": [1000], "dtype": "float64"},
            {"id": "tcv:results:/ne", "success": false, "error": "TreeNNF: Node not found"},
            ...
        ],
        "stats": {
            "total": 100,
            "groups": 5,
            "success": 85,
            "failed": 15
        }
    }
"""

import json
import signal
import sys
from collections import defaultdict
from typing import Any


def timeout_handler(signum: int, frame: Any) -> None:
    """Handle signal timeout."""
    raise TimeoutError("Group processing timed out")


def check_signal_group(
    tree_name: str,
    shot: int,
    signals: list[dict[str, Any]],
    timeout: int = 30,
) -> list[dict[str, Any]]:
    """Check all signals for a single tree/shot combination.

    Opens the tree once and checks all signals, minimizing MDSplus overhead.

    Args:
        tree_name: MDSplus tree name
        shot: Shot number
        signals: List of signal dicts with id, accessor
        timeout: Timeout in seconds for entire group

    Returns:
        List of result dicts with id, success, and shape/dtype or error
    """
    results = []

    # Set timeout for entire group
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)

    try:
        import MDSplus

        # Open tree once for all signals in group
        tree = MDSplus.Tree(tree_name, int(shot), "readonly")

        for sig in signals:
            result: dict[str, Any] = {"id": sig["id"], "success": False}

            try:
                accessor = sig.get("accessor")
                if not accessor:
                    result["error"] = "Missing accessor"
                    results.append(result)
                    continue

                # Execute TDI and get data
                data = tree.tdiExecute(accessor).data()

                result["success"] = True

                # Get shape
                if hasattr(data, "shape"):
                    result["shape"] = list(data.shape)
                elif hasattr(data, "__len__"):
                    result["shape"] = [len(data)]
                else:
                    result["shape"] = [1]  # scalar

                # Get dtype
                if hasattr(data, "dtype"):
                    result["dtype"] = str(data.dtype)
                else:
                    result["dtype"] = type(data).__name__

            except Exception as e:
                result["error"] = str(e)[:200]

            results.append(result)

    except TimeoutError:
        # Timeout for entire group - mark remaining as failed
        checked_ids = {r["id"] for r in results}
        for sig in signals:
            if sig["id"] not in checked_ids:
                results.append(
                    {
                        "id": sig["id"],
                        "success": False,
                        "error": f"Group timeout after {timeout}s",
                    }
                )
    except Exception as e:
        # Tree open failed - all signals in group fail
        error_msg = str(e)[:200]
        for sig in signals:
            results.append(
                {
                    "id": sig["id"],
                    "success": False,
                    "error": f"Tree open failed: {error_msg}",
                }
            )
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    return results


def main() -> None:
    """Read config from stdin, validate signals in batches, output JSON to stdout."""
    try:
        config = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON input: {e}", "results": []}))
        sys.exit(0)  # Exit cleanly so caller gets JSON

    signals = config.get("signals", [])
    timeout_per_group = config.get("timeout_per_group", 30)

    if not signals:
        print(
            json.dumps(
                {
                    "results": [],
                    "stats": {"total": 0, "groups": 0, "success": 0, "failed": 0},
                }
            )
        )
        return

    # Group signals by (tree_name, shot) for efficient batch processing
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for sig in signals:
        tree_name = sig.get("tree_name", "results")
        shot = sig.get("shot")
        if shot is not None:
            groups[(tree_name, int(shot))].append(sig)
        else:
            # No shot - immediate failure
            pass

    # Process each group
    all_results = []
    for (tree_name, shot), group_signals in groups.items():
        group_results = check_signal_group(
            tree_name,
            shot,
            group_signals,
            timeout=timeout_per_group,
        )
        all_results.extend(group_results)

    # Compute stats
    success_count = sum(1 for r in all_results if r.get("success"))
    failed_count = len(all_results) - success_count

    output = {
        "results": all_results,
        "stats": {
            "total": len(signals),
            "groups": len(groups),
            "success": success_count,
            "failed": failed_count,
        },
    }
    print(json.dumps(output))


if __name__ == "__main__":
    main()
