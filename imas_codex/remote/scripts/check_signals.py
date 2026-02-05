#!/usr/bin/env python3
"""Remote signal check script.

This script is executed on remote facilities via SSH. It checks MDSplus
signals by testing data access and returns structured results for the batch.

Requirements:
- Python 3.8+ (stdlib only except MDSplus)
- MDSplus Python bindings

Usage:
    echo '{"signals": [...]}' | python3 check_signals.py

Input (JSON on stdin):
    {
        "signals": [
            {
                "id": "tcv:results:/ip",
                "accessor": "\\ip",
                "tree_name": "results",
                "shot": 80000
            },
            ...
        ],
        "timeout_per_signal": 10
    }

Output (JSON on stdout):
    {
        "results": [
            {
                "id": "tcv:results:/ip",
                "success": true,
                "shape": [1000],
                "dtype": "float64"
            },
            {
                "id": "tcv:results:/bad_signal",
                "success": false,
                "error": "TreeNNF: Node not found"
            },
            ...
        ]
    }
"""

import json
import signal
import sys
from typing import Any


def timeout_handler(signum: int, frame: Any) -> None:
    """Handle signal timeout."""
    raise TimeoutError("Signal validation timed out")


def validate_signal(
    sig: dict[str, Any],
    timeout: int = 10,
) -> dict[str, Any]:
    """Validate a single signal by testing data access.

    Args:
        sig: Signal dict with id, accessor, tree_name, shot
        timeout: Timeout in seconds for this signal

    Returns:
        Result dict with id, success, and shape/dtype or error
    """
    result: dict[str, Any] = {"id": sig["id"], "success": False}

    accessor = sig.get("accessor")
    tree_name = sig.get("tree_name", "results")
    shot = sig.get("shot")

    if not accessor or not shot:
        result["error"] = "Missing accessor or shot"
        return result

    # Set timeout for this signal
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)

    try:
        import MDSplus

        tree = MDSplus.Tree(tree_name, int(shot), "readonly")
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

    except TimeoutError:
        result["error"] = f"Timeout after {timeout}s"
    except Exception as e:
        result["error"] = str(e)[:200]
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    return result


def main() -> None:
    """Read config from stdin, validate signals, output JSON to stdout."""
    try:
        config = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON input: {e}"}))
        sys.exit(1)

    signals = config.get("signals", [])
    timeout_per_signal = config.get("timeout_per_signal", 10)

    results = []
    for sig in signals:
        result = validate_signal(sig, timeout=timeout_per_signal)
        results.append(result)

    print(json.dumps({"results": results}))


if __name__ == "__main__":
    main()
