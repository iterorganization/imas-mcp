#!/usr/bin/env python3
"""High-throughput batched signal check script.

Optimized for checking thousands of signals by:
1. Grouping signals by (tree_name, shot) to minimize tree opens
2. Processing all signals for a tree/shot combination in one batch
3. Trying multiple check_shots per signal (all versions/epochs)

Requirements:
- Python 3.8+ (stdlib only except MDSplus)
- MDSplus Python bindings

Usage:
    echo '{"signals": [...]}' | python3 check_signals_batch.py

Input (JSON on stdin):
    {
        "signals": [
            {"id": "tcv:results:/ip", "accessor": "\\ip", "tree_name": "results",
             "check_shots": [85000]},
            {"id": "tcv:static:/r_c", "accessor": "\\STATIC::R_C", "tree_name": "static",
             "check_shots": [1, 2, 3, 4, 5, 6, 7, 8]},
            ...
        ],
        "timeout_per_group": 30
    }

    Each signal has ``check_shots`` — an ordered list of shots to try.
    The first shot is tried first; if that fails with a shot-dependent
    error, remaining shots are tried in order until one succeeds.
    Structural errors (SYNTAX, MISS_ARG) stop retries immediately.

    Legacy format (``shot`` + optional ``fallback_shots``) is still
    accepted for backward compatibility.

Output (JSON on stdout):
    {
        "results": [
            {"id": "tcv:results:/ip", "success": true, "shape": [1000], "dtype": "float64",
             "checked_shot": 85000},
            {"id": "tcv:static:/r_c", "success": true, "shape": [1], "dtype": "float64",
             "checked_shot": 3, "failed_shots": [1, 2]},
            ...
        ],
        "stats": {
            "total": 100,
            "groups": 5,
            "success": 85,
            "failed": 15,
            "retry_success": 3
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


# Errors that are shot-dependent and worth retrying on fallback shots
_SHOT_DEPENDENT_ERRORS = (
    "NODATA",  # No data for this shot
    "NNF",  # Node not found (may exist in other tree version)
    "KEYNOTFOU",  # Key not found (shot-dependent lookup)
    "TreeNOT_OPEN",  # Tree not available for this shot
    "Unable to change",  # Unable to change current shot
)

# Errors that are structural and never resolve by changing shots
_STRUCTURAL_ERRORS = (
    "MISS_ARG",  # Function signature error
    "UNKNOWN_VAR",  # Undefined variable
    "INVCLADSC",  # Invalid class descriptor
    "INVDTYDSC",  # Invalid data type descriptor
    "SYNTAX",  # TDI syntax error
)


def _is_shot_dependent_error(error: str) -> bool:
    """Check if an error might resolve on a different shot."""
    return any(tag in error for tag in _SHOT_DEPENDENT_ERRORS)


def _normalize_signal_shots(sig: dict) -> list:
    """Extract check_shots from a signal, supporting both new and legacy formats.

    New format: ``check_shots`` list (preferred).
    Legacy format: ``shot`` + optional ``fallback_shots``.

    Returns:
        Ordered list of shots to try.
    """
    if "check_shots" in sig:
        return [int(s) for s in sig["check_shots"]]
    # Legacy format
    shots = []
    if "shot" in sig and sig["shot"] is not None:
        shots.append(int(sig["shot"]))
    for s in sig.get("fallback_shots", []):
        shots.append(int(s))
    return shots


def check_signal_group(
    tree_name: str,
    shot: int,
    signals: list[dict[str, Any]],
    timeout: int = 30,
) -> list[dict[str, Any]]:
    """Check all signals for a single tree/shot combination.

    Opens the tree once and checks all signals, minimizing MDSplus overhead.
    Captures stdout/stderr during TDI execution to prevent pollution from
    functions like tile_store that print debug output.

    Args:
        tree_name: MDSplus tree name
        shot: Shot number
        signals: List of signal dicts with id, accessor
        timeout: Timeout in seconds for entire group

    Returns:
        List of result dicts with id, success, and shape/dtype or error
    """
    import io
    import os

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

                # Capture stdout during TDI execution to prevent pollution
                # from functions like tile_store that print debug output
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()
                # Also redirect fd 1 for C-level prints
                old_fd = os.dup(1)
                devnull_fd = os.open(os.devnull, os.O_WRONLY)
                os.dup2(devnull_fd, 1)
                try:
                    data = tree.tdiExecute(accessor).data()
                finally:
                    os.dup2(old_fd, 1)
                    os.close(old_fd)
                    os.close(devnull_fd)
                    sys.stdout = old_stdout

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
                # Ensure stdout is restored even if exception occurs mid-redirect
                if sys.stdout != sys.__stdout__ and hasattr(sys.stdout, "getvalue"):
                    sys.stdout = old_stdout  # type: ignore[possibly-undefined]
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
    # Redirect stderr to suppress MDSplus library loading noise
    # (e.g., libvaccess.so, TDI function debug output like tile_store).
    # This prevents contamination of JSON output on stdout.
    import os

    _original_stderr = sys.stderr
    sys.stderr = open(os.devnull, "w")  # noqa: SIM115

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

    # Group signals by (tree_name, first_check_shot) for efficient batch processing
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    # Track remaining check_shots per signal for retry
    signal_remaining_shots: dict[str, list[int]] = {}
    for sig in signals:
        tree_name = sig.get("tree_name", "results")
        check_shots = _normalize_signal_shots(sig)
        if check_shots:
            primary_shot = check_shots[0]
            groups[(tree_name, primary_shot)].append(sig)
            if len(check_shots) > 1:
                signal_remaining_shots[sig["id"]] = check_shots[1:]
        else:
            # No shots — immediate failure
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
        # Tag results with the shot that was checked
        for r in group_results:
            r["checked_shot"] = shot
        all_results.extend(group_results)

    # Retry failed signals on remaining check_shots (shot-dependent errors only)
    retry_groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    final_results: list[dict] = []
    failed_for_retry: dict[str, list[int]] = {}  # signal_id -> list of failed shots

    for result in all_results:
        sig_id = result["id"]
        if result.get("success"):
            final_results.append(result)
        elif sig_id in signal_remaining_shots and _is_shot_dependent_error(
            result.get("error", "")
        ):
            # Track the failed shot and queue for retry
            failed_for_retry.setdefault(sig_id, []).append(result["checked_shot"])
            # Find original signal data
            orig_sig = None
            for sig in signals:
                if sig["id"] == sig_id:
                    orig_sig = sig
                    break
            if orig_sig:
                tree_name = orig_sig.get("tree_name", "results")
                for retry_shot in signal_remaining_shots[sig_id]:
                    retry_groups[(tree_name, retry_shot)].append(orig_sig)
        else:
            final_results.append(result)

    # Process retry groups
    if retry_groups:
        # Track which signals have already succeeded during retries
        succeeded: set[str] = set()
        for (tree_name, shot), group_signals in retry_groups.items():
            # Skip signals that already succeeded on an earlier retry
            pending = [s for s in group_signals if s["id"] not in succeeded]
            if not pending:
                continue
            group_results = check_signal_group(
                tree_name, shot, pending, timeout=timeout_per_group
            )
            for r in group_results:
                sig_id = r["id"]
                if r.get("success") and sig_id not in succeeded:
                    succeeded.add(sig_id)
                    r["checked_shot"] = shot
                    r["failed_shots"] = failed_for_retry.get(sig_id, [])
                    final_results.append(r)
                elif not r.get("success") and sig_id not in succeeded:
                    # Track this shot as failed too
                    failed_for_retry.setdefault(sig_id, []).append(shot)

        # Add remaining failures (never succeeded on any shot)
        for sig_id, failed_shots in failed_for_retry.items():
            if sig_id not in succeeded:
                # Find original signal to get error from primary attempt
                for result in all_results:
                    if result["id"] == sig_id:
                        result["failed_shots"] = failed_shots
                        final_results.append(result)
                        break

    # Compute stats
    success_count = sum(1 for r in final_results if r.get("success"))
    failed_count = len(final_results) - success_count
    retry_success = sum(1 for r in final_results if r.get("failed_shots"))

    output = {
        "results": final_results,
        "stats": {
            "total": len(signals),
            "groups": len(groups),
            "success": success_count,
            "failed": failed_count,
            "retry_success": retry_success,
        },
    }
    print(json.dumps(output))


if __name__ == "__main__":
    main()
