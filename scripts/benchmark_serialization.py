"""Benchmark script for msgpack vs JSON array serialization.

Compares serialization speed and size for numpy arrays typical of
IMAS signal extraction data (time series, profiles, scalars).

Usage:
    python scripts/benchmark_serialization.py
"""

from __future__ import annotations

import json
import sys
import time
from io import BytesIO


def benchmark_json(arrays: dict[str, list]) -> tuple[float, int]:
    """Benchmark JSON serialization of arrays.

    Returns:
        Tuple of (time_seconds, size_bytes).
    """
    start = time.perf_counter()
    data = json.dumps({"results": arrays})
    elapsed = time.perf_counter() - start
    return elapsed, len(data.encode())


def benchmark_msgpack(arrays: dict[str, dict]) -> tuple[float, int]:
    """Benchmark msgpack serialization of arrays.

    Returns:
        Tuple of (time_seconds, size_bytes).
    """
    import msgpack

    start = time.perf_counter()
    data = msgpack.packb({"results": arrays}, use_bin_type=True)
    elapsed = time.perf_counter() - start
    return elapsed, len(data)


def benchmark_msgpack_binary(arrays: dict[str, dict]) -> tuple[float, int]:
    """Benchmark msgpack with binary numpy packing.

    Returns:
        Tuple of (time_seconds, size_bytes).
    """
    import msgpack
    import numpy as np

    from imas_codex.remote.serialization import pack_array

    packed = {}
    for key, arr_dict in arrays.items():
        packed[key] = {
            "data": pack_array(np.array(arr_dict["data"])),
            "time": pack_array(np.array(arr_dict["time"])) if arr_dict.get("time") else None,
        }

    start = time.perf_counter()
    data = msgpack.packb({"results": packed}, use_bin_type=True)
    elapsed = time.perf_counter() - start
    return elapsed, len(data)


def main():
    import numpy as np

    print("Serialization Benchmark: JSON vs msgpack")
    print("=" * 60)

    # Generate test data of increasing sizes
    test_cases = [
        ("10 signals × 1K points", 10, 1_000),
        ("10 signals × 10K points", 10, 10_000),
        ("50 signals × 10K points", 50, 10_000),
        ("50 signals × 100K points", 50, 100_000),
    ]

    for label, n_signals, n_points in test_cases:
        print(f"\n{label}")
        print("-" * 60)

        # Generate random signal data
        arrays = {}
        for i in range(n_signals):
            data = np.random.randn(n_points).tolist()
            time_arr = np.linspace(0, 10, n_points).tolist()
            arrays[f"signal_{i}"] = {
                "success": True,
                "data": data,
                "time": time_arr,
                "shape": [n_points],
                "dtype": "float64",
            }

        # JSON benchmark
        json_time, json_size = benchmark_json(arrays)
        print(f"  JSON:           {json_time:.4f}s  {json_size / 1024:.0f} KB")

        # msgpack benchmark (if available)
        try:
            msgpack_time, msgpack_size = benchmark_msgpack(arrays)
            print(f"  msgpack (list): {msgpack_time:.4f}s  {msgpack_size / 1024:.0f} KB")
            print(
                f"    → {json_size / msgpack_size:.1f}x smaller, "
                f"{json_time / msgpack_time:.1f}x faster than JSON"
            )
        except ImportError:
            print("  msgpack: not installed (pip install msgpack)")

        # msgpack binary benchmark
        try:
            bin_time, bin_size = benchmark_msgpack_binary(arrays)
            print(f"  msgpack (bin):  {bin_time:.4f}s  {bin_size / 1024:.0f} KB")
            print(
                f"    → {json_size / bin_size:.1f}x smaller, "
                f"{json_time / bin_time:.1f}x faster than JSON"
            )
        except ImportError:
            print("  msgpack binary: not installed (pip install msgpack)")


if __name__ == "__main__":
    main()
