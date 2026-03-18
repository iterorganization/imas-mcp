"""Benchmark: msgpack vs Arrow vs Protobuf for numpy array serialization.

Compares serialization throughput (MB/s) and wire size for numpy float64
arrays representative of IMAS signal extraction payloads — from small
profile data up to large multi-signal time-series streams.

Usage:
    uv run python scripts/benchmark_serialization.py
"""

from __future__ import annotations

import json
import statistics
import time
from io import BytesIO

import numpy as np

# ---------------------------------------------------------------------------
# Benchmark runners — each returns (pack_seconds, unpack_seconds, wire_bytes)
# ---------------------------------------------------------------------------

WARMUP = 2
ITERATIONS = 5


def _median(fn, n=ITERATIONS):
    """Run fn n times and return median result tuple."""
    results = [fn() for _ in range(n)]
    # Transpose list-of-tuples and take median per position
    return tuple(statistics.median(col) for col in zip(*results, strict=False))


def bench_json(arrays: list[np.ndarray]) -> tuple[float, float, int]:
    """JSON: tolist() → json.dumps / json.loads → np.array."""

    def run():
        payload = [a.tolist() for a in arrays]
        t0 = time.perf_counter()
        packed = json.dumps(payload).encode()
        t_pack = time.perf_counter() - t0

        t0 = time.perf_counter()
        decoded = json.loads(packed)
        _ = [np.array(d, dtype=np.float64) for d in decoded]
        t_unpack = time.perf_counter() - t0
        return t_pack, t_unpack, len(packed)

    # warmup
    for _ in range(WARMUP):
        run()
    return _median(run)


def bench_msgpack_list(arrays: list[np.ndarray]) -> tuple[float, float, int]:
    """msgpack with Python lists (no binary numpy encoding)."""
    import msgpack

    def run():
        payload = [a.tolist() for a in arrays]
        t0 = time.perf_counter()
        packed = msgpack.packb(payload, use_bin_type=True)
        t_pack = time.perf_counter() - t0

        t0 = time.perf_counter()
        decoded = msgpack.unpackb(packed, raw=False)
        _ = [np.array(d, dtype=np.float64) for d in decoded]
        t_unpack = time.perf_counter() - t0
        return t_pack, t_unpack, len(packed)

    for _ in range(WARMUP):
        run()
    return _median(run)


def bench_msgpack_binary(arrays: list[np.ndarray]) -> tuple[float, float, int]:
    """msgpack with raw numpy .tobytes() encoding (our wire format)."""
    import msgpack

    from imas_codex.remote.serialization import pack_array, unpack_array

    def run():
        packed_arrays = [pack_array(a) for a in arrays]
        t0 = time.perf_counter()
        packed = msgpack.packb(packed_arrays, use_bin_type=True)
        t_pack = time.perf_counter() - t0

        t0 = time.perf_counter()
        decoded = msgpack.unpackb(packed, raw=False)
        _ = [unpack_array(d) for d in decoded]
        t_unpack = time.perf_counter() - t0
        return t_pack, t_unpack, len(packed)

    for _ in range(WARMUP):
        run()
    return _median(run)


def bench_arrow_ipc(arrays: list[np.ndarray]) -> tuple[float, float, int]:
    """Apache Arrow IPC (Feather) stream serialization."""
    import pyarrow as pa

    def run():
        # Build a RecordBatch with one column per array
        cols = {f"s{i}": pa.array(a) for i, a in enumerate(arrays)}
        batch = pa.record_batch(cols)

        sink = BytesIO()
        t0 = time.perf_counter()
        writer = pa.ipc.new_stream(sink, batch.schema)
        writer.write_batch(batch)
        writer.close()
        packed = sink.getvalue()
        t_pack = time.perf_counter() - t0

        t0 = time.perf_counter()
        reader = pa.ipc.open_stream(BytesIO(packed))
        table = reader.read_all()
        _ = [table.column(i).to_numpy() for i in range(table.num_columns)]
        t_unpack = time.perf_counter() - t0
        return t_pack, t_unpack, len(packed)

    for _ in range(WARMUP):
        run()
    return _median(run)


def bench_arrow_file(arrays: list[np.ndarray]) -> tuple[float, float, int]:
    """Apache Arrow file (random-access) serialization."""
    import pyarrow as pa

    def run():
        cols = {f"s{i}": pa.array(a) for i, a in enumerate(arrays)}
        batch = pa.record_batch(cols)

        sink = BytesIO()
        t0 = time.perf_counter()
        writer = pa.ipc.new_file(sink, batch.schema)
        writer.write_batch(batch)
        writer.close()
        packed = sink.getvalue()
        t_pack = time.perf_counter() - t0

        t0 = time.perf_counter()
        reader = pa.ipc.open_file(BytesIO(packed))
        table = reader.read_all()
        _ = [table.column(i).to_numpy() for i in range(table.num_columns)]
        t_unpack = time.perf_counter() - t0
        return t_pack, t_unpack, len(packed)

    for _ in range(WARMUP):
        run()
    return _median(run)


def bench_protobuf(arrays: list[np.ndarray]) -> tuple[float, float, int]:
    """Protobuf using self-describing wire format (no .proto compilation).

    Encodes each array as: 4-byte length prefix + raw float64 bytes.
    This is the minimal viable protobuf-style binary framing.
    """

    def run():
        import struct

        # Pack: for each array, write shape + dtype tag + raw bytes
        t0 = time.perf_counter()
        parts = []
        for a in arrays:
            raw = a.tobytes()
            header = struct.pack("<II", len(a.shape), len(raw))
            shape_bytes = struct.pack(f"<{len(a.shape)}I", *a.shape)
            parts.append(header + shape_bytes + raw)
        packed = b"".join(parts)
        t_pack = time.perf_counter() - t0

        # Unpack
        t0 = time.perf_counter()
        offset = 0
        result = []
        while offset < len(packed):
            ndim, nbytes = struct.unpack_from("<II", packed, offset)
            offset += 8
            shape = struct.unpack_from(f"<{ndim}I", packed, offset)
            offset += 4 * ndim
            arr = np.frombuffer(
                packed, dtype=np.float64, count=nbytes // 8, offset=offset
            )
            arr = arr.reshape(shape)
            result.append(arr)
            offset += nbytes
        t_unpack = time.perf_counter() - t0
        return t_pack, t_unpack, len(packed)

    for _ in range(WARMUP):
        run()
    return _median(run)


def bench_numpy_save(arrays: list[np.ndarray]) -> tuple[float, float, int]:
    """numpy .npz (compressed zip of .npy files)."""

    def run():
        buf = BytesIO()
        t0 = time.perf_counter()
        np.savez(buf, *arrays)
        packed = buf.getvalue()
        t_pack = time.perf_counter() - t0

        t0 = time.perf_counter()
        loaded = np.load(BytesIO(packed))
        _ = [loaded[k] for k in loaded.files]
        t_unpack = time.perf_counter() - t0
        return t_pack, t_unpack, len(packed)

    for _ in range(WARMUP):
        run()
    return _median(run)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_throughput(data_bytes: float, seconds: float) -> str:
    """Format as MB/s or GB/s."""
    if seconds <= 0:
        return "∞"
    mbps = (data_bytes / 1e6) / seconds
    if mbps >= 1000:
        return f"{mbps / 1000:.1f} GB/s"
    return f"{mbps:.1f} MB/s"


def _fmt_size(nbytes: int) -> str:
    """Format byte count."""
    if nbytes >= 1e9:
        return f"{nbytes / 1e9:.2f} GB"
    if nbytes >= 1e6:
        return f"{nbytes / 1e6:.1f} MB"
    return f"{nbytes / 1e3:.0f} KB"


def _raw_data_size(arrays: list[np.ndarray]) -> int:
    """Total raw bytes in the numpy arrays."""
    return sum(a.nbytes for a in arrays)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


FORMATS = [
    ("JSON", bench_json),
    ("msgpack (list)", bench_msgpack_list),
    ("msgpack (binary)", bench_msgpack_binary),
    ("Arrow IPC stream", bench_arrow_ipc),
    ("Arrow IPC file", bench_arrow_file),
    ("Raw binary framing", bench_protobuf),
    ("numpy .npz", bench_numpy_save),
]


def main():
    print("=" * 90)
    print("Serialization Benchmark: msgpack vs Arrow vs Protobuf-style vs numpy")
    print("  numpy float64 arrays — IMAS signal extraction payloads")
    print(f"  {WARMUP} warmup + {ITERATIONS} iterations, median reported")
    print("=" * 90)

    # Representative IMAS data sizes:
    # - Small: 5 profile signals, 200 radial points each (equilibrium profiles)
    # - Medium: 20 diagnostic signals, 10K time points (magnetics probes)
    # - Large: 50 signals, 100K time points (full PF active extraction)
    # - XL: 100 signals, 500K time points (multi-IDS batch extraction)
    # - XXL: 50 signals, 1M time points (long-pulse tokamak discharge)
    test_cases = [
        ("5 × 200 (profiles)", 5, 200),
        ("20 × 10K (magnetics)", 20, 10_000),
        ("50 × 100K (PF active)", 50, 100_000),
        ("100 × 500K (multi-IDS)", 100, 500_000),
        ("50 × 1M (long pulse)", 50, 1_000_000),
    ]

    # Skip JSON for payloads > 50 MB raw — it's too slow and memory-hungry
    JSON_MAX_RAW = 50 * 1024 * 1024

    for label, n_signals, n_points in test_cases:
        arrays = [np.random.randn(n_points) for _ in range(n_signals)]
        raw_size = _raw_data_size(arrays)

        print(f"\n{'─' * 90}")
        print(f"  {label}  —  raw data: {_fmt_size(raw_size)}")
        print(f"{'─' * 90}")
        print(
            f"  {'Format':<22s} {'Pack':>12s} {'Unpack':>12s} "
            f"{'Wire size':>12s} {'Ratio':>8s} {'Pack MB/s':>12s} {'Unpack MB/s':>12s}"
        )
        print(f"  {'─' * 84}")

        for name, fn in FORMATS:
            if name == "JSON" and raw_size > JSON_MAX_RAW:
                print(
                    f"  {name:<22s} {'— skipped —':>12s}  (too slow for {_fmt_size(raw_size)} payload)"
                )
                continue
            if name == "msgpack (list)" and raw_size > JSON_MAX_RAW:
                print(
                    f"  {name:<22s} {'— skipped —':>12s}  (tolist() too slow for {_fmt_size(raw_size)} payload)"
                )
                continue
            try:
                t_pack, t_unpack, wire_size = fn(arrays)
                ratio = wire_size / raw_size
                print(
                    f"  {name:<22s} {t_pack:>11.4f}s {t_unpack:>11.4f}s "
                    f"{_fmt_size(wire_size):>12s} {ratio:>7.2f}x "
                    f"{_fmt_throughput(raw_size, t_pack):>12s} "
                    f"{_fmt_throughput(raw_size, t_unpack):>12s}"
                )
            except ImportError as e:
                mod = str(e).split("'")[1] if "'" in str(e) else str(e)
                print(f"  {name:<22s} {'not installed':>12s}  (pip install {mod})")
            except Exception as e:
                print(f"  {name:<22s} {'ERROR':>12s}  {e}")

    # Dependency analysis
    print(f"\n{'=' * 90}")
    print("Dependency Analysis (installability on remote facility compute nodes)")
    print(f"{'=' * 90}")
    deps = [
        (
            "msgpack",
            "Pure Python (C ext optional)",
            "uv pip install msgpack",
            "~100 KB",
        ),
        (
            "pyarrow",
            "C++/Cython compiled extension",
            "uv pip install pyarrow",
            "~150 MB",
        ),
        ("protobuf", "C++ compiled extension", "uv pip install protobuf", "~1 MB"),
        ("numpy", "C/Fortran compiled extension", "uv pip install numpy", "~30 MB"),
        ("struct (stdlib)", "No install needed", "—", "0"),
    ]
    print(f"  {'Package':<22s} {'Type':<32s} {'Install':<30s} {'Size':>10s}")
    print(f"  {'─' * 96}")
    for pkg, typ, install, size in deps:
        print(f"  {pkg:<22s} {typ:<32s} {install:<30s} {size:>10s}")


if __name__ == "__main__":
    main()
