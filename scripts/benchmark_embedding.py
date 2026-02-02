#!/usr/bin/env python3
"""Benchmark comparing local MiniLM vs remote Qwen3-Embedding-0.6B.

Compares:
- Speed: embeddings per second
- Semantic accuracy: IDS path retrieval precision

Usage:
    uv run python scripts/benchmark_embedding.py
"""

import time
from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class BenchmarkResult:
    """Benchmark results for a single model."""

    model_name: str
    dimension: int
    total_texts: int
    total_time_s: float
    embeddings_per_sec: float
    avg_latency_ms: float
    mrr: float  # Mean Reciprocal Rank for retrieval
    precision_at_1: float
    precision_at_5: float


# IDS test cases: (query, expected_top_path)
IDS_TEST_CASES = [
    ("electron temperature profile", "core_profiles.profiles_1d[:].electrons.temperature"),
    ("toroidal magnetic field", "equilibrium.time_slice[:].profiles_1d.b_field_tor"),
    ("plasma current", "equilibrium.time_slice[:].global_quantities.ip"),
    ("ion density", "core_profiles.profiles_1d[:].ion[:].density"),
    ("electron density profile", "core_profiles.profiles_1d[:].electrons.density"),
    ("magnetic axis position", "equilibrium.time_slice[:].global_quantities.magnetic_axis.r"),
    ("plasma boundary", "equilibrium.time_slice[:].boundary.outline.r"),
    ("safety factor q profile", "equilibrium.time_slice[:].profiles_1d.q"),
    ("neutron rate", "summary.fusion.neutron_fluxes.total.value"),
    ("heating power", "summary.heating_current_drive.power_launched.value"),
    ("wall temperature", "wall.wall_module.limiter_unit[:].point[:].temperature"),
    ("divertor target heat flux", "divertor.divertor_unit[:].target[:].flux.heat"),
    ("NBI power", "nbi.unit[:].power_launched.value"),
    ("ECH frequency", "ec_launchers.beam[:].frequency.value"),
    ("pellet injection velocity", "pellets.unit[:].pellet_set[:].launching_speed.value"),
]

# IDS paths corpus (subset for benchmark)
IDS_PATHS = [
    "core_profiles.profiles_1d[:].electrons.temperature",
    "core_profiles.profiles_1d[:].electrons.density",
    "core_profiles.profiles_1d[:].electrons.pressure",
    "core_profiles.profiles_1d[:].ion[:].density",
    "core_profiles.profiles_1d[:].ion[:].temperature",
    "core_profiles.profiles_1d[:].ion[:].pressure",
    "equilibrium.time_slice[:].profiles_1d.b_field_tor",
    "equilibrium.time_slice[:].profiles_1d.b_field_pol",
    "equilibrium.time_slice[:].profiles_1d.q",
    "equilibrium.time_slice[:].profiles_1d.pressure",
    "equilibrium.time_slice[:].global_quantities.ip",
    "equilibrium.time_slice[:].global_quantities.magnetic_axis.r",
    "equilibrium.time_slice[:].global_quantities.magnetic_axis.z",
    "equilibrium.time_slice[:].boundary.outline.r",
    "equilibrium.time_slice[:].boundary.outline.z",
    "summary.fusion.neutron_fluxes.total.value",
    "summary.heating_current_drive.power_launched.value",
    "wall.wall_module.limiter_unit[:].point[:].temperature",
    "divertor.divertor_unit[:].target[:].flux.heat",
    "divertor.divertor_unit[:].target[:].flux.particle",
    "nbi.unit[:].power_launched.value",
    "nbi.unit[:].species.a",
    "ec_launchers.beam[:].frequency.value",
    "ec_launchers.beam[:].power_launched.value",
    "pellets.unit[:].pellet_set[:].launching_speed.value",
    "ic_antennas.antenna[:].power_launched.value",
    "magnetics.flux_loop[:].flux.value",
    "magnetics.b_field_pol_probe[:].field.value",
    "pf_active.coil[:].current.value",
    "pf_passive.loop[:].resistance.value",
]


def benchmark_local_minilm() -> tuple[BenchmarkResult, np.ndarray, np.ndarray]:
    """Benchmark local MiniLM model."""
    print("\n=== Local MiniLM ===")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Speed benchmark
    start = time.perf_counter()
    path_embeddings = model.encode(IDS_PATHS, normalize_embeddings=True)
    query_embeddings = model.encode(
        [q for q, _ in IDS_TEST_CASES], normalize_embeddings=True
    )
    total_time = time.perf_counter() - start

    total_texts = len(IDS_PATHS) + len(IDS_TEST_CASES)
    embeddings_per_sec = total_texts / total_time

    # Semantic accuracy
    mrr, p1, p5 = compute_retrieval_metrics(
        query_embeddings, path_embeddings, IDS_PATHS
    )

    result = BenchmarkResult(
        model_name="all-MiniLM-L6-v2",
        dimension=384,
        total_texts=total_texts,
        total_time_s=total_time,
        embeddings_per_sec=embeddings_per_sec,
        avg_latency_ms=(total_time / total_texts) * 1000,
        mrr=mrr,
        precision_at_1=p1,
        precision_at_5=p5,
    )

    print(f"  Dimension: {result.dimension}")
    print(f"  Speed: {result.embeddings_per_sec:.1f} emb/s")
    print(f"  Latency: {result.avg_latency_ms:.2f} ms/text")
    print(f"  MRR: {result.mrr:.3f}")
    print(f"  P@1: {result.precision_at_1:.3f}")
    print(f"  P@5: {result.precision_at_5:.3f}")

    return result, query_embeddings, path_embeddings


def benchmark_remote_qwen(url: str = "http://localhost:18765") -> tuple[BenchmarkResult, np.ndarray, np.ndarray]:
    """Benchmark remote Qwen3-Embedding-0.6B via HTTP."""
    print("\n=== Remote Qwen3-Embedding-0.6B (iter GPU) ===")

    # Get info
    with httpx.Client(timeout=30) as client:
        info = client.get(f"{url}/info").json()
        model_name = info["model"]["name"]
        dimension = info["model"]["embedding_dimension"]

        # Speed benchmark - path embeddings
        start = time.perf_counter()
        resp = client.post(f"{url}/embed", json={"texts": IDS_PATHS})
        path_embeddings = np.array(resp.json()["embeddings"])

        # Query embeddings
        resp = client.post(
            f"{url}/embed", json={"texts": [q for q, _ in IDS_TEST_CASES]}
        )
        query_embeddings = np.array(resp.json()["embeddings"])
        total_time = time.perf_counter() - start

    total_texts = len(IDS_PATHS) + len(IDS_TEST_CASES)
    embeddings_per_sec = total_texts / total_time

    # Semantic accuracy
    mrr, p1, p5 = compute_retrieval_metrics(
        query_embeddings, path_embeddings, IDS_PATHS
    )

    result = BenchmarkResult(
        model_name=model_name,
        dimension=dimension,
        total_texts=total_texts,
        total_time_s=total_time,
        embeddings_per_sec=embeddings_per_sec,
        avg_latency_ms=(total_time / total_texts) * 1000,
        mrr=mrr,
        precision_at_1=p1,
        precision_at_5=p5,
    )

    print(f"  Dimension: {result.dimension}")
    print(f"  Speed: {result.embeddings_per_sec:.1f} emb/s")
    print(f"  Latency: {result.avg_latency_ms:.2f} ms/text")
    print(f"  MRR: {result.mrr:.3f}")
    print(f"  P@1: {result.precision_at_1:.3f}")
    print(f"  P@5: {result.precision_at_5:.3f}")

    return result, query_embeddings, path_embeddings


def compute_retrieval_metrics(
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    corpus_texts: list[str],
) -> tuple[float, float, float]:
    """Compute MRR, P@1, P@5 for retrieval."""
    similarities = cosine_similarity(query_embeddings, corpus_embeddings)

    reciprocal_ranks = []
    correct_at_1 = 0
    correct_at_5 = 0

    for i, (query, expected_path) in enumerate(IDS_TEST_CASES):
        # Get ranked indices
        ranked_indices = np.argsort(similarities[i])[::-1]
        ranked_paths = [corpus_texts[j] for j in ranked_indices]

        # Find rank of expected path
        try:
            rank = ranked_paths.index(expected_path) + 1
            reciprocal_ranks.append(1.0 / rank)
            if rank == 1:
                correct_at_1 += 1
            if rank <= 5:
                correct_at_5 += 1
        except ValueError:
            reciprocal_ranks.append(0.0)

    mrr = np.mean(reciprocal_ranks)
    p1 = correct_at_1 / len(IDS_TEST_CASES)
    p5 = correct_at_5 / len(IDS_TEST_CASES)

    return mrr, p1, p5


def print_comparison(local: BenchmarkResult, remote: BenchmarkResult):
    """Print side-by-side comparison."""
    print("\n" + "=" * 60)
    print("COMPARISON: MiniLM (local CPU) vs Qwen3 (iter GPU)")
    print("=" * 60)

    print(f"\n{'Metric':<25} {'MiniLM':<15} {'Qwen3':<15} {'Winner':<10}")
    print("-" * 65)

    # Dimension (informational)
    print(f"{'Dimension':<25} {local.dimension:<15} {remote.dimension:<15} {'Qwen3 (2.7x)':<10}")

    # Speed comparison
    speed_winner = "Qwen3" if remote.embeddings_per_sec > local.embeddings_per_sec else "MiniLM"
    speed_ratio = max(remote.embeddings_per_sec, local.embeddings_per_sec) / min(remote.embeddings_per_sec, local.embeddings_per_sec)
    print(f"{'Speed (emb/s)':<25} {local.embeddings_per_sec:<15.1f} {remote.embeddings_per_sec:<15.1f} {speed_winner} ({speed_ratio:.1f}x)")

    # Latency
    lat_winner = "Qwen3" if remote.avg_latency_ms < local.avg_latency_ms else "MiniLM"
    print(f"{'Latency (ms/text)':<25} {local.avg_latency_ms:<15.2f} {remote.avg_latency_ms:<15.2f} {lat_winner}")

    # Semantic metrics
    mrr_winner = "Qwen3" if remote.mrr > local.mrr else "MiniLM"
    print(f"{'MRR':<25} {local.mrr:<15.3f} {remote.mrr:<15.3f} {mrr_winner}")

    p1_winner = "Qwen3" if remote.precision_at_1 > local.precision_at_1 else "MiniLM"
    print(f"{'Precision@1':<25} {local.precision_at_1:<15.3f} {remote.precision_at_1:<15.3f} {p1_winner}")

    p5_winner = "Qwen3" if remote.precision_at_5 > local.precision_at_5 else "MiniLM"
    print(f"{'Precision@5':<25} {local.precision_at_5:<15.3f} {remote.precision_at_5:<15.3f} {p5_winner}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"""
Qwen3-Embedding-0.6B provides:
- {remote.dimension / local.dimension:.1f}x higher dimensional embeddings ({remote.dimension} vs {local.dimension})
- MRR improvement: {(remote.mrr - local.mrr) / local.mrr * 100:+.1f}%
- P@1 improvement: {(remote.precision_at_1 - local.precision_at_1) / max(local.precision_at_1, 0.001) * 100:+.1f}%

Speed note: Remote includes network latency (SSH tunnel to iter).
GPU batch processing may be faster for large batches despite network overhead.
""")


def main():
    """Run benchmarks."""
    print("IDS Embedding Benchmark")
    print(f"Test cases: {len(IDS_TEST_CASES)}")
    print(f"Corpus size: {len(IDS_PATHS)} paths")

    try:
        local_result, _, _ = benchmark_local_minilm()
    except Exception as e:
        print(f"  Error: {e}")
        return

    try:
        remote_result, _, _ = benchmark_remote_qwen()
    except Exception as e:
        print(f"  Error: {e}")
        print("  (Is SSH tunnel to iter active? LocalForward 18765)")
        return

    print_comparison(local_result, remote_result)


if __name__ == "__main__":
    main()
