"""
Embedding model retrieval quality tests.

Tests MiniLM (all-MiniLM-L6-v2) embedding quality for IMAS DD retrieval.

Run with:
    uv run pytest tests/embeddings/test_model_comparison.py -v -s -m slow

Or standalone:
    uv run python tests/embeddings/test_model_comparison.py
"""

import time
from dataclasses import dataclass

import numpy as np
import pytest

sentence_transformers = pytest.importorskip(
    "sentence_transformers",
    reason="sentence-transformers not installed (optional GPU dependency)",
)
SentenceTransformer = sentence_transformers.SentenceTransformer

# Exclude from default test runs — these load real models
pytestmark = pytest.mark.slow

# Test cases: (query, expected_top_paths, language)
# Expected paths should appear in top-5 results for good retrieval
DD_RETRIEVAL_CASES = [
    # English queries
    (
        "electron temperature profile",
        ["core_profiles/profiles_1d/electrons/temperature"],
        "en",
    ),
    (
        "plasma current measurement",
        ["magnetics/ip"],
        "en",
    ),
    (
        "equilibrium reconstruction toroidal flux",
        ["equilibrium/time_slice/profiles_1d/phi"],
        "en",
    ),
    (
        "neutral beam injection power",
        ["nbi/unit/power_launched"],
        "en",
    ),
    (
        "Thomson scattering density",
        ["thomson_scattering/channel/n_e"],
        "en",
    ),
]

# Sample DD paths for corpus (subset for fast testing)
SAMPLE_DD_PATHS = [
    ("core_profiles/profiles_1d/electrons/temperature", "Electron temperature profile"),
    ("core_profiles/profiles_1d/electrons/density", "Electron density profile"),
    ("core_profiles/profiles_1d/ion/temperature", "Ion temperature"),
    ("magnetics/ip", "Plasma current"),
    ("magnetics/b_field_tor_vacuum_r", "Toroidal magnetic field"),
    ("equilibrium/time_slice/profiles_1d/phi", "Toroidal flux"),
    ("equilibrium/time_slice/profiles_1d/psi", "Poloidal flux"),
    ("equilibrium/time_slice/profiles_1d/q", "Safety factor q"),
    ("nbi/unit/power_launched", "NBI power launched"),
    ("nbi/unit/energy", "NBI beam energy"),
    ("thomson_scattering/channel/n_e", "Thomson scattering electron density"),
    ("thomson_scattering/channel/t_e", "Thomson scattering electron temperature"),
    ("ece/channel/t_e", "ECE electron temperature"),
    ("bolometer/channel/power", "Bolometer power"),
    ("interferometer/channel/n_e_line", "Interferometer line-integrated density"),
    ("charge_exchange/channel/ion/temperature", "CX ion temperature"),
    ("mhd/time_slice/toroidal_mode", "MHD toroidal mode"),
    ("edge_profiles/profiles_2d/electrons/temperature", "Edge electron temperature"),
    ("pf_active/coil/current", "PF coil current"),
    ("wall/temperature_reference", "Wall temperature"),
]


@dataclass
class BenchmarkResult:
    """Results from a single model benchmark."""

    model_name: str
    load_time_s: float
    encode_time_s: float
    memory_mb: float
    embedding_dim: int
    retrieval_accuracy: float  # fraction of cases where expected path in top-5


def load_model_with_timing(model_name: str) -> tuple[SentenceTransformer, float, float]:
    """Load model and return (model, load_time, memory_mb)."""
    import gc
    import os

    gc.collect()

    # Get baseline memory
    try:
        import psutil

        process = psutil.Process(os.getpid())
        baseline_mem = process.memory_info().rss / 1024 / 1024
    except ImportError:
        baseline_mem = 0

    start = time.time()
    model = SentenceTransformer(model_name)
    load_time = time.time() - start

    try:
        import psutil

        process = psutil.Process(os.getpid())
        current_mem = process.memory_info().rss / 1024 / 1024
        memory_mb = current_mem - baseline_mem
    except ImportError:
        memory_mb = 0

    return model, load_time, memory_mb


def compute_retrieval_accuracy(
    model: SentenceTransformer,
    test_cases: list[tuple[str, list[str], str]],
    corpus: list[tuple[str, str]],
    top_k: int = 5,
) -> float:
    """Compute fraction of test cases where expected path appears in top-k."""
    # Encode corpus
    corpus_texts = [f"{path}: {desc}" for path, desc in corpus]
    corpus_embeddings = model.encode(corpus_texts, convert_to_numpy=True)

    hits = 0
    for query, expected_paths, _lang in test_cases:
        query_embedding = model.encode([query], convert_to_numpy=True)

        # Cosine similarity
        similarities = np.dot(corpus_embeddings, query_embedding.T).flatten()
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Check if any expected path in top-k
        top_paths = [corpus[i][0] for i in top_indices]
        if any(exp in top_paths for exp in expected_paths):
            hits += 1

    return hits / len(test_cases) if test_cases else 0.0


def benchmark_model(model_name: str) -> BenchmarkResult:
    """Run full benchmark for a single model."""
    import gc

    print(f"\n{'=' * 60}")
    print(f"Benchmarking: {model_name}")
    print("=" * 60)

    # Load model
    model, load_time, memory_mb = load_model_with_timing(model_name)
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"  Load time: {load_time:.2f}s")
    print(f"  Memory: {memory_mb:.0f} MB")
    print(f"  Embedding dim: {embedding_dim}")

    # Time encoding
    test_texts = [desc for _, desc in SAMPLE_DD_PATHS]
    start = time.time()
    model.encode(test_texts)
    encode_time = time.time() - start
    print(f"  Encode time ({len(test_texts)} texts): {encode_time:.3f}s")

    # Retrieval accuracy
    accuracy = compute_retrieval_accuracy(model, DD_RETRIEVAL_CASES, SAMPLE_DD_PATHS)
    print(f"  Retrieval accuracy: {accuracy:.0%}")

    # Cleanup model to free memory
    del model
    gc.collect()

    return BenchmarkResult(
        model_name=model_name,
        load_time_s=load_time,
        encode_time_s=encode_time,
        memory_mb=memory_mb,
        embedding_dim=embedding_dim,
        retrieval_accuracy=accuracy,
    )


def print_comparison(results: list[BenchmarkResult]) -> None:
    """Print comparison table."""
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    headers = ["Model", "Dim", "Load(s)", "Enc(s)", "Mem(MB)", "Acc"]
    print(
        f"{headers[0]:<35} {headers[1]:>5} {headers[2]:>8} {headers[3]:>7} "
        f"{headers[4]:>8} {headers[5]:>7}"
    )
    print("-" * 70)

    for r in results:
        print(
            f"{r.model_name:<35} {r.embedding_dim:>5} {r.load_time_s:>8.2f} "
            f"{r.encode_time_s:>7.3f} {r.memory_mb:>8.0f} "
            f"{r.retrieval_accuracy:>6.0%}"
        )


# =============================================================================
# pytest tests
# =============================================================================


@pytest.fixture(scope="module")
def minilm_model():
    """Load MiniLM model once for all tests."""
    return SentenceTransformer("all-MiniLM-L6-v2")


class TestMiniLMRetrieval:
    """Test MiniLM retrieval accuracy for IMAS DD paths."""

    def test_english_retrieval(self, minilm_model):
        """MiniLM handles English retrieval."""
        accuracy = compute_retrieval_accuracy(
            minilm_model, DD_RETRIEVAL_CASES, SAMPLE_DD_PATHS
        )
        assert accuracy >= 0.4, f"MiniLM English accuracy too low: {accuracy:.0%}"


# =============================================================================
# Standalone execution
# =============================================================================


def main():
    """Run MiniLM benchmark."""
    result = benchmark_model("all-MiniLM-L6-v2")
    print_comparison([result])


if __name__ == "__main__":
    main()
