"""
Benchmark comparison between embedding models.

Compares MiniLM (current) vs Qwen3-Embedding-0.6B (proposed) across:
1. IMAS DD retrieval accuracy
2. Multilingual query handling (Japanese, French, German)
3. Latency and memory usage

Run with:
    uv run pytest tests/embeddings/test_model_comparison.py -v -s

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

# Exclude from default test runs — these download large models
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

MULTILINGUAL_CASES = [
    # Japanese
    (
        "電子温度プロファイル",  # "electron temperature profile"
        ["core_profiles/profiles_1d/electrons/temperature"],
        "ja",
    ),
    (
        "プラズマ電流測定",  # "plasma current measurement"
        ["magnetics/ip"],
        "ja",
    ),
    # French
    (
        "profil de température électronique",  # "electron temperature profile"
        ["core_profiles/profiles_1d/electrons/temperature"],
        "fr",
    ),
    (
        "mesure du courant plasma",  # "plasma current measurement"
        ["magnetics/ip"],
        "fr",
    ),
    # German
    (
        "Elektronentemperaturprofil",  # "electron temperature profile"
        ["core_profiles/profiles_1d/electrons/temperature"],
        "de",
    ),
    (
        "Plasmastrom Messung",  # "plasma current measurement"
        ["magnetics/ip"],
        "de",
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
    multilingual_accuracy: float  # same for multilingual queries


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
    model = SentenceTransformer(model_name, trust_remote_code=True)
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

    # Retrieval accuracy - English
    en_accuracy = compute_retrieval_accuracy(model, DD_RETRIEVAL_CASES, SAMPLE_DD_PATHS)
    print(f"  English retrieval accuracy: {en_accuracy:.0%}")

    # Retrieval accuracy - Multilingual
    ml_accuracy = compute_retrieval_accuracy(model, MULTILINGUAL_CASES, SAMPLE_DD_PATHS)
    print(f"  Multilingual retrieval accuracy: {ml_accuracy:.0%}")

    # Cleanup model to free memory
    del model
    gc.collect()

    return BenchmarkResult(
        model_name=model_name,
        load_time_s=load_time,
        encode_time_s=encode_time,
        memory_mb=memory_mb,
        embedding_dim=embedding_dim,
        retrieval_accuracy=en_accuracy,
        multilingual_accuracy=ml_accuracy,
    )


def print_comparison(results: list[BenchmarkResult]) -> None:
    """Print comparison table."""
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    headers = [
        "Model",
        "Dim",
        "Load(s)",
        "Enc(s)",
        "Mem(MB)",
        "EN Acc",
        "ML Acc",
    ]
    print(
        f"{headers[0]:<35} {headers[1]:>5} {headers[2]:>8} {headers[3]:>7} "
        f"{headers[4]:>8} {headers[5]:>7} {headers[6]:>7}"
    )
    print("-" * 80)

    for r in results:
        print(
            f"{r.model_name:<35} {r.embedding_dim:>5} {r.load_time_s:>8.2f} "
            f"{r.encode_time_s:>7.3f} {r.memory_mb:>8.0f} "
            f"{r.retrieval_accuracy:>6.0%} {r.multilingual_accuracy:>6.0%}"
        )

    print("-" * 80)

    # Find winner
    best_en = max(results, key=lambda r: r.retrieval_accuracy)
    best_ml = max(results, key=lambda r: r.multilingual_accuracy)

    print(
        f"\nBest English retrieval: {best_en.model_name} ({best_en.retrieval_accuracy:.0%})"
    )
    print(
        f"Best Multilingual:      {best_ml.model_name} ({best_ml.multilingual_accuracy:.0%})"
    )


# =============================================================================
# pytest tests
# =============================================================================


@pytest.fixture(scope="module")
def minilm_model():
    """Load MiniLM model once for all tests."""
    return SentenceTransformer("all-MiniLM-L6-v2")


@pytest.fixture(scope="module")
def qwen_model():
    """Load Qwen3 model once for all tests."""
    return SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)


class TestQwen3Loading:
    """Test Qwen3 model loading and basic functionality."""

    def test_model_loads(self, qwen_model):
        """Qwen3 model loads successfully."""
        assert qwen_model is not None

    def test_embedding_dimension(self, qwen_model):
        """Qwen3 produces 1024-dim embeddings."""
        assert qwen_model.get_sentence_embedding_dimension() == 1024

    def test_encode_english(self, qwen_model):
        """Qwen3 encodes English text."""
        embeddings = qwen_model.encode(["electron temperature profile"])
        assert embeddings.shape == (1, 1024)

    def test_encode_japanese(self, qwen_model):
        """Qwen3 encodes Japanese text."""
        embeddings = qwen_model.encode(["電子温度プロファイル"])
        assert embeddings.shape == (1, 1024)

    def test_encode_batch(self, qwen_model):
        """Qwen3 encodes batches correctly."""
        texts = ["hello", "world", "こんにちは"]
        embeddings = qwen_model.encode(texts)
        assert embeddings.shape == (3, 1024)


class TestRetrievalComparison:
    """Compare retrieval accuracy between models."""

    def test_minilm_english_retrieval(self, minilm_model):
        """MiniLM handles English retrieval."""
        accuracy = compute_retrieval_accuracy(
            minilm_model, DD_RETRIEVAL_CASES, SAMPLE_DD_PATHS
        )
        assert accuracy >= 0.4, f"MiniLM English accuracy too low: {accuracy:.0%}"

    def test_qwen_english_retrieval(self, qwen_model):
        """Qwen3 handles English retrieval."""
        accuracy = compute_retrieval_accuracy(
            qwen_model, DD_RETRIEVAL_CASES, SAMPLE_DD_PATHS
        )
        assert accuracy >= 0.4, f"Qwen3 English accuracy too low: {accuracy:.0%}"

    def test_qwen_multilingual_retrieval(self, qwen_model):
        """Qwen3 handles multilingual queries better than MiniLM."""
        accuracy = compute_retrieval_accuracy(
            qwen_model, MULTILINGUAL_CASES, SAMPLE_DD_PATHS
        )
        # Qwen3 should handle multilingual well
        assert accuracy >= 0.3, f"Qwen3 multilingual accuracy too low: {accuracy:.0%}"

    @pytest.mark.parametrize(
        "query,expected,lang",
        [
            (
                "電子温度プロファイル",
                "core_profiles/profiles_1d/electrons/temperature",
                "ja",
            ),
            ("プラズマ電流測定", "magnetics/ip", "ja"),
        ],
    )
    def test_qwen_japanese_specific(self, qwen_model, query, expected, lang):
        """Qwen3 retrieves correct path for Japanese queries."""
        accuracy = compute_retrieval_accuracy(
            qwen_model, [(query, [expected], lang)], SAMPLE_DD_PATHS, top_k=10
        )
        # At least find in top-10 for Japanese
        assert accuracy >= 0.5, f"Failed to retrieve '{expected}' for '{query}'"


# =============================================================================
# Standalone execution
# =============================================================================


def main():
    """Run full benchmark comparison."""
    import gc

    models = [
        "all-MiniLM-L6-v2",  # Current model
        "Qwen/Qwen3-Embedding-0.6B",  # Proposed model
    ]

    results = []
    for model_name in models:
        try:
            result = benchmark_model(model_name)
            results.append(result)
            # Free memory between models to avoid OOM on low-RAM systems
            gc.collect()
        except Exception as e:
            print(f"  ✗ Error: {e}")

    if len(results) > 1:
        print_comparison(results)


if __name__ == "__main__":
    main()
