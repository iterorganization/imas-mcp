"""Embedding encoder performance benchmarks.

No graph dependency — measures embedding encode latency in isolation.
Forces local (CPU) embedding for reproducible CI results.
"""

from __future__ import annotations

import os

os.environ["IMAS_CODEX_EMBEDDING_LOCATION"] = "local"

from imas_codex.embeddings.encoder import Encoder  # noqa: E402

SAMPLE_TEXTS = [
    "electron temperature profile",
    "magnetic field strength",
    "plasma current measurement",
    "ion density distribution",
    "safety factor q profile",
    "toroidal rotation velocity",
    "electron cyclotron emission",
    "neutral beam injection power",
    "edge localized mode frequency",
    "bootstrap current fraction",
]

LONG_TEXT = (
    "The electron temperature profile in a tokamak plasma is a critical "
    "measurement for understanding energy confinement and transport processes. "
    "Thomson scattering diagnostics provide spatially resolved measurements "
    "of electron temperature across the plasma cross-section, from the "
    "magnetic axis to the last closed flux surface. The temperature profile "
    "shape is determined by the balance between heating sources, such as "
    "neutral beam injection and electron cyclotron resonance heating, and "
    "transport losses driven by turbulence and neoclassical effects. In "
    "H-mode plasmas, a steep temperature gradient forms at the edge, "
    "creating a pedestal that significantly improves global energy "
    "confinement. Understanding the interplay between the pedestal height, "
    "the core temperature profile stiffness, and the transport mechanisms "
    "is essential for predicting fusion performance in future devices like "
    "ITER. The temperature profile also influences the current density "
    "distribution through its effect on plasma resistivity, which in turn "
    "affects MHD stability and the safety factor profile."
)


class EmbeddingBenchmarks:
    """Benchmark embedding encode performance (CPU-only)."""

    timeout = 300

    def setup(self):
        """Create encoder and warmup."""
        self.encoder = Encoder()
        self.encoder.embed_texts(["warmup query"])

    def time_encode_single_query(self):
        """Single query latency."""
        self.encoder.embed_texts(["electron temperature"])

    def time_encode_batch_10(self):
        """Batch encode (10 texts)."""
        self.encoder.embed_texts(SAMPLE_TEXTS)

    def time_encode_batch_100(self):
        """Batch encode (100 texts)."""
        self.encoder.embed_texts(SAMPLE_TEXTS * 10)

    def time_encode_long_text(self):
        """Long text latency."""
        self.encoder.embed_texts([LONG_TEXT])

    def time_model_load(self):
        """Cold start model load."""
        Encoder()

    def peakmem_encode_batch_100(self):
        """Batch memory footprint."""
        self.encoder.embed_texts(SAMPLE_TEXTS * 10)
