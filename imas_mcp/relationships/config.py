"""Configuration for relationship extraction and clustering."""

from dataclasses import dataclass, field
from pathlib import Path

from imas_mcp.embeddings.config import EncoderConfig


@dataclass
class RelationshipExtractionConfig:
    """Configuration for relationship extraction."""

    # Embedding configuration - single source of truth (required)
    encoder_config: EncoderConfig

    # Cross-IDS clustering configuration (optimized via Latin Hypercube)
    cross_ids_eps: float = (
        0.0751  # Optimized epsilon for cross-IDS clustering (LHC Round 2)
    )
    cross_ids_min_samples: int = 2  # Minimum samples for cross-IDS clusters

    # Intra-IDS clustering configuration (optimized via Latin Hypercube)
    intra_ids_eps: float = (
        0.0319  # Optimized epsilon for intra-IDS clustering (LHC Round 2)
    )
    intra_ids_min_samples: int = 2  # Optimized minimum samples for intra-IDS clusters

    # Legacy clustering configuration (for backward compatibility)
    eps: float = 0.25  # DBSCAN epsilon parameter
    min_samples: int = 3  # DBSCAN minimum samples
    similarity_threshold: float = 0.7  # Threshold for relationship creation

    # Path filtering configuration
    min_documentation_length: int = 30
    skip_patterns: list[str] = field(
        default_factory=lambda: [
            r".*/(name|index|description|identifier)$",
            r".*/time_slice/\d+/(global_quantities|profiles_\dd)",
            r".*_index$",
            r".*_name$",
            r".*_description$",
        ]
    )

    # Generic documentation terms to skip
    generic_docs: list[str] = field(
        default_factory=lambda: [
            "Generic data field",
            "Placeholder",
            "",
        ]
    )

    # Output configuration
    max_relationships_per_path: int = 3
    max_paths_per_cluster: int = 3
    max_cluster_size: int = 10

    # Display configuration
    use_rich: bool = True

    # Filtering configuration
    ids_set: set[str] | None = None  # Specific IDS names to include

    # Paths and directories
    input_dir: Path = Path("imas_mcp/resources/schemas/detailed")
    output_file: Path = Path("imas_mcp/resources/schemas/relationships.json")
    cache_dir: Path | None = None

    def __post_init__(self):
        """Convert string paths to Path objects and sync encoder config."""
        if isinstance(self.input_dir, str):
            self.input_dir = Path(self.input_dir)
        if isinstance(self.output_file, str):
            self.output_file = Path(self.output_file)
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)

        # Sync ids_set between relationship config and encoder config
        if self.ids_set is not None and self.encoder_config.ids_set is None:
            self.encoder_config.ids_set = self.ids_set
        elif self.encoder_config.ids_set is not None and self.ids_set is None:
            self.ids_set = self.encoder_config.ids_set

    def get_encoder_config(self) -> EncoderConfig:
        """Get the encoder configuration."""
        return self.encoder_config
