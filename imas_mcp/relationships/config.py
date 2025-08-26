"""
Configuration for relationship extraction and clustering.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RelationshipExtractionConfig:
    """Configuration for relationship extraction."""

    # Model configuration
    model_name: str = "all-MiniLM-L6-v2"
    device: str | None = None  # Auto-detect GPU/CPU

    # Clustering configuration
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

    # Cache and performance
    enable_cache: bool = True
    batch_size: int = 250
    normalize_embeddings: bool = True
    use_rich: bool = True

    # Paths and directories
    input_dir: Path = Path("imas_mcp/resources/schemas/detailed")
    output_file: Path = Path("imas_mcp/resources/schemas/relationships.json")
    cache_dir: Path | None = None

    def __post_init__(self):
        """Convert string paths to Path objects."""
        if isinstance(self.input_dir, str):
            self.input_dir = Path(self.input_dir)
        if isinstance(self.output_file, str):
            self.output_file = Path(self.output_file)
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)
