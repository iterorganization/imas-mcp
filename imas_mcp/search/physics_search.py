"""
Physics embedding system for semantic search over YAML domain data.

This module provides semantic search capabilities over physics domain characteristics,
phenomena, units, and concepts using sentence transformers. It integrates with the
existing semantic search infrastructure while being specialized for physics concepts.
"""

import hashlib
import logging
import pickle
import time
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from imas_mcp.core.data_model import PhysicsDomain
from imas_mcp.core.physics_accessors import DomainAccessor, UnitAccessor
from imas_mcp.core.physics_domains import DomainCharacteristics
from imas_mcp.models.physics_models import (
    EmbeddingDocument,
    SemanticResult,
)
from imas_mcp.units import unit_registry

logger = logging.getLogger(__name__)


@dataclass
class PhysicsEmbeddingCache:
    """Cache for physics concept embeddings."""

    embeddings: np.ndarray = field(default_factory=lambda: np.array([]))
    documents: list[EmbeddingDocument] = field(default_factory=list)
    concept_ids: list[str] = field(default_factory=list)
    model_name: str = ""
    cache_version: str = "1.0"
    created_at: float = 0.0

    @property
    def size(self) -> int:
        """Get number of cached embeddings."""
        return len(self.concept_ids)

    def is_valid(self) -> bool:
        """Check if cache is valid and consistent."""
        if self.size == 0:
            return True

        return (
            len(self.embeddings) == len(self.documents) == len(self.concept_ids)
            and self.embeddings.shape[0] == self.size
        )


class PhysicsSemanticSearch:
    """Semantic search over physics domain definitions using sentence transformers."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        enable_cache: bool = True,
        cache_dir: Path | None = None,
    ):
        self.model_name = model_name
        self.device = device
        self.enable_cache = enable_cache
        self.cache_dir = cache_dir or self._get_default_cache_dir()

        self._model: SentenceTransformer | None = None
        self._cache: PhysicsEmbeddingCache | None = None
        self._domain_accessor = DomainAccessor()

    def _get_default_cache_dir(self) -> Path:
        """Get default cache directory."""
        # Use the same embeddings directory as the main semantic search system
        resources_dir = Path(str(files("imas_mcp") / "resources"))
        embeddings_dir = resources_dir / "embeddings"
        embeddings_dir.mkdir(parents=True, exist_ok=True)

        return embeddings_dir

    def _get_cache_path(self) -> Path:
        """Get cache file path."""
        # Create cache filename with model name and content hash
        domain_data = self._domain_accessor.get_all_domains()
        content_hash = self._compute_content_hash(domain_data)
        cache_filename = f"physics_embeddings_{self.model_name}_{content_hash[:8]}.pkl"
        return self.cache_dir / cache_filename

    def _compute_content_hash(self, domain_names: set[PhysicsDomain]) -> str:
        """Compute hash of domain data content for cache invalidation."""
        content_str = ""
        for domain_name in sorted(domain_names, key=lambda d: d.value):
            domain_info = self._domain_accessor.get_domain_info(domain_name)
            if domain_info:
                content_str += f"{domain_name.value}:{domain_info.description}:"
                content_str += ":".join(sorted(domain_info.primary_phenomena))
                content_str += ":".join(sorted(domain_info.typical_units))

        return hashlib.sha256(content_str.encode()).hexdigest()

    def _load_cache(self) -> bool:
        """Load embeddings cache if available and valid."""
        if not self.enable_cache:
            return False

        cache_path = self._get_cache_path()
        if not cache_path.exists():
            return False

        try:
            with open(cache_path, "rb") as f:
                self._cache = pickle.load(f)

            if self._cache is None or not self._cache.is_valid():
                logger.warning("Invalid physics embeddings cache, will rebuild")
                self._cache = None
                return False

            if self._cache.model_name != self.model_name:
                logger.info(
                    f"Model changed from {self._cache.model_name} to {self.model_name}, will rebuild"
                )
                self._cache = None
                return False

            logger.info(
                f"Loaded physics embeddings cache with {self._cache.size} concepts"
            )
            return True

        except Exception as e:
            logger.warning(f"Failed to load physics embeddings cache: {e}")
            self._cache = None
            return False

    def _save_cache(self) -> None:
        """Save embeddings cache."""
        if not self.enable_cache or self._cache is None:
            return

        cache_path = self._get_cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(self._cache, f)
            logger.info(f"Saved physics embeddings cache to {cache_path}")
        except Exception as e:
            logger.error(f"Failed to save physics embeddings cache: {e}")

    def _load_model(self) -> None:
        """Load sentence transformer model with optimized loading pattern."""
        if self._model is not None:
            return

        logger.info(f"Loading sentence transformer model: {self.model_name}")

        # Get embeddings directory for model cache (same pattern as DD semantic search)
        resources_dir = Path(str(files("imas_mcp") / "resources"))
        embeddings_dir = resources_dir / "embeddings"
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        cache_folder = str(embeddings_dir / "models")

        # Try to load with local_files_only first for speed (like DD semantic search)
        try:
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=cache_folder,
                local_files_only=True,  # Prevent internet downloads
            )
            logger.info(f"Loaded model {self.model_name} from cache")
        except Exception:
            # If local loading fails, try downloading
            logger.info(f"Model not in cache, downloading {self.model_name}...")
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=cache_folder,
                local_files_only=False,  # Allow downloads
            )
            logger.info(f"Downloaded and loaded model {self.model_name}")

    def _create_physics_documents(self) -> list[EmbeddingDocument]:
        """Create embedding documents from physics domain data."""
        documents = []
        domain_names = self._domain_accessor.get_all_domains()

        for domain_name in domain_names:
            domain_info = self._domain_accessor.get_domain_info(domain_name)
            if not domain_info:
                continue

            # Create domain document
            domain_content = self._create_domain_content(domain_name.value, domain_info)
            domain_doc = EmbeddingDocument(
                concept_id=f"domain:{domain_name.value}",
                concept_type="domain",
                domain_name=domain_name.value,
                title=f"{domain_name.value.replace('_', ' ').title()} Domain",
                description=domain_info.description,
                content=domain_content,
                metadata={
                    "complexity_level": domain_info.complexity_level.value,
                    "related_domains": domain_info.related_domains,
                    "typical_units": domain_info.typical_units,
                },
            )
            documents.append(domain_doc)

            # Create phenomenon documents
            for i, phenomenon in enumerate(domain_info.primary_phenomena):
                phenomenon_content = self._create_phenomenon_content(
                    domain_name.value, phenomenon, domain_info
                )
                phenomenon_doc = EmbeddingDocument(
                    concept_id=f"phenomenon:{domain_name.value}:{i}",
                    concept_type="phenomenon",
                    domain_name=domain_name.value,
                    title=phenomenon.replace("_", " ").title(),
                    description=f"{phenomenon} in {domain_name.value} physics",
                    content=phenomenon_content,
                    metadata={
                        "domain_description": domain_info.description,
                        "related_units": domain_info.typical_units,
                        "measurement_methods": domain_info.measurement_methods,
                    },
                )
                documents.append(phenomenon_doc)

            # Create measurement method documents
            for i, method in enumerate(domain_info.measurement_methods):
                method_content = self._create_method_content(
                    domain_name.value, method, domain_info
                )
                method_doc = EmbeddingDocument(
                    concept_id=f"method:{domain_name.value}:{i}",
                    concept_type="measurement_method",
                    domain_name=domain_name.value,
                    title=method.replace("_", " ").title(),
                    description=f"{method} measurement in {domain_name.value} physics",
                    content=method_content,
                    metadata={
                        "domain_description": domain_info.description,
                        "phenomena": domain_info.primary_phenomena,
                        "typical_units": domain_info.typical_units,
                    },
                )
                documents.append(method_doc)

        # Create unit documents with Pint-generated full names
        unit_accessor = UnitAccessor()
        all_unit_contexts = unit_accessor.get_all_unit_contexts()

        for unit_symbol, context_description in all_unit_contexts.items():
            # Get additional unit information
            category = unit_accessor.get_category_for_unit(unit_symbol)
            physics_domains = unit_accessor.get_domains_for_unit(unit_symbol)

            # Use Pint to get full unit name
            full_unit_name = self._get_unit_full_name(unit_symbol)

            unit_content = self._create_unit_content(
                unit_symbol,
                context_description,
                category,
                [
                    domain.value for domain in physics_domains
                ],  # Convert PhysicsDomain to string
                full_unit_name,
            )

            unit_doc = EmbeddingDocument(
                concept_id=f"unit:{unit_symbol}",
                concept_type="unit",
                domain_name="units",  # Special domain for units
                title=f"{unit_symbol} Unit",
                description=context_description,
                content=unit_content,
                metadata={
                    "symbol": unit_symbol,
                    "full_name": full_unit_name,
                    "category": category,
                    "physics_domains": [
                        domain.value for domain in physics_domains
                    ],  # Convert PhysicsDomain to string
                },
            )
            documents.append(unit_doc)

        return documents

    def _create_domain_content(
        self, domain_name: str, domain: DomainCharacteristics
    ) -> str:
        """Create rich content text for domain embedding."""
        content_parts = [
            f"Physics domain: {domain_name.replace('_', ' ')}",
            f"Description: {domain.description}",
            f"Primary phenomena: {', '.join(domain.primary_phenomena)}",
            f"Typical units: {', '.join(domain.typical_units)}",
            f"Measurement methods: {', '.join(domain.measurement_methods)}",
            f"Complexity level: {domain.complexity_level}",
            f"Related domains: {', '.join(domain.related_domains)}",
        ]
        return " | ".join(content_parts)

    def _create_phenomenon_content(
        self, domain_name: str, phenomenon: str, domain: DomainCharacteristics
    ) -> str:
        """Create rich content text for phenomenon embedding."""
        content_parts = [
            f"Physics phenomenon: {phenomenon.replace('_', ' ')}",
            f"Domain: {domain_name.replace('_', ' ')}",
            f"Context: {domain.description}",
            f"Related phenomena: {', '.join([p for p in domain.primary_phenomena if p != phenomenon])}",
            f"Typical units: {', '.join(domain.typical_units)}",
            f"Measurement methods: {', '.join(domain.measurement_methods)}",
        ]
        return " | ".join(content_parts)

    def _create_method_content(
        self, domain_name: str, method: str, domain: DomainCharacteristics
    ) -> str:
        """Create rich content text for measurement method embedding."""
        content_parts = [
            f"Measurement method: {method.replace('_', ' ')}",
            f"Domain: {domain_name.replace('_', ' ')}",
            f"Measures: {', '.join(domain.primary_phenomena)}",
            f"Context: {domain.description}",
            f"Typical units: {', '.join(domain.typical_units)}",
            f"Related methods: {', '.join([m for m in domain.measurement_methods if m != method])}",
        ]
        return " | ".join(content_parts)

    def _get_unit_full_name(self, unit_symbol: str) -> str:
        """Get the full unit name using Pint's formatting."""
        try:
            # Create a Pint quantity with the unit
            unit_obj = unit_registry.Unit(unit_symbol)

            # Use custom 'U' formatter to get the full unit name
            full_name = f"{unit_obj:U}"

            # If the full name is the same as symbol, try alternative approaches
            if full_name == unit_symbol:
                # Try getting unit definition string
                unit_def = str(unit_obj)
                if unit_def != unit_symbol:
                    full_name = unit_def

            return full_name

        except Exception as e:
            logger.debug(f"Could not get full name for unit '{unit_symbol}': {e}")
            return unit_symbol

    def _create_unit_content(
        self,
        unit_symbol: str,
        context_description: str,
        category: str | None,
        physics_domains: list[str],
        full_unit_name: str,
    ) -> str:
        """Create rich content text for unit embedding."""
        content_parts = [
            f"Unit: {unit_symbol}",
            f"Full name: {full_unit_name}",
            f"Context: {context_description}",
        ]

        if category:
            content_parts.append(f"Category: {category}")

        if physics_domains:
            content_parts.append(f"Physics domains: {', '.join(physics_domains)}")

        return " | ".join(content_parts)

    def build_embeddings(self, force_rebuild: bool = False) -> None:
        """Build or load embeddings for physics concepts."""
        if not force_rebuild and self._load_cache():
            return

        logger.info("Building physics concept embeddings...")
        self._load_model()

        if self._model is None:
            raise RuntimeError("Failed to load sentence transformer model")

        # Create documents
        documents = self._create_physics_documents()

        # Generate embeddings
        content_texts = [doc.content for doc in documents]
        embeddings = self._model.encode(
            content_texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        # Create cache
        self._cache = PhysicsEmbeddingCache(
            embeddings=embeddings,
            documents=documents,
            concept_ids=[doc.concept_id for doc in documents],
            model_name=self.model_name,
            created_at=time.time(),
        )

        self._save_cache()
        logger.info(f"Built embeddings for {len(documents)} physics concepts")

    def search(
        self,
        query: str,
        max_results: int = 10,
        min_similarity: float = 0.1,
        concept_types: list[str] | None = None,
        domains: list[str] | None = None,
    ) -> list[SemanticResult]:
        """
        Search physics concepts using semantic similarity.

        Args:
            query: Search query
            max_results: Maximum number of results to return
            min_similarity: Minimum similarity score threshold
            concept_types: Filter by concept types (domain, phenomenon, measurement_method)
            domains: Filter by domain names

        Returns:
            List of semantic search results sorted by similarity score
        """
        if self._cache is None:
            self.build_embeddings()

        if self._cache is None or self._cache.size == 0:
            return []

        self._load_model()

        if self._model is None:
            raise RuntimeError("Failed to load sentence transformer model")

        # Generate query embedding
        query_embedding = self._model.encode([query], normalize_embeddings=True)[0]

        # Calculate similarities
        similarities = np.dot(self._cache.embeddings, query_embedding)

        # Get top results above threshold
        valid_indices = np.where(similarities >= min_similarity)[0]
        if len(valid_indices) == 0:
            return []

        # Sort by similarity
        sorted_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1]]

        results = []
        for rank, idx in enumerate(sorted_indices[:max_results]):
            document = self._cache.documents[idx]

            # Apply filters
            if concept_types and document.concept_type not in concept_types:
                continue
            if domains and document.domain_name not in domains:
                continue

            result = SemanticResult(
                document=document, similarity_score=float(similarities[idx]), rank=rank
            )
            results.append(result)

        return results[:max_results]

    def find_similar_concepts(
        self, concept_id: str, max_results: int = 5, min_similarity: float = 0.3
    ) -> list[SemanticResult]:
        """Find concepts similar to a given concept ID."""
        if self._cache is None:
            self.build_embeddings()

        if self._cache is None:
            return []

        try:
            idx = self._cache.concept_ids.index(concept_id)
        except ValueError:
            return []

        concept_embedding = self._cache.embeddings[idx]
        similarities = np.dot(self._cache.embeddings, concept_embedding)

        # Exclude the concept itself and filter by threshold
        valid_indices = np.where(
            (similarities >= min_similarity) & (np.arange(len(similarities)) != idx)
        )[0]

        if len(valid_indices) == 0:
            return []

        # Sort by similarity
        sorted_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1]]

        results = []
        for rank, idx in enumerate(sorted_indices[:max_results]):
            result = SemanticResult(
                document=self._cache.documents[idx],
                similarity_score=float(similarities[idx]),
                rank=rank,
            )
            results.append(result)

        return results

    def get_concept_by_id(self, concept_id: str) -> EmbeddingDocument | None:
        """Get a physics concept document by ID."""
        if self._cache is None:
            self.build_embeddings()

        if self._cache is None:
            return None

        try:
            idx = self._cache.concept_ids.index(concept_id)
            return self._cache.documents[idx]
        except ValueError:
            return None


# Global instance for easy access
_physics_search = None


def get_physics_search() -> PhysicsSemanticSearch:
    """Get global physics semantic search instance."""
    global _physics_search
    if _physics_search is None:
        _physics_search = PhysicsSemanticSearch()
    return _physics_search


def search_physics_concepts(
    query: str, max_results: int = 10, **kwargs
) -> list[SemanticResult]:
    """Search physics concepts using semantic similarity."""
    physics_search = get_physics_search()
    return physics_search.search(query, max_results=max_results, **kwargs)


def build_physics_embeddings(force_rebuild: bool = False) -> None:
    """Build physics concept embeddings."""
    physics_search = get_physics_search()
    physics_search.build_embeddings(force_rebuild=force_rebuild)


if __name__ == "__main__":
    # Test the physics embedding system
    import time

    print("=== Physics Embeddings Test ===")

    # Build embeddings
    start_time = time.time()
    build_physics_embeddings(force_rebuild=True)
    build_time = time.time() - start_time
    print(f"Built embeddings in {build_time:.2f} seconds")

    # Test searches
    test_queries = [
        "electron temperature",
        "magnetic field measurement",
        "plasma instabilities",
        "transport phenomena",
        "heating systems",
    ]

    for query in test_queries:
        print(f"\nSearch: '{query}'")
        results = search_physics_concepts(query, max_results=3)

        for i, result in enumerate(results):
            print(
                f"  {i + 1}. {result.document.title} ({result.document.concept_type})"
            )
            print(f"     Domain: {result.document.domain_name}")
            print(f"     Similarity: {result.similarity_score:.3f}")
            print(f"     Description: {result.document.description[:100]}...")
