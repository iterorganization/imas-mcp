"""
Knowledge persistence for exploration agents.

This module handles loading and saving accumulated knowledge from
facility explorations. Knowledge is persisted to the facility YAML
config files so future explorations can benefit from past discoveries.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from imas_codex.discovery.config import get_facilities_dir

logger = logging.getLogger(__name__)


@dataclass
class FacilityKnowledge:
    """
    Accumulated knowledge about a facility from explorations.

    This is loaded from and saved to the `knowledge:` section of
    facility YAML config files.
    """

    tools: list[str] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)
    python: list[str] = field(default_factory=list)
    data: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        """Check if any knowledge has been accumulated."""
        return not (self.tools or self.paths or self.python or self.data)

    def to_dict(self) -> dict[str, list[str]]:
        """Convert to dictionary for YAML serialization."""
        return {
            "tools": self.tools,
            "paths": self.paths,
            "python": self.python,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FacilityKnowledge":
        """Create from dictionary (loaded from YAML)."""
        return cls(
            tools=data.get("tools", []) or [],
            paths=data.get("paths", []) or [],
            python=data.get("python", []) or [],
            data=data.get("data", []) or [],
        )


def _get_facility_config_path(facility: str) -> Path:
    """Get the path to a facility's config file."""
    return get_facilities_dir() / f"{facility}.yaml"


def load_knowledge(facility: str) -> FacilityKnowledge:
    """
    Load accumulated knowledge for a facility.

    Args:
        facility: Facility identifier (e.g., 'epfl')

    Returns:
        FacilityKnowledge with any previously persisted knowledge
    """
    config_path = _get_facility_config_path(facility)

    if not config_path.exists():
        logger.warning(f"No config file for facility: {facility}")
        return FacilityKnowledge()

    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    knowledge_data = config.get("knowledge", {}) or {}
    return FacilityKnowledge.from_dict(knowledge_data)


def save_knowledge(facility: str, knowledge: FacilityKnowledge) -> None:
    """
    Save accumulated knowledge to a facility's config.

    Args:
        facility: Facility identifier
        knowledge: Knowledge to persist
    """
    config_path = _get_facility_config_path(facility)

    if not config_path.exists():
        logger.error(f"Cannot save knowledge: no config file for {facility}")
        return

    # Load existing config
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    # Update knowledge section
    config["knowledge"] = knowledge.to_dict()

    # Write back with preserved formatting
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved knowledge to {config_path}")


def _categorize_learning(learning: str) -> tuple[str, str]:
    """
    Categorize a learning string into a knowledge category.

    Args:
        learning: A learning string from agent exploration

    Returns:
        Tuple of (category, cleaned_learning)
    """
    learning_lower = learning.lower()

    # Tool-related learnings
    tool_keywords = [
        "rg ",
        "ripgrep",
        "grep",
        "tree",
        "find",
        "ag ",
        "ack",
        "available",
        "not available",
        "installed",
        "not installed",
        "command",
        "tool",
    ]
    if any(kw in learning_lower for kw in tool_keywords):
        return "tools", learning

    # Python-related learnings
    python_keywords = [
        "python",
        "pip",
        "conda",
        "module",
        "import",
        "package",
        "virtualenv",
        "venv",
    ]
    if any(kw in learning_lower for kw in python_keywords):
        return "python", learning

    # Data-related learnings
    data_keywords = [
        "hdf5",
        "h5",
        "netcdf",
        "mdsplus",
        "shot",
        "data",
        "format",
        "storage",
    ]
    if any(kw in learning_lower for kw in data_keywords):
        return "data", learning

    # Path-related learnings (default for path mentions)
    if "/" in learning or "directory" in learning_lower or "folder" in learning_lower:
        return "paths", learning

    # Default to paths for general learnings
    return "paths", learning


def persist_learnings(facility: str, learnings: list[str]) -> FacilityKnowledge:
    """
    Persist new learnings from an exploration run.

    Learnings are automatically categorized and deduplicated before
    being added to the facility's accumulated knowledge.

    Args:
        facility: Facility identifier
        learnings: List of learning strings from agent

    Returns:
        Updated FacilityKnowledge
    """
    if not learnings:
        return load_knowledge(facility)

    # Load existing knowledge
    knowledge = load_knowledge(facility)

    # Categorize and add new learnings
    for learning in learnings:
        learning = learning.strip()
        if not learning:
            continue

        category, cleaned = _categorize_learning(learning)

        # Get the appropriate list
        category_list = getattr(knowledge, category)

        # Add if not duplicate (case-insensitive check)
        existing_lower = [item.lower() for item in category_list]
        if cleaned.lower() not in existing_lower:
            category_list.append(cleaned)
            logger.info(f"Added {category} knowledge: {cleaned[:50]}...")

    # Save updated knowledge
    save_knowledge(facility, knowledge)

    return knowledge


def merge_knowledge(
    existing: FacilityKnowledge,
    new_learnings: list[str],
) -> FacilityKnowledge:
    """
    Merge new learnings into existing knowledge without persisting.

    Useful for in-memory updates during an exploration run.

    Args:
        existing: Current knowledge state
        new_learnings: New learning strings to merge

    Returns:
        Updated FacilityKnowledge (new instance)
    """
    # Create a copy
    merged = FacilityKnowledge(
        tools=list(existing.tools),
        paths=list(existing.paths),
        python=list(existing.python),
        data=list(existing.data),
    )

    for learning in new_learnings:
        learning = learning.strip()
        if not learning:
            continue

        category, cleaned = _categorize_learning(learning)
        category_list = getattr(merged, category)

        existing_lower = [item.lower() for item in category_list]
        if cleaned.lower() not in existing_lower:
            category_list.append(cleaned)

    return merged
