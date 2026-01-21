"""Scout module for windowed facility exploration with context management.

This module implements a "moving frontier" approach to facility discovery:
- Windowed execution with summarization between windows
- Dead-end detection to skip irrelevant paths (git repos, site-packages, etc.)
- Frontier tracking to know what's explored vs remaining
- Graph persistence after each discovery step

Uses existing node types:
- FacilityPath: Discovered paths with status and interest_score
- SourceFile: Files queued for ingestion
- AgentRun: Session tracking with checkpoints
"""

from .frontier import FrontierManager, FrontierStats
from .session import ScoutConfig, ScoutSession
from .summarizer import WindowSummarizer

__all__ = [
    "FrontierManager",
    "FrontierStats",
    "ScoutConfig",
    "ScoutSession",
    "WindowSummarizer",
]
