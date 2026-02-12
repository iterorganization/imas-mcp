"""Window summarization for scout sessions.

Uses a cheap model (Haiku) to compress window discoveries into
a concise summary that can be carried forward to the next window.
This keeps context manageable across long exploration sessions.
"""

import logging
from typing import Any

from imas_codex.settings import get_model

from ..llm import get_llm

logger = logging.getLogger(__name__)

SUMMARIZATION_PROMPT = """You are summarizing the discoveries from a scout exploration window.

The goal is to create a concise summary that captures:
1. Key paths discovered and their significance
2. Files queued for ingestion (with patterns matched)
3. Dead-ends skipped (and why)
4. Any patterns or insights about the facility structure

This summary will be used as context for the next exploration window,
so focus on information that helps guide further exploration.

Keep the summary under 500 words. Use bullet points for clarity.

## Window Data

Facility: {facility}
Window number: {window_num}
Steps taken: {steps}
Discoveries: {num_discoveries}
Files queued: {num_files}
Paths skipped: {num_skipped}

### Discovered Paths
{discoveries}

### Queued Files
{queued_files}

### Skipped Paths
{skipped_paths}

---

Provide a concise summary of this window's exploration:"""


class WindowSummarizer:
    """Summarizes window discoveries using a cheap LLM.

    Uses the summarization model (typically Haiku) to compress
    window discoveries into a concise summary for the next window.
    """

    def __init__(self, model: str | None = None) -> None:
        """Initialize the summarizer.

        Args:
            model: Override model name. If None, uses summarization preset.
        """
        self.model = model or get_model("compaction")
        self._llm = None

    def _get_llm(self) -> Any:
        """Lazy-load the LLM."""
        if self._llm is None:
            self._llm = get_llm(self.model)
        return self._llm

    def summarize_window(
        self,
        facility: str,
        window_num: int,
        steps: int,
        discoveries: list[dict],
        queued_files: list[str],
        skipped_paths: list[str],
    ) -> str:
        """Generate a summary of window discoveries.

        Args:
            facility: Facility ID
            window_num: Window number in session
            steps: Number of agent steps in window
            discoveries: List of discovery dicts from the window
            queued_files: List of file paths queued for ingestion
            skipped_paths: List of paths skipped as dead-ends

        Returns:
            Concise text summary suitable for next window context
        """
        # Format discoveries for prompt
        discovery_text = self._format_discoveries(discoveries)
        files_text = self._format_files(queued_files)
        skipped_text = self._format_skipped(skipped_paths)

        prompt = SUMMARIZATION_PROMPT.format(
            facility=facility,
            window_num=window_num,
            steps=steps,
            num_discoveries=len(discoveries),
            num_files=len(queued_files),
            num_skipped=len(skipped_paths),
            discoveries=discovery_text or "None",
            queued_files=files_text or "None",
            skipped_paths=skipped_text or "None",
        )

        try:
            llm = self._get_llm()
            response = llm.complete(prompt)
            summary = response.text.strip()
            logger.debug(
                "Generated window %d summary (%d chars)",
                window_num,
                len(summary),
            )
            return summary
        except Exception as e:
            logger.exception("Failed to generate window summary: %s", e)
            # Fallback to a basic summary
            return self._fallback_summary(
                facility,
                window_num,
                steps,
                len(discoveries),
                len(queued_files),
                len(skipped_paths),
            )

    def _format_discoveries(self, discoveries: list[dict], max_items: int = 20) -> str:
        """Format discoveries for the prompt."""
        if not discoveries:
            return ""

        lines = []
        for d in discoveries[:max_items]:
            path = d.get("path", "unknown")
            score = d.get("interest_score", 0.5)
            status = d.get("status", "discovered")
            lines.append(f"- {path} (score={score:.2f}, status={status})")

        if len(discoveries) > max_items:
            lines.append(f"- ... and {len(discoveries) - max_items} more")

        return "\n".join(lines)

    def _format_files(self, files: list[str], max_items: int = 20) -> str:
        """Format queued files for the prompt."""
        if not files:
            return ""

        lines = [f"- {f}" for f in files[:max_items]]
        if len(files) > max_items:
            lines.append(f"- ... and {len(files) - max_items} more")

        return "\n".join(lines)

    def _format_skipped(self, paths: list[str], max_items: int = 15) -> str:
        """Format skipped paths for the prompt."""
        if not paths:
            return ""

        lines = [f"- {p}" for p in paths[:max_items]]
        if len(paths) > max_items:
            lines.append(f"- ... and {len(paths) - max_items} more")

        return "\n".join(lines)

    def _fallback_summary(
        self,
        facility: str,
        window_num: int,
        steps: int,
        num_discoveries: int,
        num_files: int,
        num_skipped: int,
    ) -> str:
        """Generate a basic summary when LLM fails."""
        return f"""Window {window_num} Summary (facility: {facility})
- Steps taken: {steps}
- Paths discovered: {num_discoveries}
- Files queued: {num_files}
- Dead-ends skipped: {num_skipped}

Note: Detailed summary unavailable due to LLM error."""


def summarize_session(
    facility: str,
    windows: list[dict],
) -> str:
    """Generate an overall session summary from window summaries.

    Args:
        facility: Facility ID
        windows: List of window state dicts

    Returns:
        Overall session summary
    """
    summarizer = WindowSummarizer()

    total_steps = sum(w.get("steps_in_window", 0) for w in windows)
    total_discoveries = sum(len(w.get("discoveries", [])) for w in windows)
    total_files = sum(len(w.get("queued_files", [])) for w in windows)
    total_skipped = sum(len(w.get("skipped_paths", [])) for w in windows)

    window_summaries = [w.get("summary", "") for w in windows if w.get("summary")]

    prompt = f"""Synthesize the following window summaries into an overall session summary.

Facility: {facility}
Total windows: {len(windows)}
Total steps: {total_steps}
Total discoveries: {total_discoveries}
Total files queued: {total_files}
Total dead-ends: {total_skipped}

## Window Summaries

{chr(10).join(window_summaries) if window_summaries else "No window summaries available."}

---

Provide a comprehensive session summary focusing on:
1. Key areas explored and their significance
2. Notable patterns or code discovered
3. Recommended next steps for exploration
4. Any gaps or unexplored high-priority areas

Keep under 800 words."""

    try:
        llm = summarizer._get_llm()
        response = llm.complete(prompt)
        return response.text.strip()
    except Exception as e:
        logger.exception("Failed to generate session summary: %s", e)
        return f"""Session Summary for {facility}
- Windows completed: {len(windows)}
- Total steps: {total_steps}
- Discoveries: {total_discoveries}
- Files queued: {total_files}
- Dead-ends skipped: {total_skipped}

Note: Detailed summary unavailable due to LLM error."""
