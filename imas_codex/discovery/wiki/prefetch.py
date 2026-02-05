"""
Wiki page prefetch and summarization.

This module handles:
1. Batch HTTP fetching of page content
2. Text extraction from HTML
3. LLM summarization of page content
4. Storage of preview_text and preview_summary

Usage:
    from imas_codex.discovery.wiki.prefetch import prefetch_pages

    stats = await prefetch_pages("tcv", batch_size=50, max_pages=100)
    print(f"Fetched: {stats['fetched']}, Failed: {stats['failed']}")
"""

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime

import httpx
from bs4 import BeautifulSoup
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

from imas_codex.agentic.llm import get_llm
from imas_codex.graph import GraphClient

logger = logging.getLogger(__name__)
console = Console()


async def fetch_page_content(
    url: str,
    timeout: float = 30.0,
    auth_handler: Callable | None = None,
    verify_ssl: bool = False,
) -> tuple[str | None, str | None]:
    """
    Fetch page content via HTTP.

    Args:
        url: Page URL to fetch
        timeout: Request timeout in seconds
        auth_handler: Optional authentication handler
        verify_ssl: Whether to verify SSL certificates (default False for facility wikis)

    Returns:
        (content_text, error_message)
        If successful: (text, None)
        If failed: (None, error_message)
    """
    try:
        async with httpx.AsyncClient(
            timeout=timeout, follow_redirects=True, verify=verify_ssl
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.text, None
    except httpx.TimeoutException:
        return None, "Timeout"
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            return None, "Auth required"
        elif e.response.status_code == 404:
            return None, "Not found"
        else:
            return None, f"HTTP {e.response.status_code}"
    except Exception as e:
        return None, str(e)


def extract_text_from_html(html: str, max_chars: int = 2000) -> str:
    """
    Extract clean text from HTML content.

    - Removes scripts, styles, navigation
    - Preserves paragraph structure
    - Truncates to max_chars

    Args:
        html: Raw HTML content
        max_chars: Maximum characters to extract

    Returns:
        Clean text preview
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove unwanted elements
    for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
        element.decompose()

    # Get text
    text = soup.get_text(separator=" ", strip=True)

    # Clean up whitespace
    text = " ".join(text.split())

    # Truncate
    if len(text) > max_chars:
        text = text[:max_chars]

    return text


async def summarize_pages_batch(
    pages: list[dict],
    model: str = "anthropic/claude-3-5-haiku",
    batch_size: int = 20,
) -> list[str]:
    """
    Batch summarize page previews using LLM.

    Args:
        pages: List of {id, title, preview_text}
        model: LLM model to use (Haiku for cost efficiency)
        batch_size: Pages per LLM call

    Returns:
        List of summaries (same order as input)
    """
    llm = get_llm(model=model, temperature=0.3)
    summaries = []

    for i in range(0, len(pages), batch_size):
        batch = pages[i : i + batch_size]

        # Build prompt for batch
        prompt_parts = [
            "Summarize each wiki page in 2-3 sentences (max 300 chars each).\n"
        ]
        prompt_parts.append(
            "Focus on: What data, documentation, or information does this page provide?\n"
        )
        prompt_parts.append(
            "If this is a data source or database, mention what kind of data.\n"
        )
        prompt_parts.append("If this is documentation, mention what it documents.\n\n")

        for idx, page in enumerate(batch):
            prompt_parts.append(f"Page {idx + 1}:\n")
            prompt_parts.append(f"Title: {page['title']}\n")
            prompt_parts.append(f"Content: {page['preview_text'][:1500]}\n\n")

        prompt_parts.append("Output format: One summary per line, numbered 1-N.\n")
        prompt = "".join(prompt_parts)

        # Get summaries
        response = await llm.acomplete(prompt)
        response_text = str(response)

        # Parse numbered summaries
        lines = response_text.strip().split("\n")
        batch_summaries = []
        for line in lines:
            # Remove numbering like "1. " or "1) "
            clean_line = line.strip()
            if clean_line and (clean_line[0].isdigit() or clean_line.startswith("-")):
                # Strip leading number and punctuation
                parts = clean_line.split(". ", 1)
                if len(parts) > 1:
                    clean_line = parts[1]
                else:
                    parts = clean_line.split(") ", 1)
                    if len(parts) > 1:
                        clean_line = parts[1]

            if clean_line:
                batch_summaries.append(clean_line[:300])  # Enforce max length

        # Pad if needed (in case LLM didn't return all summaries)
        while len(batch_summaries) < len(batch):
            batch_summaries.append("[Summary generation failed]")

        summaries.extend(batch_summaries[: len(batch)])

    return summaries


async def prefetch_pages(
    facility_id: str,
    batch_size: int = 50,
    max_pages: int | None = None,
    include_scored: bool = False,
) -> dict:
    """
    Prefetch and summarize pages for a facility.

    Workflow:
    1. Query pages needing prefetch (preview_text IS NULL)
    2. Batch HTTP fetch with parallel requests
    3. Extract text from HTML
    4. Batch LLM summarize
    5. Store preview_text + preview_summary to graph

    Args:
        facility_id: Target facility
        batch_size: Pages per batch
        max_pages: Maximum pages to process (None = all)
        include_scored: Also prefetch already-scored pages

    Returns:
        Stats dict: {fetched, summarized, failed, skipped}
    """
    gc = GraphClient()

    # Query pages needing prefetch
    query = """
    MATCH (wp:WikiPage {facility_id: $facility_id})
    WHERE wp.preview_text IS NULL
    """
    if not include_scored:
        query += " AND wp.status = 'scored'"

    query += """
    RETURN wp.id AS id, wp.url AS url, wp.title AS title
    ORDER BY wp.in_degree DESC
    """

    if max_pages:
        query += f" LIMIT {max_pages}"

    pages = gc.query(query, facility_id=facility_id)

    if not pages:
        console.print(f"[yellow]No pages need prefetch for {facility_id}[/yellow]")
        return {"fetched": 0, "summarized": 0, "failed": 0, "skipped": 0}

    console.print(f"[cyan]Prefetching {len(pages)} pages for {facility_id}...[/cyan]")

    stats = {"fetched": 0, "summarized": 0, "failed": 0, "skipped": 0}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching pages...", total=len(pages))

        # Process in batches
        for i in range(0, len(pages), batch_size):
            batch = pages[i : i + batch_size]

            # Fetch pages in parallel
            fetch_tasks = [fetch_page_content(page["url"]) for page in batch]
            fetch_results = await asyncio.gather(*fetch_tasks)

            # Extract text and prepare for summarization
            pages_to_summarize = []
            for page, (html, error) in zip(batch, fetch_results, strict=False):
                if html:
                    preview_text = extract_text_from_html(html)
                    pages_to_summarize.append(
                        {
                            "id": page["id"],
                            "title": page["title"],
                            "preview_text": preview_text,
                            "error": None,
                        }
                    )
                    stats["fetched"] += 1
                else:
                    pages_to_summarize.append(
                        {
                            "id": page["id"],
                            "title": page["title"],
                            "preview_text": None,
                            "error": error,
                        }
                    )
                    stats["failed"] += 1

            # Summarize successful fetches
            summaries_batch = []
            if pages_to_summarize:
                valid_pages = [p for p in pages_to_summarize if p["preview_text"]]
                if valid_pages:
                    summaries_batch = await summarize_pages_batch(valid_pages)
                    stats["summarized"] += len(summaries_batch)

            # Update graph
            summary_idx = 0
            for page_data in pages_to_summarize:
                if page_data["preview_text"]:
                    summary = (
                        summaries_batch[summary_idx]
                        if summary_idx < len(summaries_batch)
                        else "[Summary failed]"
                    )
                    summary_idx += 1

                    gc.query(
                        """
                        MATCH (wp:WikiPage {id: $id})
                        SET wp.preview_text = $preview_text,
                            wp.preview_summary = $summary,
                            wp.preview_fetched_at = datetime($timestamp)
                        """,
                        id=page_data["id"],
                        preview_text=page_data["preview_text"],
                        summary=summary,
                        timestamp=datetime.now(UTC).isoformat(),
                    )
                else:
                    gc.query(
                        """
                        MATCH (wp:WikiPage {id: $id})
                        SET wp.preview_fetch_error = $error,
                            wp.preview_fetched_at = datetime($timestamp)
                        """,
                        id=page_data["id"],
                        error=page_data["error"],
                        timestamp=datetime.now(UTC).isoformat(),
                    )

            progress.update(task, advance=len(batch))

    console.print(
        f"[green]✓ Prefetch complete: {stats['fetched']} fetched, {stats['summarized']} summarized, {stats['failed']} failed[/green]"
    )
    return stats
