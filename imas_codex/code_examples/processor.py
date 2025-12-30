"""Async code embedding processor.

Provides background processing for the embedding queue, allowing
embeddings to be generated asynchronously without blocking
LLM agent operations.
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio

from imas_codex.graph import GraphClient

from .ingester import CodeExampleIngester, get_embed_model
from .queue import EmbeddingQueue

logger = logging.getLogger(__name__)

# Progress callback type for async processor
AsyncProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass
class AsyncEmbeddingProcessor:
    """Process embedding queue asynchronously in the background.

    This processor can run independently of agent operations,
    processing files from the staging directory and generating
    embeddings without blocking the main workflow.
    """

    queue: EmbeddingQueue = field(default_factory=EmbeddingQueue)
    batch_size: int = 5
    poll_interval: float = 5.0
    progress_callback: AsyncProgressCallback | None = None
    _running: bool = field(default=False, init=False)
    _task: asyncio.Task | None = field(default=None, init=False)

    async def start(self) -> None:
        """Start the background processor."""
        if self._running:
            logger.warning("Processor already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._process_loop())
        logger.info("Started async embedding processor")

    async def stop(self) -> None:
        """Stop the background processor gracefully."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Stopped async embedding processor")

    async def process_batch(self, max_files: int | None = None) -> dict[str, int]:
        """Process a single batch of files synchronously.

        This is useful for one-off processing without running
        a background loop.

        Args:
            max_files: Maximum number of files to process

        Returns:
            Dict with counts: {"processed": N, "failed": M, "chunks": K}
        """
        max_files = max_files or self.batch_size
        pending = self.queue.get_pending()[:max_files]

        if not pending:
            return {"processed": 0, "failed": 0, "chunks": 0}

        stats = {"processed": 0, "failed": 0, "chunks": 0}

        # Run CPU-intensive embedding in thread pool
        def _process_batch() -> dict[str, int]:
            ingester = CodeExampleIngester(
                embed_model=get_embed_model(),
                graph_client=GraphClient(),
                queue=self.queue,
            )

            with ingester.graph_client:
                for qf in pending:
                    self.queue.mark_processing(qf.id)
                    filename = Path(qf.remote_path).name

                    try:
                        local_path = Path(qf.local_path)
                        if not local_path.exists():
                            raise FileNotFoundError(
                                f"Staged file missing: {qf.local_path}"
                            )

                        result = ingester._ingest_single_file(
                            facility=qf.facility_id,
                            remote_path=qf.remote_path,
                            local_path=local_path,
                            description=qf.description,
                        )

                        self.queue.mark_completed(qf.id)
                        stats["processed"] += 1
                        stats["chunks"] += result["chunks"]
                        logger.info(
                            f"Processed: {filename} ({result['chunks']} chunks)"
                        )
                    except Exception as e:
                        self.queue.mark_failed(qf.id, str(e))
                        stats["failed"] += 1
                        logger.exception(f"Failed to process {filename}: {e}")

            return stats

        result = await anyio.to_thread.run_sync(_process_batch)  # type: ignore[attr-defined]
        self._report_progress("batch_complete", result)
        return result

    async def _process_loop(self) -> None:
        """Main processing loop - runs until stopped."""
        logger.info("Embedding processor loop started")

        while self._running:
            try:
                pending = self.queue.get_pending()

                if pending:
                    self._report_progress("batch_start", {"pending": len(pending)})
                    await self.process_batch()
                    # Small delay between batches
                    await asyncio.sleep(1.0)
                else:
                    # No work to do, wait before checking again
                    await asyncio.sleep(self.poll_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in processor loop: {e}")
                await asyncio.sleep(self.poll_interval)

        logger.info("Embedding processor loop ended")

    def _report_progress(self, event: str, data: dict[str, Any]) -> None:
        """Report progress via callback if set."""
        if self.progress_callback:
            self.progress_callback(event, data)

    def get_status(self) -> dict[str, Any]:
        """Get processor and queue status."""
        queue_status = self.queue.get_status()
        return {
            "running": self._running,
            "batch_size": self.batch_size,
            "poll_interval": self.poll_interval,
            **queue_status,
        }


async def run_processor(
    staging_dir: Path | None = None,
    batch_size: int = 5,
    poll_interval: float = 5.0,
    max_iterations: int | None = None,
) -> dict[str, int]:
    """Run the embedding processor until queue is empty or max iterations reached.

    This is a convenience function for running the processor in a script
    or CLI context.

    Args:
        staging_dir: Custom staging directory (uses default if None)
        batch_size: Number of files to process per batch
        poll_interval: Seconds to wait between checks when queue is empty
        max_iterations: Maximum number of batch iterations (None = until empty)

    Returns:
        Cumulative stats: {"processed": N, "failed": M, "chunks": K}
    """
    if staging_dir:
        queue = EmbeddingQueue(staging_dir=staging_dir)
    else:
        queue = EmbeddingQueue()

    processor = AsyncEmbeddingProcessor(
        queue=queue,
        batch_size=batch_size,
        poll_interval=poll_interval,
    )

    cumulative = {"processed": 0, "failed": 0, "chunks": 0}
    iterations = 0

    while True:
        pending = queue.get_pending()
        if not pending:
            logger.info("Queue is empty, processor finished")
            break

        if max_iterations and iterations >= max_iterations:
            logger.info(f"Reached max iterations ({max_iterations})")
            break

        stats = await processor.process_batch()
        cumulative["processed"] += stats["processed"]
        cumulative["failed"] += stats["failed"]
        cumulative["chunks"] += stats["chunks"]
        iterations += 1

    return cumulative


__all__ = ["AsyncEmbeddingProcessor", "run_processor"]
