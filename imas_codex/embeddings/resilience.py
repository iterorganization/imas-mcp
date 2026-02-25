"""Embedding server resilience for long-running discovery workers.

Provides pause-and-poll behaviour when the remote embedding server becomes
unavailable mid-run.  The worker calls ``await_ready()`` before each
embedding batch; if the server is down the call blocks, polls, and
attempts reconnection (SSH tunnel re-establishment, systemd restart on
ITER login nodes) until the server is healthy again.

The current status is exposed via ``status`` for the progress display to
show in the ``embed:`` indicator.

Usage inside an async worker::

    from imas_codex.embeddings.resilience import EmbeddingResilience

    resilience = EmbeddingResilience()

    while not should_stop():
        if not await resilience.await_ready():
            break  # gave up after max retries

        result = do_embedding_work()
        ...
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time

logger = logging.getLogger(__name__)

# Module-level status for the progress display to read without a reference
# to the EmbeddingResilience instance.
_embed_status: str = "ready"
_embed_status_lock = threading.Lock()


def get_embed_status() -> str:
    """Current resilience status: 'ready', 'reconnecting', 'waiting NNs', etc."""
    with _embed_status_lock:
        return _embed_status


def _set_embed_status(status: str) -> None:
    global _embed_status
    with _embed_status_lock:
        _embed_status = status


class EmbeddingResilience:
    """Pause-and-poll wrapper around the remote embedding server.

    Parameters
    ----------
    poll_interval:
        Seconds between health-check polls while waiting.
    max_wait:
        Maximum seconds to wait for the server to come back.
        0 means wait indefinitely.
    reconnect:
        If True, attempt SSH tunnel and systemd restart during polling.
    """

    def __init__(
        self,
        poll_interval: float = 15.0,
        max_wait: float = 0,
        reconnect: bool = True,
    ) -> None:
        self.poll_interval = poll_interval
        self.max_wait = max_wait
        self.reconnect = reconnect
        self._healthy = True

    @property
    def status(self) -> str:
        return get_embed_status()

    # ------------------------------------------------------------------
    # Public API (async)
    # ------------------------------------------------------------------

    async def await_ready(self) -> bool:
        """Block until the embedding server is healthy.

        Returns True when the server is available, False if ``max_wait``
        is exceeded (the caller should stop the worker gracefully).
        """
        if await self._check_health():
            if not self._healthy:
                logger.info("Embedding server recovered")
                _set_embed_status("ready")
                self._healthy = True
            return True

        # Server is down — enter poll loop
        self._healthy = False
        logger.warning("Embedding server unavailable, entering poll loop")
        start = time.monotonic()

        while True:
            elapsed = time.monotonic() - start
            if self.max_wait > 0 and elapsed > self.max_wait:
                logger.error(
                    "Embedding server not recovered after %.0fs, giving up",
                    elapsed,
                )
                _set_embed_status("unavailable")
                return False

            remaining = (
                f" ({int(self.max_wait - elapsed)}s left)" if self.max_wait > 0 else ""
            )
            wait_msg = f"waiting {int(elapsed)}s{remaining}"
            _set_embed_status(wait_msg)
            logger.info("Embedding poll: %s", wait_msg)

            # Attempt reconnection before the next health check
            if self.reconnect:
                _set_embed_status(f"reconnecting ({int(elapsed)}s)")
                await asyncio.to_thread(self._try_reconnect)

            await asyncio.sleep(self.poll_interval)

            if await self._check_health():
                logger.info("Embedding server recovered after %.0fs", elapsed)
                _set_embed_status("ready")
                self._healthy = True
                return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _check_health(self) -> bool:
        """Non-blocking health check via thread pool."""
        return await asyncio.to_thread(self._check_health_sync)

    @staticmethod
    def _check_health_sync() -> bool:
        from imas_codex.embeddings.client import RemoteEmbeddingClient
        from imas_codex.settings import get_embed_remote_url

        url = get_embed_remote_url()
        if not url:
            return False
        client = RemoteEmbeddingClient(url)
        try:
            return client.is_available(timeout=5.0)
        finally:
            client.close()

    @staticmethod
    def _try_reconnect() -> None:
        """Attempt to restore connectivity (SSH tunnel + systemd)."""
        from imas_codex.embeddings.readiness import (
            _ensure_ssh_tunnel,
            _is_on_facility,
            _is_on_login_node,
            _try_start_service,
        )
        from imas_codex.settings import get_embed_server_port

        port = get_embed_server_port()

        if not _is_on_facility():
            # Off-facility: re-establish SSH tunnel
            logger.debug("Attempting SSH tunnel reconnection (port %d)", port)
            _ensure_ssh_tunnel(port)
        elif _is_on_login_node():
            # On login node: restart systemd service
            logger.debug("Attempting systemd service restart")
            _try_start_service()


__all__ = ["EmbeddingResilience", "get_embed_status"]
