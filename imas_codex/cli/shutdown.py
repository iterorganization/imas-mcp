"""Graceful shutdown for discovery CLI commands.

Provides a single Ctrl+C clean exit via asyncio signal handlers
with rich shutdown progress tracking:

  First Ctrl+C:  Signal workers to stop, switch display to shutdown
                 mode showing per-group drain progress.  Workers
                 finish their current batch and exit cleanly.
  Second Ctrl+C: Stops Rich display, cancels all async tasks.
  Third Ctrl+C:  Immediate process exit (os._exit).

Usage in async discovery loops::

    from imas_codex.cli.shutdown import install_shutdown_handlers

    async def _async_discovery_loop(...):
        stop_event = asyncio.Event()

        install_shutdown_handlers(
            stop_event=stop_event,
            display=display,  # BaseProgressDisplay subclass
        )

        result = await run_parallel_discovery(
            ...,
            stop_event=stop_event,
        )

The stop_event is wired into the discovery state's should_stop() check
by each parallel runner via a watcher task.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from collections.abc import Coroutine
from typing import Any

logger = logging.getLogger(__name__)


def safe_asyncio_run[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine, suppressing 'Event loop is closed' on cleanup.

    ``asyncio.run()`` closes the event loop after the coroutine returns.
    If any ``asyncio.create_subprocess_exec()``-spawned transports are
    garbage-collected *after* the loop is closed, their ``__del__``
    methods raise ``RuntimeError: Event loop is closed`` — an ugly but
    harmless traceback on stderr.

    This wrapper installs a temporary ``sys.unraisablehook`` that
    silences only that specific error, then restores the original hook.
    """
    old_hook = sys.unraisablehook

    def _suppress_closed_loop(unraisable: sys.UnraisableHookArgs) -> None:
        if (
            isinstance(unraisable.exc_value, RuntimeError)
            and str(unraisable.exc_value) == "Event loop is closed"
        ):
            return
        old_hook(unraisable)

    sys.unraisablehook = _suppress_closed_loop
    try:
        return asyncio.run(coro)
    finally:
        sys.unraisablehook = old_hook


def install_shutdown_handlers(
    *,
    stop_event: asyncio.Event,
    display: object | None = None,
) -> None:
    """Install SIGINT handlers on the running asyncio event loop.

    Replaces asyncio's default SIGINT handling (which raises
    KeyboardInterrupt and requires multiple presses) with a
    cooperative shutdown:

    1. First Ctrl+C:  Sets stop_event, switches the progress display
       to shutdown mode (yellow border, live worker-drain tracker).
       Workers finish their current batch and exit.
    2. Second Ctrl+C: Stops Rich Live display, cancels all async tasks
       so the coroutine chain unwinds.
    3. Third Ctrl+C:  ``os._exit(130)`` (hard exit).

    Args:
        stop_event: asyncio.Event that parallel runners watch to
            set ``state.stop_requested = True``.
        display: Optional BaseProgressDisplay instance.  Switched to
            shutdown mode on first Ctrl+C and stopped on second.
    """
    loop = asyncio.get_running_loop()
    sigint_count = 0

    def _handle_sigint() -> None:
        nonlocal sigint_count
        sigint_count += 1

        if sigint_count == 1:
            logger.info("Graceful shutdown requested (Ctrl+C)")
            stop_event.set()
            # Switch display to shutdown mode
            if display is not None and hasattr(display, "begin_shutdown"):
                try:
                    display.begin_shutdown()
                except Exception:
                    pass
        elif sigint_count == 2:
            logger.warning("Forced shutdown (second Ctrl+C)")
            # Stop Rich display immediately so terminal is usable
            live = getattr(display, "_live", None) if display else None
            if live is not None and getattr(live, "is_started", False):
                try:
                    live.stop()
                except Exception:
                    pass
            # Keep our signal handler installed so the third Ctrl+C
            # reaches os._exit(130) below.  Cancel all running tasks
            # from the event loop so the coroutine chain unwinds.
            for task in asyncio.all_tasks(loop):
                task.cancel()
        else:
            # Hard exit -- reached when graceful cancel didn't work
            os._exit(130)

    loop.add_signal_handler(signal.SIGINT, _handle_sigint)


async def watch_stop_event(
    stop_event: asyncio.Event,
    state: object,
) -> None:
    """Watch a stop event and set state.stop_requested when triggered.

    Spawned as a task inside each parallel runner to bridge the CLI
    signal handler (which sets the event) to the discovery state
    (which workers poll via should_stop()).

    Args:
        stop_event: Event set by the shutdown signal handler.
        state: Discovery state object with a ``stop_requested`` attribute.
    """
    await stop_event.wait()
    state.stop_requested = True
    logger.info("Stop event received -- workers will finish current batch")
