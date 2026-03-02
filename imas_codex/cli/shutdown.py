"""Graceful shutdown for discovery CLI commands.

Provides a single Ctrl+C clean exit via asyncio signal handlers
with rich shutdown progress tracking:

  First Ctrl+C:  Signal workers to stop, switch display to shutdown
                 mode showing per-group drain progress.  Workers
                 finish their current batch and exit cleanly.
  Second Ctrl+C: Force-cancel all async tasks, stop Rich display, exit
  Third Ctrl+C:  Immediate process exit (nuclear, should never be needed)

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

logger = logging.getLogger(__name__)


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
    2. Second Ctrl+C: Stops Rich Live display, removes custom handler,
       re-delivers SIGINT to trigger asyncio's default cancellation.
    3. Third Ctrl+C:  ``os._exit(130)`` (emergency, should not be needed).

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
            # Restore asyncio's default handler and re-deliver SIGINT
            # so asyncio.run() cancels the main task and raises
            # KeyboardInterrupt to the calling code.
            try:
                loop.remove_signal_handler(signal.SIGINT)
            except Exception:
                pass
            os.kill(os.getpid(), signal.SIGINT)
        else:
            # Nuclear exit -- should never be needed
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
