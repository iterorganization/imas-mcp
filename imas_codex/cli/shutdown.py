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
import threading
from collections.abc import Coroutine
from typing import Any

logger = logging.getLogger(__name__)


# Default timeout for the executor shutdown after asyncio.run() completes.
# Threads blocked on SSH subprocesses or LLM HTTP calls may outlive the
# event loop; a short timeout prevents the process from hanging.
_EXECUTOR_SHUTDOWN_TIMEOUT = 5

# Grace period after safe_asyncio_run returns before force-exiting.
# This gives the CLI time to print summary output.
_EXIT_WATCHDOG_GRACE = 10


def safe_asyncio_run[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine, suppressing 'Event loop is closed' on cleanup.

    ``asyncio.run()`` closes the event loop after the coroutine returns.
    If any ``asyncio.create_subprocess_exec()``-spawned transports are
    garbage-collected *after* the loop is closed, their ``__del__``
    methods raise ``RuntimeError: Event loop is closed`` — an ugly but
    harmless traceback on stderr.

    This wrapper installs a temporary ``sys.unraisablehook`` that
    silences only that specific error, then restores the original hook.

    It also ensures the default executor is shut down with a short
    timeout so that leaked threads (from ``asyncio.to_thread()`` calls
    to SSH subprocesses or LLM HTTP calls) do not prevent the process
    from exiting.

    A daemon watchdog thread is started that will force-exit the process
    if it hasn't terminated within a grace period after the coroutine
    completes. This prevents the process from hanging on Python's atexit
    thread join when SSH subprocess threads are still alive.
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
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(coro)
        finally:
            try:
                # Cancel any straggling tasks
                _cancel_remaining_tasks(loop)
            finally:
                # Shut down the executor with a short timeout so leaked
                # threads from to_thread() don't block exit.
                loop.run_until_complete(
                    loop.shutdown_default_executor(timeout=_EXECUTOR_SHUTDOWN_TIMEOUT)
                )
                # Force-kill SSH subprocess pools to unblock any
                # remaining executor threads waiting on SSH I/O, so
                # the thread.join() in Python's atexit handler returns.
                _force_kill_ssh_pools()
                loop.close()

        # Start a daemon watchdog: if the process hasn't exited within
        # the grace period (e.g. because atexit is joining leaked
        # threads), force-exit so the terminal is returned.
        _start_exit_watchdog(_EXIT_WATCHDOG_GRACE)

        return result
    finally:
        sys.unraisablehook = old_hook


def _start_exit_watchdog(grace_seconds: float) -> None:
    """Start a daemon thread that force-exits after a grace period.

    Python's ``concurrent.futures.thread._python_exit`` atexit handler
    joins all executor threads.  If an SSH subprocess is still running,
    ``thread.join()`` blocks forever.  This watchdog ensures the process
    exits cleanly after the CLI has printed its output.
    """

    def _watchdog() -> None:
        threading.Event().wait(timeout=grace_seconds)
        # If we're still alive, flush and hard-exit
        _force_kill_ssh_pools()
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        os._exit(0)

    t = threading.Thread(target=_watchdog, daemon=True)
    t.start()


def _cancel_remaining_tasks(loop: asyncio.AbstractEventLoop) -> None:
    """Cancel and gather all remaining tasks on the loop."""
    tasks = asyncio.all_tasks(loop)
    if not tasks:
        return
    for task in tasks:
        task.cancel()
    loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))


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
            _force_stop_display(display)
            # Force-kill SSH worker pools so leaked threads don't
            # block process exit.
            _force_kill_ssh_pools()
            # Cancel all running tasks from the event loop so the
            # coroutine chain unwinds.
            for task in asyncio.all_tasks(loop):
                task.cancel()
        else:
            # Hard exit -- reached when graceful cancel didn't work
            os._exit(130)

    loop.add_signal_handler(signal.SIGINT, _handle_sigint)


def _force_stop_display(display: object | None) -> None:
    """Stop Rich display immediately so terminal is usable."""
    live = getattr(display, "_live", None) if display else None
    if live is not None and getattr(live, "is_started", False):
        try:
            live.stop()
        except Exception:
            pass


def _force_kill_ssh_pools() -> None:
    """Synchronously force-kill all SSH worker pools.

    Called on forced shutdown to ensure leaked threads from
    ``asyncio.to_thread()`` SSH subprocess calls don't block
    process exit.
    """
    try:
        from imas_codex.remote.ssh_worker import force_kill_all_pools

        force_kill_all_pools()
    except Exception:
        pass


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


def force_exit() -> None:
    """Force process exit, cleaning up SSH resources.

    Called after all CLI output has been printed to prevent the
    process from hanging on leaked executor threads (non-daemon
    threads spawned by ``asyncio.to_thread()`` for SSH subprocess
    calls). Python's atexit handler waits for ALL threads to
    ``join()`` which blocks indefinitely if a subprocess is still
    running on a remote host.

    This function:
    1. Force-kills all SSH worker pool subprocesses
    2. Flushes stdout/stderr
    3. Calls ``os._exit(0)`` to bypass atexit thread joins
    """
    _force_kill_ssh_pools()
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(0)
