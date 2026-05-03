"""Graceful shutdown for discovery CLI commands.

Provides cooperative shutdown for both SIGINT (Ctrl+C) and SIGTERM
with rich shutdown progress tracking:

  First SIGINT/SIGTERM: Signal workers to stop, switch display to
                        shutdown mode showing per-group drain progress.
                        Workers finish their current batch and exit
                        cleanly.
  Second SIGINT:        Stops Rich display, cancels all async tasks.
                        Starts a 45s watchdog to outlast drain +
                        finalize timeouts.
  Third SIGINT:         Immediate process exit (os._exit).

SIGTERM is now cooperative (same as first SIGINT).  Force-kill
requires ``kill -9`` (SIGKILL).

Usage in discovery CLIs::

    from imas_codex.cli.shutdown import install_shutdown_handlers, safe_asyncio_run

    with MyProgressDisplay(...) as display:
        async def run_with_display():
            stop_event = asyncio.Event()
            install_shutdown_handlers(
                stop_event=stop_event,
                display=display,  # BaseProgressDisplay subclass
            )
            try:
                return await run_parallel_discovery(
                    ..., stop_event=stop_event,
                )
            finally:
                ...  # cancel refresh/ticker tasks

        result = safe_asyncio_run(run_with_display())
        display.print_summary()  # display still alive here

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
            # Start the watchdog BEFORE cleanup — if task cancellation
            # hangs (e.g. to_thread blocked on LLM/SSH I/O), the
            # watchdog guarantees the process exits.
            _start_exit_watchdog(_EXIT_WATCHDOG_GRACE)
            try:
                # Cancel any straggling tasks (with timeout so we
                # don't hang on threads blocked in to_thread)
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


# Timeout for _cancel_remaining_tasks — must be shorter than the
# exit watchdog grace period so we proceed to executor shutdown
# rather than hanging on unkillable to_thread tasks.
_CANCEL_TASKS_TIMEOUT = 5


def _cancel_remaining_tasks(loop: asyncio.AbstractEventLoop) -> None:
    """Cancel remaining tasks on the loop with a timeout.

    Tasks stuck in ``asyncio.to_thread()`` cannot be cancelled until
    the underlying thread finishes.  A bounded ``asyncio.wait()``
    prevents the process from hanging indefinitely.
    """
    tasks = asyncio.all_tasks(loop)
    if not tasks:
        return
    for task in tasks:
        task.cancel()

    async def _wait() -> None:
        _, still_pending = await asyncio.wait(tasks, timeout=_CANCEL_TASKS_TIMEOUT)
        if still_pending:
            logger.warning(
                "%d task(s) still pending after %ss cancel timeout "
                "— forcing SSH pool shutdown",
                len(still_pending),
                _CANCEL_TASKS_TIMEOUT,
            )
            _force_kill_ssh_pools()

    loop.run_until_complete(_wait())


def install_shutdown_handlers(
    *,
    stop_event: asyncio.Event,
    display: object | None = None,
) -> None:
    """Install SIGINT and SIGTERM handlers on the running asyncio event loop.

    Replaces asyncio's default SIGINT handling (which raises
    KeyboardInterrupt and requires multiple presses) with a
    cooperative shutdown:

    1. First SIGINT/SIGTERM: Sets stop_event, switches the progress
       display to shutdown mode (yellow border, live worker-drain
       tracker).  Workers finish their current batch and exit.
    2. Second SIGINT: Stops Rich Live display, cancels all async tasks
       so the coroutine chain unwinds.  Starts a 45s watchdog to
       outlast drain_pending (30s) + finalize_sn_run (10s) + buffer.
    3. Third SIGINT: ``os._exit(130)`` (hard exit).

    SIGTERM triggers the same cooperative shutdown as the first SIGINT.

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
            # Start exit watchdog NOW — must outlast DRAIN_TIMEOUT (30s)
            # + FINALIZE_TIMEOUT (10s) + buffer so finalize_sn_run lands.
            _start_exit_watchdog(45)
            # Cancel all running tasks from the event loop so the
            # coroutine chain unwinds.
            for task in asyncio.all_tasks(loop):
                task.cancel()
        else:
            # Hard exit -- reached when graceful cancel didn't work
            os._exit(130)

    loop.add_signal_handler(signal.SIGINT, _handle_sigint)
    loop.add_signal_handler(signal.SIGTERM, _handle_sigint)


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
