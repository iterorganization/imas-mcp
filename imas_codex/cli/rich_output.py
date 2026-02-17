"""Automatic rich output detection for CLI commands.

Centralizes the decision of whether to use Rich progress displays
(spinners, live updates, tables) vs plain logging. All CLI commands
should call ``should_use_rich()`` instead of accepting a ``--no-rich``
flag.

Detection priority:
1. ``IMAS_CODEX_RICH`` env var — explicit override (``0``/``false``/``no``
   to disable, ``1``/``true``/``yes`` to force enable)
2. ``NO_COLOR`` env var — standard convention, disables rich
3. ``CI`` env var — GitHub Actions / CI, disables rich
4. ``stdout.isatty()`` — false in pipes, redirects, cron, disables rich
"""

from __future__ import annotations

import os
import sys


def should_use_rich() -> bool:
    """Determine whether to use Rich interactive output.

    Returns True when the terminal supports interactive Rich displays
    (progress bars, spinners, live tables). Returns False for CI, pipes,
    redirected output, non-TTY, or when explicitly disabled.
    """
    # 1. Explicit override via env var
    override = os.environ.get("IMAS_CODEX_RICH", "").strip().lower()
    if override in ("0", "false", "no"):
        return False
    if override in ("1", "true", "yes"):
        return True

    # 2. NO_COLOR convention (https://no-color.org/)
    if os.environ.get("NO_COLOR") is not None:
        return False

    # 3. CI environment
    if os.environ.get("CI"):
        return False

    # 4. TTY check — pipes, redirects, cron all fail this
    try:
        if not sys.stdout.isatty():
            return False
    except Exception:
        return False

    return True
