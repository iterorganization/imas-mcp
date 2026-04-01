"""Background warmup for the MCP server.

Heavy packages (linkml_runtime via graph.schema, pydantic models, discovery
pipeline) take 10–25 s to import on NFS filesystems.  Starting warmup threads
at module import time lets the MCP server complete its stdio handshake (and
start accepting tool calls) while the slow imports proceed in parallel.

Usage (server.py)::

    from imas_codex.llm._warmup import warmup

    # Kick off all background threads immediately when the module is loaded.
    warmup.start()

    # Inside a tool handler — blocks at most the remaining import time:
    disc = warmup.discovery()      # dict of facility functions
    graph_ns = warmup.graph()      # dict of graph classes/functions
    emb_ns = warmup.embeddings()   # dict of embedding classes/functions
    rem_ns = warmup.remote()       # dict of remote-tool functions

Warmup groups
-------------
discovery    imas_codex.discovery — facility config, update_infrastructure …
graph        imas_codex.graph — GraphClient, get_schema, to_cypher_props
embeddings   imas_codex.embeddings + settings — Encoder, EncoderConfig, …
remote       imas_codex.remote.tools — run, check_all_tools, install_all_tools
search       imas_codex.llm.search_tools + search_formatters
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class _WarmupGroup:
    """One import group loaded in a single background thread.

    Supports explicit dependencies: if ``depends_on`` is provided, the
    group waits for those groups to finish before running its loader.
    This prevents concurrent-import races when two groups share a
    package (e.g. ``embeddings`` and ``search`` both touch
    ``imas_codex.embeddings``).
    """

    def __init__(
        self,
        name: str,
        loader: Callable[[], dict[str, Any]],
        depends_on: list[_WarmupGroup] | None = None,
    ) -> None:
        self._name = name
        self._loader = loader
        self._depends_on = depends_on or []
        self._ready = threading.Event()
        self._result: dict[str, Any] | None = None
        self._error: BaseException | None = None

    def start(self) -> None:
        t = threading.Thread(target=self._run, daemon=True, name=f"warmup-{self._name}")
        t.start()
        logger.debug("warmup: started %s", self._name)

    def _run(self) -> None:
        try:
            for dep in self._depends_on:
                dep.wait()
            self._result = self._loader()
            logger.debug("warmup: ready %s", self._name)
        except BaseException as exc:
            self._error = exc
            logger.warning("warmup: %s failed — %s", self._name, exc)
        finally:
            self._ready.set()

    def wait(self) -> dict[str, Any]:
        """Block until ready; raises RuntimeError propagating the load error."""
        self._ready.wait()
        if self._error is not None:
            raise RuntimeError(f"Warmup '{self._name}' failed") from self._error
        assert self._result is not None
        return self._result

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set()


# ---------------------------------------------------------------------------
# Loaders — one per import group
# ---------------------------------------------------------------------------


def _load_discovery() -> dict[str, Any]:
    from imas_codex.discovery import (
        get_facility as _gf,
        get_facility_infrastructure as _gfi,
        get_facility_validated as _gfv,
        update_infrastructure as _ui,
        update_metadata as _um,
    )

    return {
        "get_facility": _gf,
        "get_facility_infrastructure": _gfi,
        "get_facility_validated": _gfv,
        "update_infrastructure": _ui,
        "update_metadata": _um,
    }


def _load_graph() -> dict[str, Any]:
    from imas_codex.graph import GraphClient, get_schema
    from imas_codex.graph.schema import to_cypher_props

    return {
        "GraphClient": GraphClient,
        "get_schema": get_schema,
        "to_cypher_props": to_cypher_props,
    }


def _load_embeddings() -> dict[str, Any]:
    from imas_codex.embeddings.config import EncoderConfig
    from imas_codex.embeddings.encoder import EmbeddingBackendError, Encoder
    from imas_codex.settings import get_embedding_location

    return {
        "EncoderConfig": EncoderConfig,
        "Encoder": Encoder,
        "EmbeddingBackendError": EmbeddingBackendError,
        "get_embedding_location": get_embedding_location,
    }


def _load_remote() -> dict[str, Any]:
    from imas_codex.remote.tools import (
        check_all_tools as _ca,
        install_all_tools as _ia,
        run as _r,
    )

    return {"run": _r, "check_all_tools": _ca, "install_all_tools": _ia}


def _load_search() -> dict[str, Any]:
    # Importing search_tools/formatters triggers ~13s of chain imports.
    # Pre-loading here ensures tool-call-time imports are instant (sys.modules hit).
    import imas_codex.llm.search_formatters  # noqa: F401
    import imas_codex.llm.search_tools  # noqa: F401

    return {}


# ---------------------------------------------------------------------------
# Public singleton
# ---------------------------------------------------------------------------


class ServerWarmup:
    """Registry of background warmup groups for the MCP server.

    Call :meth:`start` once — typically at module import time in ``server.py``
    — to kick off all background threads.  Then call the per-group accessors
    (``discovery()``, ``graph()``, etc.) inside tool handlers to block until
    that group is ready.  If warmup already finished the call returns immediately.
    """

    def __init__(self) -> None:
        self._groups: dict[str, _WarmupGroup] = {}
        self._lock = threading.Lock()
        self._started = False

    def start(self) -> None:
        """Start all background import threads.  Idempotent — safe to call multiple times.

        The ``search`` group depends on ``embeddings`` because both
        import from ``imas_codex.embeddings.*``.  Without this ordering
        the two threads race on the package ``__init__.py``, triggering
        a circular-import ``ImportError`` on partially-initialized modules.
        """
        with self._lock:
            if self._started:
                return
            self._started = True

        # Independent groups — no shared imports, safe to run in parallel.
        discovery = _WarmupGroup("discovery", _load_discovery)
        graph = _WarmupGroup("graph", _load_graph)
        embeddings = _WarmupGroup("embeddings", _load_embeddings)
        remote = _WarmupGroup("remote", _load_remote)

        # search imports search_tools.py which has a top-level
        #   from imas_codex.embeddings.encoder import ...
        # so it must wait for the embeddings group to finish first.
        search = _WarmupGroup("search", _load_search, depends_on=[embeddings])

        for g in (discovery, graph, embeddings, remote, search):
            self._groups[g._name] = g
            g.start()

    # ---- per-group accessors ------------------------------------------------

    def discovery(self) -> dict[str, Any]:
        """Wait for discovery group and return its loaded objects."""
        return self._groups["discovery"].wait()

    def graph(self) -> dict[str, Any]:
        """Wait for graph group and return its loaded objects."""
        return self._groups["graph"].wait()

    def embeddings(self) -> dict[str, Any]:
        """Wait for embeddings group and return its loaded objects."""
        return self._groups["embeddings"].wait()

    def remote(self) -> dict[str, Any]:
        """Wait for remote group and return its loaded objects."""
        return self._groups["remote"].wait()

    def wait_all(self) -> None:
        """Block until every group is ready."""
        for g in self._groups.values():
            g.wait()

    @property
    def status(self) -> dict[str, str]:
        """Snapshot of per-group readiness for health-check endpoints."""
        return {
            name: ("ready" if g.is_ready else "loading")
            for name, g in self._groups.items()
        }


#: Module-level singleton imported and started by server.py.
warmup = ServerWarmup()
