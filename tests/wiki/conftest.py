"""Test fixtures for wiki discovery tests.

Provides mocked external dependencies so that wiki discovery tests run
entirely offline — no Neo4j, no SSH, no HTTP, no LLM API calls.

Fixtures:
- Sample HTML for each platform (MediaWiki, TWiki, Confluence, static)
- Mock GraphClient / graph_ops functions
- Mock WikiDiscoveryState with configurable auth types
- Mock LLM/VLM responses matching Pydantic models
- Mock SSH subprocess results
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# =============================================================================
# Sample HTML Content (one per platform)
# =============================================================================

MEDIAWIKI_HTML = """\
<!DOCTYPE html>
<html>
<head><title>Thomson Scattering - SPCwiki</title></head>
<body>
<h1>Thomson Scattering Diagnostic</h1>
<p>The Thomson scattering diagnostic measures electron temperature
and density profiles on TCV.</p>
<h2>Data Access</h2>
<p>Data is stored in MDSplus under <code>\\tcv_shot::thomson:te</code>.
IMAS IDS: core_profiles/profiles_1d/electrons/temperature.</p>
<h3>Calibration</h3>
<p>Calibration factors are updated annually. See the calibration
procedure document attached below.</p>
<img src="/wiki/images/thomson_layout.png" alt="Thomson layout" width="600" height="400">
<img src="/wiki/images/favicon.png" alt="" width="16" height="16">
<a href="/wiki/files/calibration_report.pdf">Calibration Report (PDF)</a>
<a href="/wiki/files/analysis.xlsx">Analysis Spreadsheet</a>
</body>
</html>
"""

CONFLUENCE_HTML = """\
<!DOCTYPE html>
<html>
<head><title>ITER Disruption Mitigation - Confluence</title></head>
<body>
<h1>Disruption Mitigation System</h1>
<p>The DMS injects shattered pellets to mitigate disruptions.</p>
<h2>Signal Descriptions</h2>
<p>Plasma current: PCS signal PCS_IP measured in MA.</p>
<ac:image ac:width="800" ac:height="600" ac:alt="DMS schematic">
  <ri:attachment ri:filename="dms_diagram.png" />
</ac:image>
<ac:image ac:width="400">
  <ri:url ri:value="/download/attachments/12345/plasma_current.png" />
</ac:image>
<img src="https://confluence.example.com/images/logo.png" width="32" height="32">
</body>
</html>
"""

TWIKI_HTML = """\
<!DOCTYPE html>
<html>
<head><title>LiuqeCode</title></head>
<body>
<h1>LIUQE Equilibrium Reconstruction</h1>
<p>LIUQE is the real-time equilibrium reconstruction code used on TCV.</p>
<h2>Usage</h2>
<p>Run with: liuqe -shot 12345 -t 0.5</p>
<p>Results stored in MDSplus tree: \\tcv_shot::results:liuqe</p>
<img src="/twiki/pub/Main/LiuqeCode/equilibrium_flux.png" alt="Flux surfaces" width="500" height="500">
</body>
</html>
"""

STATIC_HTML = """\
<!DOCTYPE html>
<html>
<head><title>JT-60SA Documentation</title></head>
<body>
<h1>JT-60SA Control System</h1>
<p>Overview of the plasma control system for JT-60SA.</p>
<h2>Diagnostics</h2>
<p>The interferometer measures line-integrated density.</p>
<a href="diagnostics/interferometer.html">Interferometer Details</a>
<a href="reports/annual_report_2023.pdf">Annual Report 2023</a>
<a href="/images/control_room.jpg">
<img src="/images/control_room_thumb.jpg" alt="Control room" width="200" height="150">
</a>
</body>
</html>
"""

EMPTY_HTML = "<html><body></body></html>"

MINIMAL_HTML = "<html><head><title>Test</title></head><body><p>Hello world</p></body></html>"


# =============================================================================
# Sample Data Structures
# =============================================================================


@dataclass
class MockWorkerStats:
    """Mirrors the WorkerStats interface used by progress callbacks."""

    processed: int = 0
    cost: float = 0.0
    errors: int = 0
    rate: float | None = None


@dataclass
class MockDiscoveredPage:
    """Matches DiscoveredPage from adapters."""

    name: str
    url: str
    namespace: str = ""


@dataclass
class MockDiscoveredArtifact:
    """Matches DiscoveredArtifact from adapters."""

    filename: str
    url: str
    artifact_type: str
    size_bytes: int | None = None
    mime_type: str | None = None
    linked_pages: list[str] = field(default_factory=list)


# =============================================================================
# Sample Score / Ingest Results
# =============================================================================


def make_score_results(
    page_ids: list[str], scores: list[float] | None = None
) -> list[dict[str, Any]]:
    """Create mock LLM scoring results for pages."""
    if scores is None:
        scores = [0.7] * len(page_ids)
    return [
        {
            "id": pid,
            "score": score,
            "purpose": "data_documentation",
            "description": f"Documentation for {pid.split(':')[-1]}",
            "reasoning": "Contains physics content",
            "keywords": ["plasma", "diagnostic"],
            "physics_domain": "equilibrium",
            "should_ingest": score >= 0.5,
            "skip_reason": None if score >= 0.5 else "low relevance",
            "is_physics": True,
            "score_data_documentation": score,
            "score_physics_content": score * 0.8,
            "score_code_documentation": 0.3,
            "score_data_access": 0.5,
            "score_calibration": 0.2,
            "score_imas_relevance": score * 0.6,
            "score_cost": 0.001,
            "preview_text": f"Preview text for {pid}",
        }
        for pid, score in zip(page_ids, scores)
    ]


def make_ingest_results(page_ids: list[str], chunks: int = 5) -> list[dict[str, Any]]:
    """Create mock ingestion results."""
    return [
        {
            "id": pid,
            "chunk_count": chunks,
            "score": 0.8,
            "description": f"Ingested {pid.split(':')[-1]}",
            "physics_domain": "diagnostics",
        }
        for pid in page_ids
    ]


def make_artifact_score_results(
    artifact_ids: list[str], scores: list[float] | None = None
) -> list[dict[str, Any]]:
    """Create mock LLM artifact scoring results."""
    if scores is None:
        scores = [0.6] * len(artifact_ids)
    return [
        {
            "id": aid,
            "score": score,
            "artifact_purpose": "data_documentation",
            "description": f"Artifact {aid.split(':')[-1]}",
            "reasoning": "Relevant artifact",
            "keywords": ["calibration"],
            "physics_domain": "diagnostics",
            "should_ingest": score >= 0.5,
            "skip_reason": None,
            "score_data_documentation": score,
            "score_physics_content": 0.4,
            "score_code_documentation": 0.2,
            "score_data_access": 0.3,
            "score_calibration": 0.5,
            "score_imas_relevance": 0.3,
            "score_cost": 0.002,
            "filename": f"artifact_{aid.split(':')[-1]}.pdf",
            "artifact_type": "pdf",
            "preview_text": "Preview of artifact",
        }
        for aid, score in zip(artifact_ids, scores)
    ]


def make_image_score_results(image_ids: list[str]) -> list[dict[str, Any]]:
    """Create mock VLM image scoring results."""
    return [
        {
            "id": iid,
            "ocr_text": "",
            "ocr_mdsplus_paths": [],
            "ocr_imas_paths": [],
            "ocr_ppf_paths": [],
            "ocr_tool_mentions": [],
            "purpose": "data_plot",
            "description": f"Diagram showing {iid.split(':')[-1]}",
            "score": 0.75,
            "score_data_documentation": 0.7,
            "score_physics_content": 0.8,
            "score_code_documentation": 0.1,
            "score_data_access": 0.3,
            "score_calibration": 0.2,
            "score_imas_relevance": 0.4,
            "reasoning": "Contains physics data plot",
            "keywords": ["equilibrium", "flux"],
            "physics_domain": "equilibrium",
            "should_ingest": True,
            "skip_reason": None,
            "score_cost": 0.005,
        }
        for iid in image_ids
    ]


# =============================================================================
# Mock GraphClient
# =============================================================================


@pytest.fixture
def mock_graph_client():
    """Mock GraphClient that returns empty results by default.

    Use mock_graph_client.query.side_effect or .return_value to set responses.
    """
    mock_gc = MagicMock()
    mock_gc.__enter__ = MagicMock(return_value=mock_gc)
    mock_gc.__exit__ = MagicMock(return_value=False)
    mock_gc.query = MagicMock(return_value=[])

    with patch("imas_codex.discovery.wiki.graph_ops.GraphClient", return_value=mock_gc):
        yield mock_gc


@pytest.fixture
def mock_graph_ops():
    """Mock all graph_ops functions used by workers.

    Returns a namespace with all mocked functions for easy assertion.
    """
    patches = {}
    funcs = [
        "claim_pages_for_scoring",
        "claim_pages_for_ingesting",
        "claim_artifacts_for_scoring",
        "claim_artifacts_for_ingesting",
        "claim_images_for_scoring",
        "mark_pages_scored",
        "mark_pages_ingested",
        "mark_page_failed",
        "mark_artifacts_scored",
        "mark_artifacts_ingested",
        "mark_artifact_failed",
        "mark_artifact_deferred",
        "mark_images_scored",
        "_release_claimed_pages",
        "_release_claimed_images",
        "has_pending_work",
        "has_pending_artifact_work",
        "has_pending_artifact_score_work",
        "has_pending_artifact_ingest_work",
        "has_pending_image_work",
        "reset_transient_pages",
    ]

    mock_ns = MagicMock()
    active_patches = []

    for func_name in funcs:
        p = patch(
            f"imas_codex.discovery.wiki.graph_ops.{func_name}",
            return_value=[] if "claim" in func_name else 0,
        )
        mock_func = p.start()
        active_patches.append(p)
        setattr(mock_ns, func_name, mock_func)

    yield mock_ns

    for p in active_patches:
        p.stop()


# =============================================================================
# Mock WikiDiscoveryState
# =============================================================================


@pytest.fixture
def mock_state():
    """Create a MockWikiDiscoveryState with sensible defaults.

    Attributes can be overridden in individual tests:
        state = mock_state
        state.facility = "jet"
        state.auth_type = "basic"
    """
    state = MagicMock()
    state.facility = "tcv"
    state.site_type = "mediawiki"
    state.base_url = "https://wiki.example.com/wiki"
    state.ssh_host = None
    state.auth_type = "tequila"
    state.credential_service = "tcv"
    state.focus = None
    state.cost_limit = 10.0
    state.page_limit = None
    state.deadline = None
    state.service_monitor = None

    # Worker stats
    state.score_stats = MockWorkerStats()
    state.ingest_stats = MockWorkerStats()
    state.artifact_stats = MockWorkerStats()
    state.artifact_score_stats = MockWorkerStats()
    state.image_stats = MockWorkerStats()

    # Idle counts
    state.score_idle_count = 0
    state.ingest_idle_count = 0
    state.artifact_idle_count = 0
    state.artifact_score_idle_count = 0
    state.image_idle_count = 0

    # should_stop methods — all return False by default (run once)
    _stop_counter = {"score": 0, "ingest": 0, "artifact": 0, "ascore": 0, "image": 0}

    def _make_stop_fn(key: str, max_iterations: int = 1):
        """Return a function that returns False for max_iterations then True."""

        def fn():
            _stop_counter[key] += 1
            return _stop_counter[key] > max_iterations

        return fn

    state.should_stop_scoring = MagicMock(side_effect=_make_stop_fn("score"))
    state.should_stop_ingesting = MagicMock(side_effect=_make_stop_fn("ingest"))
    state.should_stop_artifact_worker = MagicMock(side_effect=_make_stop_fn("artifact"))
    state.should_stop_artifact_scoring = MagicMock(side_effect=_make_stop_fn("ascore"))
    state.should_stop_image_scoring = MagicMock(side_effect=_make_stop_fn("image"))

    # Semaphores
    state.effective_fetch_semaphore = MagicMock()
    state.effective_fetch_semaphore.__aenter__ = AsyncMock()
    state.effective_fetch_semaphore.__aexit__ = AsyncMock()

    # Async client getters
    state.get_async_wiki_client = AsyncMock(return_value=None)
    state.get_confluence_client = AsyncMock(return_value=None)
    state.get_keycloak_client = AsyncMock(return_value=None)
    state.get_basic_auth_client = AsyncMock(return_value=None)

    return state


# =============================================================================
# Mock SSH / Subprocess
# =============================================================================


@pytest.fixture
def mock_subprocess_run():
    """Mock subprocess.run for SSH commands.

    Returns a mock that can be configured per test:
        mock_subprocess_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=b"output", stderr=b""
        )
    """
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=b"",
            stderr=b"",
        )
        yield mock_run


# =============================================================================
# Mock LLM / VLM API
# =============================================================================


@pytest.fixture
def mock_litellm():
    """Mock litellm.acompletion for LLM/VLM scoring.

    The response object mimics OpenRouter's response format with
    cost tracking in _hidden_params.
    """
    mock_response = MagicMock()
    mock_response.usage.prompt_tokens = 1000
    mock_response.usage.completion_tokens = 500
    mock_response._hidden_params = {"response_cost": 0.005}
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "{}"

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_completion:
        mock_completion.return_value = mock_response
        yield mock_completion


# =============================================================================
# Sample Facility Configs
# =============================================================================


SAMPLE_FACILITY_CONFIG = {
    "ssh_host": "tcv",
    "wiki_sites": [
        {
            "url": "https://wiki.epfl.ch/spc-wiki",
            "site_type": "mediawiki",
            "auth_type": "tequila",
            "credential_service": "tcv",
            "description": "SPC Wiki (MediaWiki)",
            "portal_page": "Main_Page",
        },
    ],
    "data_access_patterns": {
        "primary_method": "mdsplus",
        "key_tools": ["MdsOpen", "MdsValue", "TdiExecute"],
        "code_import_patterns": ["import MDSplus"],
    },
}

SAMPLE_MULTI_SITE_CONFIG = {
    "ssh_host": "jt60sa",
    "wiki_sites": [
        {
            "url": "https://wiki.jt60sa.org/twiki",
            "site_type": "twiki_static",
            "auth_type": None,
            "description": "Main TWiki",
            "portal_page": "Main/WebHome",
        },
        {
            "url": "https://docs.jt60sa.org",
            "site_type": "static_html",
            "auth_type": None,
            "description": "Static documentation",
        },
    ],
}

SAMPLE_CONFLUENCE_CONFIG = {
    "ssh_host": "iter",
    "wiki_sites": [
        {
            "url": "https://confluence.iter.org",
            "site_type": "confluence",
            "auth_type": "session",
            "credential_service": "iter",
            "description": "ITER Confluence",
            "portal_page": "IMP",
        },
    ],
}
