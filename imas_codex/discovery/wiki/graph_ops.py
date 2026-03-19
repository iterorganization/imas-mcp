"""Graph operations for wiki discovery.

Neo4j graph helpers for wiki page and document lifecycle management:
- Bulk node creation (WikiPage, Document)
- Work claiming with claim_token pattern
- Status transitions (mark scored, ingested, failed)
- Orphan recovery and transient reset
- Pending work queries for worker coordination

All functions use the retry_on_deadlock decorator for Neo4j transient errors.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from imas_codex.discovery.base.claims import retry_on_deadlock
from imas_codex.graph import GraphClient
from imas_codex.graph.models import DocumentStatus, WikiPageStatus

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# Claim timeout - pages claimed longer than this are reclaimed
CLAIM_TIMEOUT_SECONDS = 300  # 5 minutes

# Retry configuration for Neo4j transient errors (deadlocks)
MAX_RETRY_ATTEMPTS = 5
RETRY_BASE_DELAY = 0.1  # seconds
RETRY_MAX_DELAY = 2.0  # seconds

# Document type classification using semantic names (matching DocumentType enum).
# Adapters produce these names; all downstream code uses them consistently.

# Types we can extract full text from for chunking and embedding
INGESTABLE_DOCUMENT_TYPES = {
    "pdf",
    "text_document",
    "presentation",
    "spreadsheet",
    "notebook",
    "json",
}

# Types that should become Image nodes (routed to image pipeline, not text extraction).
# Used by docs_worker for routing; Cypher queries use score_exempt property instead.
IMAGE_DOCUMENT_TYPES = {"image"}

# All types worth scoring via LLM (metadata-only scoring for data/archive/other).
# Image documents are excluded — they are score_exempt and bypass LLM text scoring,
# going directly from discovered → ingested via the VLM captioning pipeline.
SCORABLE_DOCUMENT_TYPES = INGESTABLE_DOCUMENT_TYPES | {
    "data",
    "archive",
    "other",
}


def _document_url_filter(base_url: str | None) -> str:
    """Build a Cypher URL filter for document queries.

    Matches documents by linked WikiPage URL OR by the document's own URL.
    Bulk-discovered documents may not have a HAS_DOCUMENT relationship to
    any WikiPage, so filtering only on linked page URLs leaves them orphaned.

    Uses scheme-agnostic comparison because some documents were discovered
    with ``http://`` URLs while site base URLs use ``https://``.
    """
    if not base_url:
        return ""
    # Build both http:// and https:// variants so STARTS WITH catches either scheme
    return (
        "AND (wa.url STARTS WITH $base_url OR wa.url STARTS WITH $base_url_alt"
        " OR EXISTS { MATCH (wa)<-[:HAS_DOCUMENT]-(wp:WikiPage)"
        " WHERE wp.url STARTS WITH $base_url })"
    )


def _base_url_params(base_url: str | None) -> dict[str, str]:
    """Return Cypher parameters for scheme-agnostic URL matching.

    Produces ``base_url`` (as-is) and ``base_url_alt`` (opposite scheme)
    so that ``STARTS WITH`` filters match documents regardless of whether
    they were discovered with ``http://`` or ``https://``.

    Ensures trailing ``/`` on both variants to prevent prefix collisions
    (e.g. ``tf`` matching ``tfe``).
    """
    if not base_url:
        return {}
    url = base_url.rstrip("/") + "/"
    if url.startswith("https://"):
        alt = "http://" + url[len("https://") :]
    elif url.startswith("http://"):
        alt = "https://" + url[len("http://") :]
    else:
        alt = url
    return {"base_url": url, "base_url_alt": alt}


_SOURCE_TYPE_NORMALIZATION: dict[str, str] = {
    "twiki_static": "twiki",
    "twiki_raw": "twiki",
    "static_html": "generic_html",
}


def create_doc_source(
    gc: GraphClient,
    facility: str,
    *,
    name: str,
    url: str,
    source_type: str = "wiki",
    auth_type: str = "none",
) -> str:
    """Create or update a DocSource node and return its ID.

    DocSource ID format: ``{facility}:{name}`` where name is the short site
    identifier (e.g. ``jet:wiki``, ``jet:efda-wiki``).
    """
    source_type = _SOURCE_TYPE_NORMALIZATION.get(source_type, source_type)
    doc_source_id = f"{facility}:{name}"
    gc.query(
        """
        MERGE (ds:DocSource {id: $id})
        ON CREATE SET ds.name = $name,
                      ds.url = $url,
                      ds.source_type = $source_type,
                      ds.auth_type = $auth_type,
                      ds.facility_id = $facility,
                      ds.status = 'active',
                      ds.created_at = datetime()
        ON MATCH SET ds.url = $url,
                     ds.source_type = $source_type,
                     ds.auth_type = $auth_type
        WITH ds
        MATCH (f:Facility {id: $facility})
        MERGE (ds)-[:AT_FACILITY]->(f)
        """,
        id=doc_source_id,
        name=name,
        url=url,
        source_type=source_type,
        auth_type=auth_type,
        facility=facility,
    )
    return doc_source_id


def _bulk_create_wiki_pages(
    gc: GraphClient,
    facility: str,
    batch_data: list[dict],
    *,
    batch_size: int = 500,
    on_progress: Callable | None = None,
) -> int:
    """Create WikiPage nodes with AT_FACILITY and FROM_SOURCE relationships.

    Every WikiPage gets a [:AT_FACILITY]->(:Facility) relationship at creation
    time, not deferred to ingestion. If batch items contain ``doc_source_id``,
    a [:FROM_SOURCE]->(:DocSource) relationship is also created.

    Args:
        gc: Open GraphClient
        facility: Facility ID
        batch_data: List of dicts with id, title, url keys
            (optionally doc_source_id, site_type, auth_type)
        batch_size: Nodes per batch
        on_progress: Optional progress callback

    Returns:
        Number of pages created/updated
    """
    created = 0
    total = len(batch_data)
    for i in range(0, total, batch_size):
        batch = batch_data[i : i + batch_size]
        result = gc.query(
            """
            UNWIND $pages AS page
            MERGE (wp:WikiPage {id: page.id})
            ON CREATE SET wp.title = page.title,
                          wp.url = page.url,
                          wp.facility_id = $facility,
                          wp.status = $scanned,
                          wp.link_depth = 1,
                          wp.discovered_at = datetime(),
                          wp.bulk_discovered = true,
                          wp.doc_source_id = page.doc_source_id,
                          wp.site_type = page.site_type,
                          wp.auth_type = page.auth_type,
                          wp.content_language = page.content_language
            ON MATCH SET wp.bulk_discovered = true,
                         wp.url = page.url,
                         wp.title = page.title,
                         wp.doc_source_id = coalesce(page.doc_source_id, wp.doc_source_id),
                         wp.site_type = coalesce(page.site_type, wp.site_type),
                         wp.auth_type = coalesce(page.auth_type, wp.auth_type),
                         wp.content_language = coalesce(page.content_language, wp.content_language)
            WITH wp
            MATCH (f:Facility {id: $facility})
            MERGE (wp)-[:AT_FACILITY]->(f)
            WITH wp
            WHERE wp.doc_source_id IS NOT NULL
            MATCH (ds:DocSource {id: wp.doc_source_id})
            MERGE (wp)-[:FROM_SOURCE]->(ds)
            WITH count(wp) AS dummy
            RETURN dummy + 0 AS count
            """,
            pages=batch,
            facility=facility,
            scanned=WikiPageStatus.scanned.value,
        )
        if result:
            created += result[0]["count"]

        if on_progress:
            on_progress(f"creating pages ({i + len(batch)}/{total})", None)

    return created


def _bulk_create_wiki_documents(
    gc: GraphClient,
    facility: str,
    batch_data: list[dict],
    *,
    batch_size: int = 500,
    on_progress: Callable | None = None,
) -> int:
    """Create Document nodes with AT_FACILITY, FROM_SOURCE, and HAS_DOCUMENT relationships.

    Args:
        gc: Open GraphClient
        facility: Facility ID
        batch_data: List of dicts with id, filename, url, document_type,
                    and optionally size_bytes, mime_type, linked_pages, doc_source_id
        batch_size: Nodes per batch
        on_progress: Optional progress callback

    Returns:
        Number of documents created/updated
    """
    # Pre-compute score_exempt flag for each document.
    # Image documents bypass LLM text scoring and go directly to ingestion
    # (discovered → ingested) via the VLM captioning pipeline.
    for a in batch_data:
        a.setdefault(
            "score_exempt", a.get("document_type", "").lower() in IMAGE_DOCUMENT_TYPES
        )

    created = 0
    total = len(batch_data)
    for i in range(0, total, batch_size):
        batch = batch_data[i : i + batch_size]
        result = gc.query(
            """
            UNWIND $documents AS a
            MERGE (wa:Document {id: a.id})
            ON CREATE SET wa.facility_id = $facility,
                          wa.filename = a.filename,
                          wa.url = a.url,
                          wa.document_type = a.document_type,
                          wa.size_bytes = a.size_bytes,
                          wa.mime_type = a.mime_type,
                          wa.status = $discovered,
                          wa.score_exempt = a.score_exempt,
                          wa.discovered_at = datetime(),
                          wa.bulk_discovered = true,
                          wa.doc_source_id = a.doc_source_id
            ON MATCH SET wa.bulk_discovered = true,
                         wa.doc_source_id = coalesce(a.doc_source_id, wa.doc_source_id)
            WITH wa
            MATCH (f:Facility {id: $facility})
            MERGE (wa)-[:AT_FACILITY]->(f)
            WITH wa
            WHERE wa.doc_source_id IS NOT NULL
            MATCH (ds:DocSource {id: wa.doc_source_id})
            MERGE (wa)-[:FROM_SOURCE]->(ds)
            WITH count(wa) AS dummy
            RETURN dummy + 0 AS count
            """,
            documents=batch,
            facility=facility,
            discovered=DocumentStatus.discovered.value,
        )
        if result:
            created += result[0]["count"]

        if on_progress:
            on_progress(f"created {i + len(batch)}/{total} documents", None)

    # Create HAS_DOCUMENT relationships from linked pages
    # This is done in a separate pass to handle pages that may not exist yet
    page_links = []
    for a in batch_data:
        document_id = a["id"]
        for page_name in a.get("linked_pages", []):
            page_id = f"{facility}:{page_name}"
            page_links.append({"document_id": document_id, "page_id": page_id})

    if page_links:
        # Link documents to their parent pages
        # Use MERGE for WikiPage to handle cases where the page hasn't been
        # fully ingested yet — creates a stub node that will be enriched later
        # when the page is actually crawled and ingested.
        # Always set facility_id and AT_FACILITY so referential integrity holds.
        gc.query(
            """
            UNWIND $links AS link
            MATCH (wa:Document {id: link.document_id})
            MERGE (wp:WikiPage {id: link.page_id})
            ON CREATE SET wp.facility_id = $facility,
                          wp.status = 'scanned',
                          wp.title = link.page_id,
                          wp.url = '',
                          wp.discovered_at = datetime()
            MERGE (wp)-[:HAS_DOCUMENT]->(wa)
            WITH wp
            MATCH (f:Facility {id: $facility})
            MERGE (wp)-[:AT_FACILITY]->(f)
            """,
            links=page_links,
            facility=facility,
        )

    return created


def save_page_file_references(page_id: str, filenames: list[str]) -> None:
    """Store file references on a WikiPage node for later HAS_DOCUMENT linking.

    Called during page ingestion when HTML contains file/document references.
    The DOC phase uses these references to create HAS_DOCUMENT relationships
    after Document nodes are created.
    """
    with GraphClient() as gc:
        gc.query(
            """
            MATCH (wp:WikiPage {id: $page_id})
            SET wp.referenced_files = $filenames
            """,
            page_id=page_id,
            filenames=filenames,
        )


def link_documents_from_page_refs(
    gc: GraphClient,
    facility: str,
    on_progress: Callable | None = None,
) -> int:
    """Create HAS_DOCUMENT relationships from stored WikiPage file references.

    After document discovery creates Document nodes, this function
    matches them against file references extracted during page ingestion
    (stored in WikiPage.referenced_files). This provides document-page
    linking that works regardless of whether the MediaWiki API supports
    prop=fileusage.

    Returns:
        Number of HAS_DOCUMENT links created
    """
    result = gc.query(
        """
        MATCH (wp:WikiPage {facility_id: $facility})
        WHERE wp.referenced_files IS NOT NULL
        UNWIND wp.referenced_files AS fn
        MATCH (wa:Document {facility_id: $facility})
        WHERE wa.filename = fn
        MERGE (wp)-[:HAS_DOCUMENT]->(wa)
        RETURN count(*) AS created
        """,
        facility=facility,
    )
    created = result[0]["created"] if result else 0
    if created > 0:
        logger.info("Created %d HAS_DOCUMENT links from page file references", created)
    if on_progress:
        on_progress(f"linked {created} documents from page refs", None)
    return created


def has_pending_document_work(facility: str, *, base_url: str | None = None) -> bool:
    """Check if there are pending documents (scoring or ingestion).

    Documents are pending when:
    - status = 'discovered' (needs LLM scoring or direct ingestion)
    - status = 'scored' AND score >= 0.5 AND ingestable type (needs ingestion)

    Uses ``claimed_at`` filter for multi-worker coordination.
    When ``base_url`` is provided, matches documents by linked page URL
    OR by the document's own URL (for bulk-discovered unlinked documents).
    """
    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"
    url_filter = _document_url_filter(base_url)
    with GraphClient() as gc:
        params: dict = {
            "facility": facility,
            "discovered": DocumentStatus.discovered.value,
            "scored": DocumentStatus.scored.value,
            "ingestable": list(INGESTABLE_DOCUMENT_TYPES),
            "cutoff": cutoff,
        }
        params.update(_base_url_params(base_url))
        result = gc.query(
            f"""
            MATCH (wa:Document {{facility_id: $facility}})
            WHERE ((wa.status = $discovered)
               OR (wa.status = $scored AND wa.score_composite >= 0.5
                   AND wa.document_type IN $ingestable))
              AND (wa.claimed_at IS NULL
                   OR wa.claimed_at < datetime() - duration($cutoff))
              {url_filter}
            RETURN count(wa) AS pending
            """,
            **params,
        )
        return result[0]["pending"] > 0 if result else False


def has_pending_document_score_work(
    facility: str, *, base_url: str | None = None
) -> bool:
    """Check if there are discovered documents awaiting scoring.

    Returns True if there are documents with:
    - status = 'discovered'
    - document_type in SCORABLE_DOCUMENT_TYPES

    Uses ``claimed_at`` filter for multi-worker coordination.
    When ``base_url`` is provided, matches documents by linked page URL
    OR by the document's own URL (for bulk-discovered unlinked documents).
    """
    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"
    url_filter = _document_url_filter(base_url)
    with GraphClient() as gc:
        params: dict = {
            "facility": facility,
            "discovered": DocumentStatus.discovered.value,
            "types": list(SCORABLE_DOCUMENT_TYPES),
            "cutoff": cutoff,
        }
        params.update(_base_url_params(base_url))
        result = gc.query(
            f"""
            MATCH (wa:Document {{facility_id: $facility}})
            WHERE wa.status = $discovered
              AND wa.document_type IN $types
              AND (wa.claimed_at IS NULL
                   OR wa.claimed_at < datetime() - duration($cutoff))
              {url_filter}
            RETURN count(wa) AS pending
            """,
            **params,
        )
        return result[0]["pending"] > 0 if result else False


def has_pending_document_ingest_work(
    facility: str, *, base_url: str | None = None
) -> bool:
    """Check if there are documents awaiting ingestion.

    Returns True if there are documents with:
    - status = 'scored' AND score >= 0.5 AND ingestable type
    - status = 'discovered' AND score_exempt = true (bypass LLM scoring)

    Uses ``claimed_at`` filter for multi-worker coordination.
    When ``base_url`` is provided, matches documents by linked page URL
    OR by the document's own URL (for bulk-discovered unlinked documents).
    """
    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"
    url_filter = _document_url_filter(base_url)
    with GraphClient() as gc:
        params: dict = {
            "facility": facility,
            "scored": DocumentStatus.scored.value,
            "discovered": DocumentStatus.discovered.value,
            "types": list(INGESTABLE_DOCUMENT_TYPES),
            "cutoff": cutoff,
        }
        params.update(_base_url_params(base_url))
        result = gc.query(
            f"""
            MATCH (wa:Document {{facility_id: $facility}})
            WHERE (
                (wa.status = $scored AND wa.score_composite >= 0.5
                 AND wa.document_type IN $types)
                OR (wa.status = $discovered AND wa.score_exempt = true)
              )
              AND (wa.claimed_at IS NULL
                   OR wa.claimed_at < datetime() - duration($cutoff))
              {url_filter}
            RETURN count(wa) AS pending
            """,
            **params,
        )
        return result[0]["pending"] > 0 if result else False


def has_active_claims(facility: str, *, base_url: str | None = None) -> bool:
    """Check if any pages have active (non-orphaned) claims.

    Returns True if workers are currently processing pages. Used by
    the stop condition to avoid terminating while in-flight work exists.
    ``has_pending_work`` deliberately excludes claimed pages, so a
    separate check is needed to detect work that's in progress but not
    yet complete.
    """
    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"
    url_filter = "AND wp.url STARTS WITH $base_url" if base_url else ""
    with GraphClient() as gc:
        params: dict = {
            "facility": facility,
            "cutoff": cutoff,
        }
        if base_url:
            params["base_url"] = base_url
        result = gc.query(
            f"""
            MATCH (wp:WikiPage {{facility_id: $facility}})
            WHERE wp.claimed_at IS NOT NULL
              AND wp.claimed_at >= datetime() - duration($cutoff)
              {url_filter}
            RETURN count(wp) AS active
            """,
            **params,
        )
        return result[0]["active"] > 0 if result else False


def has_pending_work(facility: str, *, base_url: str | None = None) -> bool:
    """Check if there's pending wiki work in the graph.

    Work exists if there are:
    - scanned pages awaiting scoring (unclaimed or orphaned)
    - scored pages with score >= 0.5 awaiting ingest (unclaimed or orphaned)

    The ``claimed_at`` filter is essential for multi-worker coordination:
    it prevents a stop-condition from counting in-flight pages as
    "pending", which would cause workers to spin idle while other
    workers are actively processing those pages.  Orphaned claims
    (older than the timeout) are included as reclaimable work.

    When ``base_url`` is provided, only counts pages whose URL starts
    with the given prefix.  This scopes queries to a single wiki site
    in multi-site mode.
    """
    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"
    url_filter = "AND wp.url STARTS WITH $base_url" if base_url else ""
    with GraphClient() as gc:
        params: dict = {
            "facility": facility,
            "scanned": WikiPageStatus.scanned.value,
            "scored": WikiPageStatus.scored.value,
            "cutoff": cutoff,
        }
        if base_url:
            params["base_url"] = base_url
        result = gc.query(
            f"""
            MATCH (wp:WikiPage {{facility_id: $facility}})
            WHERE (wp.status = $scanned
                   OR (wp.status = $scored AND wp.score_composite >= 0.5))
              AND (wp.claimed_at IS NULL
                   OR wp.claimed_at < datetime() - duration($cutoff))
              {url_filter}
            RETURN count(wp) AS pending
            """,
            **params,
        )
        return result[0]["pending"] > 0 if result else False


def has_pending_scan_work(facility: str, *, base_url: str | None = None) -> bool:
    """Check if there's pending scoring work in the graph.

    With bulk discovery, there's no scan phase. This checks for:
    - scanned pages awaiting content-aware scoring (unclaimed or orphaned)

    When ``base_url`` is provided, only counts pages matching the site
    URL prefix.
    """
    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"
    url_filter = "AND wp.url STARTS WITH $base_url" if base_url else ""
    with GraphClient() as gc:
        params: dict = {
            "facility": facility,
            "scanned": WikiPageStatus.scanned.value,
            "cutoff": cutoff,
        }
        if base_url:
            params["base_url"] = base_url
        result = gc.query(
            f"""
            MATCH (wp:WikiPage {{facility_id: $facility}})
            WHERE wp.status = $scanned
              AND (wp.claimed_at IS NULL
                   OR wp.claimed_at < datetime() - duration($cutoff))
              {url_filter}
            RETURN count(wp) AS pending
            """,
            **params,
        )
        return result[0]["pending"] > 0 if result else False


def has_pending_ingest_work(facility: str, *, base_url: str | None = None) -> bool:
    """Check if there's pending ingest work in the graph.

    Returns True if there are scored pages with score >= 0.3 awaiting
    ingestion (unclaimed or orphaned).

    When ``base_url`` is provided, only counts pages matching the site
    URL prefix.
    """
    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"
    url_filter = "AND wp.url STARTS WITH $base_url" if base_url else ""
    with GraphClient() as gc:
        params: dict = {
            "facility": facility,
            "scored": WikiPageStatus.scored.value,
            "cutoff": cutoff,
        }
        if base_url:
            params["base_url"] = base_url
        result = gc.query(
            f"""
            MATCH (wp:WikiPage {{facility_id: $facility}})
            WHERE wp.status = $scored
              AND wp.score_composite >= 0.3
              AND (wp.claimed_at IS NULL
                   OR wp.claimed_at < datetime() - duration($cutoff))
              {url_filter}
            RETURN count(wp) AS pending
            """,
            **params,
        )
        return result[0]["pending"] > 0 if result else False


# =============================================================================
# Orphan Recovery (timeout-based, safe for parallel instances)
# =============================================================================


def reset_transient_pages(
    facility: str, *, silent: bool = False, force: bool = False
) -> dict[str, int]:
    """Release stale claims older than CLAIM_TIMEOUT_SECONDS.

    Uses timeout-based recovery so multiple CLI instances can run
    concurrently on the same facility without wiping each other's claims.
    Only claims older than the timeout are considered orphaned.

    When ``force=True`` (startup), also resets ``fetch_retries`` on
    scanned pages so pages that accumulated retries from previous runs
    get a fresh chance.  Without this, scanned pages that transiently
    failed to fetch would eventually exceed ``max_fetch_retries`` and
    be permanently marked as failed.

    Delegates to the common claims module for both WikiPage and
    Document node types.

    Args:
        facility: Facility ID
        silent: Suppress logging
        force: Clear ALL claims regardless of age (use at startup)
    """
    from imas_codex.discovery.base.claims import reset_stale_claims

    timeout = 0 if force else CLAIM_TIMEOUT_SECONDS

    page_reset = reset_stale_claims(
        "WikiPage",
        facility,
        timeout_seconds=timeout,
        silent=True,  # We log our own combined message
    )

    document_reset = reset_stale_claims(
        "Document",
        facility,
        timeout_seconds=timeout,
        silent=True,
    )

    # At startup, reset fetch_retries on scanned pages to prevent
    # accumulation across runs.  Only scanned pages are affected —
    # failed pages are handled by recover_failed_pages() separately.
    retries_reset = 0
    if force:
        try:
            with GraphClient() as gc:
                result = gc.query(
                    """
                    MATCH (wp:WikiPage {facility_id: $facility})
                    WHERE wp.status = $scanned
                      AND wp.fetch_retries IS NOT NULL
                      AND wp.fetch_retries > 0
                    SET wp.fetch_retries = 0
                    RETURN count(wp) AS reset
                    """,
                    facility=facility,
                    scanned=WikiPageStatus.scanned.value,
                )
                retries_reset = result[0]["reset"] if result else 0
        except Exception as e:
            logger.debug("Could not reset fetch_retries: %s", e)

    total_reset = page_reset + document_reset
    if not silent and total_reset > 0:
        logger.info(
            "Released %d orphaned claims%s (%d pages, %d documents)",
            total_reset,
            "" if force else f" older than {CLAIM_TIMEOUT_SECONDS}s",
            page_reset,
            document_reset,
        )
    if not silent and retries_reset > 0:
        logger.info(
            "Reset fetch_retries on %d scanned pages",
            retries_reset,
        )

    return {"orphan_reset": page_reset, "document_reset": document_reset}


# =============================================================================
# Graph-based Work Claiming (uses claimed_at for coordination)
# =============================================================================


@retry_on_deadlock()
def claim_pages_for_scoring(
    facility: str, limit: int = 50, *, base_url: str | None = None
) -> list[dict[str, Any]]:
    """Claim scanned pages for content-aware scoring.

    Workflow: scanned + unclaimed → set claimed_at
    Score worker fetches content and scores in single pass.
    After scoring: update status to 'scored' and set score field.

    Uses claim token pattern to handle race conditions between workers:
    1. Generate unique claim token
    2. Atomically SET token on unclaimed pages
    3. Read back only pages with OUR token (pages we actually won)

    When ``base_url`` is provided, only claims pages whose URL starts
    with the given prefix (site-scoped in multi-site mode).
    """
    import uuid

    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"  # ISO 8601 duration
    claim_token = str(uuid.uuid4())
    url_filter = "AND wp.url STARTS WITH $base_url" if base_url else ""

    with GraphClient() as gc:
        params: dict = {
            "facility": facility,
            "scanned": WikiPageStatus.scanned.value,
            "cutoff": cutoff,
            "limit": limit,
            "token": claim_token,
        }
        if base_url:
            params["base_url"] = base_url
        # Step 1: Attempt to claim pages with our unique token
        # Using random() in ORDER BY reduces collision probability further
        gc.query(
            f"""
            MATCH (wp:WikiPage {{facility_id: $facility}})
            WHERE wp.status = $scanned
              AND (wp.claimed_at IS NULL
                   OR wp.claimed_at < datetime() - duration($cutoff))
              {url_filter}
            WITH wp
            ORDER BY rand()
            LIMIT $limit
            SET wp.claimed_at = datetime(), wp.claim_token = $token
            """,
            **params,
        )

        # Step 2: Read back only pages WE successfully claimed
        # If another worker raced us, they have a different token
        result = gc.query(
            """
            MATCH (wp:WikiPage {facility_id: $facility, claim_token: $token})
            RETURN wp.id AS id, wp.title AS title, wp.url AS url,
                   wp.content_language AS content_language
            """,
            facility=facility,
            token=claim_token,
        )
        claimed = list(result)

        logger.debug(
            "claim_pages_for_scoring: requested %d, won %d (token=%s)",
            limit,
            len(claimed),
            claim_token[:8],
        )
        return claimed


@retry_on_deadlock()
def claim_pages_for_ingesting(
    facility: str,
    min_score: float = 0.3,
    limit: int = 10,
    *,
    base_url: str | None = None,
) -> list[dict[str, Any]]:
    """Claim scored pages for ingestion (chunking and embedding).

    Workflow: scored + score >= min_score + unclaimed → set claimed_at
    After ingest: update status to 'ingested'.

    Uses claim token pattern to handle race conditions between workers.
    When ``base_url`` is provided, only claims pages matching the site
    URL prefix (site-scoped in multi-site mode).
    """
    import uuid

    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"
    claim_token = str(uuid.uuid4())
    url_filter = "AND wp.url STARTS WITH $base_url" if base_url else ""

    with GraphClient() as gc:
        params: dict = {
            "facility": facility,
            "scored": WikiPageStatus.scored.value,
            "min_score": min_score,
            "cutoff": cutoff,
            "limit": limit,
            "token": claim_token,
        }
        if base_url:
            params["base_url"] = base_url
        # Step 1: Attempt to claim pages with our unique token
        gc.query(
            f"""
            MATCH (wp:WikiPage {{facility_id: $facility}})
            WHERE wp.status = $scored
              AND wp.score_composite >= $min_score
              AND (wp.claimed_at IS NULL
                   OR wp.claimed_at < datetime() - duration($cutoff))
              {url_filter}
            WITH wp
            ORDER BY rand()
            LIMIT $limit
            SET wp.claimed_at = datetime(), wp.claim_token = $token
            """,
            **params,
        )

        # Step 2: Read back only pages WE successfully claimed
        result = gc.query(
            """
            MATCH (wp:WikiPage {facility_id: $facility, claim_token: $token})
            RETURN wp.id AS id, wp.title AS title, wp.url AS url,
                   wp.score_composite AS score, wp.description AS description,
                   wp.physics_domain AS physics_domain,
                   wp.preview_text AS preview,
                   wp.in_degree AS in_degree, wp.out_degree AS out_degree
            """,
            facility=facility,
            token=claim_token,
        )
        claimed = list(result)

        logger.debug(
            "claim_pages_for_ingesting: requested %d, won %d (token=%s)",
            limit,
            len(claimed),
            claim_token[:8],
        )
        return claimed


# =============================================================================
# Mark Work Complete
# =============================================================================


def mark_pages_scored(
    facility: str,
    results: list[dict[str, Any]],
    skip_threshold: float = 0.3,
) -> int:
    """Mark pages as scored or skipped based on score threshold.

    Pages with score >= skip_threshold get status='scored' (proceed to ingest).
    Pages with score < skip_threshold get status='skipped' (filtered out).
    Pages where the LLM set should_ingest=True are always scored regardless
    of composite score.

    Uses batched UNWIND for O(1) graph operations instead of O(n) individual queries.
    """
    if not results:
        return 0

    # Prepare batch data with all scoring fields, split by threshold
    scored_batch: list[dict[str, Any]] = []
    skipped_batch: list[dict[str, Any]] = []
    # Safety: collect page IDs with empty preview_text to release for reprocessing
    released_ids: list[str] = []

    for r in results:
        page_id = r.get("id")
        if not page_id:
            continue

        # Pages without preview_text should not be marked as scored —
        # release them back to scanned so they can be retried
        if not r.get("preview_text"):
            released_ids.append(page_id)
            continue

        item = {
            "id": page_id,
            "score_composite": r.get("score_composite", 0.0),
            "purpose": r.get("purpose", r.get("page_purpose", "other")),
            "description": r.get("description") or None,
            "reasoning": r.get("reasoning", ""),
            "keywords": r.get("keywords", []),
            "physics_domain": r.get("physics_domain"),
            "preview_text": r.get("preview_text", ""),
            "score_data_documentation": r.get("score_data_documentation", 0.0),
            "score_physics_content": r.get("score_physics_content", 0.0),
            "score_code_documentation": r.get("score_code_documentation", 0.0),
            "score_data_access": r.get("score_data_access", 0.0),
            "score_calibration": r.get("score_calibration", 0.0),
            "score_imas_relevance": r.get("score_imas_relevance", 0.0),
            "should_ingest": r.get("should_ingest", True),
            "skip_reason": r.get("skip_reason"),
            "is_physics": r.get("is_physics", False),
            "score_cost": r.get("score_cost", 0.0),
        }

        if item["score_composite"] >= skip_threshold or item.get("should_ingest"):
            scored_batch.append(item)
        else:
            skipped_batch.append(item)

    if not scored_batch and not skipped_batch and not released_ids:
        return 0

    with GraphClient() as gc:
        # Cypher template shared by scored and skipped batches
        _SET_SCORING_FIELDS = """
                UNWIND $batch AS item
                MATCH (wp:WikiPage {id: item.id})
                SET wp.status = $status,
                    wp.score_composite = item.score_composite,
                    wp.purpose = item.purpose,
                    wp.description = item.description,
                    wp.reasoning = item.reasoning,
                    wp.keywords = item.keywords,
                    wp.physics_domain = item.physics_domain,
                    wp.preview_text = item.preview_text,
                    wp.score_data_documentation = item.score_data_documentation,
                    wp.score_physics_content = item.score_physics_content,
                    wp.score_code_documentation = item.score_code_documentation,
                    wp.score_data_access = item.score_data_access,
                    wp.score_calibration = item.score_calibration,
                    wp.score_imas_relevance = item.score_imas_relevance,
                    wp.should_ingest = item.should_ingest,
                    wp.skip_reason = item.skip_reason,
                    wp.is_physics_content = item.is_physics,
                    wp.score_cost = item.score_cost,
                    wp.scored_at = datetime(),
                    wp.preview_fetched_at = datetime(),
                    wp.claimed_at = null
        """

        if scored_batch:
            gc.query(
                _SET_SCORING_FIELDS,
                batch=scored_batch,
                status=WikiPageStatus.scored.value,
            )

        if skipped_batch:
            gc.query(
                _SET_SCORING_FIELDS,
                batch=skipped_batch,
                status=WikiPageStatus.skipped.value,
            )
            logger.info(
                "mark_pages_scored: %d pages skipped (score < %.1f)",
                len(skipped_batch),
                skip_threshold,
            )

        # Release pages with empty preview_text back to scanned for reprocessing
        if released_ids:
            logger.info(
                "mark_pages_scored: releasing %d pages with empty preview_text "
                "back to scanned for retry",
                len(released_ids),
            )
            gc.query(
                """
                UNWIND $ids AS id
                MATCH (wp:WikiPage {id: id})
                SET wp.claimed_at = null
                """,
                ids=released_ids,
            )

    return len(scored_batch) + len(skipped_batch)


def mark_pages_ingested(
    facility: str,
    results: list[dict[str, Any]],
) -> int:
    """Mark pages as ingested with chunk data.

    Uses batched UNWIND for O(1) graph operations instead of O(n) individual queries.
    Does NOT overwrite chunk_count — the pipeline sets it when persisting chunks.
    """
    if not results:
        return 0

    # Prepare batch data
    batch_data = [{"id": r.get("id")} for r in results if r.get("id")]

    if not batch_data:
        return 0

    with GraphClient() as gc:
        gc.query(
            """
            UNWIND $batch AS item
            MATCH (wp:WikiPage {id: item.id})
            SET wp.status = $ingested,
                wp.ingested_at = datetime(),
                wp.claimed_at = null
            """,
            batch=batch_data,
            ingested=WikiPageStatus.ingested.value,
        )

    return len(batch_data)


def mark_page_failed(page_id: str, error: str, fallback_status: str) -> None:
    """Mark a page as failed with error message.

    If Neo4j is unavailable, logs a warning and silently fails.
    The page will be reclaimed after CLAIM_TIMEOUT_SECONDS.
    """
    try:
        with GraphClient() as gc:
            gc.query(
                """
                MATCH (wp:WikiPage {id: $id})
                SET wp.status = $status,
                    wp.error = $error,
                    wp.failed_at = datetime(),
                    wp.claimed_at = null
                """,
                id=page_id,
                status=fallback_status,
                error=error,
            )
    except Exception as e:
        logger.warning(
            "Could not mark page %s as failed (Neo4j unavailable): %s", page_id, e
        )


def mark_pages_insufficient_content(page_ids: list[str]) -> int:
    """Mark pages as skipped due to insufficient content for chunking.

    Called by the ingest worker when a scored page is successfully fetched
    but produces 0 chunks.  This is a permanent condition (the page has
    no useful text to ingest), not a transient fetch failure.

    Sets ``status='skipped'`` with ``skip_reason='insufficient_content'``
    so these pages are not retried by ``recover_failed_pages()``.

    Returns:
        Number of pages marked as skipped.
    """
    if not page_ids:
        return 0

    try:
        with GraphClient() as gc:
            result = gc.query(
                """
                UNWIND $ids AS id
                MATCH (wp:WikiPage {id: id})
                SET wp.status = $skipped,
                    wp.skip_reason = 'insufficient_content',
                    wp.should_ingest = false,
                    wp.claimed_at = null,
                    wp.claim_token = null
                RETURN count(wp) AS cnt
                """,
                ids=page_ids,
                skipped=WikiPageStatus.skipped.value,
            )
            count = result[0]["cnt"] if result else 0
            if count:
                logger.info("Marked %d pages as skipped (insufficient content)", count)
            return count
    except Exception as e:
        logger.warning("Could not mark pages as skipped: %s", e)
        return 0


def revert_pages_to_scored(
    page_ids: list[str], *, max_ingest_retries: int = 5
) -> dict[str, int]:
    """Revert pages to scored status after an ingest-time fetch failure.

    When a page was successfully scored (has preview_text and a composite
    score) but cannot be fetched at ingest time, the failure is transient
    rather than permanent.  This function reverts such pages back to
    ``scored`` so they re-enter the ingest queue on a subsequent cycle.

    An ``ingest_retries`` counter is incremented on each call.  Pages
    that exceed *max_ingest_retries* are permanently marked as skipped
    with ``skip_reason='fetch_exhausted'`` to prevent infinite loops.

    Returns:
        Dict with ``{"reverted": N, "exhausted": M}``.
    """
    if not page_ids:
        return {"reverted": 0, "exhausted": 0}

    try:
        with GraphClient() as gc:
            # Increment ingest retry counters first.
            gc.query(
                """
                UNWIND $ids AS id
                MATCH (wp:WikiPage {id: id})
                SET wp.ingest_retries = coalesce(wp.ingest_retries, 0) + 1
                """,
                ids=page_ids,
            )

            # Revert pages still below retry cap back to scored.
            reverted_result = gc.query(
                """
                UNWIND $ids AS id
                MATCH (wp:WikiPage {id: id})
                WHERE coalesce(wp.ingest_retries, 0) < $max_retries
                SET wp.status = $scored,
                    wp.claimed_at = null,
                    wp.claim_token = null
                RETURN count(wp) AS reverted
                """,
                ids=page_ids,
                max_retries=max_ingest_retries,
                scored=WikiPageStatus.scored.value,
            )

            # Mark exhausted pages as permanently skipped to avoid infinite churn.
            exhausted_result = gc.query(
                """
                UNWIND $ids AS id
                MATCH (wp:WikiPage {id: id})
                WHERE coalesce(wp.ingest_retries, 0) >= $max_retries
                SET wp.status = $skipped,
                    wp.skip_reason = 'fetch_exhausted',
                    wp.should_ingest = false,
                    wp.claimed_at = null,
                    wp.claim_token = null
                RETURN count(wp) AS exhausted
                """,
                ids=page_ids,
                max_retries=max_ingest_retries,
                skipped=WikiPageStatus.skipped.value,
            )

            reverted = reverted_result[0]["reverted"] if reverted_result else 0
            exhausted = exhausted_result[0]["exhausted"] if exhausted_result else 0
            if reverted:
                logger.info("Reverted %d pages to scored for ingest retry", reverted)
            if exhausted:
                logger.warning(
                    "Marked %d pages as fetch_exhausted after %d ingest retries",
                    exhausted,
                    max_ingest_retries,
                )
            return {"reverted": reverted, "exhausted": exhausted}
    except Exception as e:
        logger.warning("Could not revert pages to scored: %s", e)
        return {"reverted": 0, "exhausted": 0}


def _release_claimed_pages(page_ids: list[str], *, max_fetch_retries: int = 10) -> None:
    """Release claimed pages back for reprocessing.

    Increments fetch_retries counter on each page. Pages that exceed
    max_fetch_retries are marked as 'failed' instead of released,
    preventing infinite retry loops for unfetchable pages.

    If Neo4j is unavailable, logs a warning and silently fails.
    Pages will be reclaimed after CLAIM_TIMEOUT_SECONDS anyway.
    """
    if not page_ids:
        return

    try:
        with GraphClient() as gc:
            result = gc.query(
                """
                UNWIND $ids AS id
                MATCH (wp:WikiPage {id: id})
                SET wp.fetch_retries = coalesce(wp.fetch_retries, 0) + 1
                WITH wp
                CALL {
                    WITH wp
                    WITH wp WHERE wp.fetch_retries >= $max_retries
                    SET wp.status = $failed, wp.claimed_at = null,
                        wp.claim_token = null
                    RETURN wp.id AS failed_id
                }
                WITH wp WHERE wp.fetch_retries < $max_retries
                SET wp.claimed_at = null, wp.claim_token = null
                RETURN count(wp) AS released
                """,
                ids=page_ids,
                max_retries=max_fetch_retries,
                failed=WikiPageStatus.failed.value,
            )
            released = result[0]["released"] if result else 0
            failed_count = len(page_ids) - released
            if failed_count > 0:
                logger.warning(
                    "Marked %d pages as failed after %d fetch retries",
                    failed_count,
                    max_fetch_retries,
                )
    except Exception as e:
        logger.warning(
            "Could not release %d claimed pages (Neo4j unavailable): %s",
            len(page_ids),
            e,
        )


def release_orphaned_claims(facility: str) -> dict[str, int]:
    """Release all orphaned claims for a facility.

    Orphaned claims occur when a worker crashes without releasing its claimed
    pages. This function finds all pages with claimed_at older than the timeout
    and clears the claim.

    This is automatically called by the claim functions (they skip old claims),
    but can also be called explicitly to recover from stuck state.

    Args:
        facility: Facility ID

    Returns:
        Dict with counts: {"released": N, "pages": [...]}
    """
    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"
    try:
        with GraphClient() as gc:
            # Release stale page claims
            result = gc.query(
                """
                MATCH (wp:WikiPage {facility_id: $facility})
                WHERE wp.claimed_at IS NOT NULL
                  AND wp.claimed_at < datetime() - duration($cutoff)
                SET wp.claimed_at = null
                RETURN wp.id AS id, wp.status AS status
                """,
                facility=facility,
                cutoff=cutoff,
            )
            pages = list(result)

            # Release stale document claims
            result = gc.query(
                """
                MATCH (wa:Document {facility_id: $facility})
                WHERE wa.claimed_at IS NOT NULL
                  AND wa.claimed_at < datetime() - duration($cutoff)
                SET wa.claimed_at = null
                RETURN wa.id AS id, wa.status AS status
                """,
                facility=facility,
                cutoff=cutoff,
            )
            documents = list(result)

            total = len(pages) + len(documents)
            if total > 0:
                logger.info(
                    "Released %d orphaned claims (%d pages, %d documents) for %s",
                    total,
                    len(pages),
                    len(documents),
                    facility,
                )

            return {
                "released_pages": len(pages),
                "released_documents": len(documents),
                "page_ids": [p["id"] for p in pages],
                "document_ids": [a["id"] for a in documents],
            }
    except Exception as e:
        logger.warning("Could not release orphaned claims: %s", e)
        return {"released_pages": 0, "released_documents": 0, "error": str(e)}


def recover_failed_pages(facility: str) -> int:
    """Reset failed pages for re-processing.

    Pages marked as 'failed' due to exhausted fetch_retries (transient
    fetch failures, e.g. auth session expiry) are recovered based on
    whether they were scored before failing:

    - **Without score** (failed before scoring): reset to ``scanned``
      with ``fetch_retries`` cleared so they can be re-scored.
    - **With score** (failed during ingestion): reset to ``scored``
      with ``fetch_retries`` cleared so they can be re-ingested.

    Only recovers pages that have NO error field set — pages with an
    explicit error were genuinely broken (e.g. parse errors), not
    transient failures.

    Args:
        facility: Facility ID

    Returns:
        Number of pages recovered
    """
    try:
        with GraphClient() as gc:
            # Case 1: Failed before scoring — reset to scanned
            result1 = gc.query(
                """
                MATCH (wp:WikiPage {facility_id: $facility})
                WHERE wp.status = $failed
                  AND wp.error IS NULL
                  AND wp.score_composite IS NULL
                SET wp.status = $scanned,
                    wp.fetch_retries = 0,
                    wp.claimed_at = null,
                    wp.claim_token = null,
                    wp.failed_at = null
                RETURN count(wp) AS recovered
                """,
                facility=facility,
                failed=WikiPageStatus.failed.value,
                scanned=WikiPageStatus.scanned.value,
            )
            recovered_to_scanned = result1[0]["recovered"] if result1 else 0

            # Case 2: Failed after scoring — reset to scored for re-ingestion
            result2 = gc.query(
                """
                MATCH (wp:WikiPage {facility_id: $facility})
                WHERE wp.status = $failed
                  AND wp.error IS NULL
                  AND wp.score_composite IS NOT NULL
                SET wp.status = $scored,
                    wp.fetch_retries = 0,
                    wp.claimed_at = null,
                    wp.claim_token = null,
                    wp.failed_at = null
                RETURN count(wp) AS recovered
                """,
                facility=facility,
                failed=WikiPageStatus.failed.value,
                scored=WikiPageStatus.scored.value,
            )
            recovered_to_scored = result2[0]["recovered"] if result2 else 0

            total = recovered_to_scanned + recovered_to_scored
            if total > 0:
                parts = []
                if recovered_to_scanned:
                    parts.append(f"{recovered_to_scanned} to scanned")
                if recovered_to_scored:
                    parts.append(f"{recovered_to_scored} to scored")
                logger.info(
                    "Recovered %d failed pages for %s (%s)",
                    total,
                    facility,
                    ", ".join(parts),
                )
            return total
    except Exception as e:
        logger.warning("Could not recover failed pages: %s", e)
        return 0


def recover_failed_documents(facility: str) -> int:
    """Reset failed documents for re-processing.

    Documents marked 'failed' due to transient errors (CUDA OOM, connection
    refused, etc.) are reset based on their pre-failure state:

    - **Without score** (failed before scoring): reset to ``discovered``
    - **With score** (failed during ingestion): reset to ``scored``

    Only recovers documents whose error matches known transient patterns.
    Documents with non-transient errors (parse failures, unsupported formats)
    are left as-is.

    Args:
        facility: Facility ID

    Returns:
        Number of documents recovered
    """
    # Patterns that indicate transient/retryable errors
    transient_patterns = [
        "CUDA out of memory",
        "CUDA error",
        "Connection refused",
        "Connection reset",
        "connection was reset",
        "ServiceUnavailable",
        "Failed to establish",
        "Read timed out",
        "RemoteDisconnected",
    ]
    where_clauses = " OR ".join(f"wa.error CONTAINS '{p}'" for p in transient_patterns)

    try:
        with GraphClient() as gc:
            # Case 1: Failed before scoring — reset to discovered
            result1 = gc.query(
                f"""
                MATCH (wa:Document {{facility_id: $facility}})
                WHERE wa.status = $failed
                  AND wa.score_composite IS NULL
                  AND ({where_clauses})
                SET wa.status = $discovered,
                    wa.error = null,
                    wa.failed_at = null,
                    wa.claimed_at = null
                RETURN count(wa) AS recovered
                """,
                facility=facility,
                failed=DocumentStatus.failed.value,
                discovered=DocumentStatus.discovered.value,
            )
            recovered_to_discovered = result1[0]["recovered"] if result1 else 0

            # Case 2: Failed after scoring — reset to scored for re-ingestion
            result2 = gc.query(
                f"""
                MATCH (wa:Document {{facility_id: $facility}})
                WHERE wa.status = $failed
                  AND wa.score_composite IS NOT NULL
                  AND ({where_clauses})
                SET wa.status = $scored,
                    wa.error = null,
                    wa.failed_at = null,
                    wa.claimed_at = null
                RETURN count(wa) AS recovered
                """,
                facility=facility,
                failed=DocumentStatus.failed.value,
                scored=DocumentStatus.scored.value,
            )
            recovered_to_scored = result2[0]["recovered"] if result2 else 0

            total = recovered_to_discovered + recovered_to_scored
            if total > 0:
                parts = []
                if recovered_to_discovered:
                    parts.append(f"{recovered_to_discovered} to discovered")
                if recovered_to_scored:
                    parts.append(f"{recovered_to_scored} to scored")
                logger.info(
                    "Recovered %d failed documents for %s (%s)",
                    total,
                    facility,
                    ", ".join(parts),
                )
            return total
    except Exception as e:
        logger.warning("Could not recover failed documents: %s", e)
        return 0


# =============================================================================
# Document Claim/Mark Functions
# =============================================================================


@retry_on_deadlock()
def claim_documents_for_scoring(
    facility: str,
    limit: int = 20,
    *,
    base_url: str | None = None,
) -> list[dict[str, Any]]:
    """Claim discovered documents for scoring.

    Claims documents with status='discovered' and any scorable document_type.
    Image documents are NOT scored here — they bypass LLM scoring and go
    directly to ingestion via claim_documents_for_ingesting.

    Workflow: discovered + scorable_type + unclaimed → set claimed_at
    Score worker extracts text preview and scores with LLM.
    After scoring: update status to 'scored' and set score/description/physics_domain.

    Uses claim token pattern to handle race conditions between workers.
    When ``base_url`` is provided, matches documents by linked page URL
    OR by the document's own URL (for bulk-discovered unlinked documents).
    """
    import uuid

    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"
    claim_token = str(uuid.uuid4())
    url_filter = _document_url_filter(base_url)

    with GraphClient() as gc:
        params: dict = {
            "facility": facility,
            "discovered": DocumentStatus.discovered.value,
            "types": list(SCORABLE_DOCUMENT_TYPES),
            "cutoff": cutoff,
            "limit": limit,
            "token": claim_token,
        }
        params.update(_base_url_params(base_url))
        # Step 1: Attempt to claim documents with our unique token
        gc.query(
            f"""
            MATCH (wa:Document {{facility_id: $facility}})
            WHERE wa.status = $discovered
              AND wa.document_type IN $types
              AND (wa.claimed_at IS NULL
                   OR wa.claimed_at < datetime() - duration($cutoff))
              {url_filter}
            WITH wa
            ORDER BY rand()
            LIMIT $limit
            SET wa.claimed_at = datetime(), wa.claim_token = $token
            """,
            **params,
        )

        # Step 2: Read back only documents WE successfully claimed
        result = gc.query(
            """
            MATCH (wa:Document {facility_id: $facility, claim_token: $token})
            RETURN wa.id AS id, wa.url AS url, wa.filename AS filename,
                   wa.document_type AS document_type, wa.size_bytes AS size_bytes
            """,
            facility=facility,
            token=claim_token,
        )
        claimed = list(result)

        logger.debug(
            "claim_documents_for_scoring: requested %d, won %d (token=%s)",
            limit,
            len(claimed),
            claim_token[:8],
        )
        return claimed


@retry_on_deadlock()
def claim_documents_for_ingesting(
    facility: str,
    min_score: float = 0.5,
    limit: int = 5,
    *,
    base_url: str | None = None,
) -> list[dict[str, Any]]:
    """Claim documents for ingestion.

    Claims two categories of documents:
    1. Scored text-extractable documents with score >= min_score
    2. Score-exempt documents (e.g. images) directly from discovered status

    Score-exempt documents bypass LLM text scoring and go directly to the
    ingestion pipeline, where type-specific handlers (e.g. VLM captioning
    for images) process them.

    Data/archive types are scored but never claimed for ingestion.

    Uses claim token pattern to handle race conditions between workers.
    When ``base_url`` is provided, matches documents by linked page URL
    OR by the document's own URL (for bulk-discovered unlinked documents).
    """
    import uuid

    cutoff = f"PT{CLAIM_TIMEOUT_SECONDS}S"
    claim_token = str(uuid.uuid4())
    url_filter = _document_url_filter(base_url)

    with GraphClient() as gc:
        params: dict = {
            "facility": facility,
            "scored": DocumentStatus.scored.value,
            "discovered": DocumentStatus.discovered.value,
            "min_score": min_score,
            "types": list(INGESTABLE_DOCUMENT_TYPES),
            "cutoff": cutoff,
            "limit": limit,
            "token": claim_token,
        }
        params.update(_base_url_params(base_url))
        # Step 1: Attempt to claim documents with our unique token.
        # Score-exempt documents: claim from 'discovered' (bypass LLM scoring).
        # Text documents: claim from 'scored' with score >= threshold.
        gc.query(
            f"""
            MATCH (wa:Document {{facility_id: $facility}})
            WHERE (
                (wa.status = $scored AND wa.score_composite >= $min_score
                 AND wa.document_type IN $types)
                OR (wa.status = $discovered AND wa.score_exempt = true)
              )
              AND (wa.claimed_at IS NULL
                   OR wa.claimed_at < datetime() - duration($cutoff))
              {url_filter}
            WITH wa
            ORDER BY wa.score_composite DESC, rand()
            LIMIT $limit
            SET wa.claimed_at = datetime(), wa.claim_token = $token
            """,
            **params,
        )

        # Step 2: Read back only documents WE successfully claimed
        result = gc.query(
            """
            MATCH (wa:Document {facility_id: $facility, claim_token: $token})
            RETURN wa.id AS id, wa.url AS url, wa.filename AS filename,
                   wa.document_type AS document_type, wa.score_composite AS score,
                   wa.description AS description, wa.physics_domain AS physics_domain
            """,
            facility=facility,
            token=claim_token,
        )
        claimed = list(result)

        logger.debug(
            "claim_documents_for_ingesting: requested %d, won %d (token=%s)",
            limit,
            len(claimed),
            claim_token[:8],
        )
        return claimed


def mark_documents_scored(
    facility: str,
    results: list[dict[str, Any]],
) -> int:
    """Mark documents as scored with scoring data.

    Uses batched UNWIND for O(1) graph operations.
    All documents move to 'scored' status. The ingester filters by score >= threshold.
    """
    if not results:
        return 0

    batch_data = []
    for r in results:
        document_id = r.get("id")
        if not document_id:
            continue

        batch_data.append(
            {
                "id": document_id,
                "score_composite": r.get("score_composite", 0.0),
                "document_purpose": r.get("document_purpose", "other"),
                "description": r.get("description") or None,
                "reasoning": r.get("reasoning", ""),
                "keywords": r.get("keywords", []),
                "physics_domain": r.get("physics_domain"),
                "preview_text": r.get("preview_text", "")[:500],
                "score_data_documentation": r.get("score_data_documentation", 0.0),
                "score_physics_content": r.get("score_physics_content", 0.0),
                "score_code_documentation": r.get("score_code_documentation", 0.0),
                "score_data_access": r.get("score_data_access", 0.0),
                "score_calibration": r.get("score_calibration", 0.0),
                "score_imas_relevance": r.get("score_imas_relevance", 0.0),
                "should_ingest": r.get("should_ingest", False),
                "skip_reason": r.get("skip_reason"),
                "score_cost": r.get("score_cost", 0.0),
            }
        )

    if not batch_data:
        return 0

    with GraphClient() as gc:
        gc.query(
            """
            UNWIND $batch AS item
            MATCH (wa:Document {id: item.id})
            SET wa.status = $status,
                wa.score_composite = item.score_composite,
                wa.document_purpose = item.document_purpose,
                wa.description = item.description,
                wa.reasoning = item.reasoning,
                wa.keywords = item.keywords,
                wa.physics_domain = item.physics_domain,
                wa.preview_text = item.preview_text,
                wa.score_data_documentation = item.score_data_documentation,
                wa.score_physics_content = item.score_physics_content,
                wa.score_code_documentation = item.score_code_documentation,
                wa.score_data_access = item.score_data_access,
                wa.score_calibration = item.score_calibration,
                wa.score_imas_relevance = item.score_imas_relevance,
                wa.should_ingest = item.should_ingest,
                wa.skip_reason = item.skip_reason,
                wa.score_cost = item.score_cost,
                wa.scored_at = datetime(),
                wa.claimed_at = null
            """,
            batch=batch_data,
            status=DocumentStatus.scored.value,
        )

    return len(batch_data)


def mark_documents_ingested(
    facility: str,
    results: list[dict[str, Any]],
) -> int:
    """Mark documents as ingested with chunk data.

    Uses batched UNWIND for O(1) graph operations.
    """
    if not results:
        return 0

    batch_data = [
        {"id": r.get("id"), "chunks": r.get("chunk_count", 0)}
        for r in results
        if r.get("id")
    ]

    if not batch_data:
        return 0

    with GraphClient() as gc:
        gc.query(
            """
            UNWIND $batch AS item
            MATCH (wa:Document {id: item.id})
            SET wa.status = $ingested,
                wa.chunk_count = item.chunks,
                wa.ingested_at = datetime(),
                wa.claimed_at = null
            """,
            batch=batch_data,
            ingested=DocumentStatus.ingested.value,
        )

    return len(batch_data)


def mark_document_failed(document_id: str, error: str) -> None:
    """Mark an document as failed with error message."""
    try:
        with GraphClient() as gc:
            gc.query(
                """
                MATCH (wa:Document {id: $id})
                SET wa.status = $failed,
                    wa.error = $error,
                    wa.failed_at = datetime(),
                    wa.claimed_at = null
                """,
                id=document_id,
                failed=DocumentStatus.failed.value,
                error=error,
            )
    except Exception as e:
        logger.warning(
            "Could not mark document %s as failed (Neo4j unavailable): %s",
            document_id,
            e,
        )


def mark_document_deferred(document_id: str, reason: str) -> None:
    """Mark an document as deferred (unsupported type or too large)."""
    try:
        with GraphClient() as gc:
            gc.query(
                """
                MATCH (wa:Document {id: $id})
                SET wa.status = $deferred,
                    wa.defer_reason = $reason,
                    wa.claimed_at = null
                """,
                id=document_id,
                deferred=DocumentStatus.deferred.value,
                reason=reason,
            )
    except Exception as e:
        logger.warning(
            "Could not mark document %s as deferred (Neo4j unavailable): %s",
            document_id,
            e,
        )


# =============================================================================
# Image Claim/Mark Functions
# =============================================================================


def has_pending_image_work(facility: str) -> bool:
    """Check if there are images pending scoring (status=ingested, not yet scored)."""
    from imas_codex.discovery.base.image import (
        has_pending_image_work as _has_pending,
    )

    return _has_pending(facility)


@retry_on_deadlock()
def claim_images_for_scoring(
    facility: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Claim ingested images for VLM scoring.

    Images with status='ingested' and no description are ready for VLM processing.
    Uses same claim token pattern as page/document claiming.
    """
    from imas_codex.discovery.base.image import (
        claim_images_for_scoring as _claim,
    )

    return _claim(facility, limit)


@retry_on_deadlock()
def mark_images_scored(
    facility: str,
    results: list[dict[str, Any]],
    *,
    store_images: bool = False,
) -> int:
    """Mark images as scored with VLM results.

    Updates image status to 'captioned' and persists description + scoring fields.
    When store_images is False (default), clears image_data to free graph storage.
    Uses batched UNWIND for efficient graph updates.
    """
    from imas_codex.discovery.base.image import (
        mark_images_scored as _mark,
    )

    return _mark(facility, results, store_images=store_images)


def _release_claimed_images(image_ids: list[str]) -> None:
    """Release claimed images back to pool (e.g., on VLM failure)."""
    from imas_codex.discovery.base.image import release_claimed_images

    release_claimed_images(image_ids)
