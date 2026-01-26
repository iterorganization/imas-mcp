"""
Frontier management for graph-led discovery.

The frontier is the set of FacilityPath nodes that are ready for scanning
(status='discovered') or ready for scoring (status='listed'). This module
provides queries and utilities for managing the frontier.

State machine:
    discovered → listing → listed → scoring → scored

    Transient states (listing, scoring) auto-recover to previous state on timeout.
    Paths with score >= 0.75 are rescored after enrichment (is_enriched=true).

Key concepts:
    - Frontier: Paths awaiting work (discovered → scan, listed → score)
    - Coverage: Fraction of known paths that are scored
    - Seeding: Creating initial root paths for a facility
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from imas_codex.graph.models import PathStatus

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryStats:
    """Statistics about discovery progress for a facility."""

    facility: str
    total: int = 0
    discovered: int = 0  # Awaiting scan
    listed: int = 0  # Awaiting score
    scored: int = 0  # Scored (including rescored)
    skipped: int = 0  # Low value or dead-end
    excluded: int = 0  # Matched exclusion pattern
    max_depth: int = 0  # Maximum depth in tree
    listing: int = 0  # In-progress scan (transient)
    scoring: int = 0  # In-progress score (transient)

    @property
    def frontier_size(self) -> int:
        """Number of paths awaiting work (scan or score)."""
        return self.discovered + self.listed

    @property
    def scan_frontier(self) -> int:
        """Number of paths awaiting scan."""
        return self.discovered

    @property
    def score_frontier(self) -> int:
        """Number of paths awaiting score."""
        return self.listed

    @property
    def coverage(self) -> float:
        """Fraction of known paths that are scored."""
        if self.total == 0:
            return 0.0
        return self.scored / self.total


def get_discovery_stats(facility: str) -> dict[str, Any]:
    """Get discovery statistics for a facility.

    Returns:
        Dict with counts: total, discovered, listed, scored, skipped, excluded,
        max_depth, listing, scoring
    """
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {id: $facility})
            RETURN
                count(p) AS total,
                sum(CASE WHEN p.status = $discovered THEN 1 ELSE 0 END) AS discovered,
                sum(CASE WHEN p.status = $listing THEN 1 ELSE 0 END) AS listing,
                sum(CASE WHEN p.status = $listed THEN 1 ELSE 0 END) AS listed,
                sum(CASE WHEN p.status = $scoring THEN 1 ELSE 0 END) AS scoring,
                sum(CASE WHEN p.status = $scored THEN 1 ELSE 0 END) AS scored,
                sum(CASE WHEN p.status = $skipped THEN 1 ELSE 0 END) AS skipped,
                sum(CASE WHEN p.status = $excluded THEN 1 ELSE 0 END) AS excluded,
                max(coalesce(p.depth, 0)) AS max_depth
            """,
            facility=facility,
            discovered=PathStatus.discovered.value,
            listing=PathStatus.listing.value,
            listed=PathStatus.listed.value,
            scoring=PathStatus.scoring.value,
            scored=PathStatus.scored.value,
            skipped=PathStatus.skipped.value,
            excluded=PathStatus.excluded.value,
        )

        if result:
            return {
                "total": result[0]["total"],
                "discovered": result[0]["discovered"],
                "listing": result[0]["listing"],
                "listed": result[0]["listed"],
                "scoring": result[0]["scoring"],
                "scored": result[0]["scored"],
                "skipped": result[0]["skipped"],
                "excluded": result[0]["excluded"],
                "max_depth": result[0]["max_depth"] or 0,
            }

        return {
            "total": 0,
            "discovered": 0,
            "listing": 0,
            "listed": 0,
            "scoring": 0,
            "scored": 0,
            "skipped": 0,
            "excluded": 0,
            "max_depth": 0,
        }


def get_frontier(
    facility: str,
    limit: int = 100,
    include_rescore: bool = True,
) -> list[dict[str, Any]]:
    """Get paths in the frontier (awaiting scan or rescore).

    Frontier includes:
    1. Paths with status='discovered' (awaiting initial scan)
    2. Paths with status='scored', is_enriched=true, score >= 0.75,
       rescore_count < 1 (awaiting rescore)

    Args:
        facility: Facility ID
        limit: Maximum paths to return
        include_rescore: Include paths marked for rescore

    Returns:
        List of dicts with path info: id, path, depth, status, parent_path_id
    """
    from imas_codex.graph import GraphClient

    if include_rescore:
        where_clause = (
            f"p.status = '{PathStatus.discovered.value}' OR "
            f"(p.status = '{PathStatus.scored.value}' AND p.is_enriched = true AND "
            f"p.score >= 0.75 AND coalesce(p.rescore_count, 0) < 1)"
        )
    else:
        where_clause = f"p.status = '{PathStatus.discovered.value}'"

    with GraphClient() as gc:
        result = gc.query(
            f"""
            MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {{id: $facility}})
            WHERE {where_clause}
            RETURN p.id AS id, p.path AS path, p.depth AS depth,
                   p.status AS status, p.parent_path_id AS parent_path_id
            ORDER BY p.depth ASC, p.path ASC
            LIMIT $limit
            """,
            facility=facility,
            limit=limit,
        )

        return list(result)


def get_scorable_paths(facility: str, limit: int = 100) -> list[dict[str, Any]]:
    """Get paths that are ready for scoring (listed but not scored).

    Returns:
        List of dicts with path info, DirStats, and child_names for LLM context
    """
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {id: $facility})
            WHERE p.status = $listed AND p.score IS NULL
            RETURN p.id AS id, p.path AS path, p.depth AS depth,
                   p.total_files AS total_files, p.total_dirs AS total_dirs,
                   p.file_type_counts AS file_type_counts,
                   p.has_readme AS has_readme, p.has_makefile AS has_makefile,
                   p.has_git AS has_git, p.patterns_detected AS patterns_detected,
                   p.description AS description, p.child_names AS child_names
            ORDER BY p.depth ASC, p.path ASC
            LIMIT $limit
            """,
            facility=facility,
            limit=limit,
            listed=PathStatus.listed.value,
        )

        return list(result)


def seed_facility_roots(
    facility: str,
    root_paths: list[str] | None = None,
) -> int:
    """Create initial FacilityPath nodes for discovery.

    If no root paths provided, uses facility config's paths.actionable_paths
    or falls back to common root paths. Paths are filtered through exclusion
    rules to prevent seeding scratch/temp directories.

    Args:
        facility: Facility ID
        root_paths: Optional list of root paths to seed

    Returns:
        Number of paths created
    """
    from imas_codex.config.discovery_config import get_exclusion_config_for_facility
    from imas_codex.discovery import get_facility
    from imas_codex.graph import GraphClient

    # Get exclusion config with facility-specific patterns
    exclusion_config = get_exclusion_config_for_facility(facility)

    # Get root paths from config if not provided
    if root_paths is None:
        config = get_facility(facility)
        paths_config = config.get("paths", {})

        # Standard root paths for breadth-first discovery (always included)
        standard_roots = ["/home", "/work", "/opt", "/common", "/usr/local"]

        # Facility-specific paths from config (targeted discovery)
        config_paths = []
        if isinstance(paths_config, dict):
            # First check for explicit actionable_paths list
            actionable = paths_config.get("actionable_paths", [])
            if isinstance(actionable, list) and actionable:
                config_paths = [
                    p.get("path") if isinstance(p, dict) else p for p in actionable
                ]
            else:
                # Use path values from config, excluding personal/user paths
                # and infrastructure docs (categories that shouldn't be seeded)
                excluded_categories = {"user", "actionable_paths"}
                for key, value in paths_config.items():
                    if key in excluded_categories:
                        continue
                    if isinstance(value, str) and value.startswith("/"):
                        config_paths.append(value)
                    elif isinstance(value, dict):
                        # Nested paths like {root: "/work/imas", core: "/work/imas/core"}
                        for subvalue in value.values():
                            if isinstance(subvalue, str) and subvalue.startswith("/"):
                                config_paths.append(subvalue)

        # Combine: standard roots first (breadth-first base), then config paths (targeted)
        root_paths = standard_roots + config_paths

        # Deduplicate while preserving order (standard roots take priority)
        seen = set()
        unique_paths = []
        for p in root_paths:
            if p not in seen:
                seen.add(p)
                unique_paths.append(p)
        root_paths = unique_paths

    # Filter out paths that should be excluded (scratch, system, etc.)
    filtered_paths = []
    excluded_paths = []
    for path in root_paths:
        should_exclude, reason = exclusion_config.should_exclude(path)
        if should_exclude:
            excluded_paths.append((path, reason))
            logger.debug(f"Excluding seed path {path}: {reason}")
        else:
            filtered_paths.append(path)

    if excluded_paths:
        logger.info(
            f"Filtered {len(excluded_paths)} paths from seeding: "
            f"{[p for p, _ in excluded_paths[:5]]}"
            + ("..." if len(excluded_paths) > 5 else "")
        )

    root_paths = filtered_paths

    now = datetime.now(UTC).isoformat()
    items = []

    for path in root_paths:
        path_id = f"{facility}:{path}"
        items.append(
            {
                "id": path_id,
                "facility_id": facility,
                "path": path,
                "path_type": "code_directory",  # Default, will be updated during scan
                "status": PathStatus.discovered.value,
                "depth": 0,
                "discovered_at": now,
            }
        )

    with GraphClient() as gc:
        # First ensure facility exists
        existing = gc.get_facility(facility)
        if not existing:
            # Create minimal facility node
            gc.create_facility(facility, name=facility)

        result = gc.create_nodes("FacilityPath", items)

    logger.info(f"Seeded {result['processed']} root paths for {facility}")
    return result["processed"]


def create_child_paths(
    facility: str,
    parent_path: str,
    child_paths: list[str],
) -> int:
    """Create child FacilityPath nodes from a parent.

    Args:
        facility: Facility ID
        parent_path: Parent path string
        child_paths: List of child path strings

    Returns:
        Number of paths created
    """
    from imas_codex.graph import GraphClient

    parent_id = f"{facility}:{parent_path}"
    parent_depth_result = None

    with GraphClient() as gc:
        # Get parent depth
        result = gc.query(
            "MATCH (p:FacilityPath {id: $id}) RETURN p.depth AS depth",
            id=parent_id,
        )
        parent_depth_result = result[0]["depth"] if result else 0

    parent_depth = parent_depth_result or 0
    now = datetime.now(UTC).isoformat()
    items = []

    for child_path in child_paths:
        child_id = f"{facility}:{child_path}"
        items.append(
            {
                "id": child_id,
                "facility_id": facility,
                "path": child_path,
                "path_type": "code_directory",
                "status": PathStatus.discovered.value,
                "depth": parent_depth + 1,
                "parent_path_id": parent_id,
                "discovered_at": now,
            }
        )

    if not items:
        return 0

    with GraphClient() as gc:
        result = gc.create_nodes("FacilityPath", items)

        # Create PARENT relationships
        gc.query(
            """
            UNWIND $children AS child
            MATCH (c:FacilityPath {id: child.id})
            MATCH (p:FacilityPath {id: child.parent_path_id})
            MERGE (c)-[:PARENT]->(p)
            """,
            children=items,
        )

    return result["processed"]


def _is_repo_publicly_accessible(url: str, timeout: float = 5.0) -> bool:
    """Check if a git repository is publicly accessible via HTTP.

    Makes a lightweight HEAD request to the repository URL.
    Returns True only if repo is confirmed public (HTTP 200).
    Returns False if private (401/403/404) or can't determine.

    Args:
        url: Git remote URL
        timeout: Request timeout in seconds

    Returns:
        True if confirmed publicly accessible, False otherwise
    """
    import re
    import urllib.request
    from urllib.error import HTTPError, URLError

    # Convert SSH URL to HTTPS for visibility check
    # git@github.com:owner/repo.git -> https://github.com/owner/repo
    ssh_match = re.match(r"git@([^:]+):(.+?)(?:\.git)?$", url)
    if ssh_match:
        host, path = ssh_match.groups()
        url = f"https://{host}/{path}"
    elif not url.startswith("http"):
        # Local path - not public
        return False

    # Remove .git suffix if present
    url = re.sub(r"\.git$", "", url)

    try:
        req = urllib.request.Request(url, method="HEAD")
        req.add_header("User-Agent", "imas-codex/1.0")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            # 200 = public repo
            return resp.status == 200
    except HTTPError as e:
        # 401/403/404 = private or doesn't exist
        logger.debug(f"Repo {url} not public: HTTP {e.code}")
        return False
    except URLError as e:
        # Network error - assume not public to be conservative
        logger.debug(f"Repo {url} check failed: {e}")
        return False
    except Exception as e:
        logger.debug(f"Repo visibility check failed: {e}")
        return False


def _parse_git_remote_url(url: str) -> tuple[str, str | None, str | None]:
    """Parse git remote URL to extract source type and normalized repo ID.

    Args:
        url: Git remote URL (https, git@, or local path)

    Returns:
        Tuple of (source_type, owner, repo_name) or (source_type, None, None) for local
    """
    import re

    # SSH format: git@github.com:owner/repo.git
    ssh_match = re.match(r"git@([^:]+):([^/]+)/(.+?)(?:\.git)?$", url)
    if ssh_match:
        host, owner, repo = ssh_match.groups()
        if "github" in host.lower():
            return "github", owner, repo
        elif "gitlab" in host.lower():
            return "gitlab", owner, repo
        elif "bitbucket" in host.lower():
            return "bitbucket", owner, repo
        return "gitlab", owner, repo  # Default to gitlab for other hosts

    # HTTPS format: https://github.com/owner/repo.git
    https_match = re.match(r"https?://([^/]+)/([^/]+)/(.+?)(?:\.git)?$", url)
    if https_match:
        host, owner, repo = https_match.groups()
        if "github" in host.lower():
            return "github", owner, repo
        elif "gitlab" in host.lower():
            return "gitlab", owner, repo
        elif "bitbucket" in host.lower():
            return "bitbucket", owner, repo
        return "gitlab", owner, repo

    # Local or unknown format
    return "local", None, None


def _create_software_repo_link(
    gc: Any,
    facility: str,
    path_id: str,
    remote_url: str,
    head_commit: str | None,
    branch: str | None,
    now: str,
) -> None:
    """Create SoftwareRepo node and CLONE_OF relationship.

    Args:
        gc: GraphClient instance
        facility: Facility ID
        path_id: FacilityPath ID
        remote_url: Git remote origin URL
        head_commit: Current HEAD commit hash
        branch: Current branch name
        now: ISO timestamp
    """
    source_type, owner, repo_name = _parse_git_remote_url(remote_url)

    if source_type == "local" or not owner or not repo_name:
        # Local repo or can't parse - create with path-based ID
        repo_id = f"local:{facility}:{path_id.split(':', 1)[1]}"
        name = path_id.split("/")[-1] if "/" in path_id else path_id
    else:
        # Remote repo - create with normalized ID
        repo_id = f"{source_type}:{owner}/{repo_name}"
        name = repo_name

    # MERGE SoftwareRepo (deduplicates across clones)
    gc.query(
        """
        MERGE (r:SoftwareRepo {id: $repo_id})
        ON CREATE SET
            r.source_type = $source_type,
            r.remote_url = $remote_url,
            r.name = $name,
            r.head_commit = $head_commit,
            r.default_branch = $branch,
            r.discovered_at = $now,
            r.discovered_via = $facility,
            r.clone_count = 1
        ON MATCH SET
            r.clone_count = coalesce(r.clone_count, 0) + 1,
            r.head_commit = CASE
                WHEN $head_commit IS NOT NULL THEN $head_commit
                ELSE r.head_commit
            END
        """,
        repo_id=repo_id,
        source_type=source_type,
        remote_url=remote_url,
        name=name,
        head_commit=head_commit,
        branch=branch,
        now=now,
        facility=facility,
    )

    # Link FacilityPath to SoftwareRepo
    gc.query(
        """
        MATCH (p:FacilityPath {id: $path_id})
        MATCH (r:SoftwareRepo {id: $repo_id})
        MERGE (p)-[:CLONE_OF]->(r)
        SET p.software_repo_id = $repo_id
        """,
        path_id=path_id,
        repo_id=repo_id,
    )


def mark_paths_scored(
    facility: str,
    scores: list[dict[str, Any]],
) -> int:
    """Update multiple paths with LLM scores and create/link Evidence nodes.

    Args:
        facility: Facility ID
        scores: List of dicts from ScoredDirectory.to_graph_dict() with:
                path, path_purpose, description, evidence, score_code,
                score_data, score_docs, score_imas, score, should_expand,
                keywords, physics_domain, expansion_reason, skip_reason

    Returns:
        Number of paths updated
    """
    import hashlib
    import json

    from imas_codex.graph import GraphClient

    now = datetime.now(UTC).isoformat()
    updated = 0

    with GraphClient() as gc:
        for score_data in scores:
            path = score_data["path"]
            path_id = f"{facility}:{path}"

            # Create content-addressable Evidence node from indicators
            evidence_dict = score_data.get("evidence")
            evidence_id = None

            if evidence_dict and isinstance(evidence_dict, dict):
                # Compute stable hash from evidence content
                evidence_json = json.dumps(evidence_dict, sort_keys=True)
                hash_bytes = hashlib.sha256(evidence_json.encode()).hexdigest()[:16]
                evidence_id = f"ev:{hash_bytes}"

                # MERGE evidence node (idempotent)
                gc.query(
                    """
                    MERGE (e:Evidence {id: $ev_id})
                    ON CREATE SET
                        e.code_indicators = $code_indicators,
                        e.data_indicators = $data_indicators,
                        e.doc_indicators = $doc_indicators,
                        e.imas_indicators = $imas_indicators,
                        e.physics_indicators = $physics_indicators,
                        e.quality_indicators = $quality_indicators,
                        e.created_at = $now
                    """,
                    ev_id=evidence_id,
                    code_indicators=evidence_dict.get("code_indicators", []),
                    data_indicators=evidence_dict.get("data_indicators", []),
                    doc_indicators=evidence_dict.get("doc_indicators", []),
                    imas_indicators=evidence_dict.get("imas_indicators", []),
                    physics_indicators=evidence_dict.get("physics_indicators", []),
                    quality_indicators=evidence_dict.get("quality_indicators", []),
                    now=now,
                )

            # Update FacilityPath with scores and link to Evidence
            gc.query(
                """
                MATCH (p:FacilityPath {id: $id})
                SET p.status = $scored,
                    p.scored_at = $now,
                    p.score = $score,
                    p.score_code = $score_code,
                    p.score_data = $score_data,
                    p.score_docs = $score_docs,
                    p.score_imas = $score_imas,
                    p.description = $description,
                    p.path_purpose = $path_purpose,
                    p.evidence_id = $evidence_id,
                    p.should_expand = $should_expand,
                    p.keywords = $keywords,
                    p.physics_domain = $physics_domain,
                    p.expansion_reason = $expansion_reason,
                    p.skip_reason = $skip_reason
                WITH p
                OPTIONAL MATCH (e:Evidence {id: $evidence_id})
                FOREACH (_ IN CASE WHEN e IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (p)-[:HAS_EVIDENCE]->(e)
                )
                """,
                id=path_id,
                now=now,
                score=score_data.get("score"),
                score_code=score_data.get("score_code"),
                score_data=score_data.get("score_data"),
                score_docs=score_data.get("score_docs"),
                score_imas=score_data.get("score_imas"),
                description=score_data.get("description"),
                path_purpose=score_data.get("path_purpose"),
                evidence_id=evidence_id,
                should_expand=score_data.get("should_expand"),
                keywords=score_data.get("keywords"),
                physics_domain=score_data.get("physics_domain"),
                expansion_reason=score_data.get("expansion_reason"),
                skip_reason=score_data.get("skip_reason"),
                scored=PathStatus.scored.value,
            )
            updated += 1

    return updated


def mark_path_skipped(
    facility: str,
    path: str,
    reason: str,
) -> None:
    """Mark a path as skipped with a reason.

    Args:
        facility: Facility ID
        path: Path string
        reason: Skip reason
    """
    from imas_codex.graph import GraphClient

    path_id = f"{facility}:{path}"
    now = datetime.now(UTC).isoformat()

    with GraphClient() as gc:
        gc.query(
            """
            MATCH (p:FacilityPath {id: $id})
            SET p.status = $skipped,
                p.skipped_at = $now,
                p.skip_reason = $reason
            """,
            id=path_id,
            now=now,
            reason=reason,
            skipped=PathStatus.skipped.value,
        )


def get_high_value_paths(
    facility: str,
    min_score: float = 0.7,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Get scored paths above a threshold.

    Args:
        facility: Facility ID
        min_score: Minimum score threshold
        limit: Maximum paths to return

    Returns:
        List of dicts with path info and scores
    """
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {id: $facility})
            WHERE p.score >= $min_score
            RETURN p.id AS id, p.path AS path, p.score AS score,
                   p.description AS description, p.path_purpose AS path_purpose,
                   p.score_code AS score_code, p.score_data AS score_data,
                   p.score_imas AS score_imas
            ORDER BY p.score DESC
            LIMIT $limit
            """,
            facility=facility,
            min_score=min_score,
            limit=limit,
        )

        return list(result)


def clear_facility_paths(facility: str, batch_size: int = 5000) -> int:
    """Delete all FacilityPath nodes for a facility in batches.

    Uses batched deletion to avoid memory exhaustion on large facilities.

    Args:
        facility: Facility ID
        batch_size: Nodes to delete per batch (default 5000)

    Returns:
        Total number of paths deleted
    """
    from imas_codex.graph import GraphClient

    total_deleted = 0

    with GraphClient() as gc:
        while True:
            # Delete a batch and return count
            result = gc.query(
                """
                MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {id: $facility})
                WITH p LIMIT $batch_size
                DETACH DELETE p
                RETURN count(p) AS deleted
                """,
                facility=facility,
                batch_size=batch_size,
            )

            deleted = result[0]["deleted"] if result else 0
            total_deleted += deleted

            # If we deleted less than batch_size, we're done
            if deleted < batch_size:
                break

    return total_deleted


def persist_scan_results(
    facility: str,
    results: list[tuple[str, dict, list[str], str | None]],
    excluded: list[tuple[str, str, str]] | None = None,
) -> dict[str, int]:
    """Persist multiple scan results in a single transaction.

    Much faster than calling mark_path_scanned/create_child_paths per path.

    Args:
        facility: Facility ID
        results: List of (path, stats_dict, child_dirs, error) tuples
        excluded: Optional list of (path, parent_path, reason) for excluded dirs

    Returns:
        Dict with scanned, children_created, excluded, errors counts
    """
    from imas_codex.graph import GraphClient

    now = datetime.now(UTC).isoformat()
    scanned = 0
    children_created = 0
    excluded_count = 0
    errors = 0

    # Separate successes and errors
    successes = []
    all_children = []

    for path, stats, child_dirs, error in results:
        path_id = f"{facility}:{path}"
        if error:
            errors += 1
            # Mark as skipped
            successes.append(
                {
                    "id": path_id,
                    "status": PathStatus.skipped.value,
                    "skip_reason": error,
                    "listed_at": now,
                }
            )
        else:
            scanned += 1
            # Build update dict with child_names if available
            update_dict = {
                "id": path_id,
                "status": PathStatus.listed.value,
                "listed_at": now,
                "total_files": stats.get("total_files", 0),
                "total_dirs": stats.get("total_dirs", 0),
                "has_readme": stats.get("has_readme", False),
                "has_makefile": stats.get("has_makefile", False),
                "has_git": stats.get("has_git", False),
                "git_remote_url": stats.get("git_remote_url"),
                "git_head_commit": stats.get("git_head_commit"),
                "git_branch": stats.get("git_branch"),
            }
            # Store file_type_counts if available
            file_type_counts = stats.get("file_type_counts")
            if file_type_counts:
                import json

                if isinstance(file_type_counts, dict):
                    update_dict["file_type_counts"] = json.dumps(file_type_counts)
                else:
                    update_dict["file_type_counts"] = file_type_counts
            # Store child names for LLM scoring context (as JSON string)
            child_names = stats.get("child_names")
            if child_names:
                import json

                update_dict["child_names"] = json.dumps(child_names)
            # Store patterns detected
            patterns = stats.get("patterns_detected")
            if patterns:
                update_dict["patterns_detected"] = patterns
            successes.append(update_dict)

            # Decide whether to skip children for git repos
            # Skip ONLY verified public repos that are accessible elsewhere
            # Continue scanning private/local repos - they contain unique facility code
            is_git_repo = stats.get("has_git", False)
            git_remote_url = stats.get("git_remote_url", "")

            if is_git_repo and git_remote_url:
                # Check if this is on a public hosting service (quick heuristic)
                is_public_host = any(
                    host in git_remote_url.lower()
                    for host in ["github.com", "gitlab.com", "bitbucket.org"]
                )
                if is_public_host:
                    # Verify actual visibility via HTTP (confirms not private)
                    is_public = _is_repo_publicly_accessible(
                        git_remote_url, timeout=3.0
                    )
                    if is_public:
                        # Confirmed public - skip children, content available from source
                        logger.debug(
                            f"Skip children: public repo {path} ({git_remote_url})"
                        )
                        continue
                    else:
                        logger.debug(
                            f"Scan children: private repo on public host {path}"
                        )
                # Private/internal repo - continue scanning

            # Prepare children for non-public-git directories
            for child_path in child_dirs:
                all_children.append(
                    {
                        "id": f"{facility}:{child_path}",
                        "facility_id": facility,
                        "path": child_path,
                        "parent_id": path_id,
                    }
                )

    with GraphClient() as gc:
        # Batch update scanned/skipped paths
        if successes:
            gc.query(
                """
                UNWIND $items AS item
                MATCH (p:FacilityPath {id: item.id})
                SET p.status = item.status,
                    p.listed_at = item.listed_at,
                    p.skip_reason = item.skip_reason,
                    p.total_files = item.total_files,
                    p.total_dirs = item.total_dirs,
                    p.has_readme = item.has_readme,
                    p.has_makefile = item.has_makefile,
                    p.has_git = item.has_git,
                    p.git_remote_url = item.git_remote_url,
                    p.git_head_commit = item.git_head_commit,
                    p.git_branch = item.git_branch,
                    p.child_names = item.child_names
                """,
                items=successes,
            )

        # Batch create children
        if all_children:
            # First get parent depths
            parent_ids = list({c["parent_id"] for c in all_children})
            depth_result = gc.query(
                """
                UNWIND $ids AS id
                MATCH (p:FacilityPath {id: id})
                RETURN p.id AS id, p.depth AS depth
                """,
                ids=parent_ids,
            )
            depth_map = {r["id"]: r["depth"] or 0 for r in depth_result}

            # Add depth to children
            for child in all_children:
                child["depth"] = depth_map.get(child["parent_id"], 0) + 1
                child["status"] = PathStatus.discovered.value
                child["path_type"] = "code_directory"
                child["discovered_at"] = now

            # Create child nodes with relationships in a single query
            gc.query(
                """
                UNWIND $children AS child
                MATCH (f:Facility {id: child.facility_id})
                MATCH (parent:FacilityPath {id: child.parent_id})
                MERGE (c:FacilityPath {id: child.id})
                ON CREATE SET c.facility_id = child.facility_id,
                              c.path = child.path,
                              c.path_type = child.path_type,
                              c.status = child.status,
                              c.depth = child.depth,
                              c.parent_path_id = child.parent_id,
                              c.discovered_at = child.discovered_at
                MERGE (c)-[:FACILITY_ID]->(f)
                MERGE (c)-[:PARENT]->(parent)
                """,
                children=all_children,
            )

            children_created = len(all_children)

        # Handle excluded directories (create with status='excluded')
        if excluded:
            excluded_nodes = []
            for path, parent_path, reason in excluded:
                parent_id = f"{facility}:{parent_path}"
                # Get parent depth
                depth_result = gc.query(
                    "MATCH (p:FacilityPath {id: $id}) RETURN p.depth AS depth",
                    id=parent_id,
                )
                parent_depth = depth_result[0]["depth"] if depth_result else 0

                excluded_nodes.append(
                    {
                        "id": f"{facility}:{path}",
                        "facility_id": facility,
                        "path": path,
                        "parent_id": parent_id,
                        "depth": (parent_depth or 0) + 1,
                        "status": PathStatus.excluded.value,
                        "skip_reason": reason,
                        "discovered_at": now,
                    }
                )

            if excluded_nodes:
                gc.query(
                    """
                    UNWIND $nodes AS node
                    MERGE (p:FacilityPath {id: node.id})
                    ON CREATE SET p.facility_id = node.facility_id,
                                  p.path = node.path,
                                  p.status = node.status,
                                  p.skip_reason = node.skip_reason,
                                  p.depth = node.depth,
                                  p.parent_path_id = node.parent_id,
                                  p.discovered_at = node.discovered_at
                    """,
                    nodes=excluded_nodes,
                )

                # Create relationships
                gc.query(
                    """
                    UNWIND $nodes AS node
                    MATCH (c:FacilityPath {id: node.id})
                    MATCH (f:Facility {id: node.facility_id})
                    MERGE (c)-[:FACILITY_ID]->(f)
                    """,
                    nodes=excluded_nodes,
                )

                excluded_count = len(excluded_nodes)

        # User enrichment: extract users from discovered home paths
        # Run in same transaction for consistency
        all_paths = [path for path, _stats, _children, _error in results]
        try:
            from imas_codex.discovery.user_enrichment import enrich_users_from_paths

            facility_users = enrich_users_from_paths(facility, all_paths)
            if facility_users:
                # Create FacilityUser nodes
                gc.query(
                    """
                    UNWIND $users AS user
                    MATCH (f:Facility {id: user.facility_id})
                    MERGE (u:FacilityUser {id: user.id})
                    ON CREATE SET u.username = user.username,
                                  u.facility_id = user.facility_id,
                                  u.name = user.name,
                                  u.given_name = user.given_name,
                                  u.family_name = user.family_name,
                                  u.home_path = user.home_path,
                                  u.discovered_at = user.discovered_at,
                                  u.enriched_at = user.enriched_at
                    ON MATCH SET u.name = COALESCE(user.name, u.name),
                                 u.given_name = COALESCE(user.given_name, u.given_name),
                                 u.family_name = COALESCE(user.family_name, u.family_name),
                                 u.enriched_at = COALESCE(user.enriched_at, u.enriched_at)
                    MERGE (u)-[:FACILITY_ID]->(f)
                    """,
                    users=facility_users,
                )
                logger.debug(f"Enriched {len(facility_users)} users for {facility}")
        except Exception as e:
            # User enrichment is non-critical; don't fail scan
            logger.warning(f"User enrichment failed: {e}")

        # Create SoftwareRepo nodes for git repos with remote URLs
        git_repos = [
            (
                item["id"],
                item["git_remote_url"],
                item.get("git_head_commit"),
                item.get("git_branch"),
            )
            for item in successes
            if item.get("has_git") and item.get("git_remote_url")
        ]
        for path_id, remote_url, head_commit, branch in git_repos:
            try:
                _create_software_repo_link(
                    gc, facility, path_id, remote_url, head_commit, branch, now
                )
            except Exception as e:
                logger.debug(f"Failed to create SoftwareRepo for {path_id}: {e}")

    return {
        "scanned": scanned,
        "children_created": children_created,
        "excluded": excluded_count,
        "errors": errors,
    }


def normalize_scores(facility: str) -> dict[str, int]:
    """Compute percentile ranks for scored paths within a facility.

    Updates score_percentile field for all scored paths. Uses average rank
    method for tied scores - paths with the same score get the same percentile.

    Percentile range: 0.0 to 1.0. Top 5% means score_percentile >= 0.95.

    Args:
        facility: Facility ID

    Returns:
        Dict with count of paths updated
    """
    from itertools import groupby

    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        # Get all scored paths with their scores
        result = gc.query(
            """
            MATCH (p:FacilityPath {facility_id: $facility})
            WHERE p.status = 'scored' AND p.score IS NOT NULL
            RETURN p.id AS id, p.score AS score
            ORDER BY p.score ASC
            """,
            facility=facility,
        )

        if not result:
            return {"updated": 0}

        total = len(result)
        if total == 1:
            # Single path gets percentile 0.5
            gc.query(
                "MATCH (p:FacilityPath {id: $id}) SET p.score_percentile = 0.5",
                id=result[0]["id"],
            )
            return {"updated": 1}

        # Compute percentile ranks with average rank for ties
        updates = []
        current_rank = 0
        for _score, group in groupby(result, key=lambda r: r["score"]):
            group_list = list(group)
            # Average rank for tied scores: midpoint of the rank range
            avg_rank = current_rank + (len(group_list) - 1) / 2
            percentile = avg_rank / (total - 1)  # 0.0 to 1.0
            for row in group_list:
                updates.append({"id": row["id"], "percentile": round(percentile, 4)})
            current_rank += len(group_list)

        # Batch update
        gc.query(
            """
            UNWIND $updates AS u
            MATCH (p:FacilityPath {id: u.id})
            SET p.score_percentile = u.percentile
            """,
            updates=updates,
        )

        return {"updated": len(updates)}


def sample_scored_paths(
    facility: str, per_quartile: int = 3
) -> dict[str, list[dict[str, Any]]]:
    """Sample paths from each score quartile for LLM calibration.

    Returns representative examples from low, medium, high, and very_high
    score ranges to help the LLM calibrate its scoring decisions.

    Args:
        facility: Facility ID
        per_quartile: Number of paths to sample from each quartile

    Returns:
        Dict with keys: low, medium, high, very_high
        Each value is a list of dicts with: path, score, purpose, description
    """
    from imas_codex.graph import GraphClient

    quartiles = {
        "low": (0.0, 0.25),
        "medium": (0.25, 0.5),
        "high": (0.5, 0.75),
        "very_high": (0.75, 0.95),  # Cap at 0.95, we forbid 1.0
    }

    samples: dict[str, list[dict[str, Any]]] = {}

    with GraphClient() as gc:
        for name, (min_score, max_score) in quartiles.items():
            result = gc.query(
                """
                MATCH (p:FacilityPath {facility_id: $facility})
                WHERE p.status = 'scored'
                    AND p.score >= $min_score
                    AND p.score < $max_score
                RETURN p.path AS path,
                       p.score AS score,
                       p.path_purpose AS purpose,
                       p.description AS description
                ORDER BY rand()
                LIMIT $limit
                """,
                facility=facility,
                min_score=min_score,
                max_score=max_score,
                limit=per_quartile,
            )
            samples[name] = [
                {
                    "path": r["path"],
                    "score": r["score"],
                    "purpose": r["purpose"] or "unknown",
                    "description": (r["description"] or "")[:100],
                }
                for r in result
            ]

    return samples
