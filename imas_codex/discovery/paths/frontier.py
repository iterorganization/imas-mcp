"""
Frontier management for graph-led discovery.

The frontier is the set of FacilityPath nodes that are ready for scanning
(status='discovered') or ready for scoring (status='scanned'). This module
provides queries and utilities for managing the frontier.

State machine:
    discovered → scanning → scanned → scoring → scored

    Transient states (scanning, scoring) auto-recover to previous state on timeout.
    Paths with score >= 0.75 are rescored after enrichment (is_enriched=true).

Key concepts:
    - Frontier: Paths awaiting work (discovered → scan, scanned → score)
    - Coverage: Fraction of known paths that are scored
    - Seeding: Creating initial root paths for a facility
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from imas_codex.graph.models import PathStatus, TerminalReason

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _dedupe_paths_by_inode(
    facility: str, paths: list[str]
) -> tuple[list[str], list[tuple[str, str]]]:
    """Deduplicate paths, preferring canonical shorter paths.

    Detection methods (in order):
    1. Inode match - same device:inode means bind mount
    2. Realpath resolution - symlinks resolve to canonical path
    3. Content similarity - 80%+ shared children detects NFS re-mounts

    Args:
        facility: Facility ID for SSH execution
        paths: List of paths to deduplicate

    Returns:
        Tuple of (canonical_paths, alias_pairs) where alias_pairs is
        list of (alias_path, canonical_path) for creating ALIAS_OF links
    """
    from imas_codex.remote.tools import run

    if not paths:
        return paths, []

    # Common base directory names that might be mounted elsewhere
    canonical_bases = {"home", "work", "opt", "common", "scratch", "data", "shared"}

    # Build set of paths to check: original paths + potential canonical bases
    paths_to_check = set(paths)
    path_to_candidates: dict[str, list[str]] = {}

    for path in paths:
        # Extract potential canonical bases from path components
        components = path.strip("/").split("/")
        candidates = []
        for comp in components:
            if comp.lower() in canonical_bases:
                canonical = f"/{comp}"
                candidates.append(canonical)
                paths_to_check.add(canonical)
        path_to_candidates[path] = candidates

    # Single SSH call to get inode, realpath, and child names for all paths
    path_args = " ".join(f'"{p}"' for p in sorted(paths_to_check))
    script = f"""
for p in {path_args}; do
    if [ -d "$p" ]; then
        inode=$(stat --format="%d:%i" "$p" 2>/dev/null || echo "")
        rp=$(realpath "$p" 2>/dev/null || echo "$p")
        # Get first 30 children names (comma-separated)
        children=$(ls -1 "$p" 2>/dev/null | head -30 | tr '\\n' ',' | sed 's/,$//')
        echo "$inode|$rp|$children|$p"
    fi
done
"""

    try:
        result = run(script, facility=facility, timeout=60)
    except Exception as e:
        logger.warning(f"Failed to stat paths for dedup: {e}")
        return paths, []

    # Parse output: path → (inode, realpath, children_set)
    path_info: dict[str, tuple[str, str, set[str]]] = {}
    for line in result.strip().split("\n"):
        if not line or "|" not in line:
            continue
        parts = line.split("|")
        if len(parts) >= 4:
            inode, realpath, children_str, path = (
                parts[0],
                parts[1],
                parts[2],
                parts[3],
            )
            children = set(children_str.split(",")) if children_str else set()
            path_info[path] = (inode, realpath, children)

    def paths_are_same(p1: str, p2: str) -> bool:
        """Check if two paths point to the same content."""
        info1 = path_info.get(p1)
        info2 = path_info.get(p2)
        if not info1 or not info2:
            return False

        inode1, realpath1, children1 = info1
        inode2, realpath2, children2 = info2

        # Method 1: Same inode (bind mount)
        if inode1 and inode2 and inode1 == inode2:
            return True

        # Method 2: Realpath resolves to same location
        if realpath1 and realpath2 and realpath1 == realpath2:
            return True

        # Method 3: Content similarity (NFS re-mount with minor differences)
        # Require 80%+ children overlap AND matching base names
        if children1 and children2:
            # Extra check: last component should match to avoid false positives
            base1 = p1.rstrip("/").split("/")[-1]
            base2 = p2.rstrip("/").split("/")[-1]
            if base1 != base2:
                return False

            # Compute Jaccard similarity
            intersection = len(children1 & children2)
            union = len(children1 | children2)
            if union > 0:
                similarity = intersection / union
                if similarity >= 0.8:  # 80% overlap threshold
                    logger.debug(f"Content similarity {p1} <-> {p2}: {similarity:.0%}")
                    return True

        return False

    # Deduplicate: prefer shorter canonical paths
    canonical: list[str] = []
    aliases: list[tuple[str, str]] = []  # (alias, canonical)
    processed: set[str] = set()

    # Sort by depth (shallower first) so we pick canonical paths first
    sorted_paths = sorted(paths, key=lambda p: p.count("/"))

    for path in sorted_paths:
        if path in processed:
            continue

        # Check if any candidate canonical base is equivalent
        # Skip comparing path to itself (e.g., /home as candidate for /home)
        found_canonical = None
        for candidate in path_to_candidates.get(path, []):
            if candidate == path:
                continue  # Don't alias a path to itself
            if candidate in path_info and paths_are_same(path, candidate):
                found_canonical = candidate
                logger.info(f"Dedup: {candidate} (canonical) <- {path} (same content)")
                break

        if found_canonical:
            # Use the canonical path instead
            if found_canonical not in processed:
                canonical.append(found_canonical)
                processed.add(found_canonical)
            aliases.append((path, found_canonical))
            processed.add(path)
        else:
            # Check if matches any already-selected canonical path
            matched_existing = None
            for existing in canonical:
                if paths_are_same(path, existing):
                    matched_existing = existing
                    break

            if matched_existing:
                aliases.append((path, matched_existing))
                logger.info(
                    f"Dedup: {matched_existing} (canonical) <- {path} (same content)"
                )
            else:
                canonical.append(path)
            processed.add(path)

    return canonical, aliases


def _path_canonicality_score(path: str) -> int:
    """Score a path's canonicality - higher is more canonical.

    Uses negative path depth so shallower paths have higher scores.
    Bind mounts typically expose deep infrastructure paths at shallow
    user-facing locations.

    Example: /home/user (depth 2) → -2 (wins)
             /mnt/HPC/ITER/home/user (depth 5) → -5

    Args:
        path: Absolute path string

    Returns:
        Negative depth - higher (less negative) means shallower/more canonical
    """
    depth = path.rstrip("/").count("/")
    return -depth


@dataclass
class DiscoveryStats:
    """Statistics about discovery progress for a facility."""

    facility: str
    total: int = 0
    discovered: int = 0  # Awaiting scan
    scanned: int = 0  # Awaiting score
    scored: int = 0  # Scored (including rescored)
    skipped: int = 0  # Low value or dead-end
    excluded: int = 0  # Matched exclusion pattern
    max_depth: int = 0  # Maximum depth in tree
    scanning: int = 0  # In-progress scan (transient)
    scoring: int = 0  # In-progress score (transient)

    @property
    def frontier_size(self) -> int:
        """Number of paths awaiting work (scan or score)."""
        return self.discovered + self.scanned

    @property
    def scan_frontier(self) -> int:
        """Number of paths awaiting scan."""
        return self.discovered

    @property
    def score_frontier(self) -> int:
        """Number of paths awaiting score."""
        return self.scanned

    @property
    def coverage(self) -> float:
        """Fraction of known paths that are scored."""
        if self.total == 0:
            return 0.0
        return self.scored / self.total


def get_discovery_stats(facility: str) -> dict[str, Any]:
    """Get discovery statistics for a facility.

    Returns:
        Dict with counts: total, discovered, scanned, scored, skipped, excluded,
        max_depth, claimed (paths with active claims)
    """
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (p:FacilityPath)-[:AT_FACILITY]->(f:Facility {id: $facility})
            RETURN
                count(p) AS total,
                sum(CASE WHEN p.status = $discovered THEN 1 ELSE 0 END) AS discovered,
                sum(CASE WHEN p.status = $scanned THEN 1 ELSE 0 END) AS scanned,
                sum(CASE WHEN p.status = $scored THEN 1 ELSE 0 END) AS scored,
                sum(CASE WHEN p.status = $skipped THEN 1 ELSE 0 END) AS skipped,
                sum(CASE WHEN p.terminal_reason = $excluded_reason THEN 1 ELSE 0 END) AS excluded,
                sum(CASE WHEN p.claimed_at IS NOT NULL THEN 1 ELSE 0 END) AS claimed,
                sum(CASE WHEN p.status = $scored AND p.should_expand = true AND p.expanded_at IS NULL THEN 1 ELSE 0 END) AS expansion_ready,
                sum(CASE WHEN p.status = $scored AND p.should_enrich = true AND (p.is_enriched IS NULL OR p.is_enriched = false) THEN 1 ELSE 0 END) AS enrichment_ready,
                max(coalesce(p.depth, 0)) AS max_depth
            """,
            facility=facility,
            discovered=PathStatus.discovered.value,
            scanned=PathStatus.scanned.value,
            scored=PathStatus.scored.value,
            skipped=PathStatus.skipped.value,
            excluded_reason=TerminalReason.excluded.value,
        )

        if result:
            return {
                "total": result[0]["total"],
                "discovered": result[0]["discovered"],
                "scanned": result[0]["scanned"],
                "scored": result[0]["scored"],
                "skipped": result[0]["skipped"],
                "excluded": result[0]["excluded"],
                "claimed": result[0]["claimed"],
                "expansion_ready": result[0]["expansion_ready"],
                "enrichment_ready": result[0]["enrichment_ready"],
                "max_depth": result[0]["max_depth"] or 0,
            }

        return {
            "total": 0,
            "discovered": 0,
            "scanned": 0,
            "scored": 0,
            "skipped": 0,
            "excluded": 0,
            "claimed": 0,
            "expansion_ready": 0,
            "enrichment_ready": 0,
            "max_depth": 0,
        }


def get_purpose_distribution(facility: str) -> dict[str, int]:
    """Get count of scored paths by path_purpose.

    Returns:
        Dict mapping purpose name to count, sorted by count descending.
    """
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (p:FacilityPath)-[:AT_FACILITY]->(f:Facility {id: $facility})
            WHERE p.status = 'scored' AND p.path_purpose IS NOT NULL
            RETURN p.path_purpose AS purpose, count(*) AS count
            ORDER BY count DESC
            """,
            facility=facility,
        )

        return {row["purpose"]: row["count"] for row in result}


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
        List of dicts with path info: id, path, depth, status, in_directory
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
            MATCH (p:FacilityPath)-[:AT_FACILITY]->(f:Facility {{id: $facility}})
            WHERE {where_clause}
            RETURN p.id AS id, p.path AS path, p.depth AS depth,
                   p.status AS status, p.in_directory AS in_directory
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
            MATCH (p:FacilityPath)-[:AT_FACILITY]->(f:Facility {id: $facility})
            WHERE p.status = $scanned AND p.score IS NULL
            RETURN p.id AS id, p.path AS path, p.depth AS depth,
                   p.total_files AS total_files, p.total_dirs AS total_dirs,
                   p.file_type_counts AS file_type_counts,
                   p.has_readme AS has_readme, p.has_makefile AS has_makefile,
                   p.has_git AS has_git, p.patterns_found AS patterns_found,
                   p.description AS description, p.child_names AS child_names
            ORDER BY p.depth ASC, p.path ASC
            LIMIT $limit
            """,
            facility=facility,
            limit=limit,
            scanned=PathStatus.scanned.value,
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

        # Priority 1: Explicit discovery_roots list (preferred)
        # This is the standard key for seeding the discovery pipeline
        discovery_roots = config.get("discovery_roots", [])
        if isinstance(discovery_roots, list) and discovery_roots:
            root_paths = [
                p.get("path") if isinstance(p, dict) else p for p in discovery_roots
            ]
            logger.info(f"Using {len(root_paths)} discovery_roots from config")
        else:
            # Priority 2: Legacy paths.actionable_paths or paths dict
            paths_config = config.get("paths", {})
            root_paths = []
            if isinstance(paths_config, dict):
                # Check for explicit actionable_paths list
                actionable = paths_config.get("actionable_paths", [])
                if isinstance(actionable, list) and actionable:
                    root_paths = [
                        p.get("path") if isinstance(p, dict) else p for p in actionable
                    ]
                else:
                    # Use path values from config, excluding personal/user paths
                    excluded_categories = {"user", "actionable_paths"}
                    for key, value in paths_config.items():
                        if key in excluded_categories:
                            continue
                        if isinstance(value, str) and value.startswith("/"):
                            root_paths.append(value)
                        elif isinstance(value, dict):
                            # Nested paths like {root: "/work/imas", core: "/work/imas/core"}
                            for subvalue in value.values():
                                if isinstance(subvalue, str) and subvalue.startswith(
                                    "/"
                                ):
                                    root_paths.append(subvalue)

        # Deduplicate while preserving order
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

    # Deduplicate by inode/realpath/content-hash via single SSH call
    # Returns canonical paths to scan + alias pairs for ALIAS_OF links
    canonical_paths, alias_pairs = _dedupe_paths_by_inode(facility, root_paths)

    now = datetime.now(UTC).isoformat()
    items = []

    # Create nodes for canonical paths (status=discovered)
    for path in canonical_paths:
        path_id = f"{facility}:{path}"
        items.append(
            {
                "id": path_id,
                "facility_id": facility,
                "path": path,
                "path_type": "code_directory",
                "status": PathStatus.discovered.value,
                "depth": 0,
                "discovered_at": now,
            }
        )

    # Create nodes for alias paths (status=skipped with reason)
    alias_items = []
    for alias_path, canonical_path in alias_pairs:
        alias_id = f"{facility}:{alias_path}"
        alias_items.append(
            {
                "id": alias_id,
                "facility_id": facility,
                "path": alias_path,
                "path_type": "code_directory",
                "status": PathStatus.skipped.value,
                "terminal_reason": TerminalReason.alias.value,
                "skip_reason": f"Alias of {canonical_path}",
                "depth": 0,
                "discovered_at": now,
            }
        )

    with GraphClient() as gc:
        # Ensure facility node exists (idempotent)
        gc.ensure_facility(facility)

        result = gc.create_nodes("FacilityPath", items)

        # Create alias nodes and ALIAS_OF relationships
        if alias_items:
            gc.create_nodes("FacilityPath", alias_items)
            for alias_path, canonical_path in alias_pairs:
                alias_id = f"{facility}:{alias_path}"
                canonical_id = f"{facility}:{canonical_path}"
                gc.query(
                    """
                    MATCH (alias:FacilityPath {id: $alias_id})
                    MATCH (canonical:FacilityPath {id: $canonical_id})
                    MERGE (alias)-[:ALIAS_OF]->(canonical)
                    """,
                    alias_id=alias_id,
                    canonical_id=canonical_id,
                )
            logger.info(
                f"Created {len(alias_pairs)} ALIAS_OF links for duplicate mount points"
            )

    logger.info(f"Seeded {result['processed']} root paths for {facility}")
    return result["processed"]


def seed_missing_roots(facility: str) -> int:
    """Add discovery_roots from config that are not already in the graph.

    This is an additive operation - existing paths are preserved.
    Only paths from discovery_roots that don't exist in the graph are added.

    Args:
        facility: Facility ID

    Returns:
        Number of new paths created
    """
    from imas_codex.discovery import get_facility
    from imas_codex.graph import GraphClient

    config = get_facility(facility)
    discovery_roots = config.get("discovery_roots", [])

    if not discovery_roots:
        logger.info(f"No discovery_roots configured for {facility}")
        return 0

    # Extract paths from config (may be dicts with 'path' key or plain strings)
    config_paths = [
        p.get("path") if isinstance(p, dict) else p for p in discovery_roots
    ]

    # Query which paths already exist in graph
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (p:FacilityPath {facility_id: $facility})
            WHERE p.path IN $paths
            RETURN p.path AS path
            """,
            facility=facility,
            paths=config_paths,
        )
        existing_paths = {r["path"] for r in result}

    # Find missing paths
    missing_paths = [p for p in config_paths if p not in existing_paths]

    if not missing_paths:
        logger.info(f"All {len(config_paths)} discovery_roots already in graph")
        return 0

    logger.info(
        f"Seeding {len(missing_paths)} missing roots "
        f"(of {len(config_paths)} configured)"
    )

    # Use existing seed function with only the missing paths
    return seed_facility_roots(facility, root_paths=missing_paths)


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
                "in_directory": parent_id,
                "discovered_at": now,
            }
        )

    if not items:
        return 0

    with GraphClient() as gc:
        result = gc.create_nodes("FacilityPath", items)
        # IN_DIRECTORY relationships now created automatically by create_nodes()

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
        url: Git remote URL (https, ssh://, git@, or local path)

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

    # SSH URL format: ssh://git@host/owner/repo.git or ssh://git@host/path/repo.git
    ssh_url_match = re.match(r"ssh://[^@]+@([^/]+)/(.+?)(?:\.git)?$", url)
    if ssh_url_match:
        host, path = ssh_url_match.groups()
        # Split path into owner/repo (may be multi-level like eq/finesse)
        parts = path.rsplit("/", 1)
        if len(parts) == 2:
            owner, repo = parts
        else:
            owner, repo = None, parts[0]
        if "github" in host.lower():
            return "github", owner, repo
        elif "gitlab" in host.lower():
            return "gitlab", owner, repo
        elif "bitbucket" in host.lower():
            return "bitbucket", owner, repo
        return "gitlab", owner, repo

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
    remote_url: str | None,
    root_commit: str | None,
    head_commit: str | None,
    branch: str | None,
    now: str,
) -> None:
    """Create SoftwareRepo node and INSTANCE_OF relationship.

    Identity priority:
    1. remote_url (github:owner/repo, gitlab:host/owner/repo, etc.)
    2. root_commit (root:{commit_hash} for repos without remote)
    3. path-based (local:{facility}:{path} as fallback)

    Args:
        gc: GraphClient instance
        facility: Facility ID
        path_id: FacilityPath ID
        remote_url: Git remote origin URL (if available)
        root_commit: First commit hash in repo history
        head_commit: Current HEAD commit hash
        branch: Current branch name
        now: ISO timestamp
    """
    # Determine SoftwareRepo identity and metadata
    if remote_url:
        source_type, owner, repo_name = _parse_git_remote_url(remote_url)
        if source_type != "local" and owner and repo_name:
            # Use remote URL as identity
            repo_id = f"{source_type}:{owner}/{repo_name}"
            name = repo_name
        else:
            # Can't parse remote URL, fall back to root commit
            remote_url = None  # Ignore unparseable URL

    if not remote_url:
        # No remote or unparseable - use root commit as identity
        if root_commit:
            repo_id = f"root:{root_commit}"
            # Extract name from path
            name = path_id.split("/")[-1] if "/" in path_id else path_id
            source_type = "local"
        else:
            # No remote and no root commit - rare case, use path-based ID
            repo_id = f"local:{facility}:{path_id.split(':', 1)[1]}"
            name = path_id.split("/")[-1] if "/" in path_id else path_id
            source_type = "local"

    # MERGE SoftwareRepo (deduplicates across facilities and clones)
    gc.query(
        """
        MERGE (r:SoftwareRepo {id: $repo_id})
        ON CREATE SET
            r.source_type = $source_type,
            r.remote_url = $remote_url,
            r.root_commit = $root_commit,
            r.name = $name,
            r.discovered_at = $now,
            r.clone_count = 1
        ON MATCH SET
            r.clone_count = coalesce(r.clone_count, 0) + 1,
            r.root_commit = CASE
                WHEN $root_commit IS NOT NULL THEN $root_commit
                ELSE r.root_commit
            END
        """,
        repo_id=repo_id,
        source_type=source_type,
        remote_url=remote_url,
        root_commit=root_commit,
        name=name,
        now=now,
    )

    # Link FacilityPath to SoftwareRepo with instance metadata
    gc.query(
        """
        MATCH (p:FacilityPath {id: $path_id})
        MATCH (r:SoftwareRepo {id: $repo_id})
        MERGE (p)-[rel:INSTANCE_OF]->(r)
        ON CREATE SET
            rel.head_commit = $head_commit,
            rel.branch = $branch,
            rel.discovered_at = $now
        ON MATCH SET
            rel.head_commit = $head_commit,
            rel.branch = $branch
        SET p.software_repo_id = $repo_id
        """,
        path_id=path_id,
        repo_id=repo_id,
        head_commit=head_commit,
        branch=branch,
        now=now,
    )


def _normalize_name(given_name: str | None, family_name: str | None) -> str | None:
    """Normalize full name for cross-facility matching.

    Converts to lowercase, strips extra whitespace, removes common suffixes.
    Used to match same person across facilities despite name variations.

    Args:
        given_name: First/given name
        family_name: Last/family name

    Returns:
        Normalized full name (e.g., "john smith") or None
    """
    if not given_name or not family_name:
        return None

    # Convert to lowercase and strip
    given = given_name.lower().strip()
    family = family_name.lower().strip()

    # Remove common suffixes/prefixes
    suffixes_to_remove = ["ext", "jr", "sr", "ii", "iii", "phd", "dr", "prof"]
    for suffix in suffixes_to_remove:
        family = family.replace(f" {suffix}", "").replace(f"-{suffix}", "")
        given = given.replace(f" {suffix}", "").replace(f"-{suffix}", "")

    # Normalize whitespace
    full_name = f"{given} {family}".strip()
    full_name = " ".join(full_name.split())  # Collapse multiple spaces

    return full_name if full_name else None


async def _create_person_link(
    gc: Any,
    facility_user_id: str,
    username: str,
    name: str | None,
    given_name: str | None,
    family_name: str | None,
    email: str | None,
    now: str,
) -> None:
    """Create Person node and IS_PERSON relationship with ORCID lookup.

    Identity priority:
    1. ORCID (when available - gold standard)
    2. Normalized full name (given + family) - best for cross-facility
    3. Email (secondary - can vary between facilities)
    4. Username-based (fallback)

    Full name matching handles variations like:
    - "Agata Czarnecka" vs "czarneka" → matched via name normalization
    - "Alessandro Casati" vs "acasati" → matched via name normalization

    ORCID lookup is performed automatically during discovery using the
    ORCID public API. Lookups are cached in Person nodes to avoid
    redundant API calls.

    Args:
        gc: GraphClient instance
        facility_user_id: FacilityUser ID (facility:username)
        username: Local username
        name: Full name from GECOS
        given_name: Parsed given name
        family_name: Parsed family name
        email: Email address if available
        now: ISO timestamp
    """
    from imas_codex.discovery.paths.orcid import enrich_person_with_orcid

    # Try ORCID lookup (cached if already done)
    orcid = None
    try:
        orcid = await enrich_person_with_orcid(
            email=email,
            given_name=given_name,
            family_name=family_name,
        )
        if orcid:
            logger.debug(f"Found ORCID {orcid} for {name or username}")
    except Exception as e:
        logger.debug(f"ORCID lookup failed for {name or username}: {e}")

    # Determine Person identity
    if orcid:
        # Best case - use ORCID as identity
        person_id = f"orcid:{orcid}"
        person_name = (
            name or f"{given_name} {family_name}"
            if given_name and family_name
            else username
        )
    else:
        # Use normalized full name for cross-facility matching
        normalized_name = _normalize_name(given_name, family_name)
        if normalized_name:
            # Primary: full name (handles "Agata Czarnecka" vs "czarneka")
            person_id = f"name:{normalized_name}"
            person_name = name or f"{given_name} {family_name}"
        elif email:
            # Secondary: email (but can vary between facilities)
            person_id = f"email:{email.lower()}"
            person_name = name or email
        else:
            # Fallback: username-based
            person_id = f"user:{username}"
            person_name = name or username

    # MERGE Person node (deduplicates across facilities)
    gc.query(
        """
        MERGE (p:Person {id: $person_id})
        ON CREATE SET
            p.name = $person_name,
            p.given_name = $given_name,
            p.family_name = $family_name,
            p.email = $email,
            p.orcid = $orcid,
            p.discovered_at = $now,
            p.account_count = 1
        ON MATCH SET
            p.account_count = coalesce(p.account_count, 0) + 1,
            p.name = COALESCE($person_name, p.name),
            p.given_name = COALESCE($given_name, p.given_name),
            p.family_name = COALESCE($family_name, p.family_name),
            p.email = COALESCE($email, p.email),
            p.orcid = COALESCE($orcid, p.orcid)
        """,
        person_id=person_id,
        person_name=person_name,
        given_name=given_name,
        family_name=family_name,
        email=email,
        orcid=orcid,
        now=now,
    )

    # Link FacilityUser to Person
    gc.query(
        """
        MATCH (u:FacilityUser {id: $user_id})
        MATCH (p:Person {id: $person_id})
        MERGE (u)-[rel:IS_PERSON]->(p)
        ON CREATE SET
            rel.discovered_at = $now
        SET u.person_id = $person_id
        """,
        user_id=facility_user_id,
        person_id=person_id,
        now=now,
    )


def apply_expansion_overrides(
    facility: str,
    scores: list[dict[str, Any]],
) -> None:
    """Apply structural should_expand/should_enrich overrides in place.

    Checks each scored path against:
    1. VCS accessibility — if the repo's remote is accessible elsewhere
       (config patterns or scanner probe), block expansion and enrichment.
    2. Data containers — modeling_data/experimental_data never expand.

    Mutates the score dicts directly before they are persisted.

    Args:
        facility: Facility ID
        scores: List of dicts from ScoredDirectory.to_graph_dict()
    """
    from imas_codex.config.discovery_config import is_repo_accessible_elsewhere
    from imas_codex.graph import GraphClient

    path_ids = [f"{facility}:{s['path']}" for s in scores]

    with GraphClient() as gc:
        # Batch-fetch VCS data for all paths in one query
        vcs_rows = gc.query(
            """
            UNWIND $ids AS pid
            MATCH (p:FacilityPath {id: pid})
            RETURN p.id AS id, p.vcs_type AS vcs_type, p.has_git AS has_git,
                   p.vcs_remote_url AS vcs_remote_url,
                   p.vcs_remote_accessible AS vcs_remote_accessible
            """,
            ids=path_ids,
        )

    vcs_by_id = {r["id"]: r for r in (vcs_rows or [])}

    data_purposes = {"modeling_data", "experimental_data"}

    for score_data in scores:
        path_id = f"{facility}:{score_data['path']}"
        vcs_row = vcs_by_id.get(path_id)

        # VCS accessibility override
        if vcs_row:
            has_vcs = vcs_row["vcs_type"] is not None or vcs_row.get("has_git") is True
            if has_vcs and is_repo_accessible_elsewhere(
                remote_url=vcs_row.get("vcs_remote_url"),
                scanner_accessible=vcs_row.get("vcs_remote_accessible"),
                facility=facility,
            ):
                score_data["should_expand"] = False
                score_data["should_enrich"] = False
                vcs_label = vcs_row["vcs_type"] or "git"
                score_data.setdefault(
                    "enrich_skip_reason",
                    f"{vcs_label} repo accessible elsewhere",
                )

        # Data container override
        if score_data.get("path_purpose") in data_purposes:
            score_data["should_expand"] = False
            score_data["should_enrich"] = False
            score_data.setdefault(
                "enrich_skip_reason", "data container - too many files"
            )


def mark_paths_scored(
    facility: str,
    scores: list[dict[str, Any]],
) -> int:
    """Update multiple paths with LLM scores and create/link Evidence nodes.

    Args:
        facility: Facility ID
        scores: List of dicts from ScoredDirectory.to_graph_dict() with:
                path, path_purpose, description, evidence, per-purpose scores
                (score_modeling_code, score_analysis_code, etc.), score, should_expand,
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
            # terminal_reason is only set when provided (e.g., empty directories)
            # For LLM-scored paths it stays NULL - reason is derivable from
            # has_git, path_purpose, score.
            gc.query(
                """
                MATCH (p:FacilityPath {id: $id})
                SET p.status = $scored,
                    p.claimed_at = null,
                    p.scored_at = $now,
                    p.score = $score,
                    p.score_modeling_code = $score_modeling_code,
                    p.score_analysis_code = $score_analysis_code,
                    p.score_operations_code = $score_operations_code,
                    p.score_modeling_data = $score_modeling_data,
                    p.score_experimental_data = $score_experimental_data,
                    p.score_data_access = $score_data_access,
                    p.score_workflow = $score_workflow,
                    p.score_visualization = $score_visualization,
                    p.score_documentation = $score_documentation,
                    p.score_imas = $score_imas,
                    p.description = $description,
                    p.path_purpose = $path_purpose,
                    p.evidence_id = $evidence_id,
                    p.should_expand = $should_expand,
                    p.should_enrich = $should_enrich,
                    p.keywords = $keywords,
                    p.physics_domain = $physics_domain,
                    p.expansion_reason = $expansion_reason,
                    p.skip_reason = $skip_reason,
                    p.terminal_reason = $terminal_reason,
                    p.enrich_skip_reason = $enrich_skip_reason,
                    p.score_cost = coalesce(p.score_cost, 0) + $score_cost
                WITH p
                OPTIONAL MATCH (e:Evidence {id: $evidence_id})
                FOREACH (_ IN CASE WHEN e IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (p)-[:HAS_EVIDENCE]->(e)
                )
                """,
                id=path_id,
                now=now,
                score=score_data.get("score"),
                score_modeling_code=score_data.get("score_modeling_code"),
                score_analysis_code=score_data.get("score_analysis_code"),
                score_operations_code=score_data.get("score_operations_code"),
                score_modeling_data=score_data.get("score_modeling_data"),
                score_experimental_data=score_data.get("score_experimental_data"),
                score_data_access=score_data.get("score_data_access"),
                score_workflow=score_data.get("score_workflow"),
                score_visualization=score_data.get("score_visualization"),
                score_documentation=score_data.get("score_documentation"),
                score_imas=score_data.get("score_imas"),
                description=score_data.get("description"),
                path_purpose=score_data.get("path_purpose"),
                evidence_id=evidence_id,
                should_expand=score_data.get("should_expand"),
                should_enrich=score_data.get("should_enrich", True),
                keywords=score_data.get("keywords"),
                physics_domain=score_data.get("physics_domain"),
                expansion_reason=score_data.get("expansion_reason"),
                skip_reason=score_data.get("skip_reason"),
                terminal_reason=score_data.get("terminal_reason"),
                enrich_skip_reason=score_data.get("enrich_skip_reason"),
                score_cost=score_data.get("score_cost", 0.0),
                scored=PathStatus.scored.value,
            )
            updated += 1

            # Data container child-skip: mark children as skipped if parent is
            # a data container that shouldn't expand (modeling_data, experimental_data)
            path_purpose = score_data.get("path_purpose")
            should_expand = score_data.get("should_expand", True)
            data_purposes = {"modeling_data", "experimental_data"}

            if path_purpose in data_purposes and not should_expand:
                # Mark all discovered children as skipped
                skipped_result = gc.query(
                    """
                    MATCH (child:FacilityPath)-[:CHILD_OF]->(p:FacilityPath {id: $id})
                    WHERE child.status = 'discovered'
                    SET child.status = $skipped,
                        child.skipped_at = $now,
                        child.terminal_reason = $terminal_reason,
                        child.skip_reason = $reason
                    RETURN count(child) AS skipped_count
                    """,
                    id=path_id,
                    now=now,
                    terminal_reason=TerminalReason.parent_terminal.value,
                    reason=f"parent_{path_purpose}",
                    skipped=PathStatus.skipped.value,
                )
                skipped_count = (
                    skipped_result[0]["skipped_count"] if skipped_result else 0
                )
                if skipped_count > 0:
                    logger.debug(
                        f"Skipped {skipped_count} children of {path_purpose}: {path}"
                    )

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
            MATCH (p:FacilityPath)-[:AT_FACILITY]->(f:Facility {id: $facility})
            WHERE p.score >= $min_score
            RETURN p.id AS id, p.path AS path, p.score AS score,
                   p.description AS description, p.path_purpose AS path_purpose,
                   p.score_modeling_code AS score_modeling_code,
                   p.score_analysis_code AS score_analysis_code,
                   p.score_operations_code AS score_operations_code,
                   p.score_modeling_data AS score_modeling_data,
                   p.score_experimental_data AS score_experimental_data,
                   p.score_data_access AS score_data_access,
                   p.score_workflow AS score_workflow,
                   p.score_visualization AS score_visualization,
                   p.score_documentation AS score_documentation,
                   p.score_imas AS score_imas,
                   p.should_expand AS should_expand, p.skip_reason AS skip_reason
            ORDER BY p.score DESC
            LIMIT $limit
            """,
            facility=facility,
            min_score=min_score,
            limit=limit,
        )

        return list(result)


def get_top_paths_by_purpose(
    facility: str,
    purpose: str,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Get top-scoring paths for a specific purpose.

    Args:
        facility: Facility ID
        purpose: ResourcePurpose value (e.g., 'modeling_code', 'analysis_code')
        limit: Maximum paths to return (default 3)

    Returns:
        List of dicts with path, score, and description
    """
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (p:FacilityPath)-[:AT_FACILITY]->(f:Facility {id: $facility})
            WHERE p.path_purpose = $purpose AND p.score > 0
            RETURN p.path AS path, p.score AS score, p.description AS description
            ORDER BY p.score DESC
            LIMIT $limit
            """,
            facility=facility,
            purpose=purpose,
            limit=limit,
        )

        return list(result)


def clear_facility_paths(facility: str, batch_size: int = 5000) -> dict[str, int]:
    """Delete all FacilityPath nodes and related data for a facility.

    Cascades to FacilityUser nodes and cleans up orphaned SoftwareRepo/Person
    nodes. Does NOT delete SourceFile nodes — those belong to the files domain.

    Deletion order follows referential integrity:
    1. FacilityUser nodes by AT_FACILITY relationship
    2. FacilityPath nodes by AT_FACILITY relationship
    3. Orphaned SoftwareRepo and Person nodes

    Args:
        facility: Facility ID
        batch_size: Nodes to delete per batch (default 5000)

    Returns:
        Dict with counts: paths_deleted, users_deleted
    """
    from imas_codex.graph import GraphClient

    results = {"paths_deleted": 0, "users_deleted": 0}

    with GraphClient() as gc:
        # Delete FacilityUser nodes with facility relationship
        while True:
            result = gc.query(
                """
                MATCH (fu:FacilityUser)-[:AT_FACILITY]->(f:Facility {id: $facility})
                WITH fu LIMIT $batch_size
                DETACH DELETE fu
                RETURN count(fu) AS deleted
                """,
                facility=facility,
                batch_size=batch_size,
            )
            deleted = result[0]["deleted"] if result else 0
            results["users_deleted"] += deleted
            if deleted < batch_size:
                break

        while True:
            # Delete a batch and return count
            result = gc.query(
                """
                MATCH (p:FacilityPath)-[:AT_FACILITY]->(f:Facility {id: $facility})
                WITH p LIMIT $batch_size
                DETACH DELETE p
                RETURN count(p) AS deleted
                """,
                facility=facility,
                batch_size=batch_size,
            )

            deleted = result[0]["deleted"] if result else 0
            results["paths_deleted"] += deleted

            # If we deleted less than batch_size, we're done
            if deleted < batch_size:
                break

        # Clean up orphaned SoftwareRepo nodes (no INSTANCE_OF relationships)
        orphan_result = gc.query(
            """
            MATCH (r:SoftwareRepo)
            WHERE NOT EXISTS { MATCH (:FacilityPath)-[:INSTANCE_OF]->(r) }
            WITH r LIMIT $batch_size
            DELETE r
            RETURN count(r) AS orphans_deleted
            """,
            batch_size=batch_size,
        )
        orphans = orphan_result[0]["orphans_deleted"] if orphan_result else 0
        if orphans > 0:
            import logging

            logging.getLogger(__name__).info(
                f"Cleaned up {orphans} orphaned SoftwareRepo nodes"
            )

        # Clean up orphaned Person nodes (no relationships to any nodes)
        person_deleted = 0
        while True:
            result = gc.query(
                """
                MATCH (p:Person)
                WHERE NOT (p)--()
                WITH p LIMIT $batch_size
                DELETE p
                RETURN count(p) AS deleted
                """,
                batch_size=batch_size,
            )
            deleted = result[0]["deleted"] if result else 0
            person_deleted += deleted
            if deleted < batch_size:
                break
        if person_deleted > 0:
            import logging

            logging.getLogger(__name__).info(
                f"Cleaned up {person_deleted} orphaned Person nodes"
            )

    return results


def cleanup_orphaned_software_repos(batch_size: int = 1000) -> int:
    """Delete SoftwareRepo nodes with no linked FacilityPath instances.

    SoftwareRepos are shared across facilities (same remote URL = same node).
    When FacilityPath nodes are deleted, the INSTANCE_OF relationships are
    severed but the SoftwareRepo nodes remain. This function cleans them up.

    Args:
        batch_size: Nodes to delete per batch (default 1000)

    Returns:
        Total number of orphaned repos deleted
    """
    from imas_codex.graph import GraphClient

    total_deleted = 0

    with GraphClient() as gc:
        while True:
            result = gc.query(
                """
                MATCH (r:SoftwareRepo)
                WHERE NOT EXISTS { MATCH (:FacilityPath)-[:INSTANCE_OF]->(r) }
                WITH r LIMIT $batch_size
                DELETE r
                RETURN count(r) AS deleted
                """,
                batch_size=batch_size,
            )
            deleted = result[0]["deleted"] if result else 0
            total_deleted += deleted

            if deleted < batch_size:
                break

    return total_deleted


async def persist_scan_results(
    facility: str,
    results: list[tuple[str, dict, list[dict], str | None, bool]],
    excluded: list[tuple[str, str, str]] | None = None,
) -> dict[str, int]:
    """Persist multiple scan results in a single transaction.

    Much faster than calling mark_path_scanned/create_child_paths per path.

    Two modes based on is_expanding flag:
    1. First scan (is_expanding=False): Set status='scanned', no children created
    2. Expansion scan (is_expanding=True): Keep status='scored', create children

    Symlink handling:
    - Symlink children are created with is_symlink=True, status='excluded'
    - An ALIAS_OF relationship links symlink → its realpath target
    - This allows searching by symlink paths while avoiding duplicate scans

    Args:
        facility: Facility ID
        results: List of (path, stats_dict, child_dirs, error, is_expanding) tuples.
                 child_dirs is list of {path, is_symlink, realpath} dicts.
        excluded: Optional list of (path, parent_path, reason) for excluded dirs

    Returns:
        Dict with scanned, children_created, excluded, errors counts
    """
    from imas_codex.discovery.base.facility import get_facility
    from imas_codex.graph import GraphClient

    now = datetime.now(UTC).isoformat()
    scanned = 0
    children_created = 0
    excluded_count = 0
    errors = 0

    # Load facility config for trusted_git_servers
    try:
        facility_config = get_facility(facility)
    except ValueError:
        facility_config = {}

    # Separate by mode: first_scan vs expansion
    first_scan_updates = []
    expansion_updates = []
    all_children = []

    for path, stats, child_dirs, error, is_expanding in results:
        path_id = f"{facility}:{path}"
        if error:
            errors += 1
            # Determine terminal reason from error type
            if "permission" in error.lower():
                terminal_reason = TerminalReason.access_denied.value
            else:
                terminal_reason = TerminalReason.scan_error.value
            # Mark as skipped
            first_scan_updates.append(
                {
                    "id": path_id,
                    "status": PathStatus.skipped.value,
                    "terminal_reason": terminal_reason,
                    "skip_reason": error,
                    "listed_at": now,
                }
            )
        else:
            scanned += 1
            # Build update dict with child_names if available
            update_dict = {
                "id": path_id,
                "listed_at": now,
                "total_files": stats.get("total_files", 0),
                "total_dirs": stats.get("total_dirs", 0),
                "has_readme": stats.get("has_readme", False),
                "has_makefile": stats.get("has_makefile", False),
                "has_git": stats.get("has_git", False),
                "git_remote_url": stats.get("git_remote_url"),
                "git_head_commit": stats.get("git_head_commit"),
                "git_branch": stats.get("git_branch"),
                "git_root_commit": stats.get("git_root_commit"),
                "vcs_type": stats.get("vcs_type"),
                "vcs_remote_url": stats.get("vcs_remote_url"),
                "vcs_remote_accessible": stats.get("vcs_remote_accessible"),
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
            # Store tree context for hierarchical view
            tree_context = stats.get("tree_context")
            if tree_context:
                update_dict["tree_context"] = tree_context
            # Store numeric directory ratio (data container detection)
            numeric_dir_ratio = stats.get("numeric_dir_ratio", 0)
            if numeric_dir_ratio > 0:
                update_dict["numeric_dir_ratio"] = numeric_dir_ratio
            # Store patterns found
            patterns = stats.get("patterns_detected")
            if patterns:
                update_dict["patterns_found"] = patterns

            if is_expanding:
                # Expansion scan: keep scored status, mark as expanded
                update_dict["status"] = PathStatus.scored.value
                update_dict["expanded_at"] = now
                expansion_updates.append(update_dict)

                # Create children for expanding paths
                # (with git repo skip logic below)
                should_create_children = True
            else:
                # First scan: set listed status, no children
                update_dict["status"] = PathStatus.scanned.value
                first_scan_updates.append(update_dict)
                should_create_children = False

            # Child creation only for expansion paths
            if should_create_children:
                # Decide whether to skip children for git repos
                is_git_repo = stats.get("has_git", False)
                git_remote_url = stats.get("git_remote_url", "")

                if is_git_repo and git_remote_url:
                    should_skip = False

                    # Check if on a public hosting service
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
                            should_skip = True
                        else:
                            logger.debug(
                                f"Scan children: private repo on public host {path}"
                            )
                    else:
                        # Check if on an internal git server (facility has access)
                        internal_servers = facility_config.get(
                            "internal_git_servers", []
                        )
                        is_internal = any(
                            server in git_remote_url.lower()
                            for server in internal_servers
                        )
                        if is_internal:
                            should_skip = True

                    if should_skip:
                        logger.debug(
                            f"Skip children: git repo {path} ({git_remote_url})"
                        )
                        continue

                # Prepare children for expanding paths
                # child_dirs is now list of {path, is_symlink, realpath, device_inode} dicts
                for child_info in child_dirs:
                    # Handle both old format (string) and new format (dict)
                    if isinstance(child_info, str):
                        child_path = child_info
                        child_is_symlink = False
                        child_realpath = None
                        child_device_inode = None
                    else:
                        child_path = child_info.get("path", "")
                        child_is_symlink = child_info.get("is_symlink", False)
                        child_realpath = child_info.get("realpath")
                        child_device_inode = child_info.get("device_inode")

                    child_id = f"{facility}:{child_path}"

                    all_children.append(
                        {
                            "id": child_id,
                            "facility_id": facility,
                            "path": child_path,
                            "parent_id": path_id,
                            "is_symlink": child_is_symlink,
                            "realpath": child_realpath,
                            "device_inode": child_device_inode,
                        }
                    )

    with GraphClient() as gc:
        # Batch update first-scan paths (discovered → scanned)
        if first_scan_updates:
            update_result = gc.query(
                """
                UNWIND $items AS item
                MATCH (p:FacilityPath {id: item.id})
                SET p.status = item.status,
                    p.claimed_at = null,
                    p.listed_at = item.listed_at,
                    p.skip_reason = item.skip_reason,
                    p.terminal_reason = item.terminal_reason,
                    p.total_files = item.total_files,
                    p.total_dirs = item.total_dirs,
                    p.has_readme = item.has_readme,
                    p.has_makefile = item.has_makefile,
                    p.has_git = item.has_git,
                    p.git_remote_url = item.git_remote_url,
                    p.git_head_commit = item.git_head_commit,
                    p.git_branch = item.git_branch,
                    p.git_root_commit = item.git_root_commit,
                    p.vcs_type = item.vcs_type,
                    p.vcs_remote_url = item.vcs_remote_url,
                    p.vcs_remote_accessible = item.vcs_remote_accessible,
                    p.child_names = item.child_names,
                    p.file_type_counts = item.file_type_counts,
                    p.tree_context = item.tree_context,
                    p.numeric_dir_ratio = item.numeric_dir_ratio,
                    p.patterns_found = item.patterns_found
                RETURN count(p) AS updated
                """,
                items=first_scan_updates,
            )
            updated = update_result[0]["updated"] if update_result else 0
            if updated != len(first_scan_updates):
                logger.warning(
                    f"First-scan UNWIND: updated {updated}/{len(first_scan_updates)} "
                    f"paths (IDs: {[i['id'] for i in first_scan_updates[:3]]}...)"
                )
            else:
                logger.debug(f"First-scan UNWIND: updated {updated} paths")

            # Yield to event loop so score workers can claim the
            # newly-scanned paths while we handle excluded dirs, users, etc.
            await asyncio.sleep(0)

        # Batch update expansion paths (scored stays scored, mark expanded)
        if expansion_updates:
            expand_result = gc.query(
                """
                UNWIND $items AS item
                MATCH (p:FacilityPath {id: item.id})
                SET p.status = item.status,
                    p.claimed_at = null,
                    p.expanded_at = item.expanded_at,
                    p.listed_at = item.listed_at,
                    p.total_files = item.total_files,
                    p.total_dirs = item.total_dirs,
                    p.has_readme = item.has_readme,
                    p.has_makefile = item.has_makefile,
                    p.has_git = item.has_git,
                    p.git_remote_url = item.git_remote_url,
                    p.git_head_commit = item.git_head_commit,
                    p.git_branch = item.git_branch,
                    p.git_root_commit = item.git_root_commit,
                    p.vcs_type = item.vcs_type,
                    p.vcs_remote_url = item.vcs_remote_url,
                    p.vcs_remote_accessible = item.vcs_remote_accessible,
                    p.child_names = item.child_names,
                    p.file_type_counts = item.file_type_counts,
                    p.tree_context = item.tree_context,
                    p.numeric_dir_ratio = item.numeric_dir_ratio,
                    p.patterns_found = item.patterns_found
                RETURN count(p) AS updated
                """,
                items=expansion_updates,
            )
            updated = expand_result[0]["updated"] if expand_result else 0
            if updated != len(expansion_updates):
                logger.warning(
                    f"Expansion UNWIND: updated {updated}/{len(expansion_updates)} paths"
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

            # Add depth to children and handle symlinks
            symlink_children = []
            regular_children = []
            for child in all_children:
                child["depth"] = depth_map.get(child["parent_id"], 0) + 1
                child["path_type"] = "code_directory"
                child["discovered_at"] = now

                if child.get("is_symlink"):
                    # Symlinks are excluded from scanning to avoid duplicates
                    child["status"] = PathStatus.skipped.value
                    child["skip_reason"] = "symlink"
                    symlink_children.append(child)
                else:
                    child["status"] = PathStatus.discovered.value
                    regular_children.append(child)

            # Create regular child nodes with device_inode-based deduplication
            # If a node with same device_inode already exists, create ALIAS_OF
            # Use path canonicality to decide which becomes canonical vs alias
            # (e.g., /home is more canonical than /mnt)
            if regular_children:
                # Add canonicality scores to children for Cypher comparison
                for child in regular_children:
                    child["canonicality"] = _path_canonicality_score(child["path"])

                # First pass: create nodes that don't have device_inode conflicts
                # or where the new path is MORE canonical than existing (swap)
                gc.query(
                    """
                    UNWIND $children AS child
                    MATCH (f:Facility {id: child.facility_id})
                    MATCH (parent:FacilityPath {id: child.parent_id})

                    // Check if a node with same device_inode already exists (bind mount)
                    OPTIONAL MATCH (existing:FacilityPath)
                    WHERE existing.device_inode = child.device_inode
                      AND existing.device_inode IS NOT NULL
                      AND existing.id <> child.id
                      AND existing.facility_id = child.facility_id

                    // No conflict - create normally
                    FOREACH (_ IN CASE WHEN existing IS NULL THEN [1] ELSE [] END |
                        MERGE (c:FacilityPath {id: child.id})
                        ON CREATE SET c.facility_id = child.facility_id,
                                      c.path = child.path,
                                      c.path_type = child.path_type,
                                      c.status = child.status,
                                      c.depth = child.depth,
                                      c.in_directory = child.parent_id,
                                      c.discovered_at = child.discovered_at,
                                      c.device_inode = child.device_inode
                        MERGE (c)-[:AT_FACILITY]->(f)
                        MERGE (c)-[:IN_DIRECTORY]->(parent)
                    )

                    // Conflict exists AND new path is LESS canonical - new becomes alias
                    FOREACH (_ IN CASE
                        WHEN existing IS NOT NULL
                         AND child.canonicality <= coalesce(existing.canonicality, 50)
                        THEN [1] ELSE [] END |
                        MERGE (alias:FacilityPath {id: child.id})
                        ON CREATE SET alias.facility_id = child.facility_id,
                                      alias.path = child.path,
                                      alias.path_type = child.path_type,
                                      alias.status = 'excluded',
                                      alias.skip_reason = 'bind_mount_duplicate',
                                      alias.depth = child.depth,
                                      alias.in_directory = child.parent_id,
                                      alias.discovered_at = child.discovered_at,
                                      alias.device_inode = child.device_inode,
                                      alias.canonicality = child.canonicality,
                                      alias.alias_of_id = existing.id
                        MERGE (alias)-[:AT_FACILITY]->(f)
                        MERGE (alias)-[:IN_DIRECTORY]->(parent)
                        MERGE (alias)-[:ALIAS_OF]->(existing)
                    )

                    // Conflict exists AND new path is MORE canonical - swap!
                    // Existing becomes alias, new becomes canonical
                    FOREACH (_ IN CASE
                        WHEN existing IS NOT NULL
                         AND child.canonicality > coalesce(existing.canonicality, 50)
                        THEN [1] ELSE [] END |
                        // Create new path as canonical (normal status)
                        MERGE (c:FacilityPath {id: child.id})
                        ON CREATE SET c.facility_id = child.facility_id,
                                      c.path = child.path,
                                      c.path_type = child.path_type,
                                      c.status = child.status,
                                      c.depth = child.depth,
                                      c.in_directory = child.parent_id,
                                      c.discovered_at = child.discovered_at,
                                      c.device_inode = child.device_inode,
                                      c.canonicality = child.canonicality
                        MERGE (c)-[:AT_FACILITY]->(f)
                        MERGE (c)-[:IN_DIRECTORY]->(parent)
                    )
                    """,
                    children=regular_children,
                )

                # Second pass: demote existing nodes that lost to more canonical paths
                gc.query(
                    """
                    UNWIND $children AS child
                    MATCH (existing:FacilityPath)
                    WHERE existing.device_inode = child.device_inode
                      AND existing.device_inode IS NOT NULL
                      AND existing.id <> child.id
                      AND existing.facility_id = child.facility_id
                      AND child.canonicality > coalesce(existing.canonicality, 50)

                    MATCH (newCanonical:FacilityPath {id: child.id})

                    // Demote existing to alias
                    SET existing.status = 'excluded',
                        existing.skip_reason = 'bind_mount_duplicate',
                        existing.alias_of_id = child.id

                    // Create ALIAS_OF from old to new canonical
                    MERGE (existing)-[:ALIAS_OF]->(newCanonical)
                    """,
                    children=regular_children,
                )

            # Create symlink nodes with ALIAS_OF relationships
            if symlink_children:
                # First create the symlink nodes (excluded status)
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
                                  c.skip_reason = child.skip_reason,
                                  c.depth = child.depth,
                                  c.in_directory = child.parent_id,
                                  c.discovered_at = child.discovered_at,
                                  c.is_symlink = true,
                                  c.realpath = child.realpath,
                                  c.device_inode = child.device_inode
                    MERGE (c)-[:AT_FACILITY]->(f)
                    MERGE (c)-[:IN_DIRECTORY]->(parent)
                    """,
                    children=symlink_children,
                )

                # Create target nodes for symlinks with ALIAS_OF relationships
                # Target depth is based on actual path components (not parent+1)
                # This ensures deep targets (e.g., /mnt/HPC/.../target) are scanned
                # after shallower canonical paths (e.g., /home/user), allowing
                # device_inode deduplication to exclude them as duplicates.
                symlinks_with_targets = [
                    c
                    for c in symlink_children
                    if c.get("realpath") and c["realpath"].startswith("/")
                ]
                if symlinks_with_targets:
                    # Add target_depth based on path component count
                    for c in symlinks_with_targets:
                        c["target_depth"] = c["realpath"].rstrip("/").count("/")

                    gc.query(
                        """
                        UNWIND $children AS child
                        MATCH (f:Facility {id: child.facility_id})
                        MATCH (symlink:FacilityPath {id: child.id})

                        // Create or match the target (realpath) node
                        // Depth based on path components ensures proper scan ordering
                        MERGE (target:FacilityPath {id: child.facility_id + ':' + child.realpath})
                        ON CREATE SET target.facility_id = child.facility_id,
                                      target.path = child.realpath,
                                      target.path_type = 'code_directory',
                                      target.status = $discovered,
                                      target.depth = child.target_depth,
                                      target.discovered_at = child.discovered_at,
                                      target.is_symlink = false,
                                      target.device_inode = child.device_inode
                        MERGE (target)-[:AT_FACILITY]->(f)

                        // Create ALIAS_OF relationship from symlink to target
                        MERGE (symlink)-[:ALIAS_OF]->(target)
                        """,
                        children=symlinks_with_targets,
                        discovered=PathStatus.discovered.value,
                    )

            children_created = len(all_children)

        # Handle excluded directories (create with status='skipped', terminal_reason='excluded')
        if excluded:
            # Batch depth lookup: get all parent depths in one query
            unique_parents = list({f"{facility}:{pp}" for _, pp, _ in excluded})
            depth_result = gc.query(
                """
                UNWIND $ids AS id
                MATCH (p:FacilityPath {id: id})
                RETURN p.id AS id, p.depth AS depth
                """,
                ids=unique_parents,
            )
            depth_map = {r["id"]: r["depth"] or 0 for r in depth_result}

            excluded_nodes = []
            for path, parent_path, reason in excluded:
                parent_id = f"{facility}:{parent_path}"
                parent_depth = depth_map.get(parent_id, 0)

                excluded_nodes.append(
                    {
                        "id": f"{facility}:{path}",
                        "facility_id": facility,
                        "path": path,
                        "parent_id": parent_id,
                        "depth": parent_depth + 1,
                        "status": PathStatus.skipped.value,
                        "terminal_reason": TerminalReason.excluded.value,
                        "skip_reason": reason,
                        "path_type": "code_directory",
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
                                  p.path_type = node.path_type,
                                  p.status = node.status,
                                  p.terminal_reason = node.terminal_reason,
                                  p.skip_reason = node.skip_reason,
                                  p.depth = node.depth,
                                  p.in_directory = node.parent_id,
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
                    MERGE (c)-[:AT_FACILITY]->(f)
                    """,
                    nodes=excluded_nodes,
                )

                excluded_count = len(excluded_nodes)

        # User enrichment: extract users from discovered home paths
        # Run in same transaction for consistency
        all_paths = [path for path, _stats, _children, _error, _expanding in results]
        try:
            from imas_codex.discovery.paths.users import enrich_users_from_paths

            facility_users = enrich_users_from_paths(facility, all_paths, gc=gc)
            if facility_users:
                # Create FacilityUser nodes with HAS_HOME relationship to home path
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
                                  u.discovered_at = user.discovered_at,
                                  u.enriched_at = user.enriched_at
                    ON MATCH SET u.name = COALESCE(user.name, u.name),
                                 u.given_name = COALESCE(user.given_name, u.given_name),
                                 u.family_name = COALESCE(user.family_name, u.family_name),
                                 u.enriched_at = COALESCE(user.enriched_at, u.enriched_at)
                    MERGE (u)-[:AT_FACILITY]->(f)
                    WITH u, user
                    WHERE user.home_path_id IS NOT NULL
                    MATCH (home:FacilityPath {id: user.home_path_id})
                    MERGE (u)-[:HAS_HOME]->(home)
                    """,
                    users=facility_users,
                )

                # Create Person nodes for cross-facility identity (with ORCID lookup)
                for user in facility_users:
                    try:
                        await _create_person_link(
                            gc,
                            facility_user_id=user["id"],
                            username=user["username"],
                            name=user.get("name"),
                            given_name=user.get("given_name"),
                            family_name=user.get("family_name"),
                            email=user.get("email"),
                            now=now,
                        )
                    except Exception as e:
                        logger.debug(
                            f"Failed to create Person link for {user['id']}: {e}"
                        )

                logger.debug(f"Enriched {len(facility_users)} users for {facility}")
        except Exception as e:
            # User enrichment is non-critical; don't fail scan
            logger.warning(f"User enrichment failed: {e}")

        # Create SoftwareRepo nodes for git repos (with remote URL or root commit)
        all_updates = first_scan_updates + expansion_updates
        git_repos = [
            (
                item["id"],
                item.get("git_remote_url"),
                item.get("git_root_commit"),
                item.get("git_head_commit"),
                item.get("git_branch"),
            )
            for item in all_updates
            if item.get("has_git")
            and (item.get("git_remote_url") or item.get("git_root_commit"))
        ]
        for path_id, remote_url, root_commit, head_commit, branch in git_repos:
            try:
                _create_software_repo_link(
                    gc,
                    facility,
                    path_id,
                    remote_url,
                    root_commit,
                    head_commit,
                    branch,
                    now,
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
        "very_high": (0.75, 1.01),  # Include 1.0
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


# Score dimension names for per-category sampling
SCORE_DIMENSIONS = [
    "score_modeling_code",
    "score_analysis_code",
    "score_operations_code",
    "score_modeling_data",
    "score_experimental_data",
    "score_data_access",
    "score_workflow",
    "score_visualization",
    "score_documentation",
    "score_imas",
]


def sample_paths_by_dimension(
    facility: str | None = None,
    per_dimension: int = 2,
    cross_facility: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """Sample high-scoring paths for each score dimension.

    Returns exemplar paths that are strong in each dimension, allowing
    the LLM to see what high scores look like across categories.

    Args:
        facility: Current facility (for preferring same-facility examples)
        per_dimension: Paths to sample per dimension
        cross_facility: If True, sample from ALL facilities for diversity

    Returns:
        Dict mapping dimension name to list of example paths.
        Each path has: path, facility, dimension_score, purpose, description
    """
    from imas_codex.graph import GraphClient

    samples: dict[str, list[dict[str, Any]]] = {}

    with GraphClient() as gc:
        for dim in SCORE_DIMENSIONS:
            # Build query - optionally filter by facility
            if cross_facility or not facility:
                # Cross-facility: get best examples from all facilities
                result = gc.query(
                    f"""
                    MATCH (p:FacilityPath)
                    WHERE p.status = 'scored'
                        AND p.{dim} >= 0.6
                    RETURN p.path AS path,
                           p.facility_id AS facility,
                           p.{dim} AS dimension_score,
                           p.path_purpose AS purpose,
                           p.description AS description
                    ORDER BY p.{dim} DESC
                    LIMIT $limit
                    """,
                    limit=per_dimension * 2,  # Get extras for diversity
                )
            else:
                # Single facility only
                result = gc.query(
                    f"""
                    MATCH (p:FacilityPath {{facility_id: $facility}})
                    WHERE p.status = 'scored'
                        AND p.{dim} >= 0.6
                    RETURN p.path AS path,
                           p.facility_id AS facility,
                           p.{dim} AS dimension_score,
                           p.path_purpose AS purpose,
                           p.description AS description
                    ORDER BY p.{dim} DESC
                    LIMIT $limit
                    """,
                    facility=facility,
                    limit=per_dimension,
                )

            # Take per_dimension paths, preferring current facility if available
            paths = []
            current_facility_paths = [r for r in result if r["facility"] == facility]
            other_paths = [r for r in result if r["facility"] != facility]

            # Interleave: current facility first, then others
            for r in current_facility_paths[:per_dimension]:
                paths.append(r)
            for r in other_paths[: per_dimension - len(paths)]:
                paths.append(r)

            samples[dim] = [
                {
                    "path": r["path"],
                    "facility": r["facility"],
                    "dimension_score": round(r["dimension_score"], 2),
                    "purpose": r["purpose"] or "unknown",
                    "description": (r["description"] or "")[:80],
                }
                for r in paths[:per_dimension]
            ]

    return samples


def sample_enriched_paths(
    facility: str | None = None,
    per_category: int = 2,
    cross_facility: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """Sample paths with enrichment data for rescore calibration.

    Returns examples that show how enrichment data (LOC, languages,
    multiformat) correlates with scores, AND examples from different
    score ranges to help the rescorer understand score distribution.

    Args:
        facility: Current facility (for preferring same-facility examples)
        per_category: Paths per category (high_loc, fortran, multiformat, etc.)
        cross_facility: If True, sample from ALL facilities

    Returns:
        Dict mapping category to list of enriched path examples.
        Categories include both enrichment-based and score-distribution-based.
    """
    from imas_codex.graph import GraphClient

    # Enrichment-based categories (what characteristics paths have)
    categories = {
        "high_loc": {
            "filter": "p.total_lines >= 5000",
            "order": "p.total_lines DESC",
            "desc": "High lines of code (5000+)",
        },
        "fortran_heavy": {
            "filter": "p.language_breakdown CONTAINS 'Fortran'",
            "order": "p.score_modeling_code DESC",
            "desc": "Fortran-dominant directories",
        },
        "python_heavy": {
            "filter": "p.language_breakdown CONTAINS 'Python' AND NOT p.language_breakdown CONTAINS 'Fortran'",
            "order": "p.score_analysis_code DESC",
            "desc": "Python-dominant directories",
        },
        "multiformat": {
            "filter": "p.is_multiformat = true",
            "order": "p.score_data_access DESC",
            "desc": "Multi-format conversion code",
        },
        "small_code": {
            "filter": "p.total_lines > 0 AND p.total_lines < 500",
            "order": "p.score DESC",
            "desc": "Smaller codebases (under 500 LOC)",
        },
        # Score distribution categories (how scores distribute)
        "score_high": {
            "filter": "p.score >= 0.75",
            "order": "p.score DESC",
            "desc": "High-scoring enriched paths (0.75+)",
        },
        "score_medium": {
            "filter": "p.score >= 0.5 AND p.score < 0.75",
            "order": "rand()",
            "desc": "Medium-scoring enriched paths (0.5-0.75)",
        },
        "score_low": {
            "filter": "p.score >= 0.25 AND p.score < 0.5",
            "order": "rand()",
            "desc": "Lower-scoring enriched paths (0.25-0.5)",
        },
    }

    samples: dict[str, list[dict[str, Any]]] = {}

    with GraphClient() as gc:
        for cat_name, cat_def in categories.items():
            # Build facility filter
            facility_filter = (
                "" if cross_facility else f"AND p.facility_id = '{facility}'"
            )

            try:
                result = gc.query(
                    f"""
                    MATCH (p:FacilityPath)
                    WHERE p.status = 'scored'
                        AND p.total_lines IS NOT NULL
                        AND {cat_def["filter"]}
                        {facility_filter}
                    RETURN p.path AS path,
                           p.facility_id AS facility,
                           p.score AS score,
                           p.total_lines AS total_lines,
                           p.language_breakdown AS language_breakdown,
                           p.is_multiformat AS is_multiformat,
                           p.path_purpose AS purpose,
                           p.description AS description
                    ORDER BY {cat_def["order"]}
                    LIMIT $limit
                    """,
                    limit=per_category * 2,
                )
            except Exception:
                # Query might fail if enrichment fields don't exist yet
                result = []

            # Prefer current facility, then others
            paths = []
            current_facility_paths = [r for r in result if r["facility"] == facility]
            other_paths = [r for r in result if r["facility"] != facility]

            for r in current_facility_paths[:per_category]:
                paths.append(r)
            for r in other_paths[: per_category - len(paths)]:
                paths.append(r)

            samples[cat_name] = [
                {
                    "path": r["path"],
                    "facility": r["facility"],
                    "score": round(r["score"], 2),
                    "total_lines": r["total_lines"] or 0,
                    "language_breakdown": r["language_breakdown"] or "{}",
                    "is_multiformat": r["is_multiformat"] or False,
                    "purpose": r["purpose"] or "unknown",
                    "description": (r["description"] or "")[:60],
                }
                for r in paths[:per_category]
            ]

    return samples


def get_accumulated_cost(facility: str) -> dict[str, Any]:
    """Get accumulated LLM cost for a facility from score_cost fields.

    Sums all score_cost values across scored paths. This represents the
    total historical cost of scoring this facility across all runs.

    Args:
        facility: Facility ID

    Returns:
        Dict with: total_cost, paths_with_cost, scored_paths
    """
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (p:FacilityPath {facility_id: $facility})
            WHERE p.status = 'scored' OR p.status = 'skipped'
            RETURN
                sum(coalesce(p.score_cost, 0)) AS total_cost,
                sum(CASE WHEN p.score_cost IS NOT NULL AND p.score_cost > 0 THEN 1 ELSE 0 END) AS paths_with_cost,
                count(p) AS scored_paths
            """,
            facility=facility,
        )

        if result:
            return {
                "total_cost": result[0]["total_cost"] or 0.0,
                "paths_with_cost": result[0]["paths_with_cost"] or 0,
                "scored_paths": result[0]["scored_paths"] or 0,
            }

        return {
            "total_cost": 0.0,
            "paths_with_cost": 0,
            "scored_paths": 0,
        }


# ============================================================================
# Enrichment Frontier Functions
# ============================================================================


def claim_paths_for_enriching(facility: str, limit: int = 25) -> list[dict[str, Any]]:
    """Claim paths ready for enrichment (deep analysis: du, tokei, patterns).

    Paths ready for enrichment:
    - status = 'scored' (already valued by LLM)
    - should_enrich = true (LLM decided it's worth deep analysis)
    - is_enriched IS NULL OR is_enriched = false (not yet enriched)

    Uses claimed_at timestamp to prevent concurrent workers from claiming
    the same paths. Paths are claimed for 5 minutes - if not completed,
    they become claimable again (orphan recovery).

    Args:
        facility: Facility ID
        limit: Maximum paths to claim (default 25, SSH batch size)

    Returns:
        List of dicts with path info for enrichment
    """
    from imas_codex.graph import GraphClient

    now = datetime.now(UTC)
    cutoff = (now - __import__("datetime").timedelta(minutes=5)).isoformat()
    now_iso = now.isoformat()

    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (p:FacilityPath)-[:AT_FACILITY]->(f:Facility {id: $facility})
            WHERE p.status = $scored
              AND p.should_enrich = true
              AND (p.is_enriched IS NULL OR p.is_enriched = false)
              AND (p.claimed_at IS NULL OR p.claimed_at < datetime($cutoff))
            WITH p
            ORDER BY p.score DESC, p.depth ASC
            LIMIT $limit
            SET p.claimed_at = $now
            RETURN p.id AS id, p.path AS path, p.score AS score,
                   p.total_files AS total_files, p.total_dirs AS total_dirs
            """,
            facility=facility,
            scored=PathStatus.scored.value,
            cutoff=cutoff,
            now=now_iso,
            limit=limit,
        )

        return list(result)


def mark_enrichment_complete(
    facility: str,
    results: list[dict[str, Any]],
) -> int:
    """Mark paths as enriched with deep analysis results.

    Updates paths with:
    - is_enriched = true
    - enriched_at = current timestamp
    - total_bytes, total_lines, language_breakdown from du/tokei
    - is_multiformat from pattern analysis
    - Clears claimed_at

    Args:
        facility: Facility ID
        results: List of dicts with enrichment data:
            - path: Path string
            - total_bytes: Size from du (optional)
            - total_lines: Lines from tokei (optional)
            - language_breakdown: Language stats from tokei (optional, dict or JSON)
            - is_multiformat: Multi-format detection (optional)
            - error: Error message if enrichment failed (optional)

    Returns:
        Number of paths updated
    """
    import json

    from imas_codex.graph import GraphClient

    now = datetime.now(UTC).isoformat()
    updated = 0

    with GraphClient() as gc:
        for result in results:
            path = result["path"]
            path_id = f"{facility}:{path}"

            if result.get("error"):
                # Mark as unenrichable
                gc.query(
                    """
                    MATCH (p:FacilityPath {id: $id})
                    SET p.claimed_at = null,
                        p.should_enrich = false,
                        p.enrich_skip_reason = $reason
                    """,
                    id=path_id,
                    reason=result["error"],
                )
                continue

            # Prepare language breakdown as JSON string (Neo4j rejects empty dicts)
            lang_breakdown = result.get("language_breakdown")
            if isinstance(lang_breakdown, dict):
                lang_breakdown = json.dumps(lang_breakdown) if lang_breakdown else None

            # Serialize warnings to string for Neo4j
            warnings = result.get("warnings", [])
            warn_str = ", ".join(warnings) if warnings else None

            gc.query(
                """
                MATCH (p:FacilityPath {id: $id})
                SET p.is_enriched = true,
                    p.enriched_at = $now,
                    p.claimed_at = null,
                    p.total_bytes = $total_bytes,
                    p.total_lines = $total_lines,
                    p.language_breakdown = $language_breakdown,
                    p.is_multiformat = $is_multiformat,
                    p.enrich_warnings = $enrich_warnings
                """,
                id=path_id,
                now=now,
                total_bytes=result.get("total_bytes"),
                total_lines=result.get("total_lines"),
                language_breakdown=lang_breakdown,
                is_multiformat=result.get("is_multiformat"),
                enrich_warnings=warn_str,
            )
            updated += 1

    return updated


def claim_paths_for_rescoring(facility: str, limit: int = 10) -> list[dict[str, Any]]:
    """Claim enriched paths ready for rescoring with full context.

    Paths ready for rescoring:
    - status = 'scored' (base scoring complete)
    - is_enriched = true (deep analysis done)
    - score >= 0.5 (only bother rescoring potentially valuable paths)
    - rescored_at IS NULL (not yet rescored)

    Uses claimed_at for worker coordination.

    Args:
        facility: Facility ID
        limit: Maximum paths to claim (default 10, LLM batch size)

    Returns:
        List of dicts with path info, per-dimension scores, and enrichment data
    """
    from imas_codex.graph import GraphClient

    now = datetime.now(UTC)
    cutoff = (now - __import__("datetime").timedelta(minutes=5)).isoformat()
    now_iso = now.isoformat()

    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (p:FacilityPath)-[:AT_FACILITY]->(f:Facility {id: $facility})
            WHERE p.status = $scored
              AND p.is_enriched = true
              AND p.score >= 0.5
              AND p.rescored_at IS NULL
              AND (p.claimed_at IS NULL OR p.claimed_at < datetime($cutoff))
            WITH p
            ORDER BY p.score DESC
            LIMIT $limit
            SET p.claimed_at = $now
            RETURN p.id AS id, p.path AS path, p.score AS score,
                   p.total_bytes AS total_bytes, p.total_lines AS total_lines,
                   p.language_breakdown AS language_breakdown,
                   p.is_multiformat AS is_multiformat,
                   p.description AS description, p.path_purpose AS path_purpose,
                   p.keywords AS keywords, p.child_names AS child_names,
                   p.expansion_reason AS expansion_reason,
                   p.score_modeling_code AS score_modeling_code,
                   p.score_analysis_code AS score_analysis_code,
                   p.score_operations_code AS score_operations_code,
                   p.score_modeling_data AS score_modeling_data,
                   p.score_experimental_data AS score_experimental_data,
                   p.score_data_access AS score_data_access,
                   p.score_workflow AS score_workflow,
                   p.score_visualization AS score_visualization,
                   p.score_documentation AS score_documentation,
                   p.score_imas AS score_imas
            """,
            facility=facility,
            scored=PathStatus.scored.value,
            cutoff=cutoff,
            now=now_iso,
            limit=limit,
        )

        return list(result)


def sample_dimension_calibration_examples(
    facility: str | None = None,
    per_level: int = 3,
    tolerance: float = 0.1,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Sample calibration examples for each dimension at 5 score levels.

    Returns representative examples at score levels 0.0, 0.2, 0.5, 0.8, and 1.0
    for each scoring dimension. This enables LLMs to calibrate their scoring
    by seeing what historically received each score level.

    Args:
        facility: Current facility (preferring same-facility examples)
        per_level: Number of examples per score level (default 3)
        tolerance: Score range tolerance (default ±0.1)

    Returns:
        Nested dict: dimension -> level -> list of examples
        Each example has: path, facility, score, purpose, description
        Level keys: "lowest", "low", "medium", "high", "highest"

    Example structure:
        {
            "score_modeling_code": {
                "lowest": [{"path": "...", "score": 0.05, ...}, ...],
                "low": [{"path": "...", "score": 0.22, ...}, ...],
                "medium": [{"path": "...", "score": 0.48, ...}, ...],
                "high": [{"path": "...", "score": 0.78, ...}, ...],
                "highest": [{"path": "...", "score": 0.92, ...}, ...],
            },
            ...
        }
    """
    from imas_codex.graph import GraphClient

    # Target score levels with descriptive names
    score_levels = {
        "lowest": 0.0,  # Range: 0.0-0.1
        "low": 0.2,  # Range: 0.1-0.3
        "medium": 0.5,  # Range: 0.4-0.6
        "high": 0.8,  # Range: 0.7-0.9
        "highest": 1.0,  # Range: 0.9-1.0
    }

    samples: dict[str, dict[str, list[dict[str, Any]]]] = {}

    with GraphClient() as gc:
        for dim in SCORE_DIMENSIONS:
            samples[dim] = {}

            for level_name, target_score in score_levels.items():
                # Calculate score range based on level (different ranges for edge cases)
                if level_name == "lowest":
                    min_score, max_score = 0.0, 0.15
                elif level_name == "highest":
                    min_score, max_score = 0.9, 1.01  # Include 1.0
                else:
                    min_score = max(0.0, target_score - tolerance)
                    max_score = min(0.96, target_score + tolerance)

                # Query for examples at this score level for this dimension
                # Only consider paths where this dimension has a meaningful score
                result = gc.query(
                    f"""
                    MATCH (p:FacilityPath)
                    WHERE p.status = 'scored'
                        AND p.{dim} >= $min_score
                        AND p.{dim} < $max_score
                        AND p.{dim} IS NOT NULL
                    RETURN p.path AS path,
                           p.facility_id AS facility,
                           p.{dim} AS score,
                           p.path_purpose AS purpose,
                           p.description AS description
                    ORDER BY rand()
                    LIMIT $limit
                    """,
                    min_score=min_score,
                    max_score=max_score,
                    limit=per_level * 3,  # Get extras for facility preference
                )

                # Prefer current facility examples, then others
                paths: list[dict[str, Any]] = []
                current_facility_paths = [
                    r for r in result if r["facility"] == facility
                ]
                other_paths = [r for r in result if r["facility"] != facility]

                # Interleave: current facility first
                for r in current_facility_paths[:per_level]:
                    paths.append(r)
                for r in other_paths[: per_level - len(paths)]:
                    paths.append(r)

                samples[dim][level_name] = [
                    {
                        "path": r["path"],
                        "facility": r["facility"],
                        "score": round(r["score"], 2),
                        "purpose": r["purpose"] or "unknown",
                        "description": (r["description"] or "")[:100],
                    }
                    for r in paths[:per_level]
                ]

    return samples


def mark_rescore_complete(
    facility: str,
    results: list[dict[str, Any]],
) -> int:
    """Mark paths as rescored with refined per-dimension scores.

    Updates paths with:
    - Combined score and individual dimension scores (refined based on enrichment)
    - rescore_reason: explanation of why scores changed
    - rescored_at = current timestamp
    - Augments score_cost (doesn't replace)
    - Clears claimed_at

    Args:
        facility: Facility ID
        results: List of dicts with:
            - path: Path string
            - score: New combined score
            - score_cost: LLM cost for rescoring (added to existing)
            - adjustment_reason: Why scores changed (stored as rescore_reason)
            - score_modeling_code, score_analysis_code, etc. (optional per-dimension)

    Returns:
        Number of paths updated
    """
    from imas_codex.graph import GraphClient

    now = datetime.now(UTC).isoformat()
    updated = 0

    # Dimension fields to update
    dimensions = [
        "score_modeling_code",
        "score_analysis_code",
        "score_operations_code",
        "score_modeling_data",
        "score_experimental_data",
        "score_data_access",
        "score_workflow",
        "score_visualization",
        "score_documentation",
        "score_imas",
    ]

    with GraphClient() as gc:
        for result in results:
            path = result["path"]
            path_id = f"{facility}:{path}"

            # Build SET clause dynamically based on which dimensions are present
            set_parts = [
                "p.score = $score",
                "p.rescored_at = $now",
                "p.claimed_at = null",
                "p.score_cost = coalesce(p.score_cost, 0) + $cost",
            ]
            params = {
                "id": path_id,
                "now": now,
                "score": result.get("score"),
                "cost": result.get("score_cost", 0.0),
            }

            # Store adjustment reason
            if result.get("adjustment_reason"):
                set_parts.append("p.adjustment_reason = $adjustment_reason")
                params["adjustment_reason"] = result["adjustment_reason"][:200]

            # Add each dimension that has a value
            for dim in dimensions:
                if dim in result:
                    set_parts.append(f"p.{dim} = ${dim}")
                    params[dim] = result[dim]

            # Build and execute query
            set_clause = ", ".join(set_parts)
            gc.query(
                f"""
                MATCH (p:FacilityPath {{id: $id}})
                SET {set_clause}
                """,
                **params,
            )
            updated += 1

    return updated


def get_hierarchy_context(
    facility: str,
    paths: list[str],
    max_siblings: int = 8,
) -> dict[str, dict[str, Any]]:
    """Get parent and sibling context for a batch of paths.

    For each path, returns the parent directory's score/purpose and
    already-scored sibling directories. This context helps the LLM
    make relative scoring decisions — seeing that /home/user1 was
    scored as analysis_code(0.72) helps calibrate /home/user2.

    Args:
        facility: Facility ID
        paths: List of paths to get context for
        max_siblings: Maximum scored siblings to return per path

    Returns:
        Dict mapping path -> {
            "parent": {"path": ..., "purpose": ..., "score": ..., "description": ...} | None,
            "siblings": [{"path": ..., "purpose": ..., "score": ...}],
            "depth": int,
        }
    """
    import posixpath

    from imas_codex.graph import GraphClient

    if not paths:
        return {}

    # Compute parent paths
    path_parents: dict[str, str] = {}
    parent_set: set[str] = set()
    for p in paths:
        parent = posixpath.dirname(p.rstrip("/"))
        if parent and parent != p.rstrip("/"):
            path_parents[p] = parent
            parent_set.add(parent)

    context: dict[str, dict[str, Any]] = {
        p: {"parent": None, "siblings": [], "depth": 0} for p in paths
    }

    if not parent_set:
        return context

    with GraphClient() as gc:
        # Batch query: get parent info + scored children for all parent paths
        parent_ids = [f"{facility}:{p}" for p in parent_set]
        result = gc.query(
            """
            UNWIND $parent_ids AS pid
            OPTIONAL MATCH (parent:FacilityPath {id: pid})
            WHERE parent.status = 'scored'
            OPTIONAL MATCH (sibling:FacilityPath {facility_id: $facility})
            WHERE sibling.status = 'scored'
              AND sibling.path STARTS WITH parent.path + '/'
              AND NOT sibling.path CONTAINS substring(parent.path + '/', size(parent.path) + 1) + '/'
            WITH parent, pid,
                 collect({
                     path: sibling.path,
                     purpose: sibling.path_purpose,
                     score: sibling.score,
                     description: left(coalesce(sibling.description, ''), 60)
                 })[0..$max_siblings] AS siblings
            RETURN pid,
                   parent.path AS parent_path,
                   parent.path_purpose AS parent_purpose,
                   parent.score AS parent_score,
                   left(coalesce(parent.description, ''), 80) AS parent_description,
                   parent.depth AS parent_depth,
                   siblings
            """,
            parent_ids=parent_ids,
            facility=facility,
            max_siblings=max_siblings,
        )

        # Build lookup: parent_path -> parent_info + siblings
        parent_info: dict[str, dict[str, Any]] = {}
        for r in result:
            parent_path_from_id = r["pid"].split(":", 1)[1] if ":" in r["pid"] else ""
            if r["parent_path"]:
                parent_info[parent_path_from_id] = {
                    "parent": {
                        "path": r["parent_path"],
                        "purpose": r["parent_purpose"],
                        "score": r["parent_score"],
                        "description": r["parent_description"],
                    },
                    "siblings": [
                        s for s in (r["siblings"] or []) if s.get("path") is not None
                    ],
                    "depth": (r["parent_depth"] or 0) + 1,
                }

        # Map back to input paths
        for p in paths:
            parent = path_parents.get(p)
            if parent and parent in parent_info:
                info = parent_info[parent]
                context[p]["parent"] = info["parent"]
                # Exclude the current path from siblings
                context[p]["siblings"] = [
                    s for s in info["siblings"] if s["path"] != p
                ][:max_siblings]
                context[p]["depth"] = info["depth"]

    return context
