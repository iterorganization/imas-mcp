"""Base infrastructure for discovery operations.

Provides:
- Facility configuration management
- Parallel command execution
- Progress display utilities
- Worker supervision with crash recovery
"""

from imas_codex.discovery.base.executor import CommandResult, ParallelExecutor
from imas_codex.discovery.base.facility import (
    add_exploration_note,
    filter_private_fields,
    get_facilities_dir,
    get_facility,
    get_facility_infrastructure,
    get_facility_metadata,
    list_facilities,
    update_infrastructure,
    update_metadata,
    validate_no_private_fields,
)
from imas_codex.discovery.base.progress import (
    BaseProgressDisplay,
    BaseProgressState,
    ProgressConfig,
    StreamQueue,
    WorkerStats,
    build_header,
    build_resource_section,
    build_worker_row,
    clean_text,
    clip_path,
    format_bytes,
    format_count,
    format_time,
    make_bar,
    make_gradient_bar,
    make_resource_gauge,
)
from imas_codex.discovery.base.supervision import (
    DEFAULT_INITIAL_BACKOFF,
    DEFAULT_MAX_BACKOFF,
    DEFAULT_MAX_RESTARTS,
    OrphanRecoveryResult,
    SupervisedWorkerGroup,
    WorkerState,
    WorkerStatus,
    is_infrastructure_error,
    make_orphan_recovery_query,
    release_orphaned_claims_generic,
    supervised_worker,
)

__all__ = [
    # Facility
    "get_facility",
    "get_facility_metadata",
    "get_facility_infrastructure",
    "update_infrastructure",
    "update_metadata",
    "add_exploration_note",
    "list_facilities",
    "get_facilities_dir",
    "filter_private_fields",
    "validate_no_private_fields",
    # Executor
    "ParallelExecutor",
    "CommandResult",
    # Progress
    "ProgressConfig",
    "BaseProgressState",
    "BaseProgressDisplay",
    "StreamQueue",
    "WorkerStats",
    "format_time",
    "format_bytes",
    "format_count",
    "clip_path",
    "clean_text",
    "make_bar",
    "make_gradient_bar",
    "make_resource_gauge",
    "build_header",
    "build_resource_section",
    "build_worker_row",
    # Supervision
    "supervised_worker",
    "is_infrastructure_error",
    "WorkerState",
    "WorkerStatus",
    "SupervisedWorkerGroup",
    "OrphanRecoveryResult",
    "make_orphan_recovery_query",
    "release_orphaned_claims_generic",
    "DEFAULT_MAX_RESTARTS",
    "DEFAULT_INITIAL_BACKOFF",
    "DEFAULT_MAX_BACKOFF",
]
