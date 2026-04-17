"""Pydantic models for code migration guides."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CodeUpdateAction(BaseModel):
    """A single code update action."""

    path: str = Field(description="IMAS path affected")
    ids: str = Field(description="IDS name")
    change_type: str = Field(
        description="Type of change: cocos_sign_flip, type_change, path_rename, "
        "path_removed, unit_change, new_path, definition_clarification, "
        "definition_change"
    )
    severity: str = Field(
        description="'required' (will break) or 'optional' (best practice)"
    )

    search_patterns: list[str] = Field(
        default_factory=list,
        description="Language-agnostic patterns to find affected code",
    )
    path_fragments: list[str] = Field(
        default_factory=list,
        description="Path segments to search for",
    )

    description: str = Field(description="Human-readable description of the change")
    before: str = Field(default="", description="What the code does now")
    after: str = Field(default="", description="What the code should do")

    cocos_label: str | None = Field(
        default=None, description="COCOS label (e.g., 'psi_like')"
    )
    cocos_factor: float | None = Field(
        default=None, description="COCOS multiplication factor"
    )

    old_path: str | None = Field(default=None, description="For renames: old path")
    new_path: str | None = Field(default=None, description="For renames: new path")

    old_type: str | None = Field(default=None, description="For type changes: old type")
    new_type: str | None = Field(default=None, description="For type changes: new type")

    old_units: str | None = Field(
        default=None, description="For unit changes: old units"
    )
    new_units: str | None = Field(
        default=None, description="For unit changes: new units"
    )


class CocosMigrationAdvice(BaseModel):
    """COCOS-specific migration advice."""

    from_cocos: int = Field(description="Source COCOS convention")
    to_cocos: int = Field(description="Target COCOS convention")
    sign_flips: list[dict] = Field(
        default_factory=list,
        description="Paths needing sign change (factor != 1)",
    )
    no_change: list[dict] = Field(
        default_factory=list,
        description="Paths with factor=1 (verify only)",
    )


class PathUpdateAdvice(BaseModel):
    """Path-level update advice with search patterns."""

    renamed_paths: list[dict] = Field(
        default_factory=list, description="old_path -> new_path"
    )
    removed_paths: list[dict] = Field(
        default_factory=list, description="path + replacement suggestion"
    )
    new_paths: list[dict] = Field(
        default_factory=list, description="Newly available paths"
    )


class TypeUpdateAdvice(BaseModel):
    """Type change advice."""

    type_changes: list[dict] = Field(
        default_factory=list,
        description="path, old_type, new_type changes",
    )


class CodeMigrationGuide(BaseModel):
    """Complete code migration guide between DD versions."""

    from_version: str = Field(description="Source DD version")
    to_version: str = Field(description="Target DD version")
    cocos_change: str | None = Field(
        default=None, description="COCOS change description (e.g., '11 -> 17')"
    )

    required_actions: list[CodeUpdateAction] = Field(
        default_factory=list,
        description="Changes that WILL break code if not applied",
    )
    optional_actions: list[CodeUpdateAction] = Field(
        default_factory=list,
        description="Best-practice changes for full DD compliance",
    )

    total_actions: int = Field(default=0)
    required_count: int = Field(default=0)
    optional_count: int = Field(default=0)
    ids_affected: list[str] = Field(default_factory=list)

    global_search_patterns: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Search patterns grouped by IDS",
    )

    include_recipes: bool = Field(
        default=True,
        description="Whether to include code update recipes and search patterns",
    )

    cocos_advice: CocosMigrationAdvice | None = None
    path_update_advice: PathUpdateAdvice | None = None
    type_update_advice: TypeUpdateAdvice | None = None

    rename_lineages: list[dict] = Field(
        default_factory=list,
        description=(
            "Multi-hop rename chains (2+ hops). Each entry has: "
            "'ids', 'start_path', 'end_path', 'hops', "
            "'chain' (list of {path, introduced} dicts)."
        ),
    )


def generate_search_patterns(path: str, change_type: str) -> list[str]:
    """Generate language-agnostic search patterns to find code accessing a path."""
    segments = path.split("/")
    if len(segments) < 2:
        return [path]

    ids_name = segments[0]
    leaf_name = segments[-1]
    parent_name = segments[-2] if len(segments) > 2 else ""
    sub_path = "/".join(segments[1:])

    patterns = [
        # Exact path string (any language)
        f"'{sub_path}'",
        f'"{sub_path}"',
        # Leaf name in accessor context
        leaf_name,
    ]

    if parent_name:
        patterns.append(f"{parent_name}.*{leaf_name}")
        patterns.append(f"{leaf_name}.*{parent_name}")

    # IDS-level patterns
    patterns.append(f"{ids_name}.*{leaf_name}")

    # Common IMAS access patterns
    if change_type in ("cocos_sign_flip", "path_rename", "path_removed"):
        patterns.extend(
            [
                f"get.*{leaf_name}",
                f"put.*{leaf_name}",
            ]
        )

    return patterns
