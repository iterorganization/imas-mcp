"""Prompt loading for agentic workflows.

Prompts are markdown files with YAML frontmatter:
- name: Prompt identifier (can include path like "discovery/scorer")
- description: Short description
- task: Task type for model selection (optional)

Prompts support Jinja2 templating:
- Includes: {% include "safety.md" %} (resolved from shared/)
- Variables: {{ facility }} (from render context)
- Schema loops: {% for cat in discovery_categories %}...{% endfor %}

Rendering modes:
1. Static: parse_prompt_file() - includes only, for MCP registration
2. Dynamic: render_prompt() - full Jinja2 with schema context

Directory structure:
    prompts/
    ├── shared/           # Reusable includes
    │   ├── safety.md
    │   └── schema/       # Schema-derived templates
    ├── discovery/        # Discovery pipeline prompts
    │   └── scorer.md
    └── enrich-system.md  # Root-level prompts
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from jinja2 import Environment

# Pattern for include directives: {% include "filename.md" %}
INCLUDE_PATTERN = re.compile(r'\{%\s*include\s+"([^"]+)"\s*%\}')


@dataclass
class PromptDefinition:
    """A loaded prompt definition."""

    name: str
    description: str
    content: str
    task: str = "default"
    metadata: dict = field(default_factory=dict)


def _resolve_includes(text: str, prompts_dir: Path) -> str:
    """Resolve include directives in prompt content.

    Args:
        text: Prompt content with potential includes
        prompts_dir: Base directory for prompts (shared/ is relative to this)

    Returns:
        Content with includes resolved
    """
    shared_dir = prompts_dir / "shared"

    def replace_include(match: re.Match) -> str:
        include_name = match.group(1)
        include_path = shared_dir / include_name
        if include_path.exists():
            return include_path.read_text().strip()
        return f"<!-- Include not found: {include_name} -->"

    return INCLUDE_PATTERN.sub(replace_include, text)


def parse_prompt_file(path: Path, prompts_dir: Path | None = None) -> PromptDefinition:
    """Parse a markdown prompt file with YAML frontmatter.

    Args:
        path: Path to the prompt file
        prompts_dir: Base directory for resolving includes (default: path.parent)

    Returns:
        Parsed PromptDefinition
    """
    if prompts_dir is None:
        prompts_dir = path.parent

    text = path.read_text()

    match = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
    if not match:
        content = _resolve_includes(text.strip(), prompts_dir)
        return PromptDefinition(name=path.stem, description="", content=content)

    frontmatter_text, content = match.groups()
    frontmatter = yaml.safe_load(frontmatter_text)

    # Resolve includes in content
    resolved_content = _resolve_includes(content.strip(), prompts_dir)

    return PromptDefinition(
        name=frontmatter.get("name", path.stem),
        description=frontmatter.get("description", ""),
        content=resolved_content,
        task=frontmatter.get("task", "default"),
        metadata={
            k: v
            for k, v in frontmatter.items()
            if k not in ("name", "description", "task")
        },
    )


def load_prompts(prompts_dir: Path | None = None) -> dict[str, PromptDefinition]:
    """Load all prompts from a directory and subdirectories.

    Loads .md files from:
    - prompts_dir/*.md (root level prompts)
    - prompts_dir/<subdir>/*.md (categorized prompts, excluding shared/)

    Prompt names include the subdirectory path (e.g., "discovery/scorer").
    """
    if prompts_dir is None:
        prompts_dir = Path(__file__).parent / "prompts"

    prompts = {}
    if prompts_dir.exists():
        # Load root-level prompts
        for md_file in sorted(prompts_dir.glob("*.md")):
            if md_file.is_file():
                try:
                    prompt = parse_prompt_file(md_file, prompts_dir)
                    prompts[prompt.name] = prompt
                except Exception as e:
                    import logging

                    logging.getLogger(__name__).warning(
                        f"Failed to parse {md_file}: {e}"
                    )

        # Load prompts from subdirectories (excluding shared/)
        for subdir in sorted(prompts_dir.iterdir()):
            if subdir.is_dir() and subdir.name != "shared":
                for md_file in sorted(subdir.glob("*.md")):
                    if md_file.is_file():
                        try:
                            prompt = parse_prompt_file(md_file, prompts_dir)
                            # Use subdir/name as the key if name doesn't include it
                            key = prompt.name
                            if "/" not in key:
                                key = f"{subdir.name}/{prompt.name}"
                            prompts[key] = prompt
                        except Exception as e:
                            import logging

                            logging.getLogger(__name__).warning(
                                f"Failed to parse {md_file}: {e}"
                            )

    return prompts


def list_prompts_summary(prompts_dir: Path | None = None) -> list[dict[str, str]]:
    """Get a summary list of available prompts."""
    prompts = load_prompts(prompts_dir)
    return [
        {"name": p.name, "description": p.description, "task": p.task}
        for p in prompts.values()
    ]


# =============================================================================
# Schema Context for Dynamic Rendering
# =============================================================================


def get_schema_context() -> dict[str, Any]:
    """Get schema-derived context for prompt rendering.

    Uses LinkML schema introspection to get enum values with their descriptions.
    This is the single source of truth - descriptions come from the schema YAML,
    not from generated Python code.

    Returns a dict with:
    - discovery_categories: List of DiscoveryRootCategory enum values
    - path_purposes: List of PathPurpose enum values
    - score_dimensions: List of per-purpose score field names with descriptions
    - Each with 'value' and 'description' keys

    This allows prompts to loop over schema-defined enums without hardcoding.
    """
    from imas_codex.graph.schema import get_schema

    schema = get_schema()

    # Get enum values with descriptions directly from LinkML schema
    discovery_categories = (
        schema.get_enum_with_descriptions("DiscoveryRootCategory") or []
    )
    path_purposes = schema.get_enum_with_descriptions("PathPurpose") or []

    # Get score dimensions from FacilityPath schema
    # These are the score_* fields that map to DiscoveryRootCategory taxonomy
    facility_path_slots = schema.get_all_slots("FacilityPath")
    score_dimensions = []
    for slot_name, slot_info in facility_path_slots.items():
        if slot_name.startswith("score_") and slot_info.get("type") == "float":
            # Extract description and build dimension info
            desc = slot_info.get("description", "")
            score_dimensions.append(
                {
                    "field": slot_name,
                    "label": slot_name.replace("score_", "").replace("_", " ").title(),
                    "description": desc,
                }
            )

    # Group path purposes for convenience in templates
    code_purposes = [
        p
        for p in path_purposes
        if p["value"] in ("modeling_code", "analysis_code", "operations_code")
    ]
    data_purposes = [
        p for p in path_purposes if p["value"] in ("modeling_data", "experimental_data")
    ]
    infra_purposes = [
        p
        for p in path_purposes
        if p["value"] in ("data_access", "workflow", "visualization")
    ]
    support_purposes = [
        p
        for p in path_purposes
        if p["value"] in ("documentation", "configuration", "test_suite")
    ]
    structural_purposes = [
        p
        for p in path_purposes
        if p["value"] in ("container", "archive", "build_artifact", "system")
    ]

    return {
        "discovery_categories": discovery_categories,
        "path_purposes": path_purposes,
        "score_dimensions": score_dimensions,
        # Grouped path purposes for cleaner template organization
        "path_purposes_code": code_purposes,
        "path_purposes_data": data_purposes,
        "path_purposes_infra": infra_purposes,
        "path_purposes_support": support_purposes,
        "path_purposes_structural": structural_purposes,
        # Legacy groupings for discover-roots compatibility
        "discovery_categories_modeling": [
            c
            for c in discovery_categories
            if c["value"] in ("modeling_code", "modeling_data")
        ],
        "discovery_categories_experimental": [
            c
            for c in discovery_categories
            if c["value"] in ("analysis_code", "experimental_data")
        ],
        "discovery_categories_shared": [
            c
            for c in discovery_categories
            if c["value"] in ("data_access", "workflow", "documentation")
        ],
    }


def _get_jinja_env(prompts_dir: Path) -> Environment:
    """Create Jinja2 environment with custom loader for includes."""
    from jinja2 import BaseLoader, Environment, TemplateNotFound

    class PromptsLoader(BaseLoader):
        """Loader that resolves includes from shared/ directory."""

        def get_source(
            self, environment: Environment, template: str
        ) -> tuple[str, str, Any]:
            # Try shared/ first, then root
            for base in [prompts_dir / "shared", prompts_dir]:
                path = base / template
                if path.exists():
                    source = path.read_text()
                    return source, str(path), lambda: True
            raise TemplateNotFound(template)

    return Environment(loader=PromptsLoader())


def render_prompt(
    name: str,
    context: dict[str, Any] | None = None,
    prompts_dir: Path | None = None,
) -> str:
    """Render a prompt with full Jinja2 templating and schema context.

    This is the dynamic rendering mode that injects schema-derived values
    and custom context variables into the prompt template.

    Args:
        name: Prompt name (e.g., "discover-roots", "discovery/scorer")
        context: Additional context variables (e.g., {"facility": "tcv"})
        prompts_dir: Base directory for prompts

    Returns:
        Fully rendered prompt content

    Example:
        >>> render_prompt("discover-roots", {"facility": "tcv"})
        # Returns prompt with {{ facility }} replaced and schema loops expanded
    """
    if prompts_dir is None:
        prompts_dir = Path(__file__).parent / "prompts"

    # Load the raw prompt (with frontmatter)
    prompts = load_prompts(prompts_dir)
    if name not in prompts:
        raise KeyError(f"Prompt '{name}' not found. Available: {list(prompts.keys())}")

    prompt_def = prompts[name]

    # Build full context: schema + user-provided
    full_context = get_schema_context()
    if context:
        full_context.update(context)

    # Render with Jinja2
    env = _get_jinja_env(prompts_dir)
    template = env.from_string(prompt_def.content)
    return template.render(full_context)


def get_prompt_content_hash(name: str, prompts_dir: Path | None = None) -> str:
    """Get a hash of prompt content for cache invalidation."""
    if prompts_dir is None:
        prompts_dir = Path(__file__).parent / "prompts"

    prompts = load_prompts(prompts_dir)
    if name not in prompts:
        raise KeyError(f"Prompt '{name}' not found")

    content = prompts[name].content
    return hashlib.sha256(content.encode()).hexdigest()[:16]
