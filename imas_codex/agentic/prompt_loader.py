"""Prompt loading for agentic workflows.

Prompts are markdown files with YAML frontmatter:
- name: Prompt identifier (can include path like "discovery/scorer")
- description: Short description
- task: Task type for model selection (optional)
- schema_needs: List of schema providers to include (optional)

Prompts support Jinja2 templating:
- Includes: {% include "safety.md" %} (resolved from shared/)
- Variables: {{ facility }} (from render context)
- Schema loops: {% for cat in discovery_categories %}...{% endfor %}

Schema Provider Architecture:
    Each prompt declares what schema context it needs via schema_needs frontmatter.
    Providers are cached and only loaded when requested:

    - path_purposes: PathPurpose enum values (grouped by category)
    - discovery_categories: DiscoveryRootCategory enum values
    - score_dimensions: score_* fields from FacilityPath schema
    - scoring_schema: DirectoryScoringBatch Pydantic schema
    - rescore_schema: RescoreBatch Pydantic schema
    - access_method_fields: AccessMethod schema fields
    - access_methods_graph: Existing AccessMethod nodes from graph

    Providers use @lru_cache so schema is loaded once per process.

Rendering modes:
1. Static: parse_prompt_file() - includes only, for MCP registration
2. Dynamic: render_prompt() - full Jinja2 with schema context

Directory structure:
    prompts/
    ├── shared/           # Reusable includes
    │   ├── safety.md
    │   └── schema/       # Schema-derived templates
    ├── discovery/        # Discovery pipeline prompts
    │   ├── roots.md      # Seed discovery frontier
    │   ├── scorer.md     # Score directories
    │   ├── rescorer.md   # Rescore with enrichment
    │   └── enricher.md   # Extract metadata
    ├── exploration/      # Interactive exploration
    │   └── facility.md   # Facility exploration agent
    └── wiki/             # Wiki operations
        ├── scout.md      # Discover wiki pages
        └── scorer.md     # Score wiki pages
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from functools import lru_cache
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


def get_pydantic_schema_json(
    model_class: type, *, indent: int = 2, example: bool = True
) -> str:
    """Generate a JSON representation of a Pydantic model schema for prompts.

    This creates a human-readable JSON example that shows the expected structure
    and field descriptions, suitable for embedding in LLM prompts.

    Args:
        model_class: Pydantic model class to generate schema for
        indent: JSON indentation level
        example: If True, generate an example instance; if False, generate JSON Schema

    Returns:
        JSON string suitable for embedding in prompts
    """
    import json
    from typing import get_args, get_origin

    def generate_example_value(field_info, field_type) -> Any:
        """Generate a reasonable example value for a field type."""
        # Handle Optional types
        origin = get_origin(field_type)
        args = get_args(field_type)

        if origin is list:
            # List type - return example list
            inner_type = args[0] if args else str
            if hasattr(inner_type, "model_fields"):
                return [generate_example_for_model(inner_type)]
            return []

        if origin in (type(None) | str, str | None):
            # Optional[str] or str | None
            inner = args[0] if args else str
            if inner is type(None):
                inner = args[1] if len(args) > 1 else str
            if inner is str:
                return None
            return generate_example_value(field_info, inner)

        # Check if it's an enum
        if hasattr(field_type, "__members__"):
            # Return first enum value as example
            members = list(field_type.__members__.values())
            return members[0].value if members else "unknown"

        # Primitive types
        if field_type is str:
            desc = getattr(field_info, "description", "") or ""
            if "path" in desc.lower():
                return "/absolute/path/to/directory"
            if "description" in desc.lower():
                return "Concise description (1-2 sentences)"
            return ""
        if field_type is bool:
            return True
        if field_type is float:
            return 0.0
        if field_type is int:
            return 0

        # Nested Pydantic model
        if hasattr(field_type, "model_fields"):
            return generate_example_for_model(field_type)

        return None

    def generate_example_for_model(model: type) -> dict[str, Any]:
        """Generate example dict for a Pydantic model."""
        result = {}
        for field_name, field_info in model.model_fields.items():
            field_type = field_info.annotation
            result[field_name] = generate_example_value(field_info, field_type)
        return result

    if example:
        # Generate an example instance
        example_data = generate_example_for_model(model_class)
        return json.dumps(example_data, indent=indent)
    else:
        # Return the JSON Schema
        schema = model_class.model_json_schema()
        return json.dumps(schema, indent=indent)


def get_pydantic_schema_description(model_class: type) -> str:
    """Generate a human-readable field description for a Pydantic model.

    Creates a markdown-formatted list of fields with their types and descriptions,
    suitable for embedding in LLM prompts.

    Args:
        model_class: Pydantic model class

    Returns:
        Markdown string describing all fields
    """
    from typing import get_args, get_origin

    lines = []

    def format_type(field_type) -> str:
        """Format a type annotation as a readable string."""
        origin = get_origin(field_type)
        args = get_args(field_type)

        if origin is list:
            inner = args[0] if args else "Any"
            inner_name = getattr(inner, "__name__", str(inner))
            return f"list[{inner_name}]"

        if origin in (type(None) | str, str | None):
            # Union with None (Optional)
            non_none = [a for a in args if a is not type(None)]
            if non_none:
                inner_name = getattr(non_none[0], "__name__", str(non_none[0]))
                return f"{inner_name} | null"
            return "string | null"

        if hasattr(field_type, "__members__"):
            # Enum - list allowed values
            values = [m.value for m in field_type.__members__.values()]
            if len(values) <= 5:
                return f"enum: {values}"
            return f"enum: {values[:5]}..."

        return getattr(field_type, "__name__", str(field_type))

    for field_name, field_info in model_class.model_fields.items():
        field_type = field_info.annotation
        type_str = format_type(field_type)
        desc = field_info.description or ""
        required = field_info.is_required()
        req_str = "(required)" if required else "(optional)"

        lines.append(f"- **{field_name}**: `{type_str}` {req_str}")
        if desc:
            lines.append(f"  - {desc}")

    return "\n".join(lines)


# =============================================================================
# Cached Schema Providers
# =============================================================================
#
# Each provider is cached with @lru_cache so schema data is loaded once per process.
# Prompts declare what they need via schema_needs frontmatter, and only those
# providers are invoked.


@lru_cache(maxsize=1)
def _get_linkml_schema():
    """Cached LinkML schema accessor."""
    from imas_codex.graph.schema import get_schema

    return get_schema()


@lru_cache(maxsize=1)
def _provide_path_purposes() -> dict[str, Any]:
    """Provide PathPurpose enum values grouped by category."""
    schema = _get_linkml_schema()
    path_purposes = schema.get_enum_with_descriptions("PathPurpose") or []

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
        "path_purposes": path_purposes,
        "path_purposes_code": code_purposes,
        "path_purposes_data": data_purposes,
        "path_purposes_infra": infra_purposes,
        "path_purposes_support": support_purposes,
        "path_purposes_structural": structural_purposes,
    }


@lru_cache(maxsize=1)
def _provide_discovery_categories() -> dict[str, Any]:
    """Provide DiscoveryRootCategory enum values grouped."""
    schema = _get_linkml_schema()
    discovery_categories = (
        schema.get_enum_with_descriptions("DiscoveryRootCategory") or []
    )

    return {
        "discovery_categories": discovery_categories,
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


@lru_cache(maxsize=1)
def _provide_score_dimensions() -> dict[str, Any]:
    """Provide score_* field definitions from FacilityPath schema."""
    schema = _get_linkml_schema()
    facility_path_slots = schema.get_all_slots("FacilityPath")

    score_dimensions = []
    for slot_name, slot_info in facility_path_slots.items():
        if slot_name.startswith("score_") and slot_info.get("type") == "float":
            desc = slot_info.get("description", "")
            score_dimensions.append(
                {
                    "field": slot_name,
                    "label": slot_name.replace("score_", "").replace("_", " ").title(),
                    "description": desc,
                }
            )

    return {"score_dimensions": score_dimensions}


@lru_cache(maxsize=1)
def _provide_scoring_schema() -> dict[str, Any]:
    """Provide DirectoryScoringBatch Pydantic schema for LLM prompts."""
    from imas_codex.discovery.paths.models import (
        DirectoryScoringBatch,
        DirectoryScoringResult,
    )

    return {
        "scoring_schema_example": get_pydantic_schema_json(DirectoryScoringBatch),
        "scoring_schema_fields": get_pydantic_schema_description(
            DirectoryScoringResult
        ),
    }


@lru_cache(maxsize=1)
def _provide_rescore_schema() -> dict[str, Any]:
    """Provide RescoreBatch Pydantic schema for LLM prompts."""
    from imas_codex.discovery.paths.models import RescoreBatch, RescoreResult

    return {
        "rescore_schema_example": get_pydantic_schema_json(RescoreBatch),
        "rescore_schema_fields": get_pydantic_schema_description(RescoreResult),
    }


@lru_cache(maxsize=1)
def _provide_access_method_fields() -> dict[str, Any]:
    """Provide AccessMethod schema fields grouped by purpose."""
    schema = _get_linkml_schema()
    access_method_slots = schema.get_all_slots("AccessMethod")

    access_method_fields = {
        "required": [],
        "environment": [],
        "templates": [],
        "validation": [],
        "documentation": [],
    }

    # Categorize fields by purpose
    required_fields = {"id", "facility_id", "method_type", "library", "access_type"}
    env_fields = {"setup_commands", "environment_variables"}
    template_fields = {
        "imports_template",
        "connection_template",
        "data_template",
        "time_template",
        "cleanup_template",
        "decimation_template",
    }
    validation_fields = {
        "data_source",
        "data_source_pattern",
        "discovery_shot",
        "discovery_template",
        "full_example",
        "verified_date",
    }
    doc_fields = {"name", "documentation_url", "documentation_local", "output_format"}

    for slot_name, slot_info in access_method_slots.items():
        field_info = {
            "name": slot_name,
            "description": slot_info.get("description", ""),
            "required": slot_info.get("required", False),
            "type": slot_info.get("type", "string"),
        }
        if slot_name in required_fields:
            access_method_fields["required"].append(field_info)
        elif slot_name in env_fields:
            access_method_fields["environment"].append(field_info)
        elif slot_name in template_fields:
            access_method_fields["templates"].append(field_info)
        elif slot_name in validation_fields:
            access_method_fields["validation"].append(field_info)
        elif slot_name in doc_fields:
            access_method_fields["documentation"].append(field_info)

    return {"access_method_fields": access_method_fields}


# Registry mapping schema_needs names to provider functions
_SCHEMA_PROVIDERS: dict[str, Any] = {
    "path_purposes": _provide_path_purposes,
    "discovery_categories": _provide_discovery_categories,
    "score_dimensions": _provide_score_dimensions,
    "scoring_schema": _provide_scoring_schema,
    "rescore_schema": _provide_rescore_schema,
    "access_method_fields": _provide_access_method_fields,
}

# Default schema needs per prompt (when not specified in frontmatter)
# Only load what's actually used by each prompt
_DEFAULT_SCHEMA_NEEDS: dict[str, list[str]] = {
    "discovery/scorer": ["path_purposes", "score_dimensions", "scoring_schema"],
    "discovery/rescorer": ["rescore_schema"],
    "discovery/roots": ["discovery_categories"],
    "discovery/data_access": ["access_method_fields"],
    # wiki prompts don't need schema context
}


def get_schema_for_prompt(
    prompt_name: str, schema_needs: list[str] | None = None
) -> dict[str, Any]:
    """Get only the schema context needed for a specific prompt.

    Uses prompt frontmatter schema_needs if available, otherwise uses
    defaults based on prompt name. Only invokes the required providers.

    Args:
        prompt_name: Prompt identifier (e.g., "discovery/scorer")
        schema_needs: Explicit list of needs (overrides frontmatter/defaults)

    Returns:
        Dict with only the requested schema context
    """
    if schema_needs is None:
        schema_needs = _DEFAULT_SCHEMA_NEEDS.get(prompt_name, [])

    context: dict[str, Any] = {}
    for need in schema_needs:
        provider = _SCHEMA_PROVIDERS.get(need)
        if provider:
            context.update(provider())

    return context


def get_schema_context() -> dict[str, Any]:
    """Get full schema context for prompt rendering.

    DEPRECATED: Use get_schema_for_prompt() for targeted loading.
    This function loads everything for backwards compatibility.
    """
    context: dict[str, Any] = {}
    for provider in _SCHEMA_PROVIDERS.values():
        context.update(provider())
    return context


def get_access_methods_context() -> dict[str, Any]:
    """Get existing AccessMethod nodes from the graph for prompt context.

    Queries the graph for all AccessMethod nodes and returns them in a format
    suitable for Jinja2 templates. Used by discovery/data_access prompt to
    provide working examples from other facilities.

    Returns:
        Dict with 'existing_access_methods' list containing node properties.
    """
    try:
        from imas_codex.graph import GraphClient

        with GraphClient() as client:
            # Query existing methods with key fields for examples
            result = client.query("""
                MATCH (m:AccessMethod)-[:FACILITY_ID]->(f:Facility)
                RETURN m.id AS id,
                       m.name AS name,
                       f.id AS facility,
                       m.method_type AS method_type,
                       m.library AS library,
                       m.data_template AS data_template,
                       m.setup_commands AS setup_commands,
                       m.full_example AS full_example
                ORDER BY f.id, m.method_type
            """)

            methods = []
            for row in result:
                methods.append(
                    {
                        "id": row["id"],
                        "name": row["name"] or row["id"],
                        "facility": row["facility"],
                        "method_type": row["method_type"],
                        "library": row["library"],
                        "data_template": row["data_template"],
                        "setup_commands": row["setup_commands"],
                        "full_example": row["full_example"],
                    }
                )

            return {"existing_access_methods": methods}

    except Exception:
        # Graph not available - return empty context
        return {"existing_access_methods": []}


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
        name: Prompt name (e.g., "discovery/roots", "discovery/scorer")
        context: Additional context variables (e.g., {"facility": "tcv"})
        prompts_dir: Base directory for prompts

    Returns:
        Fully rendered prompt content

    Example:
        >>> render_prompt("discovery/roots", {"facility": "tcv"})
        # Returns prompt with {{ facility }} replaced and schema loops expanded
    """
    if prompts_dir is None:
        prompts_dir = Path(__file__).parent / "prompts"

    # Load the raw prompt (with frontmatter)
    prompts = load_prompts(prompts_dir)
    if name not in prompts:
        raise KeyError(f"Prompt '{name}' not found. Available: {list(prompts.keys())}")

    prompt_def = prompts[name]

    # Build targeted context: only load schema providers the prompt needs
    # Check frontmatter for explicit schema_needs, otherwise use defaults
    schema_needs = prompt_def.metadata.get("schema_needs")
    full_context = get_schema_for_prompt(name, schema_needs)

    # Add prompt-specific context (backwards compat)
    if name == "discovery/data_access":
        full_context.update(get_access_methods_context())

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
