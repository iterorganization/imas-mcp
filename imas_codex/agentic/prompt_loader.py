"""Prompt loading for agentic workflows.

Prompts are markdown files with YAML frontmatter:
- name: Prompt identifier (can include path like "discovery/scorer")
- description: Short description
- task: Task type for model selection (optional)

Prompts support includes via Jinja-style syntax:
    {% include "safety.md" %}

Includes are resolved relative to the `shared/` subdirectory.

Directory structure:
    prompts/
    ├── shared/           # Reusable includes
    │   └── safety.md
    ├── discovery/        # Discovery pipeline prompts
    │   └── scorer.md
    ├── wiki/             # Wiki processing prompts (future)
    └── enrich-system.md  # Root-level prompts
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

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
