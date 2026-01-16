"""Prompt loading for exploration prompts.

Prompts are markdown files with YAML frontmatter:
- name: Prompt identifier
- description: Short description
"""

import re
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class PromptDefinition:
    """A loaded prompt definition."""

    name: str
    description: str
    content: str


def parse_prompt_file(path: Path) -> PromptDefinition:
    """Parse a markdown prompt file with YAML frontmatter."""
    text = path.read_text()

    match = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
    if not match:
        return PromptDefinition(name=path.stem, description="", content=text.strip())

    frontmatter_text, content = match.groups()
    frontmatter = yaml.safe_load(frontmatter_text)

    return PromptDefinition(
        name=frontmatter.get("name", path.stem),
        description=frontmatter.get("description", ""),
        content=content.strip(),
    )


def load_prompts(prompts_dir: Path | None = None) -> dict[str, PromptDefinition]:
    """Load all prompts from a directory."""
    if prompts_dir is None:
        prompts_dir = Path(__file__).parent / "prompts"

    prompts = {}
    if prompts_dir.exists():
        for md_file in sorted(prompts_dir.glob("*.md")):
            try:
                prompt = parse_prompt_file(md_file)
                prompts[prompt.name] = prompt
            except Exception as e:
                import logging

                logging.getLogger(__name__).warning(f"Failed to parse {md_file}: {e}")

    return prompts


def list_prompts_summary(prompts_dir: Path | None = None) -> list[dict[str, str]]:
    """Get a summary list of available prompts."""
    prompts = load_prompts(prompts_dir)
    return [{"name": p.name, "description": p.description} for p in prompts.values()]
