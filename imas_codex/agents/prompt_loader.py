"""Prompt loading and management for exploration prompts.

Prompts are markdown files with YAML frontmatter that define:
- name: Prompt identifier
- description: Short description for listing
- arguments: Dict of argument definitions with type, description, default, required
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PromptArgument:
    """Definition of a prompt argument."""

    name: str
    type: str = "string"
    description: str = ""
    default: Any = None
    required: bool = False


@dataclass
class PromptDefinition:
    """A loaded prompt definition."""

    name: str
    description: str
    template: str
    arguments: list[PromptArgument] = field(default_factory=list)
    source_file: Path | None = None

    def render(self, **kwargs: Any) -> str:
        """Render the prompt template with arguments.

        Missing optional arguments use their defaults.
        """
        # Build context with defaults
        context = {}
        for arg in self.arguments:
            if arg.name in kwargs:
                context[arg.name] = kwargs[arg.name]
            elif arg.default is not None:
                context[arg.name] = arg.default
            elif arg.required:
                msg = f"Missing required argument: {arg.name}"
                raise ValueError(msg)

        # Simple string formatting (handles {name} placeholders)
        try:
            return self.template.format(**context)
        except KeyError:
            # Some placeholders might be code examples, not arguments
            # Fall back to partial formatting
            return self.template.format_map(_SafeDict(context))


class _SafeDict(dict):
    """Dict that returns the key wrapped in braces for missing keys."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def parse_prompt_file(path: Path) -> PromptDefinition:
    """Parse a markdown prompt file with YAML frontmatter.

    Args:
        path: Path to the markdown file

    Returns:
        PromptDefinition with parsed metadata and template
    """
    content = path.read_text()

    # Split frontmatter from body
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", content, re.DOTALL)
    if not match:
        # No frontmatter, use filename as name
        return PromptDefinition(
            name=path.stem,
            description="",
            template=content,
            source_file=path,
        )

    frontmatter_text, template = match.groups()
    frontmatter = yaml.safe_load(frontmatter_text)

    # Parse arguments
    arguments = []
    for arg_name, arg_def in frontmatter.get("arguments", {}).items():
        if isinstance(arg_def, dict):
            arguments.append(
                PromptArgument(
                    name=arg_name,
                    type=arg_def.get("type", "string"),
                    description=arg_def.get("description", ""),
                    default=arg_def.get("default"),
                    required=arg_def.get("required", False),
                )
            )
        else:
            # Simple value is the default
            arguments.append(PromptArgument(name=arg_name, default=arg_def))

    return PromptDefinition(
        name=frontmatter.get("name", path.stem),
        description=frontmatter.get("description", ""),
        template=template.strip(),
        arguments=arguments,
        source_file=path,
    )


def load_prompts(prompts_dir: Path | None = None) -> dict[str, PromptDefinition]:
    """Load all prompts from a directory.

    Args:
        prompts_dir: Directory containing markdown prompt files.
                    Defaults to agents/prompts/.

    Returns:
        Dict mapping prompt name to PromptDefinition.
    """
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

                logging.getLogger(__name__).warning(
                    f"Failed to parse prompt {md_file}: {e}"
                )

    return prompts


def list_prompts_summary(prompts_dir: Path | None = None) -> list[dict[str, Any]]:
    """Get a summary list of available prompts.

    Returns:
        List of dicts with name, description, and argument info.
    """
    prompts = load_prompts(prompts_dir)
    result = []
    for prompt in prompts.values():
        args_summary = []
        for arg in prompt.arguments:
            arg_info = {"name": arg.name, "type": arg.type}
            if arg.required:
                arg_info["required"] = True
            if arg.default is not None:
                arg_info["default"] = arg.default
            if arg.description:
                arg_info["description"] = arg.description
            args_summary.append(arg_info)

        result.append(
            {
                "name": prompt.name,
                "description": prompt.description,
                "arguments": args_summary,
            }
        )
    return result
