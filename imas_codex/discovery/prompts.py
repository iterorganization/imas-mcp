"""
Runtime prompt loading and rendering for the Discovery Engine.

Loads prompt templates from markdown files with YAML frontmatter,
renders them with Jinja2 using facility-specific context.
"""

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from jinja2 import BaseLoader, Environment


def get_prompts_dir() -> Path:
    """Get the prompts directory path."""
    return Path(__file__).parent.parent / "prompts"


@dataclass
class PromptMetadata:
    """Metadata from prompt YAML frontmatter."""

    name: str
    description: str = ""
    model: str = "anthropic/claude-opus-4.5"
    max_iterations: int = 5
    output_schema: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class Prompt:
    """A loaded prompt template."""

    metadata: PromptMetadata
    template: str
    source: Path

    def render(self, **context: Any) -> str:
        """Render the template with the given context."""
        env = Environment(loader=BaseLoader())

        # Add custom filters
        env.filters["yaml"] = lambda x: yaml.dump(x, default_flow_style=False)
        env.filters["join"] = lambda x, sep=", ": sep.join(str(i) for i in x)

        template = env.from_string(self.template)
        return template.render(**context)


class PromptLoader:
    """
    Loads and manages prompt templates from markdown files.

    Prompts are loaded from imas_codex/prompts/ directory.
    Each prompt is a markdown file with YAML frontmatter.
    """

    FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

    def __init__(self, prompts_dir: Path | None = None):
        self.prompts_dir = prompts_dir or get_prompts_dir()
        self._cache: dict[str, Prompt] = {}

    def _parse_frontmatter(self, content: str) -> tuple[dict[str, Any], str]:
        """Parse YAML frontmatter from markdown content."""
        match = self.FRONTMATTER_PATTERN.match(content)
        if not match:
            raise ValueError("No YAML frontmatter found")

        frontmatter = yaml.safe_load(match.group(1))
        body = content[match.end() :]
        return frontmatter, body

    def load(self, name: str) -> Prompt:
        """
        Load a prompt by name.

        Args:
            name: Prompt name (from frontmatter) or relative path without .md

        Returns:
            Loaded Prompt object

        Raises:
            ValueError: If prompt not found
        """
        # Check cache first
        if name in self._cache:
            return self._cache[name]

        # Try to find by name in all prompts
        for md_file in self.prompts_dir.rglob("*.md"):
            try:
                content = md_file.read_text(encoding="utf-8")
                frontmatter, body = self._parse_frontmatter(content)

                prompt_name = frontmatter.get("name")
                if prompt_name == name:
                    prompt = Prompt(
                        metadata=PromptMetadata(
                            name=prompt_name,
                            description=frontmatter.get("description", ""),
                            model=frontmatter.get("model", "anthropic/claude-opus-4.5"),
                            max_iterations=frontmatter.get("max_iterations", 5),
                            output_schema=frontmatter.get("output_schema"),
                            tags=frontmatter.get("tags", []),
                        ),
                        template=body,
                        source=md_file,
                    )
                    self._cache[name] = prompt
                    return prompt

            except Exception:
                continue

        # Try loading by path
        path_candidates = [
            self.prompts_dir / f"{name}.md",
            self.prompts_dir / name / "index.md",
        ]
        for path in path_candidates:
            if path.exists():
                return self._load_file(path)

        raise ValueError(f"Prompt not found: {name}")

    def _load_file(self, path: Path) -> Prompt:
        """Load a prompt from a specific file path."""
        content = path.read_text(encoding="utf-8")
        frontmatter, body = self._parse_frontmatter(content)

        prompt_name = frontmatter.get("name", path.stem)
        prompt = Prompt(
            metadata=PromptMetadata(
                name=prompt_name,
                description=frontmatter.get("description", ""),
                model=frontmatter.get("model", "anthropic/claude-opus-4.5"),
                max_iterations=frontmatter.get("max_iterations", 5),
                output_schema=frontmatter.get("output_schema"),
                tags=frontmatter.get("tags", []),
            ),
            template=body,
            source=path,
        )
        self._cache[prompt_name] = prompt
        return prompt

    def list_prompts(self) -> list[PromptMetadata]:
        """List all available prompts."""
        prompts = []
        for md_file in self.prompts_dir.rglob("*.md"):
            try:
                content = md_file.read_text(encoding="utf-8")
                frontmatter, _ = self._parse_frontmatter(content)

                prompts.append(
                    PromptMetadata(
                        name=frontmatter.get("name", md_file.stem),
                        description=frontmatter.get("description", ""),
                        model=frontmatter.get("model", "anthropic/claude-opus-4.5"),
                        max_iterations=frontmatter.get("max_iterations", 5),
                        output_schema=frontmatter.get("output_schema"),
                        tags=frontmatter.get("tags", []),
                    )
                )
            except Exception:
                continue

        return prompts

    def render(
        self,
        name: str,
        **context: Any,
    ) -> tuple[str, str]:
        """
        Load and render a prompt, returning system and user messages.

        Args:
            name: Prompt name
            **context: Template context variables

        Returns:
            Tuple of (system_message, user_message)
        """
        prompt = self.load(name)
        rendered = prompt.render(**context)

        # Build system message with safety rules
        system_message = self._build_system_message(prompt.metadata)

        return system_message, rendered

    def _build_system_message(self, metadata: PromptMetadata) -> str:
        """Build the system message for the LLM."""
        return f"""You are an expert system administrator and code analyst exploring remote fusion research facilities.

Your role: {metadata.description}

## Response Format

You operate in an agentic loop. Each turn you must respond with ONE of:

### Option 1: Generate a bash script
When you need to gather information, wrap your script in a bash code block:

```bash
#!/bin/bash
# Your script here
```

I will execute this script on the remote system and show you the results.

### Option 2: Signal completion
When you have gathered enough information, respond with a JSON block:

```json
{{"done": true, "findings": {{...}}}}
```

## Safety Rules

- All scripts must be READ-ONLY - no file modifications
- Never use: rm, mv, cp, chmod, sudo, dd, or any destructive commands
- Handle errors gracefully - don't fail on permission denied
- Limit output size - use head/tail when appropriate

## Guidelines

- Be systematic and thorough
- Explain your reasoning before each script
- If a script fails, analyze the error and try a different approach
- You may generate multiple scripts iteratively to explore
"""


@lru_cache(maxsize=1)
def get_prompt_loader() -> PromptLoader:
    """Get the global prompt loader instance (cached)."""
    return PromptLoader()


def load_prompt(name: str) -> Prompt:
    """Convenience function to load a prompt."""
    return get_prompt_loader().load(name)


def render_prompt(name: str, **context: Any) -> tuple[str, str]:
    """Convenience function to render a prompt."""
    return get_prompt_loader().render(name, **context)
