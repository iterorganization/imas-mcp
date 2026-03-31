"""
LLM module — prompt rendering, MCP server, and search tools.

Submodules:
- prompt_loader: Jinja2 prompt rendering with schema injection
- server: MCP server for LLM-driven facility exploration
- search_tools: MCP search tool implementations
- search_formatters: Format search results for MCP responses
- prompts/: Jinja2 prompt templates

Import AgentsServer directly to avoid loading the full server at package import time:

    from imas_codex.llm.server import AgentsServer
"""
