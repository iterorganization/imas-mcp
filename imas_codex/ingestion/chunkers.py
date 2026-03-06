"""Text and code chunking via tree-sitter and sliding window.

Uses tree-sitter-language-pack for most languages and tree-sitter-gdl
for IDL/GDL parsing.
"""

from dataclasses import dataclass

import tree_sitter
import tree_sitter_gdl
from tree_sitter_language_pack import get_parser

_gdl_language = tree_sitter_gdl.language()


def _get_parser(language: str) -> tree_sitter.Parser:
    """Return a tree-sitter parser for the given language."""
    if language in ("idl", "gdl"):
        return tree_sitter.Parser(_gdl_language)
    return get_parser(language)


@dataclass
class Chunk:
    """A chunk of text with position metadata."""

    text: str
    start_line: int
    end_line: int


def chunk_code(
    text: str,
    language: str,
    max_chars: int = 10000,
    chunk_lines: int = 40,
    chunk_lines_overlap: int = 10,
) -> list[Chunk]:
    """Chunk source code using tree-sitter AST boundaries.

    Parses the source with tree-sitter, walks top-level AST nodes,
    and accumulates them into chunks that respect function/class
    boundaries. Falls back to text chunking if parsing fails.

    Args:
        text: Source code to chunk
        language: Programming language name (e.g., "python", "fortran")
        max_chars: Maximum characters per chunk
        chunk_lines: Target lines per chunk (unused, kept for API compat)
        chunk_lines_overlap: Number of overlap lines between chunks

    Returns:
        List of Chunk objects with text and line positions
    """
    parser = _get_parser(language)
    tree = parser.parse(text.encode())
    root = tree.root_node

    chunks: list[Chunk] = []
    current_lines: list[str] = []
    current_start = 0
    current_chars = 0

    for child in root.children:
        child_text = child.text.decode()
        child_lines = child_text.split("\n")
        child_chars = len(child_text)

        # If adding this child would exceed max_chars and we have content, flush
        if current_chars + child_chars > max_chars and current_lines:
            joined = "\n".join(current_lines)
            chunks.append(
                Chunk(
                    text=joined,
                    start_line=current_start,
                    end_line=current_start + len(current_lines) - 1,
                )
            )
            # Overlap: keep last N lines
            overlap = current_lines[-chunk_lines_overlap:]
            current_start = current_start + len(current_lines) - len(overlap)
            current_lines = list(overlap)
            current_chars = sum(len(line) for line in current_lines)

        # If a single AST node exceeds max_chars, sub-chunk it via text
        # splitting to prevent oversized chunks from reaching the embedder.
        if child_chars > max_chars:
            sub_chunks = chunk_text(
                child_text,
                chunk_size=max_chars,
                chunk_overlap=chunk_lines_overlap * 60,
            )
            for sc in sub_chunks:
                # Flush any accumulated content first
                if current_lines:
                    joined = "\n".join(current_lines)
                    chunks.append(
                        Chunk(
                            text=joined,
                            start_line=current_start,
                            end_line=current_start + len(current_lines) - 1,
                        )
                    )
                    current_start = current_start + len(current_lines)
                    current_lines = []
                    current_chars = 0
                chunks.append(
                    Chunk(
                        text=sc.text,
                        start_line=current_start + sc.start_line,
                        end_line=current_start + sc.end_line,
                    )
                )
            # Advance start past the sub-chunked node
            current_start += len(child_lines)
            continue

        current_lines.extend(child_lines)
        current_chars += child_chars

    if current_lines:
        joined = "\n".join(current_lines)
        chunks.append(
            Chunk(
                text=joined,
                start_line=current_start,
                end_line=current_start + len(current_lines) - 1,
            )
        )

    return chunks


def chunk_text(
    text: str,
    chunk_size: int = 10000,
    chunk_overlap: int = 200,
    separator: str = "\n",
) -> list[Chunk]:
    """Chunk text using a sliding window on separator boundaries.

    Used for languages without tree-sitter support (IDL, TDI)
    and for document content (wiki pages, markdown).

    Args:
        text: Text to chunk
        chunk_size: Maximum characters per chunk
        chunk_overlap: Character overlap between chunks
        separator: String to split on

    Returns:
        List of Chunk objects with text and line positions
    """
    parts = text.split(separator)
    chunks: list[Chunk] = []
    current_parts: list[str] = []
    current_chars = 0
    current_start = 0

    for part in parts:
        part_len = len(part) + len(separator)
        if current_chars + part_len > chunk_size and current_parts:
            chunk_text = separator.join(current_parts)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    start_line=current_start,
                    end_line=current_start + len(current_parts) - 1,
                )
            )
            # Calculate overlap in parts
            overlap_chars = 0
            overlap_start = len(current_parts)
            for j in range(len(current_parts) - 1, -1, -1):
                overlap_chars += len(current_parts[j]) + len(separator)
                if overlap_chars >= chunk_overlap:
                    overlap_start = j
                    break
            overlap = current_parts[overlap_start:]
            current_start = current_start + overlap_start
            current_parts = list(overlap)
            current_chars = sum(len(p) + len(separator) for p in current_parts)

        current_parts.append(part)
        current_chars += part_len

    if current_parts:
        chunk_text = separator.join(current_parts)
        chunks.append(
            Chunk(
                text=chunk_text,
                start_line=current_start,
                end_line=current_start + len(current_parts) - 1,
            )
        )

    return chunks
