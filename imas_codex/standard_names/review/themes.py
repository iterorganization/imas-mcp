"""Reviewer theme extraction for compose prompt enrichment (L4).

Extracts recurring themes from reviewer comments on StandardName nodes
in a given physics domain using n-gram frequency analysis, then renders
them as concise bullet points for injection into compose system prompts.
"""

from __future__ import annotations

import logging
import re
from collections import Counter

logger = logging.getLogger(__name__)

# Common stopwords to skip in n-gram extraction
_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "must",
        "can",
        "could",
        "of",
        "in",
        "to",
        "for",
        "with",
        "on",
        "at",
        "from",
        "by",
        "as",
        "or",
        "and",
        "but",
        "if",
        "not",
        "no",
        "so",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "which",
        "who",
        "whom",
        "whose",
        "when",
        "where",
        "how",
        "what",
        "why",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "than",
        "too",
        "very",
        "just",
        "also",
        "about",
        "up",
        "out",
        "into",
        "over",
        "after",
        "before",
        "between",
        "under",
        "through",
        "during",
        "any",
        "own",
        "same",
    }
)


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, return tokens."""
    return [
        w
        for w in re.findall(r"[a-z_]+", text.lower())
        if w not in _STOPWORDS and len(w) > 2
    ]


def _extract_ngrams(tokens: list[str], n: int) -> list[str]:
    """Extract n-grams from token list."""
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def extract_reviewer_themes(
    domain: str | None,
    limit: int = 50,
) -> list[str]:
    """Extract top recurring themes from reviewer comments in a domain.

    Queries the graph for StandardName nodes with non-null
    ``reviewer_comments`` in the specified domain, runs bigram/trigram
    frequency analysis, and returns the top ~10 concise themes.

    Args:
        domain: Physics domain filter. Returns empty list if None.
        limit: Maximum number of names to sample reviewer comments from.

    Returns:
        List of concise theme strings (e.g., "missing sign convention",
        "inconsistent boundary naming"). Empty if no comments found.
    """
    if not domain:
        return []

    try:
        from imas_codex.graph.client import GraphClient

        with GraphClient() as gc:
            rows = gc.query(
                """
                MATCH (sn:StandardName)
                WHERE sn.physics_domain = $domain
                  AND sn.reviewer_comments IS NOT NULL
                RETURN sn.reviewer_comments AS comments
                LIMIT $limit
                """,
                domain=domain,
                limit=limit,
            )
            if not rows:
                return []

            # Collect all comment texts
            all_comments: list[str] = []
            for row in rows:
                comment = row.get("comments") or ""
                if isinstance(comment, list):
                    all_comments.extend(comment)
                elif comment:
                    all_comments.append(comment)

            if not all_comments:
                return []

            return _extract_themes_from_texts(all_comments)
    except Exception:
        logger.debug("Reviewer theme extraction unavailable", exc_info=True)
        return []


def _extract_themes_from_texts(texts: list[str], top_n: int = 10) -> list[str]:
    """Extract top themes from a list of comment texts using n-gram frequency.

    Uses bigram and trigram frequency counting with basic TF-IDF weighting
    (document frequency penalty). Falls back to pure frequency if sklearn
    is not available.
    """
    # Tokenize all texts
    all_tokens: list[list[str]] = [_tokenize(t) for t in texts]

    # Count bigrams and trigrams across all documents
    bigram_counter: Counter[str] = Counter()
    trigram_counter: Counter[str] = Counter()

    # Also track document frequency for weighting
    bigram_df: Counter[str] = Counter()
    trigram_df: Counter[str] = Counter()

    n_docs = len(texts)

    for tokens in all_tokens:
        bigrams = _extract_ngrams(tokens, 2)
        trigrams = _extract_ngrams(tokens, 3)

        bigram_counter.update(bigrams)
        trigram_counter.update(trigrams)

        # Document frequency (unique per document)
        bigram_df.update(set(bigrams))
        trigram_df.update(set(trigrams))

    # Combine with TF-IDF-like scoring
    import math

    scored: list[tuple[str, float]] = []

    for gram, count in bigram_counter.items():
        if count < 2:
            continue
        df = bigram_df[gram]
        # IDF = log(N/df) — downweight terms appearing in all documents
        idf = math.log(max(n_docs, 1) / max(df, 1)) + 1.0
        scored.append((gram, count * idf))

    for gram, count in trigram_counter.items():
        if count < 2:
            continue
        df = trigram_df[gram]
        idf = math.log(max(n_docs, 1) / max(df, 1)) + 1.0
        # Boost trigrams slightly — they are more specific
        scored.append((gram, count * idf * 1.2))

    # Sort by score descending, deduplicate overlapping n-grams
    scored.sort(key=lambda x: x[1], reverse=True)

    themes: list[str] = []
    seen_tokens: set[str] = set()

    for gram, _score in scored:
        gram_tokens = set(gram.split())
        # Skip if significantly overlapping with already-selected theme
        if (
            gram_tokens & seen_tokens
            and len(gram_tokens & seen_tokens) >= len(gram_tokens) // 2 + 1
        ):
            continue
        themes.append(gram)
        seen_tokens.update(gram_tokens)
        if len(themes) >= top_n:
            break

    return themes
