"""Scoring functions for wiki discovery.

LLM-based scoring for wiki pages, artifacts, and images:
- Page scoring: content preview + LLM structured output
- Artifact scoring: text extraction + LLM structured output
- Image scoring: VLM captioning + structured output
- HTML fetching: SSH proxy, Tequila, Keycloak, HTTP Basic auth
- Heuristic fallback scoring for non-LLM path

All scoring functions return (results, cost) tuples where cost
is the actual LLM cost from OpenRouter API.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from typing import Any

logger = logging.getLogger(__name__)


async def _extract_artifact_preview(
    url: str,
    artifact_type: str,
    facility: str,
    max_chars: int = 1500,
) -> str:
    """Extract a text preview from an artifact for LLM scoring.

    Downloads the artifact content and extracts text from the first
    portion. This is a lightweight extraction for scoring purposes only.

    For PDFs: extracts text from first few pages.
    For documents: extracts text paragraphs.
    For presentations: extracts slide text.
    For notebooks: extracts cell content.

    Args:
        url: Artifact download URL
        artifact_type: Type of artifact (pdf, docx, pptx, xlsx, ipynb)
        facility: Facility ID (for SSH proxy)
        max_chars: Maximum characters to extract

    Returns:
        Extracted text preview or empty string on failure
    """
    from imas_codex.discovery.wiki.pipeline import fetch_artifact_content

    try:
        _, content_bytes = await fetch_artifact_content(url, facility=facility)
    except Exception as e:
        logger.debug("Failed to download %s: %s", url, e)
        return ""

    try:
        text = _extract_text_from_bytes(content_bytes, artifact_type)
        return text[:max_chars] if text else ""
    except Exception as e:
        logger.debug("Failed to extract text from %s: %s", url, e)
        return ""


def _extract_text_from_bytes(content_bytes: bytes, artifact_type: str) -> str:
    """Extract text from artifact bytes based on type.

    Lightweight extraction for scoring preview. Does not use LlamaIndex
    to avoid heavyweight dependencies in the scoring path.

    Uses semantic artifact type names matching ArtifactType enum values.
    """
    import tempfile
    from pathlib import Path

    at = artifact_type.lower()

    if at == "pdf":
        # Extract text from first few pages
        if b"%PDF" not in content_bytes[:1024]:
            return ""
        try:
            import logging as _logging

            import pypdf

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                f.write(content_bytes)
                temp_path = Path(f.name)
            try:
                # Suppress pypdf's verbose warnings and errors.
                # pypdf._cmap uses logger_error() for benign "Advanced
                # encoding" messages at ERROR level — CRITICAL silences them.
                pypdf_logger = _logging.getLogger("pypdf")
                original_level = pypdf_logger.level
                pypdf_logger.setLevel(_logging.CRITICAL)
                try:
                    reader = pypdf.PdfReader(temp_path)
                    text_parts = []
                    for page in reader.pages[:5]:  # First 5 pages
                        text = page.extract_text()
                        if text:
                            text_parts.append(text)
                finally:
                    pypdf_logger.setLevel(original_level)
                return "\n\n".join(text_parts)
            finally:
                temp_path.unlink(missing_ok=True)
        except Exception:
            return ""

    elif at == "document":
        try:
            from docx import Document as DocxDocument

            with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
                f.write(content_bytes)
                temp_path = Path(f.name)
            try:
                doc = DocxDocument(temp_path)
                paragraphs = [p.text for p in doc.paragraphs[:50] if p.text.strip()]
                return "\n\n".join(paragraphs)
            finally:
                temp_path.unlink(missing_ok=True)
        except Exception:
            return ""

    elif at == "presentation":
        try:
            from pptx import Presentation

            with tempfile.NamedTemporaryFile(suffix=".pptx", delete=False) as f:
                f.write(content_bytes)
                temp_path = Path(f.name)
            try:
                prs = Presentation(temp_path)
                text_parts = []
                for slide in list(prs.slides)[:10]:  # First 10 slides
                    for shape in slide.shapes:
                        if hasattr(shape, "text") and shape.text.strip():
                            text_parts.append(shape.text.strip())
                return "\n".join(text_parts)
            finally:
                temp_path.unlink(missing_ok=True)
        except Exception:
            return ""

    elif at == "spreadsheet":
        try:
            from imas_codex.discovery.wiki.excel import extract_excel_preview

            return extract_excel_preview(content_bytes)
        except Exception:
            return ""

    elif at == "notebook":
        try:
            import json

            nb = json.loads(content_bytes.decode("utf-8"))
            cells = nb.get("cells", [])
            text_parts = []
            for cell in cells[:20]:  # First 20 cells
                source = "".join(cell.get("source", []))
                if source.strip():
                    text_parts.append(source)
            return "\n\n".join(text_parts)
        except Exception:
            return ""

    elif at == "json":
        try:
            import json

            text = content_bytes.decode("utf-8", errors="replace")
            # Validate it's parseable JSON
            data = json.loads(text)
            # For large JSON, extract a structured preview
            preview = json.dumps(data, indent=2, ensure_ascii=False, default=str)
            return preview[:5000]  # Cap preview at 5000 chars
        except Exception:
            return ""

    return ""


async def _score_artifacts_batch(
    artifacts: list[dict[str, Any]],
    model: str,
    focus: str | None = None,
) -> tuple[list[dict[str, Any]], float]:
    """Score a batch of artifacts using LLM with structured output.

    Uses litellm.acompletion with ArtifactScoreBatch Pydantic model for
    structured output. Content-based scoring with per-dimension scores.

    Args:
        artifacts: List of artifact dicts with id, filename, preview_text, etc.
        model: Model identifier from get_model()
        focus: Optional focus area for scoring

    Returns:
        (results, cost) tuple where cost is actual LLM cost from OpenRouter.
    """
    import os
    import re

    import litellm

    from imas_codex.agentic.prompt_loader import render_prompt
    from imas_codex.discovery.wiki.models import (
        ArtifactScoreBatch,
        grounded_artifact_score,
    )

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set.")

    model_id = model
    if not model_id.startswith("openrouter/"):
        model_id = f"openrouter/{model_id}"

    # Build system prompt using artifact-scorer template
    context: dict[str, Any] = {}
    if focus:
        context["focus"] = focus

    system_prompt = render_prompt("wiki/artifact-scorer", context)

    # Build user prompt with artifact content
    lines = [
        f"Score these {len(artifacts)} wiki artifacts based on their content.",
        "(Use the preview text to assess value for the IMAS knowledge graph.)\n",
    ]

    for i, a in enumerate(artifacts, 1):
        lines.append(f"\n## Artifact {i}")
        lines.append(f"ID: {a['id']}")
        lines.append(f"Filename: {a.get('filename', 'Unknown')}")
        lines.append(f"Type: {a.get('artifact_type', 'unknown')}")

        if a.get("size_bytes"):
            size_mb = a["size_bytes"] / (1024 * 1024)
            lines.append(f"Size: {size_mb:.1f} MB")

        preview = a.get("preview_text", "")
        if preview:
            lines.append(f"Content Preview:\n{preview[:800]}")
        else:
            lines.append("Content Preview: (not available - score from filename/type)")

        if a.get("url"):
            lines.append(f"URL: {a['url']}")

    lines.append(
        "\n\nReturn results for each artifact in order. "
        "The response format is enforced by the schema."
    )

    user_prompt = "\n".join(lines)

    # Retry loop
    max_retries = 5
    retry_base_delay = 4.0
    last_error = None
    total_cost = 0.0

    for attempt in range(max_retries):
        try:
            response = await litellm.acompletion(
                model=model_id,
                api_key=api_key,
                response_format=ArtifactScoreBatch,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=16000,
                timeout=120,
            )

            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

            if (
                hasattr(response, "_hidden_params")
                and "response_cost" in response._hidden_params
            ):
                cost = response._hidden_params["response_cost"]
            else:
                cost = (input_tokens * 3 + output_tokens * 15) / 1_000_000

            total_cost += cost

            content = response.choices[0].message.content
            if not content:
                logger.warning("LLM returned empty response for artifacts")
                return [], total_cost

            content = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", content)
            content = content.encode("utf-8", errors="surrogateescape").decode(
                "utf-8", errors="replace"
            )

            batch = ArtifactScoreBatch.model_validate_json(content)
            llm_results = batch.results
            break

        except Exception as e:
            last_error = e
            error_msg = str(e).lower()

            is_retryable = any(
                x in error_msg
                for x in [
                    "overloaded",
                    "rate",
                    "429",
                    "503",
                    "timeout",
                    "eof",
                    "json",
                    "truncated",
                    "validation",
                ]
            )

            if is_retryable and attempt < max_retries - 1:
                delay = retry_base_delay * (2**attempt)
                logger.debug(
                    "LLM error (attempt %d/%d): %s. Retrying in %.1fs...",
                    attempt + 1,
                    max_retries,
                    str(e)[:100],
                    delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "LLM failed for artifact batch of %d after %d attempts: %s. "
                    "Artifacts reverted to discovered status for retry.",
                    len(artifacts),
                    attempt + 1,
                    e,
                )
                raise ValueError(
                    f"LLM failed after {attempt + 1} attempts: {e}. "
                    f"Artifacts reverted to discovered status."
                ) from e
    else:
        raise last_error  # type: ignore[misc]

    # Convert to result dicts
    cost_per_artifact = total_cost / len(artifacts) if artifacts else 0.0
    results = []

    for r in llm_results[: len(artifacts)]:
        scores = {
            "score_data_documentation": r.score_data_documentation,
            "score_physics_content": r.score_physics_content,
            "score_code_documentation": r.score_code_documentation,
            "score_data_access": r.score_data_access,
            "score_calibration": r.score_calibration,
            "score_imas_relevance": r.score_imas_relevance,
        }

        combined_score = grounded_artifact_score(scores, r.artifact_purpose)

        # Find the matching artifact for filename
        matching = next((a for a in artifacts if a["id"] == r.id), {})

        results.append(
            {
                "id": r.id,
                "score": combined_score,
                "artifact_purpose": r.artifact_purpose.value,
                "description": r.description[:150],
                "reasoning": r.reasoning[:80],
                "keywords": r.keywords[:5],
                "physics_domain": r.physics_domain.value if r.physics_domain else None,
                "should_ingest": r.should_ingest,
                "skip_reason": r.skip_reason or None,
                "score_data_documentation": r.score_data_documentation,
                "score_physics_content": r.score_physics_content,
                "score_code_documentation": r.score_code_documentation,
                "score_data_access": r.score_data_access,
                "score_calibration": r.score_calibration,
                "score_imas_relevance": r.score_imas_relevance,
                "score_cost": cost_per_artifact,
                # Pass through filename for display
                "filename": matching.get("filename", ""),
                "artifact_type": matching.get("artifact_type", ""),
            }
        )

    return results, total_cost


# =============================================================================


async def _score_images_batch(
    images: list[dict[str, Any]],
    model: str,
    focus: str | None = None,
) -> tuple[list[dict[str, Any]], float]:
    """Score a batch of images using VLM with structured output.

    Sends image bytes + context to VLM and receives caption + scoring
    in a single pass. Uses ImageScoreBatch Pydantic model.

    Returns:
        (results, cost) tuple
    """
    import os
    import re

    import litellm

    from imas_codex.agentic.prompt_loader import render_prompt
    from imas_codex.discovery.wiki.models import (
        ImageScoreBatch,
        grounded_image_score,
    )

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set.")

    model_id = model
    if not model_id.startswith("openrouter/"):
        model_id = f"openrouter/{model_id}"

    # Build system prompt
    context: dict[str, Any] = {}
    if focus:
        context["focus"] = focus
    system_prompt = render_prompt("wiki/image-captioner", context)

    # Build user message with image content
    user_content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": f"Score and caption these {len(images)} images from fusion facility documentation.\n",
        }
    ]

    for i, img in enumerate(images, 1):
        # Add text context for this image
        context_parts = [f"\n## Image {i}", f"ID: {img['id']}"]
        if img.get("page_title"):
            context_parts.append(f"Page: {img['page_title']}")
        if img.get("section"):
            context_parts.append(f"Section: {img['section']}")
        if img.get("surrounding_text"):
            context_parts.append(f"Context: {img['surrounding_text'][:500]}")
        if img.get("alt_text"):
            context_parts.append(f"Alt text: {img['alt_text']}")

        user_content.append({"type": "text", "text": "\n".join(context_parts)})

        # Add image data
        img_format = img.get("image_format", "webp")
        mime_type = f"image/{img_format}"
        user_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{img['image_data']}",
                },
            }
        )

    user_content.append(
        {
            "type": "text",
            "text": "\n\nReturn results for each image in order. "
            "The response format is enforced by the schema.",
        }
    )

    # Retry loop
    max_retries = 5
    retry_base_delay = 4.0
    last_error = None
    total_cost = 0.0

    for attempt in range(max_retries):
        try:
            response = await litellm.acompletion(
                model=model_id,
                api_key=api_key,
                response_format=ImageScoreBatch,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.3,
                max_tokens=32000,
                timeout=180,  # VLM may need more time for images
            )

            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

            if (
                hasattr(response, "_hidden_params")
                and "response_cost" in response._hidden_params
            ):
                cost = response._hidden_params["response_cost"]
            else:
                # Fallback VLM rates
                cost = (input_tokens * 3 + output_tokens * 15) / 1_000_000

            total_cost += cost

            content = response.choices[0].message.content
            if not content:
                return [], total_cost

            content = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", content)
            content = content.encode("utf-8", errors="surrogateescape").decode(
                "utf-8", errors="replace"
            )

            batch = ImageScoreBatch.model_validate_json(content)
            llm_results = batch.results
            break

        except Exception as e:
            last_error = e
            error_msg = str(e).lower()
            is_retryable = any(
                x in error_msg
                for x in [
                    "overloaded",
                    "rate",
                    "429",
                    "503",
                    "timeout",
                    "eof",
                    "json",
                    "truncated",
                    "validation",
                ]
            )
            if is_retryable and attempt < max_retries - 1:
                delay = retry_base_delay * (2**attempt)
                logger.debug(
                    "VLM error (attempt %d/%d): %s. Retrying in %.1fs...",
                    attempt + 1,
                    max_retries,
                    str(e)[:100],
                    delay,
                )
                await asyncio.sleep(delay)
            else:
                raise ValueError(f"VLM failed after {attempt + 1} attempts: {e}") from e
    else:
        raise last_error  # type: ignore[misc]

    # Convert to result dicts with grounded scoring
    cost_per_image = total_cost / len(images) if images else 0.0
    results = []

    for r in llm_results[: len(images)]:
        scores = {
            "score_data_documentation": r.score_data_documentation,
            "score_physics_content": r.score_physics_content,
            "score_code_documentation": r.score_code_documentation,
            "score_data_access": r.score_data_access,
            "score_calibration": r.score_calibration,
            "score_imas_relevance": r.score_imas_relevance,
        }
        combined_score = grounded_image_score(scores, r.purpose)

        results.append(
            {
                "id": r.id,
                "caption": r.caption,
                "ocr_text": r.ocr_text,
                "purpose": r.purpose.value,
                "description": r.description[:150],
                "score": combined_score,
                "score_data_documentation": r.score_data_documentation,
                "score_physics_content": r.score_physics_content,
                "score_code_documentation": r.score_code_documentation,
                "score_data_access": r.score_data_access,
                "score_calibration": r.score_calibration,
                "score_imas_relevance": r.score_imas_relevance,
                "reasoning": r.reasoning[:80],
                "keywords": r.keywords[:5],
                "physics_domain": r.physics_domain.value if r.physics_domain else None,
                "should_ingest": r.should_ingest,
                "skip_reason": r.skip_reason or None,
                "score_cost": cost_per_image,
            }
        )

    return results, total_cost


async def _fetch_html(
    url: str,
    ssh_host: str | None,
    auth_type: str | None = None,
    credential_service: str | None = None,
    async_wiki_client: Any = None,
    keycloak_client: Any = None,
    basic_auth_client: Any = None,
) -> str:
    """Fetch HTML content from URL.

    Args:
        url: Page URL
        ssh_host: Optional SSH host for proxied fetching
        auth_type: Authentication type (tequila, session, keycloak, basic, etc.)
        credential_service: Keyring service for credentials
        async_wiki_client: Shared AsyncMediaWikiClient for Tequila auth
        keycloak_client: Shared httpx.AsyncClient for Keycloak auth
        basic_auth_client: Shared httpx.AsyncClient with HTTP Basic auth

    Returns:
        HTML content string or empty string on error
    """

    def _ssh_fetch() -> str:
        """Fetch via SSH proxy."""
        cmd = f'curl -sk --noproxy "*" "{url}" 2>/dev/null'
        try:
            result = subprocess.run(
                ["ssh", ssh_host, cmd],
                capture_output=True,
                timeout=30,
            )
            if result.returncode == 0:
                # Wiki pages may be ISO-8859 encoded despite claiming UTF-8
                return result.stdout.decode("utf-8", errors="replace")
            return ""
        except Exception as e:
            logger.warning("SSH fetch failed for %s: %s", url, e)
            return ""

    async def _async_tequila_fetch() -> str:
        """Fetch with Tequila authentication using async client."""
        import urllib.parse as urlparse

        # Extract page name from URL
        page_name = url.split("/wiki/")[-1] if "/wiki/" in url else url.split("/")[-1]
        if "?" in page_name:
            parsed = urlparse.parse_qs(urlparse.urlparse(url).query)
            page_name = parsed.get("title", [page_name])[0]
        page_name = urlparse.unquote(page_name)

        # Use provided async client
        if async_wiki_client is not None:
            try:
                page = await async_wiki_client.get_page(page_name)
                if page:
                    return page.content_html
                return ""
            except Exception as e:
                logger.debug("Async client fetch failed for %s: %s", url, e)
                return ""

        # No client provided - create a new async client
        from imas_codex.discovery.wiki.mediawiki import AsyncMediaWikiClient

        base_url_local = (
            url.rsplit("/", 1)[0] if "/wiki/" in url else url.rsplit("/", 1)[0]
        )
        if "/wiki" in base_url_local:
            base_url_local = base_url_local.rsplit("/wiki", 1)[0] + "/wiki"

        async with AsyncMediaWikiClient(
            base_url=base_url_local,
            credential_service=credential_service or "tcv",
            verify_ssl=False,
        ) as client:
            if not await client.authenticate():
                logger.warning("Tequila auth failed for %s", url)
                return ""
            page = await client.get_page(page_name)
            return page.content_html if page else ""

    async def _async_keycloak_fetch() -> str:
        """Fetch with Keycloak auth using shared httpx.AsyncClient."""
        if keycloak_client is None:
            logger.warning("No Keycloak client available for %s", url)
            return ""
        try:
            response = await keycloak_client.get(url)
            if response.status_code == 200:
                return response.text
            logger.debug(
                "Keycloak fetch returned HTTP %d for %s", response.status_code, url
            )
            return ""
        except Exception as e:
            logger.debug("Keycloak fetch failed for %s: %s", url, e)
            return ""

    async def _async_basic_auth_fetch() -> str:
        """Fetch with HTTP Basic auth using shared httpx.AsyncClient."""
        if basic_auth_client is None:
            logger.warning("No HTTP Basic auth client available for %s", url)
            return ""
        try:
            response = await basic_auth_client.get(url)
            if response.status_code == 200:
                return response.text
            logger.debug(
                "Basic auth fetch returned HTTP %d for %s", response.status_code, url
            )
            return ""
        except Exception as e:
            logger.debug("Basic auth fetch failed for %s: %s", url, e)
            return ""

    # Determine fetch strategy - prefer direct HTTP over SSH when credentials available
    if auth_type in ("tequila", "session"):
        return await _async_tequila_fetch()
    elif auth_type == "keycloak" and keycloak_client:
        return await _async_keycloak_fetch()
    elif auth_type == "basic" and basic_auth_client:
        return await _async_basic_auth_fetch()
    elif ssh_host:
        return await asyncio.to_thread(_ssh_fetch)
    else:
        # Direct HTTP fetch (no auth) - already async
        from imas_codex.discovery.wiki.prefetch import fetch_page_content

        html, error = await fetch_page_content(url)
        if html:
            return html
        if error:
            logger.debug("HTTP fetch failed for %s: %s", url, error)
        return ""


async def _fetch_and_summarize(
    url: str,
    ssh_host: str | None,
    auth_type: str | None = None,
    credential_service: str | None = None,
    max_chars: int = 2000,
    async_wiki_client: Any = None,
    keycloak_client: Any = None,
    basic_auth_client: Any = None,
) -> str:
    """Fetch page content and extract text preview.

    No LLM is used here - prefetch extracts text deterministically.
    The summary is just cleaned text for the scorer to evaluate.

    Delegates HTML fetching to _fetch_html and applies text extraction.
    """
    from imas_codex.discovery.wiki.prefetch import extract_text_from_html

    html = await _fetch_html(
        url,
        ssh_host,
        auth_type=auth_type,
        credential_service=credential_service,
        async_wiki_client=async_wiki_client,
        keycloak_client=keycloak_client,
        basic_auth_client=basic_auth_client,
    )

    if html:
        return extract_text_from_html(html, max_chars=max_chars)
    return ""


async def _score_pages_batch(
    pages: list[dict[str, Any]],
    model: str,
    focus: str | None = None,
    data_access_patterns: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], float]:
    """Score a batch of pages using LLM with structured output.

    Uses litellm.acompletion with WikiScoreBatch Pydantic model for
    structured output. Content-based scoring with per-dimension scores.

    The retry loop includes response parsing because JSON/validation
    errors from truncated responses are retryable — a fresh LLM call
    often returns valid output. Cost is accumulated across all attempts
    since API calls are billed regardless of parsing success.

    Args:
        pages: List of page dicts with id, title, summary, preview_text, etc.
        model: Model identifier from get_model()
        focus: Optional focus area for scoring
        data_access_patterns: Optional facility-specific data access patterns
            from facility config. When provided, injected into the prompt
            template so the LLM can boost pages matching facility tools/APIs.

    Returns:
        (results, cost) tuple where cost is actual LLM cost from OpenRouter.
    """
    from imas_codex.agentic.prompt_loader import render_prompt
    from imas_codex.discovery.base.llm import acall_llm_structured
    from imas_codex.discovery.wiki.models import (
        WikiScoreBatch,
        grounded_wiki_score,
    )

    # Build system prompt using dynamic template with schema injection
    context: dict[str, Any] = {}
    if focus:
        context["focus"] = focus
    if data_access_patterns:
        context["data_access_patterns"] = data_access_patterns

    system_prompt = render_prompt("wiki/scorer", context)

    # Build user prompt with page content (not graph metrics)
    lines = [
        f"Score these {len(pages)} wiki pages based on their content.",
        "(Use the preview text to infer value - graph metrics like in_degree are NOT indicators.)\n",
    ]

    for i, p in enumerate(pages, 1):
        lines.append(f"\n## Page {i}")
        lines.append(f"ID: {p['id']}")
        lines.append(f"Title: {p.get('title', 'Unknown')}")

        # Use preview_text for content-based scoring (preferred over summary)
        preview = p.get("preview_text") or p.get("summary") or ""
        if preview:
            lines.append(f"Preview: {preview[:800]}")

        # Include URL for context (Confluence vs MediaWiki structure hints)
        url = p.get("url")
        if url:
            lines.append(f"URL: {url}")

    lines.append(
        "\n\nReturn results for each page in order. "
        "The response format is enforced by the schema."
    )

    user_prompt = "\n".join(lines)

    # Shared retry+parse loop handles both API errors and JSON/validation
    # errors from truncated responses. Cost accumulated across retries.
    # Model-aware token limits + timeout applied automatically.
    batch, total_cost, _tokens = await acall_llm_structured(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_model=WikiScoreBatch,
        temperature=0.3,
    )

    llm_results = batch.results

    # Convert to result dicts, computing combined scores
    cost_per_page = total_cost / len(pages) if pages else 0.0
    results = []

    for r in llm_results[: len(pages)]:
        # Build per-dimension scores dict
        scores = {
            "score_data_documentation": r.score_data_documentation,
            "score_physics_content": r.score_physics_content,
            "score_code_documentation": r.score_code_documentation,
            "score_data_access": r.score_data_access,
            "score_calibration": r.score_calibration,
            "score_imas_relevance": r.score_imas_relevance,
        }

        # Compute combined score using grounded function
        combined_score = grounded_wiki_score(scores, r.page_purpose)

        results.append(
            {
                "id": r.id,
                "score": combined_score,
                "purpose": r.page_purpose.value,
                "description": r.description[:150],
                "reasoning": r.reasoning[:80],
                "keywords": r.keywords[:5],
                "physics_domain": r.physics_domain.value if r.physics_domain else None,
                "should_ingest": r.should_ingest,
                "skip_reason": r.skip_reason or None,
                # Per-dimension scores
                "score_data_documentation": r.score_data_documentation,
                "score_physics_content": r.score_physics_content,
                "score_code_documentation": r.score_code_documentation,
                "score_data_access": r.score_data_access,
                "score_calibration": r.score_calibration,
                "score_imas_relevance": r.score_imas_relevance,
                # Legacy fields for compatibility
                "page_type": r.page_purpose.value,
                "is_physics": r.physics_domain is not None
                and r.physics_domain.value != "general",
                "score_cost": cost_per_page,
            }
        )

    return results, total_cost


def _score_pages_heuristic(
    pages: list[dict[str, Any]],
    data_access_patterns: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Heuristic fallback scoring based on keywords. Zero cost.

    When data_access_patterns is provided, facility-specific key_tools
    and code_import_patterns are added to the physics keywords for
    boosted matching.
    """
    # Build facility-specific keywords from data_access_patterns
    facility_keywords: list[str] = []
    if data_access_patterns:
        for tool in data_access_patterns.get("key_tools") or []:
            # Normalize: lowercase, strip parens/dots for matching
            kw = tool.lower().rstrip("(").split(".")[-1]
            if kw and kw not in facility_keywords:
                facility_keywords.append(kw)
        for pattern in data_access_patterns.get("code_import_patterns") or []:
            # Extract the key identifier from import patterns
            # e.g., "import ppf" -> "ppf", "ppfget(" -> "ppfget"
            kw = pattern.lower().rstrip("(").split()[-1].split(".")[-1]
            if kw and len(kw) > 2 and kw not in facility_keywords:
                facility_keywords.append(kw)

    results = []
    for page in pages:
        title = page.get("title", "").lower()
        summary = page.get("summary", "") or ""

        score = 0.5
        reasoning = "Default score"

        physics_keywords = [
            "thomson",
            "liuqe",
            "equilibrium",
            "mhd",
            "plasma",
            "diagnostic",
            "calibration",
            "signal",
            "node",
        ]
        low_value_keywords = [
            "meeting",
            "workshop",
            "todo",
            "draft",
            "notes",
            "personal",
            "test",
            "sandbox",
        ]

        for kw in physics_keywords:
            if kw in title or kw in summary.lower():
                score = min(score + 0.15, 1.0)
                reasoning = f"Contains physics keyword: {kw}"

        for kw in low_value_keywords:
            if kw in title:
                score = max(score - 0.2, 0.0)
                reasoning = f"Contains low-value keyword: {kw}"

        # Boost for facility-specific data access patterns
        for kw in facility_keywords:
            if kw in title or kw in summary.lower():
                score = min(score + 0.15, 1.0)
                reasoning = f"Contains facility data access keyword: {kw}"

        results.append(
            {
                "id": page["id"],
                "score": score,
                "reasoning": reasoning,
                "page_type": "documentation",
                "is_physics": score >= 0.6,
            }
        )

    return results
