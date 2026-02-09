"""Wiki page content utilities.

Provides HTTP fetching and HTML text extraction used by the scoring pipeline.
"""

import logging
from collections.abc import Callable

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


async def fetch_page_content(
    url: str,
    timeout: float = 30.0,
    auth_handler: Callable | None = None,
    verify_ssl: bool = False,
) -> tuple[str | None, str | None]:
    """Fetch page content via HTTP.

    Args:
        url: Page URL to fetch
        timeout: Request timeout in seconds
        auth_handler: Optional authentication handler
        verify_ssl: Whether to verify SSL certificates (default False for facility wikis)

    Returns:
        (content_text, error_message)
        If successful: (text, None)
        If failed: (None, error_message)
    """
    try:
        async with httpx.AsyncClient(
            timeout=timeout, follow_redirects=True, verify=verify_ssl
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.text, None
    except httpx.TimeoutException:
        return None, "Timeout"
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            return None, "Auth required"
        elif e.response.status_code == 404:
            return None, "Not found"
        else:
            return None, f"HTTP {e.response.status_code}"
    except Exception as e:
        return None, str(e)


def extract_text_from_html(html: str, max_chars: int = 2000) -> str:
    """Extract clean text from HTML content.

    - Removes scripts, styles, navigation
    - Preserves paragraph structure
    - Truncates to max_chars

    Args:
        html: Raw HTML content
        max_chars: Maximum characters to extract

    Returns:
        Clean text preview
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove unwanted elements
    for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
        element.decompose()

    # Get text
    text = soup.get_text(separator=" ", strip=True)

    # Clean up whitespace
    text = " ".join(text.split())

    # Truncate
    if len(text) > max_chars:
        text = text[:max_chars]

    return text
