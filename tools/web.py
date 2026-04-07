"""Web fetching tool."""

from __future__ import annotations

import httpx
import trafilatura

from agent_core.tools import tool


_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)
_TEXT_CONTENT_TYPES = ("text/plain", "text/markdown")
_HTML_CONTENT_TYPES = ("text/html", "application/xhtml+xml")


@tool("Fetch the content of a web page from a URL and return readable text.", name="WebFetch")
async def web_fetch(url: str) -> str:
    timeout = httpx.Timeout(180.0, connect=15.0)
    headers = {"User-Agent": _USER_AGENT}

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, headers=headers, follow_redirects=True)
    except httpx.TimeoutException:
        return "Error: Failed to fetch URL: request timed out."
    except httpx.RequestError as exc:
        return f"Error: Failed to fetch URL due to network error: {exc}"

    if response.status_code >= 400:
        return f"Error: Failed to fetch URL. Status: {response.status_code}."

    response_text = response.text
    if not response_text:
        return "Error: Empty response body."

    content_type = response.headers.get("content-type", "").lower()
    if content_type.startswith(_TEXT_CONTENT_TYPES):
        return response_text
    if content_type.startswith(_HTML_CONTENT_TYPES):
        extracted_text = trafilatura.extract(
            response_text,
            include_comments=True,
            include_tables=True,
            include_formatting=False,
            output_format="txt",
            with_metadata=True,
        )
        if extracted_text:
            return extracted_text
        return "Error: Failed to extract meaningful content from the page."
    return response_text
