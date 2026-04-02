"""Web fetching tool."""

from __future__ import annotations

import html
from html.parser import HTMLParser
import re

import httpx

from agent_core.tools import tool


_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)
_TEXT_CONTENT_TYPES = ("text/plain", "text/markdown")
_HTML_CONTENT_TYPES = ("text/html", "application/xhtml+xml")


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized_tag = tag.lower()
        if normalized_tag in {"script", "style", "noscript"}:
            self._ignored_depth += 1
            return
        if normalized_tag in {"p", "div", "section", "article", "main", "br", "li", "h1", "h2", "h3"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        normalized_tag = tag.lower()
        if normalized_tag in {"script", "style", "noscript"} and self._ignored_depth > 0:
            self._ignored_depth -= 1
            return
        if normalized_tag in {
            "p",
            "div",
            "section",
            "article",
            "main",
            "li",
            "ul",
            "ol",
            "h1",
            "h2",
            "h3",
        }:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth > 0:
            return
        if data.strip():
            self._parts.append(data)

    def get_text(self) -> str:
        normalized_lines = [
            re.sub(r"\s+", " ", html.unescape(line)).strip()
            for line in "".join(self._parts).splitlines()
        ]
        return "\n".join(line for line in normalized_lines if line)


def _extract_html_text(content: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(content)
    parser.close()
    return parser.get_text()


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
        extracted_text = _extract_html_text(response_text)
        if extracted_text:
            return extracted_text
        return "Error: Failed to extract meaningful content from the page."
    return response_text
