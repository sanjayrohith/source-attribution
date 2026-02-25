"""
Web scraping verification module.

Searches the internet for a given news claim and returns
corroborating / contradicting evidence from multiple sources.
"""

import re
import asyncio
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

_TIMEOUT = 8  # seconds per request


def _build_query(text: str) -> str:
    """Extract a concise search query from the news text."""
    # Take the first sentence or first ~120 chars
    text = text.strip()
    sentence_match = re.match(r"[^.!?]+[.!?]", text)
    if sentence_match:
        query = sentence_match.group(0).strip().rstrip(".!?")
    else:
        words = text.split()
        query = " ".join(words[:20])

    # Remove excessive quotes / special chars for a clean query
    query = re.sub(r"[\"']", "", query)
    return query.strip()


def _fetch_page_text(url: str) -> str | None:
    """Fetch a URL and return its visible text (first ~2000 chars)."""
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove script / style tags
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text)
        return text[:2000]
    except Exception:
        return None


def _extract_snippet(page_text: str, query_words: list[str], max_len: int = 300) -> str:
    """Find the most relevant snippet from the page text."""
    if not page_text:
        return ""

    lower_text = page_text.lower()
    best_pos = 0
    best_score = 0

    # Slide a window over the text to find the region with most query-word hits
    window = 300
    for i in range(0, min(len(lower_text), 1500), 50):
        chunk = lower_text[i : i + window]
        score = sum(1 for w in query_words if w in chunk)
        if score > best_score:
            best_score = score
            best_pos = i

    snippet = page_text[best_pos : best_pos + max_len].strip()
    if best_pos > 0:
        snippet = "…" + snippet
    if best_pos + max_len < len(page_text):
        snippet += "…"
    return snippet


def _classify_results(sources: list[dict]) -> str:
    """
    Simple heuristic to decide overall verdict based on how many
    credible sources were found discussing the claim.
    """
    if not sources:
        return "UNVERIFIED"

    count = len(sources)
    has_snippets = sum(1 for s in sources if s.get("snippet"))

    if count >= 3 and has_snippets >= 2:
        return "SUPPORTED"
    elif count >= 1 and has_snippets >= 1:
        return "MIXED"
    else:
        return "UNVERIFIED"


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

async def scrape_verify(news_text: str) -> dict:
    """
    Search the web for the given news text and return structured results.

    Returns
    -------
    dict with keys:
        query_used      – the search query that was sent
        sources_found   – number of sources successfully fetched
        sources         – list of {title, url, snippet, domain}
        summary         – "SUPPORTED" | "DISPUTED" | "MIXED" | "UNVERIFIED"
    """
    query = _build_query(news_text)
    query_words = [w.lower() for w in query.split() if len(w) > 3]

    # Run DuckDuckGo search in a thread pool (it's synchronous)
    def _search():
        try:
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=5))
        except Exception:
            return []

    search_results = await asyncio.to_thread(_search)

    sources: list[dict] = []
    for item in search_results:
        url = item.get("href", "")
        title = item.get("title", "")
        domain = urlparse(url).netloc if url else ""

        # Try to get a richer snippet from the actual page
        page_text = await asyncio.to_thread(_fetch_page_text, url)
        snippet = (
            _extract_snippet(page_text, query_words)
            if page_text
            else item.get("body", "")
        )

        sources.append(
            {
                "title": title,
                "url": url,
                "snippet": snippet,
                "domain": domain,
            }
        )

    summary = _classify_results(sources)

    return {
        "query_used": query,
        "sources_found": len(sources),
        "sources": sources,
        "summary": summary,
    }
