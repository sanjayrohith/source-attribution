"""
Multi-API web scraping verification module.

Uses multiple search providers (GNews, Google Fact Check, DuckDuckGo)
to cross-reference news claims and produce a REAL / FAKE / UNVERIFIED verdict.
"""

import os
import re
import asyncio
import logging
from urllib.parse import urlparse, quote_plus

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}
_TIMEOUT = 8

# Reputable news domains get higher credibility weight
_REPUTABLE_DOMAINS = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
    "nytimes.com", "washingtonpost.com", "theguardian.com",
    "cnn.com", "aljazeera.com", "npr.org", "pbs.org",
    "abcnews.go.com", "cbsnews.com", "nbcnews.com",
    "usatoday.com", "bloomberg.com", "economist.com",
    "forbes.com", "time.com", "thehindu.com", "ndtv.com",
    "hindustantimes.com", "indianexpress.com",
    "snopes.com", "factcheck.org", "politifact.com",
    "fullfact.org", "boomlive.in", "altnews.in",
}


# ---------------------------------------------------------------------------
# Query Builder
# ---------------------------------------------------------------------------

def _build_query(text: str) -> str:
    """Extract a concise, search-friendly query from news text."""
    text = text.strip()
    # Try first sentence
    sentence_match = re.match(r"[^.!?]+[.!?]", text)
    if sentence_match:
        query = sentence_match.group(0).strip().rstrip(".!?")
    else:
        words = text.split()
        query = " ".join(words[:20])

    # Clean up for search
    query = re.sub(r"[\"']", "", query)
    query = re.sub(r"\s+", " ", query)
    return query.strip()


# ---------------------------------------------------------------------------
# Provider 1: GNews API
# ---------------------------------------------------------------------------

async def _search_gnews(query: str) -> list[dict]:
    """Search GNews API for news articles. Requires GNEWS_API_KEY."""
    api_key = os.getenv("GNEWS_API_KEY", "").strip()
    if not api_key:
        logger.info("GNews: No API key configured, skipping")
        return []

    def _fetch():
        try:
            url = (
                f"https://gnews.io/api/v4/search"
                f"?q={quote_plus(query)}"
                f"&lang=en"
                f"&max=10"
                f"&apikey={api_key}"
            )
            resp = requests.get(url, timeout=_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()

            results = []
            for article in data.get("articles", []):
                domain = urlparse(article.get("url", "")).netloc
                results.append({
                    "title": article.get("title", ""),
                    "url": article.get("url", ""),
                    "snippet": article.get("description", ""),
                    "domain": domain,
                    "provider": "gnews",
                    "published_at": article.get("publishedAt", ""),
                    "source_name": article.get("source", {}).get("name", ""),
                })
            return results
        except Exception as e:
            logger.warning(f"GNews search failed: {e}")
            return []

    return await asyncio.to_thread(_fetch)


# ---------------------------------------------------------------------------
# Provider 2: Google Fact Check API
# ---------------------------------------------------------------------------

async def _search_factcheck(query: str) -> list[dict]:
    """Search Google Fact Check Tools API. Requires GOOGLE_FACTCHECK_API_KEY."""
    api_key = os.getenv("GOOGLE_FACTCHECK_API_KEY", "").strip()
    if not api_key:
        logger.info("Google Fact Check: No API key configured, skipping")
        return []

    def _fetch():
        try:
            url = (
                f"https://factchecktools.googleapis.com/v1alpha1/claims:search"
                f"?query={quote_plus(query)}"
                f"&key={api_key}"
                f"&languageCode=en"
                f"&pageSize=10"
            )
            resp = requests.get(url, timeout=_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()

            results = []
            for claim in data.get("claims", []):
                for review in claim.get("claimReview", []):
                    domain = urlparse(review.get("url", "")).netloc
                    results.append({
                        "title": review.get("title", claim.get("text", "")),
                        "url": review.get("url", ""),
                        "snippet": claim.get("text", ""),
                        "domain": domain,
                        "provider": "factcheck",
                        "claim_text": claim.get("text", ""),
                        "claimant": claim.get("claimant", ""),
                        "rating": review.get("textualRating", ""),
                        "publisher": review.get("publisher", {}).get("name", ""),
                    })
            return results
        except Exception as e:
            logger.warning(f"Google Fact Check search failed: {e}")
            return []

    return await asyncio.to_thread(_fetch)


# ---------------------------------------------------------------------------
# Provider 3: DuckDuckGo (fallback, no key)
# ---------------------------------------------------------------------------

async def _search_duckduckgo(query: str) -> list[dict]:
    """Search DuckDuckGo. No API key needed but can be rate-limited."""
    def _fetch():
        try:
            with DDGS() as ddgs:
                raw = list(ddgs.text(query, max_results=8))
            results = []
            for item in raw:
                url = item.get("href", "")
                domain = urlparse(url).netloc if url else ""
                results.append({
                    "title": item.get("title", ""),
                    "url": url,
                    "snippet": item.get("body", ""),
                    "domain": domain,
                    "provider": "duckduckgo",
                })
            return results
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")
            return []

    return await asyncio.to_thread(_fetch)


# ---------------------------------------------------------------------------
# Cross-Reference Engine
# ---------------------------------------------------------------------------

def _deduplicate(sources: list[dict]) -> list[dict]:
    """Remove duplicate URLs, keeping the first occurrence."""
    seen_urls = set()
    unique = []
    for s in sources:
        url = s.get("url", "").rstrip("/")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique.append(s)
    return unique


def _compute_verdict(
    all_sources: list[dict],
    fact_checks: list[dict],
    providers_used: list[str],
) -> dict:
    """
    Cross-reference all sources and fact-checks to produce a verdict.

    Returns {verdict, confidence, explanation}.
    """
    # --- Phase 1: Check existing fact-checks (strongest signal) ---
    if fact_checks:
        fake_ratings = []
        real_ratings = []
        for fc in fact_checks:
            rating = fc.get("rating", "").lower()
            # Common fact-check rating keywords
            if any(w in rating for w in [
                "false", "fake", "pants on fire", "incorrect",
                "misleading", "mostly false", "not true", "hoax",
                "fabricated", "satire", "scam",
            ]):
                fake_ratings.append(fc)
            elif any(w in rating for w in [
                "true", "correct", "accurate", "mostly true",
                "verified", "confirmed", "real",
            ]):
                real_ratings.append(fc)

        if fake_ratings and not real_ratings:
            confidence = min(0.95, 0.70 + 0.05 * len(fake_ratings))
            publishers = ", ".join(set(fc.get("publisher", "?") for fc in fake_ratings[:3]))
            return {
                "verdict": "FAKE",
                "confidence": round(confidence, 2),
                "explanation": (
                    f"This claim has been fact-checked and rated FALSE by {len(fake_ratings)} "
                    f"fact-checker(s) including: {publishers}."
                ),
            }

        if real_ratings and not fake_ratings:
            confidence = min(0.95, 0.70 + 0.05 * len(real_ratings))
            publishers = ", ".join(set(fc.get("publisher", "?") for fc in real_ratings[:3]))
            return {
                "verdict": "REAL",
                "confidence": round(confidence, 2),
                "explanation": (
                    f"This claim has been fact-checked and rated TRUE by {len(real_ratings)} "
                    f"fact-checker(s) including: {publishers}."
                ),
            }

        if fake_ratings and real_ratings:
            return {
                "verdict": "FAKE" if len(fake_ratings) > len(real_ratings) else "REAL",
                "confidence": 0.55,
                "explanation": (
                    f"Mixed fact-check results: {len(fake_ratings)} rated it false, "
                    f"{len(real_ratings)} rated it true. Leaning towards the majority."
                ),
            }

    # --- Phase 2: Analyze news sources (weaker signal) ---
    non_fc_sources = [s for s in all_sources if s.get("provider") != "factcheck"]

    if not non_fc_sources:
        return {
            "verdict": "UNVERIFIED",
            "confidence": 0.0,
            "explanation": "No web sources found discussing this claim. Unable to verify.",
        }

    reputable_count = sum(
        1 for s in non_fc_sources
        if any(rd in s.get("domain", "") for rd in _REPUTABLE_DOMAINS)
    )
    total = len(non_fc_sources)

    if reputable_count >= 3:
        confidence = min(0.85, 0.50 + 0.07 * reputable_count)
        return {
            "verdict": "REAL",
            "confidence": round(confidence, 2),
            "explanation": (
                f"Found {total} source(s) discussing this claim, including "
                f"{reputable_count} reputable news outlet(s). "
                f"The claim appears to be widely reported by credible sources."
            ),
        }
    elif reputable_count >= 1:
        confidence = min(0.65, 0.35 + 0.10 * reputable_count)
        return {
            "verdict": "REAL",
            "confidence": round(confidence, 2),
            "explanation": (
                f"Found {total} source(s) discussing this claim, including "
                f"{reputable_count} reputable outlet(s). Some credible coverage exists."
            ),
        }
    elif total >= 3:
        return {
            "verdict": "REAL",
            "confidence": 0.40,
            "explanation": (
                f"Found {total} source(s) discussing this claim but none from "
                f"major reputable outlets. The claim has some web presence but "
                f"limited credible coverage."
            ),
        }
    elif total >= 1:
        return {
            "verdict": "UNVERIFIED",
            "confidence": 0.25,
            "explanation": (
                f"Found only {total} source(s) with no reputable outlets. "
                f"Insufficient evidence to verify this claim."
            ),
        }
    else:
        return {
            "verdict": "UNVERIFIED",
            "confidence": 0.0,
            "explanation": "No web sources found discussing this claim.",
        }


# ---------------------------------------------------------------------------
# Main Public Function
# ---------------------------------------------------------------------------

async def scrape_verify(news_text: str) -> dict:
    """
    Search multiple APIs, cross-reference results, and produce a verdict.
    """
    query = _build_query(news_text)

    # Run all providers in parallel
    gnews_task = _search_gnews(query)
    factcheck_task = _search_factcheck(query)
    ddg_task = _search_duckduckgo(query)

    gnews_results, factcheck_results, ddg_results = await asyncio.gather(
        gnews_task, factcheck_task, ddg_task
    )

    # Track which providers returned results
    providers_used = []
    if gnews_results:
        providers_used.append("gnews")
    if factcheck_results:
        providers_used.append("factcheck")
    if ddg_results:
        providers_used.append("duckduckgo")

    # Also note which were configured but returned nothing
    if os.getenv("GNEWS_API_KEY", "").strip() and not gnews_results:
        providers_used.append("gnews")  # attempted
    if os.getenv("GOOGLE_FACTCHECK_API_KEY", "").strip() and not factcheck_results:
        providers_used.append("factcheck")  # attempted
    providers_used = list(dict.fromkeys(providers_used))  # dedupe, preserve order

    # Merge and deduplicate
    all_sources = _deduplicate(gnews_results + ddg_results)

    # Compute verdict
    verdict_data = _compute_verdict(all_sources, factcheck_results, providers_used)

    return {
        "query_used": query,
        "verdict": verdict_data["verdict"],
        "confidence": verdict_data["confidence"],
        "explanation": verdict_data["explanation"],
        "providers_used": providers_used,
        "fact_checks": factcheck_results,
        "sources": all_sources,
        "sources_found": len(all_sources),
    }
