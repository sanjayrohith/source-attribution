"""
Dynamic headlines module.

Fetches one live news headline per category from GNews API.
Falls back to curated static headlines when no API key is configured.
"""

import os
import asyncio
import logging
from datetime import datetime, timezone
from urllib.parse import quote_plus

import requests

logger = logging.getLogger(__name__)

_TIMEOUT = 8

# Categories to fetch — one headline each
CATEGORIES = ["politics", "technology", "business", "entertainment", "world"]

# Display labels for the frontend
CATEGORY_LABELS = {
    "politics": "POLITICS",
    "technology": "TECH",
    "business": "BUSINESS",
    "entertainment": "ENTERTAINMENT",
    "world": "WORLD",
}

# Static fallback when GNews is unavailable
_FALLBACK_HEADLINES = [
    {
        "headline": "Global Leaders Discuss New Trade Agreements",
        "category": "POLITICS",
        "url": "",
        "source": "World Affairs Daily",
        "published_at": "",
    },
    {
        "headline": "AI Breakthrough Enables Faster Drug Discovery",
        "category": "TECH",
        "url": "",
        "source": "Tech Review",
        "published_at": "",
    },
    {
        "headline": "Stock Markets Rally on Strong Earnings Reports",
        "category": "BUSINESS",
        "url": "",
        "source": "Finance Today",
        "published_at": "",
    },
    {
        "headline": "Award-Winning Film Director Announces New Project",
        "category": "ENTERTAINMENT",
        "url": "",
        "source": "Entertainment Wire",
        "published_at": "",
    },
    {
        "headline": "UN Launches Initiative for Climate Resilience",
        "category": "WORLD",
        "url": "",
        "source": "Global News",
        "published_at": "",
    },
]


def _time_ago(published_at: str) -> str:
    """Convert ISO timestamp to a human-friendly 'X hours ago' string."""
    if not published_at:
        return ""
    try:
        pub = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        diff = now - pub
        minutes = int(diff.total_seconds() / 60)
        if minutes < 60:
            return f"{minutes} min ago"
        hours = minutes // 60
        if hours < 24:
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        days = hours // 24
        return f"{days} day{'s' if days != 1 else ''} ago"
    except Exception:
        return ""


def _fetch_gnews_category(category: str, api_key: str) -> dict | None:
    """Fetch one headline for a single category from GNews."""
    try:
        url = (
            f"https://gnews.io/api/v4/top-headlines"
            f"?category={quote_plus(category)}"
            f"&lang=en"
            f"&max=1"
            f"&apikey={api_key}"
        )
        resp = requests.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        articles = data.get("articles", [])
        if not articles:
            return None

        article = articles[0]
        return {
            "headline": article.get("title", ""),
            "category": CATEGORY_LABELS.get(category, category.upper()),
            "url": article.get("url", ""),
            "source": article.get("source", {}).get("name", ""),
            "published_at": article.get("publishedAt", ""),
            "time_ago": _time_ago(article.get("publishedAt", "")),
            "image": article.get("image", ""),
        }
    except Exception as e:
        logger.warning(f"GNews headline fetch failed for {category}: {e}")
        return None


async def fetch_headlines() -> list[dict]:
    """
    Fetch one headline per category. Uses GNews if API key is set,
    otherwise returns static fallback headlines.
    """
    api_key = os.getenv("GNEWS_API_KEY", "").strip()

    if not api_key:
        logger.info("Headlines: No GNews API key, using fallback")
        return _FALLBACK_HEADLINES

    def _fetch_all():
        results = []
        for cat in CATEGORIES:
            item = _fetch_gnews_category(cat, api_key)
            if item:
                results.append(item)
            else:
                # Use fallback for this category
                fb = next(
                    (h for h in _FALLBACK_HEADLINES
                     if h["category"] == CATEGORY_LABELS.get(cat, "")),
                    None,
                )
                if fb:
                    results.append(fb)
        return results

    return await asyncio.to_thread(_fetch_all)
