"""
GDELT 2.0 News adapter.
Fetches news articles via the GDELT DOC API (open, no key required).
Stores only metadata: title, snippet, source, publishedAt, url, query.
"""

import logging
from datetime import datetime
from typing import Any
from urllib.parse import urlencode

import requests

logger = logging.getLogger(__name__)

GDELT_DOC_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"


class GdeltNewsError(Exception):
    pass


def fetch_articles(
    query: str,
    max_records: int = 20,
    timespan: str = "1440",  # minutes (24h = 1440)
    mode: str = "ArtList",
    source_lang: str = "english",
) -> list[dict[str, Any]]:
    """
    Fetch news articles from GDELT 2.0 DOC API.

    Args:
        query: Search query string
        max_records: Max articles (up to 250)
        timespan: Lookback in minutes (default 24h)
        mode: ArtList (article list) or TimelineVol etc.
        source_lang: Language filter

    Returns:
        List of article dicts with keys:
          title, snippet, url, published_at, source_name, query
    """
    params = {
        "query": query,
        "mode": mode,
        "maxrecords": min(max_records, 250),
        "timespan": timespan,
        "format": "json",
        "sourcelang": source_lang,
        "sort": "DateDesc",
    }

    url = f"{GDELT_DOC_ENDPOINT}?{urlencode(params)}"

    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if resp.status_code == 429:
            logger.warning("GDELT rate limit hit. Backing off.")
            return []
        raise GdeltNewsError(f"GDELT API HTTP error: {resp.status_code}") from e
    except requests.exceptions.RequestException as e:
        raise GdeltNewsError(f"GDELT API request failed: {e}") from e

    # GDELT returns JSON with "articles" key
    try:
        data = resp.json()
    except (ValueError, TypeError):
        logger.warning(f"GDELT returned non-JSON response for query '{query}'")
        return []

    raw_articles = data.get("articles", [])
    if not raw_articles:
        logger.info(f"GDELT: no articles for query '{query}'")
        return []

    articles = []
    for item in raw_articles:
        # Parse GDELT date format (YYYYMMDDTHHMMSSZ or similar)
        raw_date = item.get("seendate", "")
        published = None
        if raw_date:
            try:
                # GDELT format: "20260518T143000Z"
                published = datetime.strptime(
                    raw_date[:15], "%Y%m%dT%H%M%S"
                ).isoformat()
            except (ValueError, TypeError):
                published = raw_date

        articles.append(
            {
                "title": item.get("title", "").strip(),
                "snippet": item.get("title", "").strip(),  # GDELT doesn't always have snippets
                "url": item.get("url", "").strip(),
                "published_at": published,
                "source_name": item.get("domain", item.get("sourcecountry", "")),
                "query": query,
            }
        )

    logger.info(f"GDELT: fetched {len(articles)} articles for query '{query}'")
    return articles


def is_available() -> bool:
    """GDELT is open - always available (no key needed)."""
    return True
