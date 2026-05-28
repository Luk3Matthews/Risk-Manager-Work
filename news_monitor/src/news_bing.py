"""
Bing News Search API adapter.
Fetches news articles via the Bing News Search v7 API.
Stores only metadata: title, snippet, provider, publishedAt, url, query.
"""

import logging
import os
from datetime import datetime
from typing import Any

import requests

logger = logging.getLogger(__name__)

BING_ENDPOINT = "https://api.bing.microsoft.com/v7.0/news/search"
ENV_KEY = "BING_NEWS_KEY"


class BingNewsError(Exception):
    pass


def _get_api_key() -> str:
    key = os.environ.get(ENV_KEY)
    if not key:
        raise BingNewsError(
            f"Bing News API key not set. Set environment variable '{ENV_KEY}'."
        )
    return key


def fetch_articles(
    query: str,
    count: int = 20,
    freshness: str = "Day",
    market: str = "en-AU",
    sort_by: str = "Date",
) -> list[dict[str, Any]]:
    """
    Fetch news articles from Bing News Search API.

    Args:
        query: Search query string
        count: Max articles to return (1-100)
        freshness: Day, Week, or Month
        market: Market code (e.g. en-AU, en-US)
        sort_by: Date or Relevance

    Returns:
        List of article dicts with keys:
          title, snippet, url, published_at, source_name, query
    """
    api_key = _get_api_key()

    headers = {"Ocp-Apim-Subscription-Key": api_key}
    params = {
        "q": query,
        "count": min(count, 100),
        "freshness": freshness,
        "mkt": market,
        "sortBy": sort_by,
        "textFormat": "Raw",
    }

    try:
        resp = requests.get(
            BING_ENDPOINT, headers=headers, params=params, timeout=15
        )
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if resp.status_code == 401:
            raise BingNewsError("Invalid Bing API key (401 Unauthorized).") from e
        if resp.status_code == 429:
            logger.warning("Bing API rate limit hit. Backing off.")
            return []
        raise BingNewsError(f"Bing API HTTP error: {resp.status_code}") from e
    except requests.exceptions.RequestException as e:
        raise BingNewsError(f"Bing API request failed: {e}") from e

    data = resp.json()
    articles = []

    for item in data.get("value", []):
        published = item.get("datePublished")
        if published:
            try:
                published = datetime.fromisoformat(
                    published.replace("Z", "+00:00")
                ).isoformat()
            except (ValueError, TypeError):
                pass

        provider_name = ""
        providers = item.get("provider", [])
        if providers:
            provider_name = providers[0].get("name", "")

        articles.append(
            {
                "title": item.get("name", "").strip(),
                "snippet": item.get("description", "").strip(),
                "url": item.get("url", "").strip(),
                "published_at": published,
                "source_name": provider_name,
                "query": query,
            }
        )

    logger.info(f"Bing: fetched {len(articles)} articles for query '{query}'")
    return articles


def is_available() -> bool:
    """Check if the Bing API key is configured."""
    return bool(os.environ.get(ENV_KEY))
