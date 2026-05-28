"""
News Ingestion Scheduler — polls configured news sources on a timer,
deduplicates, tags articles, and persists to DB.
"""

import logging
import threading
import time
from pathlib import Path
from typing import Any

import yaml

from . import db, news_bing, news_gdelt
from .bloomberg import compute_market_confirmation
from .tagger import tag_article
from .taxonomy import get_taxonomy

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class NewsScheduler:
    """
    Periodically polls news sources, deduplicates, tags, and stores articles.
    """

    def __init__(self, config: dict | None = None, db_path: str | Path | None = None):
        self._config = config or load_config()
        self._db_path = db_path
        self._running = False
        self._thread: threading.Thread | None = None
        self._poll_interval = self._config.get("ingestion", {}).get("poll_interval_minutes", 10)
        self._queries = self._config.get("queries", [])
        self._sources_config = self._config.get("sources", {})
        self._taxonomy = get_taxonomy()

    @property
    def enabled(self) -> bool:
        return self._config.get("ingestion", {}).get("enabled", True)

    def start(self):
        """Start the scheduler in a background thread."""
        if not self.enabled:
            logger.info("News ingestion is disabled in config.")
            return

        db.init_db(self._db_path)
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info(
            f"News scheduler started — polling every {self._poll_interval} min, "
            f"{len(self._queries)} queries"
        )

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("News scheduler stopped.")

    def poll_once(self):
        """Run a single poll cycle (useful for testing or manual trigger)."""
        db.init_db(self._db_path)
        self._do_poll()

    def _run_loop(self):
        while self._running:
            try:
                self._do_poll()
            except Exception as e:
                logger.error(f"Poll cycle error: {e}", exc_info=True)
                db.audit("poll_error", {"error": str(e)}, db_path=self._db_path)

            # Sleep in small increments so we can stop quickly
            for _ in range(self._poll_interval * 60):
                if not self._running:
                    return
                time.sleep(1)

    def _do_poll(self):
        """Execute one poll cycle across all sources and queries."""
        total_new = 0
        total_dupes = 0

        for i, query in enumerate(self._queries):
            if not self._running and i > 0:
                break
            # Rate-limit: GDELT allows ~1 req/5s; pause between queries
            if i > 0:
                time.sleep(5)
            articles = self._fetch_all_sources(query)
            for article in articles:
                result = self._process_article(article)
                if result == "new":
                    total_new += 1
                else:
                    total_dupes += 1

        db.audit(
            "poll_complete",
            {"new_articles": total_new, "duplicates": total_dupes, "queries": len(self._queries)},
            db_path=self._db_path,
        )
        logger.info(f"Poll complete: {total_new} new, {total_dupes} dupes")

    def _fetch_all_sources(self, query: str) -> list[dict]:
        """Fetch articles from all enabled sources for a query."""
        articles: list[dict] = []
        max_per_query = self._config.get("ingestion", {}).get("max_articles_per_query", 20)

        # Bing
        bing_cfg = self._sources_config.get("bing", {})
        if bing_cfg.get("enabled", False) and news_bing.is_available():
            try:
                bing_articles = news_bing.fetch_articles(query, count=max_per_query)
                for a in bing_articles:
                    a["source"] = "bing"
                    a["reliability"] = bing_cfg.get("reliability_weight", 0.8)
                articles.extend(bing_articles)
            except Exception as e:
                logger.warning(f"Bing fetch failed for '{query}': {e}")

        # GDELT
        gdelt_cfg = self._sources_config.get("gdelt", {})
        if gdelt_cfg.get("enabled", False) and news_gdelt.is_available():
            try:
                gdelt_articles = news_gdelt.fetch_articles(query, max_records=max_per_query)
                for a in gdelt_articles:
                    a["source"] = "gdelt"
                    a["reliability"] = gdelt_cfg.get("reliability_weight", 0.6)
                articles.extend(gdelt_articles)
            except Exception as e:
                logger.warning(f"GDELT fetch failed for '{query}': {e}")

        return articles

    def _process_article(self, article: dict) -> str:
        """
        Process a single article: deduplicate, tag, and store.
        Returns 'new' or 'duplicate'.
        """
        url = article.get("url", "")
        if not url:
            return "duplicate"

        # Check for duplicate
        if db.article_exists(url, db_path=self._db_path):
            return "duplicate"

        # Insert article
        article_id = db.insert_article(
            source=article.get("source", "unknown"),
            title=article.get("title", ""),
            snippet=article.get("snippet"),
            url=url,
            published_at=article.get("published_at"),
            query=article.get("query"),
            db_path=self._db_path,
        )

        if not article_id:
            return "duplicate"

        # Tag the article
        source_reliability = article.get("reliability", 0.7)
        tags = tag_article(
            title=article.get("title", ""),
            snippet=article.get("snippet"),
            source_reliability=source_reliability,
            taxonomy_instance=self._taxonomy,
        )

        if tags:
            # Try market confirmation (will return no_data if Bloomberg offline)
            tags = compute_market_confirmation(tags)
            db.insert_tags(article_id, tags, db_path=self._db_path)

        return "new"
