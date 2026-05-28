"""Theme Engine — News Sifter: Article Processing & Theme Tagging.

Bridges the existing news_monitor pipeline with the Theme Engine by:
  1. Pulling tagged articles from news_monitor's SQLite database
  2. Converting articles → EvidenceItem objects
  3. Matching articles to themes via keyword/factor mapping
  4. Scoring credibility, timeliness, and corroboration
  5. Bulk-processing large volumes of articles efficiently

Also supports standalone operation (direct from RSS, CSV, or API)
without the news_monitor database.
"""
from __future__ import annotations

import hashlib
import re
import sqlite3
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

from .config import EngineConfig, get_config
from .models import (
    EvidenceItem,
    Theme,
    ThemeCategory,
    ThemeStatus,
)
from .ingestion import add_evidence_to_theme, score_evidence


# ---------------------------------------------------------------------------
# Category ↔ keyword mappings for article → theme matching
# ---------------------------------------------------------------------------

THEME_KEYWORDS: dict[ThemeCategory, list[str]] = {
    ThemeCategory.GEOPOLITICAL: [
        "geopolitical", "war", "conflict", "sanctions", "military",
        "territory", "invasion", "tensions", "nato", "nuclear",
        "strait of hormuz", "south china sea", "taiwan", "ukraine",
        "middle east", "iran", "north korea", "coup", "espionage",
        "arms race", "missile", "naval", "embargo", "diplomacy",
    ],
    ThemeCategory.GROWTH: [
        "gdp", "recession", "economic growth", "employment", "unemployment",
        "payrolls", "pmi", "manufacturing", "services", "consumer spending",
        "retail sales", "housing starts", "industrial production",
        "business confidence", "leading indicators", "soft landing",
        "hard landing", "slowdown", "expansion", "recovery",
    ],
    ThemeCategory.INFLATION: [
        "inflation", "cpi", "pce", "deflation", "disinflation",
        "stagflation", "price pressure", "wage growth", "cost of living",
        "producer prices", "breakeven", "inflation expectations",
        "core inflation", "sticky inflation", "transitory",
        "food prices", "energy prices", "rent inflation",
    ],
    ThemeCategory.LIQUIDITY: [
        "liquidity", "fed funds", "repo", "money market", "tightening",
        "quantitative easing", "qe", "qt", "balance sheet", "reserves",
        "ted spread", "libor", "sofr", "funding", "credit crunch",
        "financial conditions", "margin call", "collateral",
    ],
    ThemeCategory.STRUCTURAL: [
        "deglobalization", "reshoring", "nearshoring", "ai revolution",
        "automation", "demographics", "aging population", "productivity",
        "supply chain", "green transition", "energy transition",
        "digital transformation", "fintech", "blockchain",
    ],
    ThemeCategory.POLICY: [
        "fed", "ecb", "boj", "rba", "pboc", "rate hike", "rate cut",
        "monetary policy", "fiscal policy", "stimulus", "deficit",
        "debt ceiling", "tax", "tariff", "trade war", "regulation",
        "central bank", "forward guidance", "dot plot", "pivot",
    ],
    ThemeCategory.VALUATION: [
        "overvalued", "undervalued", "bubble", "valuation", "pe ratio",
        "cape", "earnings yield", "buyback", "ipo", "spac",
        "market cap", "buffett indicator", "price to book",
        "price to sales", "equity premium", "rich", "cheap",
    ],
    ThemeCategory.CONTAGION: [
        "contagion", "systemic risk", "bank run", "credit event",
        "default", "sovereign debt", "banking crisis", "lehman",
        "cascade", "counterparty", "too big to fail", "bailout",
        "stress test", "capital adequacy", "npls",
        "pandemic", "epidemic", "outbreak", "virus", "huntervirus",
        "ebola", "who emergency", "quarantine", "lockdown",
        "zoonotic", "pathogen", "vaccine", "public health emergency",
    ],
}

# Factor names from news_monitor → ThemeCategory mapping
_NEWS_MONITOR_FACTOR_MAP: dict[str, ThemeCategory] = {
    "Inflation": ThemeCategory.INFLATION,
    "Growth": ThemeCategory.GROWTH,
    "Rates": ThemeCategory.POLICY,
    "CreditSpreads": ThemeCategory.LIQUIDITY,
    "FX": ThemeCategory.LIQUIDITY,
    "Liquidity": ThemeCategory.LIQUIDITY,
    "CommoditySupply": ThemeCategory.GEOPOLITICAL,
    "Policy": ThemeCategory.POLICY,
    "Geopolitics": ThemeCategory.GEOPOLITICAL,
}

# news_monitor theme → ThemeCategory mapping
_NEWS_MONITOR_THEME_MAP: dict[str, ThemeCategory] = {
    "Geopolitics": ThemeCategory.GEOPOLITICAL,
    "Energy": ThemeCategory.GEOPOLITICAL,  # Energy is often geopolitical
    "Inflation": ThemeCategory.INFLATION,
    "Rates": ThemeCategory.POLICY,
    "Credit": ThemeCategory.LIQUIDITY,
    "FX": ThemeCategory.LIQUIDITY,
    "Liquidity": ThemeCategory.LIQUIDITY,
    "Growth": ThemeCategory.GROWTH,
    "Policy": ThemeCategory.POLICY,
}


# ---------------------------------------------------------------------------
# Article representation (raw, before conversion to EvidenceItem)
# ---------------------------------------------------------------------------

class RawArticle:
    """Lightweight container for a raw news article."""
    __slots__ = ("title", "snippet", "url", "source", "published_at",
                 "fetched_at", "query", "tags")

    def __init__(
        self,
        title: str,
        snippet: str = "",
        url: str = "",
        source: str = "",
        published_at: str | date | None = None,
        fetched_at: str | date | None = None,
        query: str = "",
        tags: list[dict[str, Any]] | None = None,
    ):
        self.title = title
        self.snippet = snippet
        self.url = url
        self.source = source
        self.published_at = self._to_date(published_at)
        self.fetched_at = self._to_date(fetched_at)
        self.query = query
        self.tags = tags or []

    @staticmethod
    def _to_date(val: str | date | None) -> date:
        if val is None:
            return date.today()
        if isinstance(val, date):
            return val
        try:
            return datetime.fromisoformat(val.replace("Z", "+00:00")).date()
        except (ValueError, AttributeError):
            return date.today()


# ---------------------------------------------------------------------------
# News Monitor DB reader
# ---------------------------------------------------------------------------

class NewsMonitorReader:
    """Read articles and tags from the news_monitor SQLite database."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"News monitor DB not found: {self.db_path}")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def get_recent_articles(
        self,
        hours: int = 72,
        source: str | None = None,
        theme: str | None = None,
        macro_factor: str | None = None,
        limit: int = 500,
    ) -> list[RawArticle]:
        """Fetch recent articles with their tags from the news_monitor DB."""
        conn = self._connect()
        try:
            cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()

            query = """
                SELECT a.id, a.title, a.snippet, a.url, a.source,
                       a.published_at, a.fetched_at, a.query
                FROM articles a
                WHERE a.fetched_at >= ?
            """
            params: list[Any] = [cutoff]

            if source:
                query += " AND a.source = ?"
                params.append(source)

            query += " ORDER BY a.fetched_at DESC LIMIT ?"
            params.append(limit)

            rows = conn.execute(query, params).fetchall()
            articles = []

            # Batch-load all tags for fetched articles (avoid N+1 queries)
            article_ids = [row["id"] for row in rows]
            tags_by_id: dict[int, list[dict]] = {aid: [] for aid in article_ids}
            if article_ids:
                placeholders = ",".join("?" for _ in article_ids)
                tag_rows = conn.execute(
                    f"""SELECT article_id, theme, macro_factor, direction,
                               horizon, confidence
                        FROM article_tags
                        WHERE article_id IN ({placeholders})""",
                    article_ids,
                ).fetchall()
                for tr in tag_rows:
                    tags_by_id[tr["article_id"]].append({
                        "theme": tr["theme"],
                        "macro_factor": tr["macro_factor"],
                        "direction": tr["direction"],
                        "horizon": tr["horizon"],
                        "confidence": tr["confidence"],
                    })

            for row in rows:
                tags = tags_by_id.get(row["id"], [])

                # Filter by theme/factor if specified
                if theme and not any(t["theme"] == theme for t in tags):
                    continue
                if macro_factor and not any(t["macro_factor"] == macro_factor for t in tags):
                    continue

                articles.append(RawArticle(
                    title=row["title"],
                    snippet=row["snippet"] or "",
                    url=row["url"],
                    source=row["source"],
                    published_at=row["published_at"],
                    fetched_at=row["fetched_at"],
                    query=row["query"] or "",
                    tags=tags,
                ))

            return articles
        finally:
            conn.close()

    def get_article_count(self, hours: int = 72) -> int:
        """Get count of articles in the last N hours."""
        conn = self._connect()
        try:
            cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM articles WHERE fetched_at >= ?",
                [cutoff]
            ).fetchone()
            return row["cnt"] if row else 0
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Article → EvidenceItem conversion
# ---------------------------------------------------------------------------

def article_to_evidence(
    article: RawArticle,
    cfg: EngineConfig | None = None,
) -> EvidenceItem:
    """Convert a RawArticle into an EvidenceItem with scored fields.

    Credibility: based on source reliability mapping
    Timeliness: based on article age
    Corroboration: estimated from tag count / query overlap
    """
    cfg = cfg or get_config()
    source_creds = cfg.news_sifter.get("source_credibility", {})

    # Credibility from source
    credibility = source_creds.get(article.source, 0.5)

    # Timeliness: 1.0 if today, decays over 7 days to 0.3
    days_old = (date.today() - article.published_at).days
    timeliness = max(0.3, 1.0 - (days_old / 7.0) * 0.7)

    # Corroboration: count of distinct confirming tags
    corroboration = len(article.tags)

    evidence = EvidenceItem(
        source=article.source,
        date=article.published_at,
        title=article.title,
        url=article.url or None,
        credibility_score=credibility,
        timeliness_score=timeliness,
        corroboration_count=corroboration,
    )
    score_evidence(evidence, cfg)
    return evidence


# ---------------------------------------------------------------------------
# Theme matching — keyword-based
# ---------------------------------------------------------------------------

def match_article_to_categories(
    article: RawArticle,
    min_score: float = 0.1,
) -> list[tuple[ThemeCategory, float]]:
    """Match an article to theme categories using keyword scoring.

    Returns list of (category, match_score) sorted by score desc.
    Score is proportion of category keywords found in the text.
    """
    text = (article.title + " " + article.snippet).lower()
    results = []

    for cat, keywords in THEME_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw.lower() in text)
        if hits > 0:
            score = min(hits / max(len(keywords) * 0.3, 1), 1.0)
            results.append((cat, score))

    # Also check news_monitor tags if available
    for tag in article.tags:
        nm_theme = tag.get("theme", "")
        mapped_cat = _NEWS_MONITOR_THEME_MAP.get(nm_theme)
        if mapped_cat and not any(c == mapped_cat for c, _ in results):
            results.append((mapped_cat, tag.get("confidence", 0.5)))
        nm_factor = tag.get("macro_factor", "")
        mapped_cat_f = _NEWS_MONITOR_FACTOR_MAP.get(nm_factor)
        if mapped_cat_f and not any(c == mapped_cat_f for c, _ in results):
            results.append((mapped_cat_f, tag.get("confidence", 0.3)))

    results.sort(key=lambda x: x[1], reverse=True)
    return [(c, s) for c, s in results if s >= min_score]


def match_article_to_themes(
    article: RawArticle,
    themes: list[Theme],
    min_score: float = 0.1,
) -> list[tuple[Theme, float]]:
    """Match an article to specific active themes.

    Uses category matching + keyword overlap with theme name/narrative.
    """
    cat_matches = match_article_to_categories(article, min_score)
    cat_scores = {cat: score for cat, score in cat_matches}

    text = (article.title + " " + article.snippet).lower()
    results = []

    for theme in themes:
        score = 0.0

        # Category match
        if theme.category in cat_scores:
            score += cat_scores[theme.category] * 0.6

        # Name overlap
        name_words = theme.name.lower().split()
        name_hits = sum(1 for w in name_words if w in text and len(w) > 3)
        if name_words:
            score += (name_hits / len(name_words)) * 0.3

        # Narrative overlap
        if theme.narrative:
            narr_words = set(w for w in theme.narrative.lower().split() if len(w) > 4)
            narr_hits = sum(1 for w in narr_words if w in text)
            if narr_words:
                score += (narr_hits / len(narr_words)) * 0.1

        if score >= min_score:
            results.append((theme, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Bulk processing pipeline
# ---------------------------------------------------------------------------

def sift_articles(
    articles: Sequence[RawArticle],
    themes: list[Theme],
    min_match_score: float = 0.1,
    cfg: EngineConfig | None = None,
) -> dict[str, list[EvidenceItem]]:
    """Process a batch of articles and assign as evidence to themes.

    Returns {theme_id: [EvidenceItem, ...]} mapping.
    Articles can match multiple themes.
    """
    cfg = cfg or get_config()
    results: dict[str, list[EvidenceItem]] = defaultdict(list)

    for article in articles:
        matches = match_article_to_themes(article, themes, min_match_score)
        if not matches:
            continue

        evidence = article_to_evidence(article, cfg)

        for theme, score in matches:
            # Boost usefulness by match quality
            boosted = EvidenceItem(
                source=evidence.source,
                date=evidence.date,
                title=evidence.title,
                url=evidence.url,
                credibility_score=evidence.credibility_score * score,
                timeliness_score=evidence.timeliness_score,
                corroboration_count=evidence.corroboration_count,
            )
            score_evidence(boosted, cfg)
            results[theme.theme_id].append(boosted)

    return dict(results)


def ingest_from_news_monitor(
    themes: list[Theme],
    db_path: str | Path | None = None,
    hours: int | None = None,
    min_match_score: float = 0.1,
    cfg: EngineConfig | None = None,
) -> dict[str, int]:
    """Pull articles from news_monitor DB and ingest into themes.

    Returns {theme_id: count_of_new_evidence} mapping.
    """
    cfg = cfg or get_config()
    ns = cfg.news_sifter
    db = db_path or ns.get("news_monitor_db", "../news_monitor/news_monitor.db")
    lookback = hours or ns.get("default_lookback_hours", 72)
    batch_size = ns.get("batch_size", 500)

    try:
        reader = NewsMonitorReader(db)
    except FileNotFoundError:
        return {}

    articles = reader.get_recent_articles(hours=lookback, limit=batch_size)
    if not articles:
        return {}

    evidence_map = sift_articles(articles, themes, min_match_score, cfg)

    counts: dict[str, int] = {}
    for theme in themes:
        items = evidence_map.get(theme.theme_id, [])
        if items:
            add_evidence_to_theme(theme, items, cfg)
            counts[theme.theme_id] = len(items)

    return counts


def ingest_from_csv(
    csv_path: str | Path,
    themes: list[Theme],
    min_match_score: float = 0.1,
    cfg: EngineConfig | None = None,
) -> dict[str, int]:
    """Ingest articles from a CSV file.

    Expected columns: title, snippet, url, source, published_at
    """
    import csv as csv_mod

    cfg = cfg or get_config()
    articles = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            articles.append(RawArticle(
                title=row.get("title", ""),
                snippet=row.get("snippet", ""),
                url=row.get("url", ""),
                source=row.get("source", "csv"),
                published_at=row.get("published_at"),
            ))

    evidence_map = sift_articles(articles, themes, min_match_score, cfg)

    counts: dict[str, int] = {}
    for theme in themes:
        items = evidence_map.get(theme.theme_id, [])
        if items:
            add_evidence_to_theme(theme, items, cfg)
            counts[theme.theme_id] = len(items)

    return counts


def create_themes_from_articles(
    articles: Sequence[RawArticle],
    min_articles_per_theme: int = 3,
    cfg: EngineConfig | None = None,
) -> list[Theme]:
    """Auto-discover themes from a batch of articles (bottom-up).

    Groups articles by category match, extracts dominant keywords to
    build descriptive theme names, infers direction from article tone,
    and creates Theme objects with evidence attached.
    """
    cfg = cfg or get_config()

    # Group articles by best-matching category
    cat_articles: dict[ThemeCategory, list[RawArticle]] = defaultdict(list)

    for article in articles:
        matches = match_article_to_categories(article)
        if matches:
            best_cat, _ = matches[0]
            cat_articles[best_cat].append(article)

    # Create themes for categories with enough articles
    themes = []
    for cat, arts in cat_articles.items():
        if len(arts) < min_articles_per_theme:
            continue

        # Extract top keywords from articles for naming
        theme_name, direction, narrative = _summarise_article_cluster(cat, arts)

        theme = Theme(
            name=theme_name,
            category=cat,
            narrative=narrative,
            direction=direction,
            status=ThemeStatus.ACTIVE,
            likelihood=min(len(arts) / 15.0, 0.85),
        )

        # Convert articles to evidence
        evidence_items = [article_to_evidence(a, cfg) for a in arts]
        add_evidence_to_theme(theme, evidence_items, cfg)

        themes.append(theme)

    return themes


# Direction-signal keywords for bearish/bullish classification
_BEARISH_WORDS = {
    "crisis", "crash", "selloff", "sell-off", "default", "recession",
    "slowdown", "risk", "fear", "warning", "collapse", "plunge", "drop",
    "decline", "turmoil", "contagion", "threat", "sanctions", "war",
    "escalation", "spike", "surge", "pressure", "stress", "tighten",
    "outflow", "cut", "downgrade", "debt", "deficit",
}
_BULLISH_WORDS = {
    "growth", "recovery", "rally", "surge", "gain", "expansion", "boost",
    "optimism", "confidence", "easing", "stimulus", "inflow", "upgrade",
    "rebound", "strong", "resilient", "bullish", "dovish", "cut rates",
}


def _summarise_article_cluster(
    category: ThemeCategory,
    articles: list[RawArticle],
) -> tuple[str, "Direction", str]:
    """Build a descriptive name, direction and narrative from articles.

    Returns (theme_name, direction, narrative).
    """
    from .models import Direction

    # Collect all text
    all_text = " ".join(
        (a.title + " " + a.snippet).lower() for a in articles
    )
    words = re.findall(r"[a-z]{4,}", all_text)
    word_freq: dict[str, int] = defaultdict(int)
    # Stop words to exclude
    stop = {
        "that", "this", "with", "from", "have", "been", "will", "would",
        "could", "should", "their", "they", "what", "when", "were", "which",
        "about", "than", "more", "also", "other", "some", "after", "before",
        "over", "into", "most", "such", "like", "just", "only", "very",
        "says", "said", "year", "years", "market", "markets", "news",
        "according", "report", "reports", "article", "articles",
    }
    for w in words:
        if w not in stop:
            word_freq[w] += 1

    # Top keywords (exclude category name itself)
    cat_words = set(category.value.lower().split("_"))
    top_kw = [
        w for w, _ in sorted(word_freq.items(), key=lambda x: -x[1])
        if w not in cat_words
    ][:8]

    # Determine direction from bearish/bullish keyword balance
    bearish_score = sum(1 for w in words if w in _BEARISH_WORDS)
    bullish_score = sum(1 for w in words if w in _BULLISH_WORDS)
    if bearish_score > bullish_score * 1.5:
        direction = Direction.BEARISH
    elif bullish_score > bearish_score * 1.5:
        direction = Direction.BULLISH
    else:
        direction = Direction.AMBIGUOUS

    # Build descriptive name from category + top keywords
    category_labels = {
        ThemeCategory.GEOPOLITICAL: "Geopolitical",
        ThemeCategory.GROWTH: "Growth",
        ThemeCategory.INFLATION: "Inflation",
        ThemeCategory.LIQUIDITY: "Liquidity",
        ThemeCategory.STRUCTURAL: "Structural",
        ThemeCategory.POLICY: "Policy",
        ThemeCategory.VALUATION: "Valuation",
        ThemeCategory.CONTAGION: "Contagion",
    }
    cat_label = category_labels.get(category, category.value.title())

    # Pick 2-3 distinctive keywords for the name
    name_kw = [w.capitalize() for w in top_kw[:3]]
    theme_name = f"{cat_label}: {' / '.join(name_kw)}"

    # Build narrative from top headlines
    # Use English-looking titles only
    en_titles = [
        a.title for a in articles
        if a.title and all(ord(c) < 128 for c in a.title[:20])
    ]
    top_headlines = en_titles[:5] if en_titles else [a.title for a in articles[:5]]

    narrative = (
        f"Auto-detected {cat_label.lower()} theme based on {len(articles)} "
        f"recent articles. Direction: {direction.value}. "
        f"Key headlines: {'; '.join(h.strip()[:100] for h in top_headlines[:3])}"
    )

    return theme_name, direction, narrative
