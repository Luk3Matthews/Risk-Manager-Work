"""
Database layer — SQLite persistence for the News Monitor.
Tables: articles, article_tags, regimes, audit_log
"""

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).parent.parent / "news_monitor.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id TEXT PRIMARY KEY,
    fetched_at TEXT NOT NULL,
    published_at TEXT,
    source TEXT NOT NULL,
    title TEXT NOT NULL,
    snippet TEXT,
    url TEXT NOT NULL,
    query TEXT,
    url_hash TEXT NOT NULL,
    UNIQUE(url_hash)
);

CREATE TABLE IF NOT EXISTS article_tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id TEXT NOT NULL,
    theme TEXT NOT NULL,
    macro_factor TEXT NOT NULL,
    direction TEXT NOT NULL DEFAULT 'neutral',
    horizon TEXT NOT NULL DEFAULT 'short',
    confidence REAL NOT NULL DEFAULT 0.5,
    entities_json TEXT,
    keywords_json TEXT,
    bloomberg_tickers_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (article_id) REFERENCES articles(id)
);

CREATE TABLE IF NOT EXISTS regimes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    flags_json TEXT,
    key_movers_json TEXT,
    commentary TEXT
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    action TEXT NOT NULL,
    details_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source);
CREATE INDEX IF NOT EXISTS idx_articles_fetched ON articles(fetched_at);
CREATE INDEX IF NOT EXISTS idx_articles_url_hash ON articles(url_hash);
CREATE INDEX IF NOT EXISTS idx_tags_article ON article_tags(article_id);
CREATE INDEX IF NOT EXISTS idx_tags_theme ON article_tags(theme);
CREATE INDEX IF NOT EXISTS idx_tags_factor ON article_tags(macro_factor);
CREATE INDEX IF NOT EXISTS idx_tags_created ON article_tags(created_at);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
"""


def _url_hash(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def _title_hash(title: str) -> str:
    normalized = title.lower().strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _article_id(url: str, title: str) -> str:
    combined = f"{url}|{title.lower().strip()}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:20]


@contextmanager
def get_connection(db_path: str | Path | None = None):
    """Context manager for DB connections with WAL mode."""
    path = str(db_path or DB_PATH)
    conn = sqlite3.connect(path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str | Path | None = None):
    """Create tables if they don't exist."""
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA)
    audit("db_init", {"path": str(db_path or DB_PATH)}, db_path=db_path)


def insert_article(
    source: str,
    title: str,
    snippet: str | None,
    url: str,
    published_at: str | None,
    query: str | None,
    db_path: str | Path | None = None,
) -> str | None:
    """
    Insert an article. Returns article_id if new, None if duplicate.
    Deduplicates by URL hash.
    """
    uh = _url_hash(url)
    aid = _article_id(url, title)
    now = datetime.utcnow().isoformat()

    with get_connection(db_path) as conn:
        existing = conn.execute(
            "SELECT id FROM articles WHERE url_hash = ?", (uh,)
        ).fetchone()
        if existing:
            return None

        conn.execute(
            """INSERT INTO articles (id, fetched_at, published_at, source, title, snippet, url, query, url_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (aid, now, published_at, source, title, snippet, url, query, uh),
        )
    return aid


def insert_tags(
    article_id: str,
    tags: list[dict[str, Any]],
    db_path: str | Path | None = None,
):
    """
    Insert tag records for an article.
    Each tag dict: {theme, macro_factor, direction, horizon, confidence, entities, keywords, bloomberg_tickers}
    """
    now = datetime.utcnow().isoformat()
    with get_connection(db_path) as conn:
        for t in tags:
            conn.execute(
                """INSERT INTO article_tags
                   (article_id, theme, macro_factor, direction, horizon, confidence,
                    entities_json, keywords_json, bloomberg_tickers_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    article_id,
                    t["theme"],
                    t["macro_factor"],
                    t.get("direction", "neutral"),
                    t.get("horizon", "short"),
                    t.get("confidence", 0.5),
                    json.dumps(t.get("entities", [])),
                    json.dumps(t.get("keywords", [])),
                    json.dumps(t.get("bloomberg_tickers", [])),
                    now,
                ),
            )


def insert_regime(
    flags: dict,
    key_movers: dict | None = None,
    commentary: str | None = None,
    db_path: str | Path | None = None,
):
    """Insert a regime snapshot."""
    now = datetime.utcnow().isoformat()
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO regimes (timestamp, flags_json, key_movers_json, commentary) VALUES (?, ?, ?, ?)",
            (now, json.dumps(flags), json.dumps(key_movers or {}), commentary),
        )


def audit(
    action: str,
    details: dict | None = None,
    db_path: str | Path | None = None,
):
    """Write to audit log."""
    now = datetime.utcnow().isoformat()
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO audit_log (timestamp, action, details_json) VALUES (?, ?, ?)",
            (now, action, json.dumps(details or {})),
        )


def get_recent_articles(
    hours: int = 24,
    source: str | None = None,
    theme: str | None = None,
    macro_factor: str | None = None,
    limit: int = 200,
    db_path: str | Path | None = None,
) -> list[dict]:
    """Retrieve recent articles with their tags."""
    with get_connection(db_path) as conn:
        query_parts = [
            """SELECT DISTINCT a.id, a.fetched_at, a.published_at, a.source,
                      a.title, a.snippet, a.url, a.query
               FROM articles a"""
        ]
        joins = []
        wheres = [f"a.fetched_at >= datetime('now', '-{hours} hours')"]
        params: list[Any] = []

        if theme or macro_factor:
            joins.append("JOIN article_tags t ON t.article_id = a.id")
            if theme:
                wheres.append("t.theme = ?")
                params.append(theme)
            if macro_factor:
                wheres.append("t.macro_factor = ?")
                params.append(macro_factor)

        if source:
            wheres.append("a.source = ?")
            params.append(source)

        sql = " ".join(query_parts + joins)
        if wheres:
            sql += " WHERE " + " AND ".join(wheres)
        sql += " ORDER BY a.fetched_at DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(sql, params).fetchall()
        results = []
        for r in rows:
            article = dict(r)
            tags = conn.execute(
                "SELECT * FROM article_tags WHERE article_id = ?", (r["id"],)
            ).fetchall()
            article["tags"] = [dict(t) for t in tags]
            results.append(article)
        return results


def get_factor_heatmap(hours: int = 24, db_path: str | Path | None = None) -> dict:
    """
    Returns {macro_factor: {count, avg_confidence, directions: {up, down, neutral}}}
    """
    with get_connection(db_path) as conn:
        rows = conn.execute(
            f"""SELECT macro_factor, direction, COUNT(*) as cnt, AVG(confidence) as avg_conf
                FROM article_tags
                WHERE created_at >= datetime('now', '-{hours} hours')
                GROUP BY macro_factor, direction"""
        ).fetchall()

    heatmap: dict[str, dict] = {}
    for r in rows:
        f = r["macro_factor"]
        if f not in heatmap:
            heatmap[f] = {"count": 0, "avg_confidence": 0.0, "directions": {}}
        heatmap[f]["count"] += r["cnt"]
        heatmap[f]["directions"][r["direction"]] = r["cnt"]
        # Weighted average
        total = heatmap[f]["count"]
        heatmap[f]["avg_confidence"] = (
            (heatmap[f]["avg_confidence"] * (total - r["cnt"]) + r["avg_conf"] * r["cnt"])
            / total
        )
    return heatmap


def get_audit_since(hours: int = 1, db_path: str | Path | None = None) -> list[dict]:
    """Get audit log entries from the last N hours."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            f"""SELECT * FROM audit_log
                WHERE timestamp >= datetime('now', '-{hours} hours')
                ORDER BY timestamp DESC""",
        ).fetchall()
        return [dict(r) for r in rows]


def article_exists(url: str, db_path: str | Path | None = None) -> bool:
    """Check if an article URL already exists."""
    uh = _url_hash(url)
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT 1 FROM articles WHERE url_hash = ?", (uh,)
        ).fetchone()
        return row is not None


def get_factor_timeseries(hours: int = 24, bucket_minutes: int = 60, db_path: str | Path | None = None) -> list[dict]:
    """
    Returns time-bucketed signal counts per factor.
    Each row: {bucket, macro_factor, direction, count, avg_confidence}
    """
    with get_connection(db_path) as conn:
        rows = conn.execute(
            f"""SELECT
                    strftime('%Y-%m-%dT%H:', created_at) ||
                        CAST((CAST(strftime('%M', created_at) AS INTEGER) / {bucket_minutes}) * {bucket_minutes} AS TEXT) || ':00' AS bucket,
                    macro_factor,
                    direction,
                    COUNT(*) as cnt,
                    AVG(confidence) as avg_conf
                FROM article_tags
                WHERE created_at >= datetime('now', '-{hours} hours')
                GROUP BY bucket, macro_factor, direction
                ORDER BY bucket"""
        ).fetchall()
        return [dict(r) for r in rows]


def get_signal_timeline(hours: int = 24, db_path: str | Path | None = None) -> list[dict]:
    """
    Returns individual signals with timestamps for timeline charts.
    """
    with get_connection(db_path) as conn:
        rows = conn.execute(
            f"""SELECT t.created_at, t.theme, t.macro_factor, t.direction,
                       t.confidence, a.title, a.source
                FROM article_tags t
                JOIN articles a ON a.id = t.article_id
                WHERE t.created_at >= datetime('now', '-{hours} hours')
                ORDER BY t.created_at DESC
                LIMIT 500"""
        ).fetchall()
        return [dict(r) for r in rows]


def get_theme_summary(hours: int = 24, db_path: str | Path | None = None) -> dict:
    """
    Returns per-theme summary: signal counts, directions, top articles, factors, entities.
    {theme: {count, avg_conf, directions, factors, articles: [{title, direction, confidence, source}], entities}}
    """
    with get_connection(db_path) as conn:
        # Aggregate stats per theme
        stats = conn.execute(
            f"""SELECT t.theme, t.direction, COUNT(*) as cnt, AVG(t.confidence) as avg_conf
                FROM article_tags t
                WHERE t.created_at >= datetime('now', '-{hours} hours')
                GROUP BY t.theme, t.direction"""
        ).fetchall()

        # Top articles per theme (deduplicated by title, highest confidence first)
        article_rows = conn.execute(
            f"""SELECT t.theme, t.macro_factor, t.direction, t.confidence,
                       t.entities_json, t.keywords_json, a.title, a.source, a.url
                FROM article_tags t
                JOIN articles a ON a.id = t.article_id
                WHERE t.created_at >= datetime('now', '-{hours} hours')
                ORDER BY t.confidence DESC"""
        ).fetchall()

    result: dict[str, dict] = {}

    for r in stats:
        theme = r["theme"]
        if theme not in result:
            result[theme] = {
                "count": 0, "avg_confidence": 0.0,
                "directions": {}, "factors": set(), "articles": [],
                "entities": set(), "keywords": set(),
            }
        result[theme]["count"] += r["cnt"]
        result[theme]["directions"][r["direction"]] = r["cnt"]

    # Compute weighted average confidence
    for r in stats:
        theme = r["theme"]
        total = result[theme]["count"]
        if total > 0:
            result[theme]["avg_confidence"] = (
                (result[theme]["avg_confidence"] * (total - r["cnt"]) + r["avg_conf"] * r["cnt"])
                / total
            )

    seen_titles: dict[str, set] = {}
    for r in article_rows:
        theme = r["theme"]
        if theme not in result:
            continue
        result[theme]["factors"].add(r["macro_factor"])

        # Parse entities and keywords
        try:
            entities = json.loads(r["entities_json"]) if r["entities_json"] else []
            for e in entities:
                if isinstance(e, str):
                    result[theme]["entities"].add(e)
                elif isinstance(e, dict):
                    result[theme]["entities"].add(e.get("name", str(e)))
        except (json.JSONDecodeError, TypeError):
            pass

        try:
            kws = json.loads(r["keywords_json"]) if r["keywords_json"] else []
            for k in kws:
                result[theme]["keywords"].add(str(k))
        except (json.JSONDecodeError, TypeError):
            pass

        # Deduplicated top articles
        if theme not in seen_titles:
            seen_titles[theme] = set()
        title = r["title"]
        if title not in seen_titles[theme] and len(result[theme]["articles"]) < 8:
            seen_titles[theme].add(title)
            result[theme]["articles"].append({
                "title": title,
                "direction": r["direction"],
                "confidence": r["confidence"],
                "source": r["source"],
                "url": r["url"],
                "factor": r["macro_factor"],
            })

    # Convert sets to sorted lists for serialization
    for theme in result:
        result[theme]["factors"] = sorted(result[theme]["factors"])
        result[theme]["entities"] = sorted(result[theme]["entities"])[:15]
        result[theme]["keywords"] = sorted(result[theme]["keywords"])[:20]

    return result
