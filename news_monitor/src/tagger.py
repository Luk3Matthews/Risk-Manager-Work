"""
Tagging Pipeline — deterministic keyword/phrase matching + entity extraction.
Given article title + snippet, assigns themes, macro factors, direction, horizon, confidence.
"""

import logging
import re
from typing import Any

from .taxonomy import Taxonomy, get_taxonomy

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple word tokenization (lowercase, alphanumeric only)."""
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())


def _extract_keywords(text: str, taxonomy: Taxonomy) -> dict[str, list[str]]:
    """
    Match taxonomy keywords against text.
    Returns {theme: [matched_keywords]}
    """
    tokens = set(_tokenize(text))
    text_lower = text.lower()
    theme_hits: dict[str, list[str]] = {}

    # Single-word keyword matches
    for token in tokens:
        themes = taxonomy.get_themes_for_keyword(token)
        for theme in themes:
            theme_hits.setdefault(theme, []).append(token)

    # Multi-word phrase matches
    for phrase, themes in taxonomy._phrase_to_themes.items():
        if phrase in text_lower:
            for theme in themes:
                theme_hits.setdefault(theme, []).append(phrase)

    return theme_hits


def _extract_entities(text: str, taxonomy: Taxonomy) -> list[dict[str, str]]:
    """
    Extract known entities (central banks, countries, commodities) from text.
    Uses regex word-boundary matching against alias lists.
    """
    found: list[dict[str, str]] = []
    seen: set[str] = set()

    for alias_lower, entity_info in taxonomy._entity_aliases.items():
        if len(alias_lower) < 3:
            # Very short aliases (US, UK, FX) — require word boundaries
            pattern = r"\b" + re.escape(alias_lower) + r"\b"
            if re.search(pattern, text, re.IGNORECASE):
                key = entity_info["name"]
                if key not in seen:
                    found.append(entity_info)
                    seen.add(key)
        else:
            if alias_lower in text.lower():
                key = entity_info["name"]
                if key not in seen:
                    found.append(entity_info)
                    seen.add(key)

    return found


def _detect_direction(text: str, taxonomy: Taxonomy) -> str:
    """
    Detect directional bias from text using direction signal words.
    Returns 'up', 'down', or 'neutral'.
    """
    text_lower = text.lower()
    up_score = 0
    down_score = 0

    for word in taxonomy.direction_signals.get("up", []):
        if word in text_lower:
            up_score += 1
    for word in taxonomy.direction_signals.get("down", []):
        if word in text_lower:
            down_score += 1

    if up_score > down_score and up_score >= 1:
        return "up"
    elif down_score > up_score and down_score >= 1:
        return "down"
    return "neutral"


def _detect_horizon(text: str, taxonomy: Taxonomy) -> str:
    """
    Detect time horizon from text.
    Returns 'short' (0-6m), 'medium' (6-12m), or 'long' (12-24m+).
    """
    text_lower = text.lower()

    for signal in taxonomy.horizon_signals.get("long", []):
        if signal in text_lower:
            return "long"
    for signal in taxonomy.horizon_signals.get("medium", []):
        if signal in text_lower:
            return "medium"
    for signal in taxonomy.horizon_signals.get("short", []):
        if signal in text_lower:
            return "short"

    return "short"  # default


def _compute_confidence(
    keyword_count: int,
    entity_count: int,
    source_reliability: float = 0.7,
    corroboration_count: int = 0,
) -> float:
    """
    Compute confidence score (0-1) based on:
    - Number of keyword matches
    - Entity recognition
    - Source reliability weight
    - Corroboration count (same theme from other sources)
    """
    # Base from keyword density
    kw_score = min(keyword_count / 5.0, 1.0) * 0.4

    # Entity boost
    ent_score = min(entity_count / 3.0, 1.0) * 0.2

    # Source reliability
    src_score = source_reliability * 0.25

    # Corroboration
    corr_score = min(corroboration_count / 3.0, 1.0) * 0.15

    confidence = kw_score + ent_score + src_score + corr_score
    return round(min(max(confidence, 0.05), 1.0), 3)


def _get_bloomberg_tickers(
    themes: list[str], macro_factors: list[str], taxonomy: Taxonomy
) -> list[dict]:
    """Get relevant Bloomberg tickers for the detected themes and factors."""
    tickers: list[dict] = []
    seen_tickers: set[str] = set()

    for theme in themes:
        for t in taxonomy.get_tickers_for_theme(theme):
            if t["ticker"] not in seen_tickers:
                tickers.append({"ticker": t["ticker"], "name": t["name"],
                                "direction_confirms": t.get("direction_confirms", "")})
                seen_tickers.add(t["ticker"])

    for factor in macro_factors:
        for t in taxonomy.get_tickers_for_factor(factor):
            if t["ticker"] not in seen_tickers:
                tickers.append({"ticker": t["ticker"], "name": t["name"],
                                "direction_confirms": t.get("direction_confirms", "")})
                seen_tickers.add(t["ticker"])

    return tickers


def tag_article(
    title: str,
    snippet: str | None,
    source_reliability: float = 0.7,
    corroboration_count: int = 0,
    taxonomy_instance: Taxonomy | None = None,
) -> list[dict[str, Any]]:
    """
    Main tagging function. Processes title + snippet and returns tag records.

    Returns list of dicts, each with:
      theme, macro_factor, direction, horizon, confidence, entities, keywords, bloomberg_tickers
    """
    taxonomy = taxonomy_instance or get_taxonomy()
    text = f"{title} {snippet or ''}".strip()

    if not text:
        return []

    # 1. Extract keywords -> themes
    theme_keywords = _extract_keywords(text, taxonomy)
    if not theme_keywords:
        return []

    # 2. Extract entities
    entities = _extract_entities(text, taxonomy)
    entity_names = [e["name"] for e in entities]

    # 3. Detect direction
    direction = _detect_direction(text, taxonomy)

    # 4. Detect horizon
    horizon = _detect_horizon(text, taxonomy)

    # 5. For each theme, map to macro factors
    tags: list[dict[str, Any]] = []
    seen_factor_combos: set[tuple] = set()

    for theme, matched_kws in theme_keywords.items():
        # Get mapping rules
        rules = taxonomy.get_mapping_rules_for_theme(theme, direction)

        if rules:
            for rule in rules:
                for factor_map in rule["maps_to"]:
                    factor = factor_map["factor"]
                    factor_dir = factor_map["direction"]
                    combo = (theme, factor, factor_dir)
                    if combo in seen_factor_combos:
                        continue
                    seen_factor_combos.add(combo)

                    conf = _compute_confidence(
                        keyword_count=len(matched_kws),
                        entity_count=len(entities),
                        source_reliability=source_reliability,
                        corroboration_count=corroboration_count,
                    )
                    conf += factor_map.get("confidence_boost", 0)
                    conf = round(min(conf, 1.0), 3)

                    # Get relevant tickers
                    tickers = _get_bloomberg_tickers([theme], [factor], taxonomy)

                    tags.append({
                        "theme": theme,
                        "macro_factor": factor,
                        "direction": factor_dir,
                        "horizon": rule.get("typical_horizon", horizon),
                        "confidence": conf,
                        "entities": entity_names,
                        "keywords": matched_kws[:10],
                        "bloomberg_tickers": tickers,
                    })
        else:
            # No mapping rule — use direct theme->factor from macro_factors config
            for factor_name, factor_cfg in taxonomy.macro_factors.items():
                if theme in factor_cfg.get("themes", []):
                    combo = (theme, factor_name, direction)
                    if combo in seen_factor_combos:
                        continue
                    seen_factor_combos.add(combo)

                    conf = _compute_confidence(
                        keyword_count=len(matched_kws),
                        entity_count=len(entities),
                        source_reliability=source_reliability,
                        corroboration_count=corroboration_count,
                    )

                    tickers = _get_bloomberg_tickers([theme], [factor_name], taxonomy)

                    tags.append({
                        "theme": theme,
                        "macro_factor": factor_name,
                        "direction": direction,
                        "horizon": horizon,
                        "confidence": conf,
                        "entities": entity_names,
                        "keywords": matched_kws[:10],
                        "bloomberg_tickers": tickers,
                    })

    return tags
