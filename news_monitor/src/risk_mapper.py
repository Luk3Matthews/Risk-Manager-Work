"""
Risk Mapper — links news themes/factors to Bloomberg tickers and asset classes.
Provides the bridge between the tagging pipeline and market confirmation.
"""

import logging
from typing import Any

from .taxonomy import Taxonomy, get_taxonomy

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# ASSET CLASS MAPPING
# Maps macro factors to VFMC asset classes with expected sensitivity direction
# ═══════════════════════════════════════════════════════════════════════════════

FACTOR_ASSET_CLASS_MAP: dict[str, list[dict[str, Any]]] = {
    "Inflation": [
        {"asset_class": "FI", "sensitivity": "negative", "note": "Duration hurt by rising inflation"},
        {"asset_class": "ILB", "sensitivity": "positive", "note": "Inflation-linked benefit"},
        {"asset_class": "AEQ", "sensitivity": "mixed", "note": "Depends on pass-through"},
        {"asset_class": "IEQ", "sensitivity": "mixed", "note": "Sector-dependent"},
        {"asset_class": "Property", "sensitivity": "negative_lag", "note": "Higher discount rates"},
    ],
    "Growth": [
        {"asset_class": "AEQ", "sensitivity": "positive", "note": "Earnings growth"},
        {"asset_class": "IEQ", "sensitivity": "positive", "note": "Global earnings"},
        {"asset_class": "Credit", "sensitivity": "positive", "note": "Lower defaults"},
        {"asset_class": "Property", "sensitivity": "positive", "note": "Rental growth"},
        {"asset_class": "FI", "sensitivity": "negative", "note": "Yields rise with growth"},
    ],
    "Rates": [
        {"asset_class": "FI", "sensitivity": "negative", "note": "Price falls on rate rises"},
        {"asset_class": "Property", "sensitivity": "negative", "note": "Higher cap rates"},
        {"asset_class": "AEQ", "sensitivity": "negative", "note": "Higher discount rate"},
        {"asset_class": "IEQ", "sensitivity": "negative", "note": "Growth stocks hurt more"},
        {"asset_class": "Credit", "sensitivity": "negative", "note": "Higher cost of debt"},
    ],
    "CreditSpreads": [
        {"asset_class": "Credit", "sensitivity": "negative", "note": "Direct spread impact"},
        {"asset_class": "AEQ", "sensitivity": "negative", "note": "Risk-off signal"},
        {"asset_class": "IEQ", "sensitivity": "negative", "note": "Risk-off signal"},
        {"asset_class": "FI_IG", "sensitivity": "negative", "note": "Spread widening"},
    ],
    "FX": [
        {"asset_class": "IEQ", "sensitivity": "mixed", "note": "AUD hedging impact"},
        {"asset_class": "FX_Overlay", "sensitivity": "direct", "note": "P&L on hedges"},
        {"asset_class": "Alternatives", "sensitivity": "mixed", "note": "USD-denominated assets"},
    ],
    "Liquidity": [
        {"asset_class": "AEQ", "sensitivity": "negative", "note": "Selling pressure"},
        {"asset_class": "IEQ", "sensitivity": "negative", "note": "EM outflows"},
        {"asset_class": "Credit", "sensitivity": "negative", "note": "Bid-ask widens"},
        {"asset_class": "Alternatives", "sensitivity": "negative", "note": "Redemption risk"},
    ],
    "CommoditySupply": [
        {"asset_class": "Alternatives", "sensitivity": "mixed", "note": "Commodity exposure"},
        {"asset_class": "AEQ", "sensitivity": "positive", "note": "Resources sector"},
        {"asset_class": "IEQ", "sensitivity": "mixed", "note": "Net importers hurt"},
    ],
    "Policy": [
        {"asset_class": "FI", "sensitivity": "direct", "note": "Front-end driven by policy"},
        {"asset_class": "AEQ", "sensitivity": "mixed", "note": "Fiscal vs monetary"},
        {"asset_class": "IEQ", "sensitivity": "mixed", "note": "Regional divergence"},
    ],
    "Geopolitics": [
        {"asset_class": "IEQ", "sensitivity": "negative", "note": "Risk-off, EM vulnerable"},
        {"asset_class": "AEQ", "sensitivity": "mixed", "note": "Less direct unless China"},
        {"asset_class": "Alternatives", "sensitivity": "mixed", "note": "Commodity upside possible"},
        {"asset_class": "FI", "sensitivity": "positive", "note": "Safe-haven bid"},
    ],
}


def get_affected_asset_classes(macro_factor: str, direction: str) -> list[dict]:
    """
    Given a macro factor and its direction, return affected asset classes
    with expected impact.
    """
    mappings = FACTOR_ASSET_CLASS_MAP.get(macro_factor, [])
    result = []
    for m in mappings:
        impact = m["sensitivity"]
        # Flip if direction is down (e.g., rates DOWN is positive for FI)
        if direction == "down":
            if impact == "positive":
                impact = "negative"
            elif impact == "negative":
                impact = "positive"
        result.append({
            "asset_class": m["asset_class"],
            "impact": impact,
            "note": m["note"],
        })
    return result


def get_tickers_for_article_tags(tags: list[dict], taxonomy: Taxonomy | None = None) -> list[dict]:
    """
    Given article tags, return all relevant Bloomberg tickers with context.
    Deduplicates and adds relevance info.
    """
    taxonomy = taxonomy or get_taxonomy()
    tickers: list[dict] = []
    seen: set[str] = set()

    for tag in tags:
        theme = tag.get("theme", "")
        factor = tag.get("macro_factor", "")
        direction = tag.get("direction", "neutral")

        # From taxonomy ticker mappings
        for t in taxonomy.get_tickers_for_theme(theme):
            if t["ticker"] not in seen:
                tickers.append({
                    "ticker": t["ticker"],
                    "name": t["name"],
                    "category": next(
                        (cat for cat, tlist in taxonomy.bloomberg_tickers.items() if t in tlist),
                        "other"
                    ),
                    "relevance_theme": theme,
                    "relevance_factor": factor,
                    "expected_direction": _infer_expected_move(t, direction),
                    "direction_confirms": t.get("direction_confirms", ""),
                })
                seen.add(t["ticker"])

        for t in taxonomy.get_tickers_for_factor(factor):
            if t["ticker"] not in seen:
                tickers.append({
                    "ticker": t["ticker"],
                    "name": t["name"],
                    "category": next(
                        (cat for cat, tlist in taxonomy.bloomberg_tickers.items() if t in tlist),
                        "other"
                    ),
                    "relevance_theme": theme,
                    "relevance_factor": factor,
                    "expected_direction": _infer_expected_move(t, direction),
                    "direction_confirms": t.get("direction_confirms", ""),
                })
                seen.add(t["ticker"])

    return tickers


def _infer_expected_move(ticker_config: dict, news_direction: str) -> str:
    """
    Based on the ticker's direction_confirms rule and the news direction,
    infer what market move we'd expect to confirm the news signal.
    """
    rule = ticker_config.get("direction_confirms", "")
    if not rule:
        return "unknown"

    # Simple heuristic parsing of rules like "price_up = growth_up"
    if news_direction == "up":
        if "price_up" in rule or "yield_up" in rule or "spread_up" in rule or "rate_up" in rule:
            return "up"
        if "price_down" in rule:
            return "down"
    elif news_direction == "down":
        if "price_down" in rule or "yield_down" in rule:
            return "down"
        if "price_up" in rule or "yield_up" in rule:
            return "up"  # inverse: e.g., VIX up = liquidity down

    return "neutral"


def build_risk_summary(tags: list[dict]) -> dict[str, Any]:
    """
    Build a risk summary from a collection of article tags.
    Returns factor-level aggregation with Bloomberg ticker links.
    """
    taxonomy = get_taxonomy()
    factor_agg: dict[str, dict] = {}

    for tag in tags:
        factor = tag["macro_factor"]
        if factor not in factor_agg:
            factor_agg[factor] = {
                "count": 0,
                "avg_confidence": 0.0,
                "directions": {"up": 0, "down": 0, "neutral": 0},
                "themes": set(),
                "tickers": [],
                "asset_classes": [],
            }

        agg = factor_agg[factor]
        agg["count"] += 1
        agg["avg_confidence"] = (
            (agg["avg_confidence"] * (agg["count"] - 1) + tag["confidence"])
            / agg["count"]
        )
        agg["directions"][tag.get("direction", "neutral")] += 1
        agg["themes"].add(tag["theme"])

    # Add tickers and asset class impacts
    for factor, agg in factor_agg.items():
        agg["themes"] = list(agg["themes"])
        # Determine dominant direction
        dirs = agg["directions"]
        if dirs["up"] > dirs["down"]:
            dom_dir = "up"
        elif dirs["down"] > dirs["up"]:
            dom_dir = "down"
        else:
            dom_dir = "neutral"
        agg["dominant_direction"] = dom_dir

        # Bloomberg tickers
        agg["tickers"] = [
            {"ticker": t["ticker"], "name": t["name"], "direction_confirms": t.get("direction_confirms", "")}
            for t in taxonomy.get_tickers_for_factor(factor)
        ]

        # Asset class impacts
        agg["asset_classes"] = get_affected_asset_classes(factor, dom_dir)

    return factor_agg
