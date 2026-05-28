"""Theme Engine -- Module 7: Dashboard / Output Layer.

Generates outputs for the live pipeline:
  1. Pipeline Header (data sources, timestamp)
  2. Theme Dashboard (table)
  3. Market Indicator Summary (composite z-scores per family)
  4. Scenario Cards (one per active theme)
  5. Portfolio Positioning Summary
  6. News Evidence Summary (articles matched to themes)

Outputs to console, CSV, and JSON.
"""
from __future__ import annotations

import csv
import io
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import EngineConfig, get_config
from .models import (
    ASSET_CLASS_LABELS,
    DRIVER_LABELS,
    MACRO_DRIVERS,
    AssetReturn,
    FamilyComposite,
    HedgeRecommendation,
    PortfolioPosition,
    PortfolioSummary,
    ScenarioCard,
    Theme,
)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_pct(v: float, dp: int = 2) -> str:
    return f"{v * 100:+.{dp}f}%"


def _fmt_sigma(v: float, dp: int = 2) -> str:
    return f"{v:+.{dp}f}s"


def _fmt_score(v: float, dp: int = 2) -> str:
    return f"{v:.{dp}f}"


def _bar(v: float, width: int = 20, char: str = "#") -> str:
    """Simple inline bar chart for a 0-1 value."""
    filled = int(round(abs(v) * width))
    return char * filled + "." * (width - filled)


# ---------------------------------------------------------------------------
# OUTPUT 0: Pipeline Header
# ---------------------------------------------------------------------------

def format_pipeline_header(
    data_source: str = "bloomberg",
    themes_source: str = "file",
    news_db_path: str | None = None,
    n_themes: int = 0,
    n_articles_ingested: int = 0,
    n_indicators: int = 32,
) -> str:
    """Render the pipeline header showing live data sources and timestamp."""
    now = datetime.now()
    lines = []
    lines.append("=" * 80)
    lines.append("  THEME-DRIVEN MACRO SCENARIO ENGINE")
    lines.append(f"  Generated: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    lines.append("")
    lines.append("  DATA SOURCES")
    lines.append(f"  {'Indicators:':<24s} {data_source.upper()}"
                 + (" (BLPAPI //blp/refdata)" if data_source == "bloomberg" else ""))
    lines.append(f"  {'Themes:':<24s} {themes_source}"
                 + f" ({n_themes} active)")
    if news_db_path:
        lines.append(f"  {'News Articles:':<24s} {news_db_path}")
        lines.append(f"  {'Articles Matched:':<24s} {n_articles_ingested}")
    lines.append(f"  {'Indicator Families:':<24s} {n_indicators} indicators across 5 families")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# OUTPUT 1: Theme Dashboard
# ---------------------------------------------------------------------------

def theme_dashboard_table(themes: list[Theme]) -> list[dict[str, str]]:
    """Build theme dashboard as a list of dicts (for tabular rendering).

    Columns: Theme | Category | Direction | Horizon | Strength |
             Confirmation | Status | Evidence | Last Updated
    """
    rows = []
    for t in sorted(themes, key=lambda x: x.strength, reverse=True):
        # Count evidence by source type
        sources = {}
        for e in t.evidence:
            src = e.source
            sources[src] = sources.get(src, 0) + 1
        src_str = ", ".join(f"{s}:{n}" for s, n in sorted(sources.items())) if sources else "-"

        rows.append({
            "Theme": t.name,
            "Category": t.category.value,
            "Direction": t.direction.value,
            "Horizon": t.horizon.value,
            "Likelihood": _fmt_score(t.likelihood),
            "Strength": _fmt_score(t.strength),
            "Confirmation": _fmt_score(t.confirmation_score),
            "Status": t.status.value,
            "Evidence": f"{len(t.evidence)} ({src_str})",
            "Last Updated": t.last_updated.isoformat(),
        })
    return rows


def print_theme_dashboard(themes: list[Theme]) -> str:
    """Render theme dashboard as a formatted text table."""
    rows = theme_dashboard_table(themes)
    if not rows:
        return "No active themes."

    cols = list(rows[0].keys())
    widths = {c: max(len(c), max(len(r[c]) for r in rows)) for c in cols}

    lines = []
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    sep = "-+-".join("-" * widths[c] for c in cols)
    lines.append("=" * len(header))
    lines.append("THEME DASHBOARD")
    lines.append("=" * len(header))
    lines.append(header)
    lines.append(sep)
    for r in rows:
        lines.append(" | ".join(r[c].ljust(widths[c]) for c in cols))

    # Strength bar chart
    lines.append("")
    lines.append("  THEME STRENGTH")
    for t in sorted(themes, key=lambda x: x.strength, reverse=True):
        bar = _bar(t.strength, width=30)
        lines.append(f"  {t.name:<45s} [{bar}] {t.strength:.3f}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# OUTPUT 2: Market Indicator Summary
# ---------------------------------------------------------------------------

def format_indicator_summary(
    composites: list[FamilyComposite],
    overall_market_stress: float,
    latest_z: dict[str, float] | None = None,
) -> str:
    """Render the market indicator composite summary."""
    lines = []
    lines.append("=" * 80)
    lines.append("MARKET INDICATOR SUMMARY  (Bloomberg Live)")
    lines.append("=" * 80)
    lines.append("")

    # Family composites table
    lines.append(f"  {'Family':<28s} {'Composite Z':>12s} {'Percentile':>11s} "
                 f"{'Indicators':>11s}  {'Regime':>12s}")
    lines.append(f"  {'-'*28} {'-'*12} {'-'*11} {'-'*11}  {'-'*12}")

    for fc in composites:
        regime = _regime_label(fc.composite_z, fc.percentile)
        bar = _bar(fc.percentile, width=10)
        lines.append(
            f"  {fc.family.value:<28s} "
            f"{_fmt_sigma(fc.composite_z):>12s} "
            f"{_fmt_pct(fc.percentile):>11s} "
            f"{fc.n_indicators:>11d}  "
            f"{regime:>12s}"
        )

    lines.append("")
    stress_label = "ELEVATED" if overall_market_stress > 0.5 else (
        "MODERATE" if overall_market_stress > 0 else "LOW"
    )
    lines.append(f"  Overall Market Stress: {_fmt_sigma(overall_market_stress)}  ({stress_label})")
    lines.append("")

    # Individual indicator z-scores (if available)
    if latest_z:
        lines.append("  TOP INDICATOR READINGS:")
        sorted_z = sorted(latest_z.items(), key=lambda x: abs(x[1]), reverse=True)
        for name, z in sorted_z[:15]:
            direction = ">>>" if z > 1.0 else ("<<<" if z < -1.0 else "   ")
            lines.append(f"    {direction} {name:<25s} z={_fmt_sigma(z)}")
        lines.append("")

    return "\n".join(lines)


def _regime_label(z: float, pctile: float) -> str:
    """Classify the indicator regime."""
    if z > 1.5 or pctile > 0.95:
        return "EXTREME HIGH"
    if z > 0.75 or pctile > 0.85:
        return "ELEVATED"
    if z < -1.5 or pctile < 0.05:
        return "EXTREME LOW"
    if z < -0.75 or pctile < 0.15:
        return "DEPRESSED"
    return "NORMAL"


# ---------------------------------------------------------------------------
# OUTPUT 3: Scenario Cards
# ---------------------------------------------------------------------------

def format_scenario_card(card: ScenarioCard) -> str:
    """Render a single scenario card as formatted text."""
    lines = []
    t = card.theme
    lines.append("=" * 80)
    lines.append(f"SCENARIO CARD: {t.name}")
    lines.append("=" * 80)
    lines.append(f"  Category:           {t.category.value}")
    lines.append(f"  Direction:          {t.direction.value}")
    lines.append(f"  Horizon:            {t.horizon.value}")
    lines.append(f"  Likelihood:         {_fmt_score(t.likelihood)}")
    lines.append(f"  Strength:           {_fmt_score(t.strength)}  [{_bar(t.strength, 20)}]")
    lines.append(f"  Confirmation:       {_fmt_score(t.confirmation_score)}")
    lines.append(f"  Historical Analogue: {t.historical_analogue or 'N/A'}")
    lines.append(f"  Evidence Items:     {len(t.evidence)}")
    lines.append("")
    lines.append("  NARRATIVE:")
    lines.append(f"    {t.narrative}")
    lines.append("")

    # Evidence summary (top items by usefulness)
    if t.evidence:
        lines.append("  SUPPORTING EVIDENCE:")
        sorted_ev = sorted(
            t.evidence, key=lambda e: e.usefulness_score, reverse=True
        )[:5]
        for e in sorted_ev:
            age = (datetime.now().date() - e.date).days
            lines.append(
                f"    [{e.source:<12s}] {e.title[:60]:<60s}  "
                f"U={_fmt_score(e.usefulness_score)}  {age}d ago"
            )
        if len(t.evidence) > 5:
            lines.append(f"    ... and {len(t.evidence) - 5} more")
        lines.append("")

    # Shock vector
    lines.append("  MACRO DRIVER SHOCKS:")
    lines.append(f"    {'Driver':<25s} {'Direction':>10s} {'Magnitude':>12s}")
    lines.append(f"    {'-'*25} {'-'*10} {'-'*12}")
    for driver in MACRO_DRIVERS:
        val = card.shock_vector.get(driver, 0.0)
        if abs(val) < 0.001:
            continue
        direction = "UP" if val > 0 else "DOWN"
        lines.append(f"    {DRIVER_LABELS.get(driver, driver):<25s} {direction:>10s} {_fmt_sigma(val):>12s}")
    lines.append("")

    # Indicator confirmation
    if card.indicator_confirmations:
        lines.append("  MARKET INDICATOR CONFIRMATION:")
        lines.append(f"    {'Family':<25s} {'Composite Z':>12s} {'Percentile':>11s} {'Consistent?':>12s}")
        lines.append(f"    {'-'*25} {'-'*12} {'-'*11} {'-'*12}")
        for ic in card.indicator_confirmations:
            lines.append(
                f"    {ic.get('family', ''):<25s} "
                f"{_fmt_sigma(ic.get('composite_z', 0)):>12s} "
                f"{_fmt_pct(ic.get('percentile', 0)):>11s} "
                f"{'YES' if ic.get('consistent', False) else 'NO':>12s}"
            )
        lines.append("")

    # Asset-class returns
    if card.asset_returns:
        lines.append("  ASSET-CLASS RETURN ESTIMATES:")
        lines.append(f"    {'Asset Class':<30s} {'Expected':>10s} {'95% CI Low':>11s} {'95% CI High':>12s}")
        lines.append(f"    {'-'*30} {'-'*10} {'-'*11} {'-'*12}")
        for ar in card.asset_returns:
            label = ASSET_CLASS_LABELS.get(ar.asset_class, ar.asset_class)
            lines.append(
                f"    {label:<30s} "
                f"{_fmt_pct(ar.scenario_return):>10s} "
                f"{_fmt_pct(ar.ci_lower):>11s} "
                f"{_fmt_pct(ar.ci_upper):>12s}"
            )
        lines.append("")

    # Key risks
    if card.key_risks:
        lines.append("  KEY RISKS TO SCENARIO:")
        for risk in card.key_risks:
            lines.append(f"    - {risk}")
        lines.append("")

    return "\n".join(lines)


def print_all_scenario_cards(cards: list[ScenarioCard]) -> str:
    """Render all scenario cards."""
    return "\n\n".join(format_scenario_card(c) for c in cards)


# ---------------------------------------------------------------------------
# OUTPUT 3: Portfolio Positioning Summary
# ---------------------------------------------------------------------------

def format_portfolio_summary(summary: PortfolioSummary) -> str:
    """Render the portfolio positioning summary as formatted text."""
    lines = []
    lines.append("=" * 100)
    lines.append("PORTFOLIO POSITIONING SUMMARY")
    lines.append("=" * 100)
    lines.append(f"  Portfolio Expected Return:  {_fmt_pct(summary.portfolio_return)}")
    lines.append(f"  Portfolio Risk (vol):       {_fmt_pct(summary.portfolio_risk)}")
    if summary.portfolio_risk > 0:
        sharpe = summary.portfolio_return / summary.portfolio_risk
        lines.append(f"  Risk-Adjusted (Sharpe):     {sharpe:.3f}")
    lines.append("")

    # Positions table
    lines.append("  POSITION RECOMMENDATIONS")
    cols = ["Asset Class", "Weight", "Scen Return", "Scen Risk",
            "Risk-Adj Ret", "MCTR", "Signal", "Key Driver"]
    lines.append("  " + "  ".join(f"{c:<16s}" for c in cols))
    lines.append("  " + "  ".join("-" * 16 for _ in cols))

    for p in summary.positions:
        label = ASSET_CLASS_LABELS.get(p.asset_class, p.asset_class)
        signal_icon = {"OVERWEIGHT": "OW+", "UNDERWEIGHT": "UW-", "NEUTRAL": " N "}.get(
            p.signal.value, p.signal.value
        )
        row = [
            f"{label:<16s}",
            f"{_fmt_pct(p.current_weight):>16s}",
            f"{_fmt_pct(p.scenario_return):>16s}",
            f"{_fmt_pct(p.scenario_risk):>16s}",
            f"{_fmt_score(p.risk_adj_return):>16s}",
            f"{_fmt_score(p.mctr):>16s}",
            f"{signal_icon:>16s}",
            f"{p.key_theme_driver:<16s}",
        ]
        lines.append("  " + "  ".join(row))
    lines.append("")

    # Signal summary
    ow = [p for p in summary.positions if p.signal.value == "OVERWEIGHT"]
    uw = [p for p in summary.positions if p.signal.value == "UNDERWEIGHT"]
    if ow or uw:
        lines.append("  SIGNAL SUMMARY")
        if ow:
            ow_names = [ASSET_CLASS_LABELS.get(p.asset_class, p.asset_class) for p in ow]
            lines.append(f"    Overweight:  {', '.join(ow_names)}")
        if uw:
            uw_names = [ASSET_CLASS_LABELS.get(p.asset_class, p.asset_class) for p in uw]
            lines.append(f"    Underweight: {', '.join(uw_names)}")
        lines.append("")

    # Hedge recommendations
    if summary.hedges:
        lines.append("  " + "-" * 80)
        lines.append("  HEDGE RECOMMENDATIONS")
        lines.append("  " + "-" * 80)
        for h in summary.hedges:
            lines.append(f"    Theme: {h.theme_name}")
            lines.append(f"      Confirmation:    {_fmt_score(h.confirmation_score)}")
            lines.append(f"      Portfolio Impact: {_fmt_pct(h.portfolio_impact)}")
            lines.append(f"      Instruments:     {', '.join(h.suggested_instruments)}")
            lines.append(f"      Rationale:       {h.rationale}")
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# OUTPUT 5: News Evidence Summary
# ---------------------------------------------------------------------------

def format_news_evidence(themes: list[Theme], max_per_theme: int = 5) -> str:
    """Render a summary of news evidence matched to each theme."""
    lines = []
    lines.append("=" * 80)
    lines.append("NEWS EVIDENCE SUMMARY")
    lines.append("=" * 80)

    total_evidence = sum(len(t.evidence) for t in themes)
    if total_evidence == 0:
        lines.append("  No news evidence ingested.")
        return "\n".join(lines)

    lines.append(f"  Total evidence items across all themes: {total_evidence}")
    lines.append("")

    for t in sorted(themes, key=lambda x: x.strength, reverse=True):
        if not t.evidence:
            continue
        lines.append(f"  {t.name} ({t.category.value}) -- {len(t.evidence)} items")
        sorted_ev = sorted(
            t.evidence, key=lambda e: e.usefulness_score, reverse=True
        )
        for e in sorted_ev[:max_per_theme]:
            age = (datetime.now().date() - e.date).days
            lines.append(
                f"    [{e.source:<10s}] {e.title[:65]:<65s} "
                f"U={_fmt_score(e.usefulness_score)}  {age}d"
            )
        if len(t.evidence) > max_per_theme:
            lines.append(f"    ... +{len(t.evidence) - max_per_theme} more")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------

def export_to_csv(
    summary: PortfolioSummary,
    themes: list[Theme],
    output_dir: str | Path = "outputs",
    composites: list[FamilyComposite] | None = None,
    overall_market_stress: float | None = None,
) -> dict[str, Path]:
    """Export all outputs to CSV files. Returns {name: path}."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths: dict[str, Path] = {}

    # Theme dashboard
    path = out / f"theme_dashboard_{ts}.csv"
    rows = theme_dashboard_table(themes)
    if rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        paths["theme_dashboard"] = path

    # Portfolio positions
    path = out / f"portfolio_positions_{ts}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Asset Class", "Weight", "Scenario Return",
                         "Scenario Risk", "Risk-Adj Return", "MCTR",
                         "Signal", "Key Theme Driver"])
        for p in summary.positions:
            writer.writerow([
                ASSET_CLASS_LABELS.get(p.asset_class, p.asset_class),
                f"{p.current_weight:.4f}",
                f"{p.scenario_return:.6f}",
                f"{p.scenario_risk:.6f}",
                f"{p.risk_adj_return:.4f}",
                f"{p.mctr:.4f}",
                p.signal.value,
                p.key_theme_driver,
            ])
    paths["portfolio_positions"] = path

    # Hedge recommendations
    if summary.hedges:
        path = out / f"hedge_recommendations_{ts}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Theme", "Confirmation", "Portfolio Impact",
                             "Instruments", "Rationale"])
            for h in summary.hedges:
                writer.writerow([
                    h.theme_name,
                    f"{h.confirmation_score:.4f}",
                    f"{h.portfolio_impact:.6f}",
                    "; ".join(h.suggested_instruments),
                    h.rationale,
                ])
        paths["hedge_recommendations"] = path

    # Indicator composites
    if composites:
        path = out / f"indicator_composites_{ts}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Family", "Composite Z", "Percentile",
                             "N Indicators", "Regime"])
            for fc in composites:
                regime = _regime_label(fc.composite_z, fc.percentile)
                writer.writerow([
                    fc.family.value,
                    f"{fc.composite_z:.4f}",
                    f"{fc.percentile:.4f}",
                    fc.n_indicators,
                    regime,
                ])
            if overall_market_stress is not None:
                writer.writerow(["OVERALL_MARKET_STRESS",
                                 f"{overall_market_stress:.4f}", "", "", ""])
        paths["indicator_composites"] = path

    # Evidence items
    all_evidence = []
    for t in themes:
        for e in t.evidence:
            all_evidence.append({
                "theme": t.name,
                "source": e.source,
                "title": e.title,
                "date": e.date.isoformat(),
                "usefulness": f"{e.usefulness_score:.4f}",
                "credibility": f"{e.credibility:.4f}",
            })
    if all_evidence:
        path = out / f"evidence_items_{ts}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_evidence[0].keys())
            writer.writeheader()
            writer.writerows(all_evidence)
        paths["evidence_items"] = path

    return paths


def export_scenario_cards_json(
    cards: list[ScenarioCard],
    output_dir: str | Path = "outputs",
) -> Path:
    """Export scenario cards to JSON."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out / f"scenario_cards_{ts}.json"

    data = []
    for card in cards:
        data.append({
            "theme": card.theme.model_dump(mode="json"),
            "shock_vector": card.shock_vector,
            "indicator_confirmations": card.indicator_confirmations,
            "asset_returns": [ar.model_dump() for ar in card.asset_returns],
            "key_risks": card.key_risks,
        })

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    return path


def generate_full_report(
    themes: list[Theme],
    summary: PortfolioSummary,
    composites: list[FamilyComposite] | None = None,
    overall_market_stress: float | None = None,
    latest_z: dict[str, float] | None = None,
    data_source: str = "bloomberg",
    themes_source: str = "file",
    news_db_path: str | None = None,
    n_articles_ingested: int = 0,
) -> str:
    """Generate the complete text report combining all outputs."""
    sections = []

    # Pipeline header
    sections.append(format_pipeline_header(
        data_source=data_source,
        themes_source=themes_source,
        news_db_path=news_db_path,
        n_themes=len(themes),
        n_articles_ingested=n_articles_ingested,
    ))

    # Theme dashboard
    sections.append(print_theme_dashboard(themes))

    # Market indicator summary
    if composites:
        sections.append(format_indicator_summary(
            composites,
            overall_market_stress or 0.0,
            latest_z,
        ))

    # Scenario cards
    sections.append(print_all_scenario_cards(summary.scenario_cards))

    # Portfolio summary
    sections.append(format_portfolio_summary(summary))

    # News evidence
    total_evidence = sum(len(t.evidence) for t in themes)
    if total_evidence > 0:
        sections.append(format_news_evidence(themes))

    return "\n\n".join(sections)
