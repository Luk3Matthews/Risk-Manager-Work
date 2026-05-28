"""Theme Engine -- Main entry point.

Run the full pipeline end-to-end using live data:
  - Themes from JSON file or auto-discovered from news_monitor DB
  - Indicators from Bloomberg BLPAPI
  - News articles from news_monitor SQLite DB

Pipeline:
  Theme -> Scenario -> Signal -> Indicator -> Asset-Class Outcome
  -> Portfolio Positioning
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from .config import get_config
from .dashboard import (
    export_scenario_cards_json,
    export_to_csv,
    generate_full_report,
)
from .factor_model import scenario_confidence_interval, scenario_returns
from .indicators import (
    DataFrameLoader,
    IndicatorDataLoader,
    adjust_shock_for_confirmation,
    compute_theme_confirmation,
    run_indicator_pipeline,
)
from .ingestion import build_active_ledger
from .models import MACRO_DRIVERS, FamilyComposite, ScenarioCard, Theme
from .news_sifter import (
    NewsMonitorReader,
    RawArticle,
    create_themes_from_articles,
    ingest_from_news_monitor,
    sift_articles,
)
from .portfolio import build_portfolio_summary
from .scenario import compute_shock_vector, run_scenario_shocks

logger = logging.getLogger(__name__)


def _build_indicator_loader(
    source: str,
    bbg_host: str = "localhost",
    bbg_port: int = 8194,
    csv_path: str | None = None,
) -> IndicatorDataLoader:
    """Build the appropriate data loader based on *source*.

    source: 'bloomberg' | 'csv'
    """
    if source == "bloomberg":
        from .bloomberg_loader import BloombergDataLoader, BLPAPI_AVAILABLE
        if not BLPAPI_AVAILABLE:
            raise RuntimeError("xbbg is not installed")
        loader = BloombergDataLoader(host=bbg_host, port=bbg_port)
        print("  [OK] Bloomberg loader ready (xbbg)")
        return loader

    if source == "csv":
        import pandas as pd
        if not csv_path:
            raise ValueError("csv_path required for CSV source")
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        print(f"  [OK] Loaded CSV data: {df.shape[1]} indicators x {df.shape[0]} rows")
        return DataFrameLoader(df)

    raise ValueError(f"Unknown indicator source: {source}")


def _load_themes(
    themes_file: str | None = None,
    news_db: str | None = None,
    auto_discover: bool = False,
    lookback_hours: int = 168,
) -> list[Theme]:
    """Load themes from a JSON file, or auto-discover from news_monitor DB.

    Parameters
    ----------
    themes_file : str | None
        Path to a JSON file containing theme definitions.
        Defaults to data/themes/example_themes.json.
    news_db : str | None
        Path to news_monitor SQLite DB (for auto-discover mode).
    auto_discover : bool
        If True, auto-discover themes from news articles instead of loading
        from file.
    lookback_hours : int
        How many hours back to look for articles (auto-discover mode).
    """
    if auto_discover:
        db = news_db or str(
            Path(__file__).parent.parent / "news_monitor" / "news_monitor.db"
        )
        print(f"  Auto-discovering themes from news_monitor DB...")
        reader = NewsMonitorReader(db)
        articles = reader.get_recent_articles(hours=lookback_hours, limit=1000)
        if not articles:
            raise RuntimeError(
                f"No articles found in {db} within last {lookback_hours}h. "
                "Run the news_monitor ingestion first, or supply --themes-file."
            )
        print(f"  Found {len(articles)} articles in last {lookback_hours}h")
        themes = create_themes_from_articles(articles, min_articles_per_theme=2)
        if not themes:
            raise RuntimeError(
                "Could not auto-discover any themes from articles. "
                "Supply themes manually via --themes-file."
            )
        return themes

    # Load from JSON file
    default_path = Path(__file__).parent / "data" / "themes" / "example_themes.json"
    path = Path(themes_file) if themes_file else default_path
    if not path.exists():
        raise FileNotFoundError(f"Themes file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    themes = [Theme.model_validate(d) for d in data]
    print(f"  Loaded {len(themes)} themes from {path.name}")
    return themes


def run_full_pipeline(
    config_dir: str | None = None,
    output_dir: str = "outputs",
    data_source: str = "bloomberg",
    bbg_host: str = "localhost",
    bbg_port: int = 8194,
    csv_path: str | None = None,
    themes_file: str | None = None,
    news_db: str | None = None,
    auto_discover_themes: bool = False,
    news_lookback_hours: int = 168,
    verbose: bool = True,
) -> None:
    """Execute the complete engine pipeline with live data.

    Parameters
    ----------
    config_dir : str | None
        Path to config directory (defaults to built-in).
    output_dir : str
        Output directory for CSV/JSON exports.
    data_source : str
        'bloomberg' (default) or 'csv'.
    bbg_host : str
        Bloomberg terminal host.
    bbg_port : int
        Bloomberg terminal port.
    csv_path : str | None
        Path to CSV file (only used when data_source='csv').
    themes_file : str | None
        Path to themes JSON file (default: data/themes/example_themes.json).
    news_db : str | None
        Path to news_monitor SQLite DB.
    auto_discover_themes : bool
        If True, auto-discover themes from news articles.
    news_lookback_hours : int
        Hours of news history to scan (default 168 = 1 week).
    verbose : bool
        Print progress to stdout.

    Steps:
      1. Load configuration
      2. Load themes (from file or auto-discover from news)
      3. Load indicator data from Bloomberg
      4. Build active theme ledger (score evidence, compute strengths)
      5. Ingest live news articles to enrich theme evidence
      6. Compute macro driver shock vectors per theme
      7. Process market indicators & compute family composites
      8. Compute theme confirmation scores
      9. Adjust shocks for market confirmation
     10. Aggregate shocks across themes
     11. Compute asset-class scenario returns
     12. Build portfolio summary with positions & hedge recommendations
     13. Generate dashboard outputs & export
    """
    cfg = get_config(config_dir)
    print("=" * 72)
    print("THEME-DRIVEN MACRO SCENARIO ENGINE  [LIVE]")
    print("=" * 72)
    print()

    # ---- Step 1: Load themes ----
    print("[1/11] Loading themes...")
    themes = _load_themes(
        themes_file=themes_file,
        news_db=news_db,
        auto_discover=auto_discover_themes,
        lookback_hours=news_lookback_hours,
    )
    for t in themes:
        print(f"  - {t.name} ({t.category.value}) - {t.direction.value}")

    # ---- Step 2: Build active ledger ----
    print("\n[2/11] Building active theme ledger (scoring evidence)...")
    active_themes = build_active_ledger(themes, cfg=cfg)
    for t in active_themes:
        print(f"  - {t.name}: strength={t.strength:.3f}, "
              f"confidence={t.confidence:.3f}, likelihood={t.likelihood:.2f}")

    # ---- Step 3: Ingest live news articles ----
    print("\n[3/11] Ingesting live news articles from news_monitor DB...")
    db_path = news_db or str(
        Path(__file__).parent.parent / "news_monitor" / "news_monitor.db"
    )
    news_counts: dict[str, int] = {}
    try:
        news_counts = ingest_from_news_monitor(
            active_themes,
            db_path=db_path,
            hours=news_lookback_hours,
            cfg=cfg,
        )
        total_articles = sum(news_counts.values())
        if total_articles > 0:
            print(f"  Ingested {total_articles} article-theme matches:")
            for theme in active_themes:
                count = news_counts.get(theme.theme_id, 0)
                if count > 0:
                    print(f"    - {theme.name}: +{count} evidence items")
            # Re-score after new evidence
            active_themes = build_active_ledger(active_themes, cfg=cfg)
            print("  Re-scored themes with new evidence")
        else:
            print("  No matching articles found -- proceeding with existing evidence")
    except FileNotFoundError:
        print(f"  WARNING: news_monitor DB not found at {db_path}")
        print("  Proceeding without live news ingestion")

    # ---- Step 4: Load indicator data ----
    print(f"\n[4/11] Loading indicator data (source={data_source})...")
    loader = _build_indicator_loader(
        source=data_source,
        bbg_host=bbg_host,
        bbg_port=bbg_port,
        csv_path=csv_path,
    )

    # ---- Step 5: Process indicators ----
    print("\n[5/11] Processing market indicators & computing composites...")
    composites, latest_z, oms = run_indicator_pipeline(loader, cfg)
    for fc in composites:
        print(f"  - {fc.family.value}: z={fc.composite_z:+.2f}, "
              f"pctile={fc.percentile:.1%}, n={fc.n_indicators}")
    print(f"  Overall Market Stress: {oms:+.3f}")

    # ---- Step 6: Compute shock vectors ----
    print("\n[6/11] Computing macro driver shock vectors...")
    individual_shocks, agg_shock = run_scenario_shocks(active_themes, cfg=cfg)
    for i, t in enumerate(active_themes):
        shock = individual_shocks[i]
        top_drivers = sorted(
            zip(MACRO_DRIVERS, shock), key=lambda x: abs(x[1]), reverse=True
        )[:3]
        top_str = ", ".join(f"{d}={v:+.2f}s" for d, v in top_drivers)
        print(f"  - {t.name}: [{top_str}]")

    # ---- Step 7: Theme confirmation ----
    print("\n[7/11] Computing theme confirmation scores...")
    confirmations = []
    for i, t in enumerate(active_themes):
        conf = compute_theme_confirmation(individual_shocks[i], latest_z)
        t.confirmation_score = conf
        confirmations.append(conf)
        label = "PRICED" if conf > 0.3 else ("OPPOSITE" if conf < -0.3 else "AGNOSTIC")
        print(f"  - {t.name}: confirmation={conf:+.3f} ({label})")

    # ---- Step 8: Adjust shocks ----
    print("\n[8/11] Adjusting shocks for market confirmation...")
    adjusted_shocks = []
    for i, shock in enumerate(individual_shocks):
        adj = adjust_shock_for_confirmation(shock, confirmations[i], cfg=cfg)
        adjusted_shocks.append(adj)

    # Re-aggregate with adjusted shocks
    from .scenario import aggregate_shocks
    agg_adjusted = aggregate_shocks(active_themes, adjusted_shocks, cfg)
    print(f"  Aggregate adjusted shock vector:")
    for d, v in zip(MACRO_DRIVERS, agg_adjusted):
        if abs(v) > 0.01:
            print(f"    {d}: {v:+.3f}s")

    # ---- Step 9: Asset-class returns ----
    print("\n[9/11] Computing asset-class scenario returns...")
    asset_returns = scenario_confidence_interval(agg_adjusted, cfg=cfg)
    for ar in asset_returns:
        print(f"  - {ar.asset_class}: return={ar.scenario_return:+.2%} "
              f"[{ar.ci_lower:+.2%}, {ar.ci_upper:+.2%}]")

    # ---- Step 10: Build scenario cards ----
    print("\n[10/11] Building scenario cards...")
    scenario_cards = []
    for i, t in enumerate(active_themes):
        # Per-theme asset returns
        theme_returns = scenario_confidence_interval(adjusted_shocks[i], cfg=cfg)

        # Indicator confirmations per family
        ic_data = []
        for fc in composites:
            consistent = (
                (fc.composite_z > 0 and confirmations[i] > 0) or
                (fc.composite_z < 0 and confirmations[i] < 0) or
                abs(confirmations[i]) < 0.1
            )
            ic_data.append({
                "family": fc.family.value,
                "composite_z": fc.composite_z,
                "percentile": fc.percentile,
                "consistent": consistent,
            })

        card = ScenarioCard(
            theme=t,
            shock_vector={d: float(adjusted_shocks[i][j])
                          for j, d in enumerate(MACRO_DRIVERS)},
            indicator_confirmations=ic_data,
            asset_returns=theme_returns,
            key_risks=_get_key_risks(t),
        )
        scenario_cards.append(card)

    # ---- Step 11: Portfolio summary ----
    print("\n[11/11] Computing portfolio positions & hedge recommendations...")
    summary = build_portfolio_summary(
        themes=active_themes,
        individual_shocks=adjusted_shocks,
        aggregate_shock=agg_adjusted,
        asset_returns=asset_returns,
        confirmations=confirmations,
        scenario_cards=scenario_cards,
        cfg=cfg,
    )

    # ---- Generate report ----
    print("\n" + "=" * 72)
    themes_source = "auto-discovered" if auto_discover_themes else (
        themes_file or "example_themes.json"
    )
    total_ingested = sum(news_counts.values()) if news_counts else 0
    report = generate_full_report(
        active_themes, summary,
        composites=composites,
        overall_market_stress=oms,
        latest_z=latest_z,
        data_source=data_source,
        themes_source=themes_source,
        news_db_path=db_path,
        n_articles_ingested=total_ingested,
    )
    print(report)

    # ---- Export ----
    print(f"\n\nExporting outputs to {output_dir}/...")
    out_path = Path(output_dir)
    csv_paths = export_to_csv(
        summary, active_themes, out_path,
        composites=composites,
        overall_market_stress=oms,
    )
    json_path = export_scenario_cards_json(scenario_cards, out_path)
    for name, path in csv_paths.items():
        print(f"  [OK] {name}: {path}")
    print(f"  [OK] scenario_cards: {json_path}")

    # Close Bloomberg session if applicable
    if hasattr(loader, "close"):
        loader.close()

    print("\n[OK] Pipeline complete.")


def _get_key_risks(theme) -> list[str]:
    """Generate key risks based on theme category."""
    risks = {
        "GEOPOLITICAL": [
            "Diplomatic resolution reduces risk premium rapidly",
            "Escalation beyond priced scenario",
            "Alternative supply routes mitigate impact",
        ],
        "GROWTH": [
            "Labour market deterioration faster than expected",
            "External demand shock (China/EU slowdown)",
            "Fiscal cliff if stimulus expires",
        ],
        "INFLATION": [
            "Supply-side resolution (oil, shelter) brings faster disinflation",
            "De-anchoring of expectations triggers policy overshoot",
            "Wage-price spiral risk in services",
        ],
        "LIQUIDITY": [
            "Fed intervenes early to stabilise repo markets",
            "Bank reserve buffer larger than estimated",
            "Treasury refunding mix reduces RRP drain",
        ],
        "VALUATION": [
            "AI revenue growth exceeds expectations, justifying multiples",
            "Earnings rotation to value/cyclical sectors",
            "Multiple compression if rates stay higher for longer",
        ],
    }
    return risks.get(theme.category.value, ["Scenario does not materialise"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Theme-Driven Macro Scenario Engine (LIVE)",
    )
    parser.add_argument(
        "--source", "-s",
        choices=["bloomberg", "csv"],
        default="bloomberg",
        help="Indicator data source (default: bloomberg)",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Bloomberg terminal host (default: localhost)",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8194,
        help="Bloomberg terminal port (default: 8194)",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to CSV file (when --source=csv)",
    )
    parser.add_argument(
        "--themes-file",
        default=None,
        help="Path to themes JSON file (default: built-in themes)",
    )
    parser.add_argument(
        "--news-db",
        default=None,
        help="Path to news_monitor SQLite DB",
    )
    parser.add_argument(
        "--auto-discover",
        action="store_true",
        help="Auto-discover themes from news articles instead of loading from file",
    )
    parser.add_argument(
        "--news-lookback",
        type=int,
        default=168,
        help="Hours of news history to scan (default: 168 = 1 week)",
    )
    parser.add_argument(
        "--output", "-o",
        default="outputs",
        help="Output directory (default: outputs)",
    )
    args = parser.parse_args()
    run_full_pipeline(
        data_source=args.source,
        bbg_host=args.host,
        bbg_port=args.port,
        csv_path=args.csv,
        themes_file=args.themes_file,
        news_db=args.news_db,
        auto_discover_themes=args.auto_discover,
        news_lookback_hours=args.news_lookback,
        output_dir=args.output,
        verbose=True,
    )
