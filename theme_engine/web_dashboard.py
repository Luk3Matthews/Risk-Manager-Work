"""Theme Engine -- Interactive Web Dashboard (Streamlit).

Launch:
    streamlit run theme_engine/web_dashboard.py

Connects to the same pipeline as the CLI but renders results
in an interactive browser-based dashboard.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── Ensure the project root is importable ──────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from theme_engine.config import get_config
from theme_engine.factor_model import scenario_confidence_interval
from theme_engine.indicators import (
    DataFrameLoader,
    adjust_shock_for_confirmation,
    compute_theme_confirmation,
    run_indicator_pipeline,
)
from theme_engine.ingestion import build_active_ledger
from theme_engine.models import (
    ASSET_CLASS_CATEGORIES,
    ASSET_CLASS_LABELS,
    ASSET_CLASSES,
    DRIVER_LABELS,
    MACRO_DRIVERS,
    FamilyComposite,
    PortfolioSummary,
    ScenarioCard,
    Theme,
)
from theme_engine.news_sifter import ingest_from_news_monitor
from theme_engine.portfolio import build_portfolio_summary
from theme_engine.scenario import aggregate_shocks, compute_shock_vector, run_scenario_shocks
from theme_engine.synthetic_data import generate_synthetic_indicators

# ── Page config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Macro Scenario Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; max-width: 1400px; }
    div[data-testid="stMetric"] {
        background: #0e1117; border: 1px solid #262730;
        border-radius: 8px; padding: 12px 16px;
    }
    .risk-high { color: #ff4b4b; font-weight: 700; }
    .risk-med  { color: #ffa726; font-weight: 700; }
    .risk-low  { color: #66bb6a; font-weight: 700; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px; border-radius: 6px 6px 0 0;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# Sidebar — Configuration
# =====================================================================
def sidebar_config() -> dict:
    """Render sidebar controls and return configuration dict."""
    st.sidebar.title("⚙️ Engine Configuration")

    data_source = st.sidebar.selectbox(
        "Indicator Data Source",
        ["bloomberg", "csv"],
        index=0,
        help="Bloomberg pulls live data via BLPAPI. "
             "CSV loads from a file.",
    )

    bbg_host = "localhost"
    bbg_port = 8194
    csv_path = None

    if data_source == "bloomberg":
        with st.sidebar.expander("Bloomberg Settings"):
            bbg_host = st.text_input("Host", "localhost")
            bbg_port = st.number_input("Port", value=8194, step=1)
    elif data_source == "csv":
        csv_path = st.sidebar.text_input("CSV Path", "")

    st.sidebar.divider()

    theme_source = st.sidebar.selectbox(
        "Theme Source",
        ["Live News (bottom-up)", "JSON file"],
        index=0,
        help="'Live News' auto-discovers themes from news_monitor DB. "
             "'JSON file' loads from a static theme file.",
    )
    themes_file = None
    if theme_source == "JSON file":
        themes_file = st.sidebar.text_input(
            "Themes JSON File",
            "",
            help="Leave blank to use built-in example themes",
        )

    news_lookback = st.sidebar.slider(
        "News Lookback (hours)", 24, 720, 168, step=24,
        help="How far back to scan for articles",
    )

    with st.sidebar.expander("News Ingestion"):
        refresh_news = st.checkbox(
            "Fetch fresh articles on run",
            value=False,
            help="Pull latest articles from GDELT before generating themes (adds ~30s)",
        )

    st.sidebar.divider()
    run = st.sidebar.button("🚀 Run Pipeline", type="primary", width="stretch")

    return {
        "data_source": data_source,
        "bbg_host": bbg_host,
        "bbg_port": bbg_port,
        "csv_path": csv_path or None,
        "theme_source": theme_source,
        "themes_file": themes_file or None,
        "news_lookback": news_lookback,
        "refresh_news": refresh_news,
        "run": run,
    }


# =====================================================================
# News Refresh Helper
# =====================================================================
def _refresh_news_db(db_path: str, lookback_hours: int = 168) -> int:
    """Fetch fresh articles from GDELT and store in the news_monitor DB."""
    import sys
    import time
    from pathlib import Path

    nm_path = str(Path(__file__).parent.parent / "news_monitor")
    if nm_path not in sys.path:
        sys.path.insert(0, nm_path)

    try:
        from src import db as nm_db
        from src import news_gdelt
        from src.tagger import tag_article
        from src.taxonomy import get_taxonomy
    except ImportError:
        return 0

    import yaml
    config_path = Path(__file__).parent.parent / "news_monitor" / "config.yaml"
    if not config_path.exists():
        return 0

    config = yaml.safe_load(open(config_path))
    queries = config.get("queries", [])

    nm_db.init_db(Path(db_path))
    tax = get_taxonomy()

    total = 0
    for i, q in enumerate(queries):
        try:
            articles = news_gdelt.fetch_articles(
                q, max_records=10, timespan=str(lookback_hours * 60),
            )
        except Exception:
            articles = []
        for a in articles:
            aid = nm_db.insert_article(
                source="gdelt",
                title=a["title"],
                snippet=a["snippet"],
                url=a["url"],
                published_at=a["published_at"],
                query=q,
            )
            if aid:
                tags = tag_article(
                    title=a["title"],
                    snippet=a["snippet"],
                    source_reliability=0.6,
                    taxonomy_instance=tax,
                )
                if tags:
                    nm_db.insert_tags(aid, tags)
                total += 1
        if i < len(queries) - 1:
            time.sleep(1)  # GDELT rate limit
    return total


# =====================================================================
# Pipeline Runner
# =====================================================================
@st.cache_resource(show_spinner="Running pipeline...", ttl=300)
def run_pipeline(
    data_source: str,
    bbg_host: str,
    bbg_port: int,
    csv_path: str | None,
    theme_source: str = "Live News (bottom-up)",
    themes_file: str | None = None,
    news_lookback: int = 168,
    refresh_news: bool = True,
) -> dict:
    """Run the full pipeline and return all results as a dict."""

    cfg = get_config()
    db_path = str(Path(__file__).parent.parent / "news_monitor" / "news_monitor.db")

    # ── Optionally refresh news from GDELT ──
    if refresh_news:
        _refresh_news_db(db_path, news_lookback)

    # ── Load themes ──
    news_counts: dict[str, int] = {}
    if theme_source == "Live News (bottom-up)":
        # Bottom-up: auto-discover themes from live news articles
        from theme_engine.news_sifter import (
            NewsMonitorReader,
            create_themes_from_articles,
        )
        try:
            reader = NewsMonitorReader(db_path)
            articles = reader.get_recent_articles(
                hours=news_lookback, limit=1000,
            )
            if articles:
                themes = create_themes_from_articles(
                    articles, min_articles_per_theme=2, cfg=cfg,
                )
                news_counts = {t.theme_id: len(t.evidence) for t in themes}
            else:
                themes = []
        except FileNotFoundError:
            themes = []

        if not themes:
            raise RuntimeError(
                "No themes could be auto-discovered from news. "
                "Run the news_monitor ingestion first, or switch to "
                "'JSON file' theme source."
            )
    else:
        # Top-down: load from JSON file
        from theme_engine.__main__ import _load_themes
        themes = _load_themes(themes_file=themes_file)

        # Enrich static themes with news evidence
        try:
            news_counts = ingest_from_news_monitor(
                themes, db_path=db_path, hours=news_lookback, cfg=cfg,
            )
        except FileNotFoundError:
            pass

    # ── Build active ledger ──
    active_themes = build_active_ledger(themes, cfg=cfg)

    # ── Load indicators ──
    from theme_engine.__main__ import _build_indicator_loader
    try:
        loader = _build_indicator_loader(
            source=data_source, bbg_host=bbg_host,
            bbg_port=bbg_port, csv_path=csv_path,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load {data_source} data: {exc}\n\n"
            "If using Bloomberg, ensure blpapi is installed and the terminal is running."
        ) from exc

    # ── Process indicators ──
    composites, latest_z, oms = run_indicator_pipeline(loader, cfg)

    # ── Shocks ──
    individual_shocks, agg_shock = run_scenario_shocks(active_themes, cfg=cfg)

    # ── Confirmation ──
    confirmations = []
    for i, t in enumerate(active_themes):
        conf = compute_theme_confirmation(individual_shocks[i], latest_z)
        t.confirmation_score = conf
        confirmations.append(conf)

    # ── Adjust shocks ──
    adjusted_shocks = []
    for i, shock in enumerate(individual_shocks):
        adj = adjust_shock_for_confirmation(shock, confirmations[i], cfg=cfg)
        adjusted_shocks.append(adj)

    agg_adjusted = aggregate_shocks(active_themes, adjusted_shocks, cfg)

    # ── Asset returns ──
    asset_returns = scenario_confidence_interval(agg_adjusted, cfg=cfg)

    # ── Scenario cards ──
    from theme_engine.__main__ import _get_key_risks
    scenario_cards = []
    for i, t in enumerate(active_themes):
        theme_returns = scenario_confidence_interval(adjusted_shocks[i], cfg=cfg)
        ic_data = []
        for fc in composites:
            consistent = (
                (fc.composite_z > 0 and confirmations[i] > 0)
                or (fc.composite_z < 0 and confirmations[i] < 0)
                or abs(confirmations[i]) < 0.1
            )
            ic_data.append({
                "family": fc.family.value,
                "composite_z": fc.composite_z,
                "percentile": fc.percentile,
                "consistent": consistent,
            })
        card = ScenarioCard(
            theme=t,
            shock_vector={d: float(adjusted_shocks[i][j]) for j, d in enumerate(MACRO_DRIVERS)},
            indicator_confirmations=ic_data,
            asset_returns=theme_returns,
            key_risks=_get_key_risks(t),
        )
        scenario_cards.append(card)

    # ── Portfolio ──
    summary = build_portfolio_summary(
        themes=active_themes,
        individual_shocks=adjusted_shocks,
        aggregate_shock=agg_adjusted,
        asset_returns=asset_returns,
        confirmations=confirmations,
        scenario_cards=scenario_cards,
        cfg=cfg,
    )

    # Close Bloomberg session
    if hasattr(loader, "close"):
        loader.close()

    return {
        "themes": active_themes,
        "composites": composites,
        "latest_z": latest_z,
        "oms": oms,
        "individual_shocks": individual_shocks,
        "adjusted_shocks": adjusted_shocks,
        "agg_adjusted": agg_adjusted,
        "confirmations": confirmations,
        "asset_returns": asset_returns,
        "scenario_cards": scenario_cards,
        "summary": summary,
        "news_counts": news_counts,
        "data_source": data_source,
        "run_time": datetime.now().isoformat(),
    }


# =====================================================================
# Rendering Helpers
# =====================================================================

CATEGORY_COLORS = {
    "GEOPOLITICAL": "#ef5350",
    "GROWTH": "#66bb6a",
    "INFLATION": "#ffa726",
    "LIQUIDITY": "#42a5f5",
    "STRUCTURAL": "#ab47bc",
    "POLICY": "#26c6da",
    "VALUATION": "#ec407a",
    "CONTAGION": "#ff7043",
}

DIRECTION_ICONS = {"BULLISH": "🟢", "BEARISH": "🔴", "AMBIGUOUS": "🟡"}

SIGNAL_COLORS = {
    "OVERWEIGHT": "#66bb6a",
    "UNDERWEIGHT": "#ef5350",
    "NEUTRAL": "#9e9e9e",
}


def regime_label(z: float, pct: float) -> tuple[str, str]:
    """Return (label, css_class) for an indicator regime."""
    if z > 1.5 or pct > 0.95:
        return "EXTREME HIGH", "risk-high"
    if z > 0.75 or pct > 0.85:
        return "ELEVATED", "risk-med"
    if z < -1.5 or pct < 0.05:
        return "EXTREME LOW", "risk-high"
    if z < -0.75 or pct < 0.15:
        return "DEPRESSED", "risk-med"
    return "NORMAL", "risk-low"


# =====================================================================
# Emerging Risk Factors — theme-driven grouping
# =====================================================================

EMERGING_RISK_FACTORS = [
    {
        "id": "conflict",
        "label": "Conflict",
        "icon": "⚔️",
        "categories": {"GEOPOLITICAL"},
        "keywords": [
            "conflict", "war", "military", "sanctions", "escalation",
            "missile", "invasion", "ceasefire", "diplomatic", "nato",
            "iran", "israel", "russia", "ukraine", "taiwan", "strait",
            "defense", "defence", "territorial", "artillery", "nuclear",
            "proxy war", "blockade", "airstrikes",
        ],
        "primary_drivers": [
            "equity_risk_premium", "fx_risk_appetite", "policy_uncertainty",
        ],
        "colour": "#ef5350",
    },
    {
        "id": "supply_chain",
        "label": "Supply Chain",
        "icon": "🔗",
        "categories": {"GEOPOLITICAL", "INFLATION", "GROWTH", "STRUCTURAL"},
        "keywords": [
            "supply chain", "shipping", "freight", "port", "disruption",
            "commodity", "oil", "energy", "hormuz", "trade route",
            "semiconductor", "rare earth", "cobalt", "mining", "logistics",
            "embargo", "export ban", "chokepoint", "pipeline",
            "food supply", "grain", "fertiliser", "fertilizer",
        ],
        "primary_drivers": [
            "commodity_supply", "expected_inflation", "expected_growth",
        ],
        "colour": "#ffa726",
    },
    {
        "id": "health",
        "label": "Health",
        "icon": "🏥",
        "categories": {"CONTAGION"},
        "keywords": [
            "pandemic", "virus", "outbreak", "ebola", "huntervirus",
            "who", "quarantine", "vaccine", "pathogen", "zoonotic",
            "epidemic", "infection", "mortality", "health emergency",
            "lockdown", "travel ban", "biosecurity", "disease",
        ],
        "primary_drivers": [
            "expected_growth", "equity_risk_premium", "liquidity",
        ],
        "colour": "#66bb6a",
    },
    {
        "id": "inflation",
        "label": "Inflation",
        "icon": "📈",
        "categories": {"INFLATION", "POLICY"},
        "keywords": [
            "inflation", "price", "cost", "wage", "cpi", "ppi",
            "interest rate", "central bank", "fed", "ecb", "rba",
            "monetary", "tightening", "hawkish", "stagflation",
            "food price", "energy cost", "fuel", "rate hike",
            "bond yield", "term premium", "dovish",
        ],
        "primary_drivers": [
            "expected_inflation", "real_rates", "policy_uncertainty",
        ],
        "colour": "#42a5f5",
    },
]


def _map_themes_to_risk_factors(
    themes: list,
) -> dict[str, list]:
    """Map each theme to one or more Emerging Risk Factors.

    A theme matches a risk factor if:
      1. Its ThemeCategory is in the factor's category set, OR
      2. Any factor keyword appears in the theme's name or narrative.

    Returns {risk_factor_id: [theme, ...]} preserving strength-descending order.
    """
    mapping: dict[str, list] = {rf["id"]: [] for rf in EMERGING_RISK_FACTORS}
    unmapped: list = []

    for t in sorted(themes, key=lambda x: x.strength, reverse=True):
        matched = False
        searchable = (t.name + " " + (t.narrative or "")).lower()
        for rf in EMERGING_RISK_FACTORS:
            if t.category.value in rf["categories"]:
                mapping[rf["id"]].append(t)
                matched = True
                continue
            for kw in rf["keywords"]:
                if kw in searchable:
                    mapping[rf["id"]].append(t)
                    matched = True
                    break
        if not matched:
            unmapped.append(t)

    return mapping, unmapped


# =====================================================================
# Narrative helpers
# =====================================================================

def _top_evidence(theme, n=4):
    """Return top evidence items (English, URLs preferred)."""
    seen_titles: set[str] = set()
    out = []
    items = sorted(
        theme.evidence,
        key=lambda x: (x.url is not None, x.usefulness_score),
        reverse=True,
    )
    for e in items:
        title = e.title.strip()
        if not title or title in seen_titles:
            continue
        if not all(ord(c) < 128 for c in title[:30]):
            continue
        seen_titles.add(title)
        out.append(e)
        if len(out) >= n:
            break
    return out


def _format_evidence_list(evidence_items) -> str:
    lines = []
    for e in evidence_items:
        title = e.title.strip().rstrip(".")
        if e.url:
            lines.append(f"  - [{title}]({e.url})")
        else:
            lines.append(f"  - {title}")
    return "\n".join(lines)


def _regime_label_inline(z: float, pct: float) -> str:
    if z > 1.5 or pct > 0.95:
        return "at extreme highs"
    if z > 0.75 or pct > 0.85:
        return "elevated"
    if z < -1.5 or pct < 0.05:
        return "at extreme lows"
    if z < -0.75 or pct < 0.15:
        return "depressed"
    return "within normal range"


def _compute_risk_factor_shocks(
    rf_themes: list,
    all_themes: list,
    adjusted_shocks: list,
) -> np.ndarray:
    """Sum strength-weighted adjusted shocks for themes in a risk factor."""
    total = np.zeros(len(MACRO_DRIVERS), dtype=np.float64)
    for t in rf_themes:
        idx = next((i for i, at in enumerate(all_themes) if at.theme_id == t.theme_id), None)
        if idx is not None:
            total += t.strength * adjusted_shocks[idx]
    return total


# =====================================================================
# Page: Header
# =====================================================================


def render_header(results: dict):
    st.markdown("# 📊 Theme-Driven Macro Scenario Engine")
    ts = results.get("run_time", "")
    src = results.get("data_source", "unknown")
    n_themes = len(results["themes"])
    n_articles = sum(results["news_counts"].values())

    cols = st.columns(4)
    cols[0].metric("Data Source", src.upper())
    cols[1].metric("Active Themes", n_themes)
    cols[2].metric("News Articles Matched", n_articles)
    cols[3].metric("Overall Market Stress", f"{results['oms']:+.3f}σ")

    stress = results["oms"]
    if stress > 1.0:
        st.error("🚨 Market stress is EXTREME — consider de-risking")
    elif stress > 0.5:
        st.warning("⚠️ Market stress is ELEVATED — review hedges carefully")

    st.caption(f"Last run: {ts}")


# =====================================================================
# Page: Emerging Risks (Information Funnel)
# =====================================================================
def render_emerging_risks(results: dict):
    """Standalone emerging-risks page — the information funnel.

    Themes discovered from news drive the search for emerging risks.
    Themes are grouped into Emerging Risk Factors (Conflict, Supply Chain,
    Health, Inflation), and each factor shows its contributing themes,
    macro transmission, price-signal confirmation, and portfolio impact.
    """
    st.markdown("## 🔍 Emerging Risks — Information Funnel")
    st.caption(
        "Themes auto-discovered from news drive the identification of emerging risks. "
        "Each risk factor aggregates its contributing themes and shows the macro "
        "transmission channels, price-signal confirmation, and portfolio impact."
    )

    themes = results["themes"]
    composites = results["composites"]
    agg = results["agg_adjusted"]
    oms = results["oms"]
    asset_returns = results["asset_returns"]
    adjusted_shocks = results.get("adjusted_shocks", [])
    summary = results.get("summary")

    # ── Map themes → risk factors ──────────────────────────────────
    rf_mapping, unmapped = _map_themes_to_risk_factors(themes)

    # ── 1. Top-Level Dashboard ─────────────────────────────────────
    st.markdown("### Risk Dashboard")
    sorted_themes = sorted(themes, key=lambda t: t.strength, reverse=True)
    worst_asset = min(asset_returns, key=lambda a: a.scenario_return)
    best_asset = max(asset_returns, key=lambda a: a.scenario_return)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(
        "Market Stress",
        f"{oms:+.2f}σ",
        delta="ELEVATED" if oms > 0.5 else "Normal",
        delta_color="inverse" if oms > 0.5 else "off",
    )
    m2.metric("Active Themes", len(themes))
    # Count non-empty risk factors
    active_rfs = sum(1 for rf in EMERGING_RISK_FACTORS if rf_mapping[rf["id"]])
    m3.metric("Active Risk Factors", f"{active_rfs} / {len(EMERGING_RISK_FACTORS)}")
    m4.metric(
        "Most Exposed",
        ASSET_CLASS_LABELS.get(worst_asset.asset_class, worst_asset.asset_class)[:18],
        delta=f"{worst_asset.scenario_return:+.1%}",
        delta_color="inverse",
    )
    m5.metric(
        "Best Hedge",
        ASSET_CLASS_LABELS.get(best_asset.asset_class, best_asset.asset_class)[:18],
        delta=f"{best_asset.scenario_return:+.1%}",
        delta_color="normal",
    )

    st.divider()

    # ── 2. Indicator Regime Snapshot ───────────────────────────────
    st.markdown("### Indicator Regime Snapshot")
    regime_cols = st.columns(len(composites))
    for i, fc in enumerate(composites):
        label, css = regime_label(fc.composite_z, fc.percentile)
        with regime_cols[i]:
            st.metric(
                fc.family.value.replace("_", " ").title(),
                f"{fc.composite_z:+.2f}σ",
                delta=f"{fc.percentile:.0%} pctile",
            )
            st.markdown(
                f"<span class='{css}'>{label}</span>",
                unsafe_allow_html=True,
            )

    st.divider()

    # ── 3. Risk Factor Sections ────────────────────────────────────
    for rf in EMERGING_RISK_FACTORS:
        rf_themes = rf_mapping[rf["id"]]
        if not rf_themes:
            continue

        total_evidence = sum(len(t.evidence) for t in rf_themes)
        max_strength = max(t.strength for t in rf_themes)

        st.markdown(f"### {rf['icon']} {rf['label']}")

        # Summary metrics for this risk factor
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Contributing Themes", len(rf_themes))
        rc2.metric("Peak Strength", f"{max_strength:.3f}")
        rc3.metric("Evidence Items", total_evidence)

        # Narrative: list each contributing theme
        for t in rf_themes:
            icon = DIRECTION_ICONS.get(t.direction.value, "")
            ev = _top_evidence(t, 3)

            parts = []
            parts.append(
                f"**{icon} {t.name}** — "
                f"*{t.category.value}* · {t.direction.value.lower()} · "
                f"strength {t.strength:.3f} · confirmation {t.confirmation_score:+.3f} · "
                f"{len(t.evidence)} evidence items"
            )
            if t.narrative:
                parts.append(f"> {t.narrative}")
            if ev:
                parts.append("**Key articles:**\n" + _format_evidence_list(ev))

            st.markdown("\n\n".join(parts))

        # ── Health risk factor: standing analyst commentary ────────
        if rf["id"] == "health":
            st.markdown("---")
            st.markdown(
                "**Huntervirus** — A novel zoonotic pathogen first identified in "
                "late 2025, with clusters reported across Southeast Asia and sporadic "
                "cases in Europe. Human-to-human transmission rates remain uncertain, "
                "but the WHO has flagged it as a pathogen of pandemic potential. "
                "Early-stage containment measures — including targeted travel screening "
                "and quarantine protocols — are being deployed, but the lack of an "
                "approved vaccine or therapeutic creates tail-risk uncertainty. "
                "If sustained community transmission is confirmed, the macro impact "
                "would propagate through: (i) supply chain disruption in Asian "
                "manufacturing hubs, (ii) a sharp repricing of travel, hospitality, "
                "and consumer discretionary equities, and (iii) a flight-to-quality "
                "bid in sovereign bonds and gold. The CONTAGION transmission vector "
                "in our factor model implies a −0.8σ growth shock, +0.9σ equity risk "
                "premium widening, and −0.7σ liquidity drain under a materialisation "
                "scenario."
            )
            st.markdown(
                "**Ebola resurgence** — The WHO has confirmed a new Ebola outbreak "
                "in the Democratic Republic of Congo, with cases also reported in "
                "neighbouring Uganda and Rwanda. While Ebola's transmission dynamics "
                "differ from respiratory pathogens (requiring direct contact with "
                "bodily fluids), the financial implications are meaningful through "
                "several channels: disruption to cobalt and coltan mining operations "
                "in the DRC (critical for EV batteries and semiconductors), potential "
                "aid-flow redirection that pressures fiscal balances in affected "
                "nations, and a broader risk-off sentiment in frontier and emerging "
                "market allocations. Historical precedent from the 2014–2016 West "
                "Africa outbreak shows localised GDP contractions of 2–5% in affected "
                "countries, with spillovers to global commodity supply chains."
            )
            st.markdown(
                "**Combined assessment:** Neither threat is currently priced into "
                "base-case scenarios, which creates asymmetric downside risk. "
                "Monitoring points: WHO situation reports, airline booking data for "
                "early demand signals, pharmaceutical sector flows (potential "
                "beneficiaries include vaccine platform companies), and VIX/MOVE for "
                "any contagion-driven volatility spike. If either threat escalates to "
                "pandemic status, the portfolio should rotate toward: long duration "
                "sovereign bonds, gold, healthcare equities, and reduced exposure to "
                "EM, travel, and consumer discretionary."
            )

        # Macro driver transmission for this risk factor
        if adjusted_shocks:
            rf_shock = _compute_risk_factor_shocks(
                rf_themes, themes, adjusted_shocks,
            )
            # Only show drivers with meaningful shocks
            sig_drivers = [
                (DRIVER_LABELS.get(d, d), float(rf_shock[i]))
                for i, d in enumerate(MACRO_DRIVERS)
                if abs(rf_shock[i]) > 0.005
            ]
            if sig_drivers:
                sig_drivers.sort(key=lambda x: abs(x[1]), reverse=True)
                with st.expander(
                    f"📊 Macro transmission — {rf['label']}",
                    expanded=False,
                ):
                    tx_df = pd.DataFrame(sig_drivers, columns=["Driver", "Shock (σ)"])
                    st.bar_chart(
                        tx_df.set_index("Driver"),
                        horizontal=True,
                    )

        st.divider()

    # ── 4. Unmapped themes ─────────────────────────────────────────
    if unmapped:
        with st.expander(f"Other themes ({len(unmapped)})"):
            for t in unmapped:
                icon = DIRECTION_ICONS.get(t.direction.value, "")
                st.markdown(
                    f"- {icon} **{t.name}** — {t.category.value} · "
                    f"strength {t.strength:.3f} · {len(t.evidence)} evidence"
                )

    # ── 5. Aggregate Macro Driver Shocks ───────────────────────────
    st.markdown("### Aggregate Macro Driver Shocks")
    shock_left, shock_right = st.columns([2, 1])
    with shock_left:
        shock_df = pd.DataFrame({
            "Driver": [DRIVER_LABELS.get(d, d) for d in MACRO_DRIVERS],
            "Shock (σ)": [float(v) for v in agg],
        }).sort_values("Shock (σ)")
        st.bar_chart(shock_df.set_index("Driver"), horizontal=True)

    with shock_right:
        st.dataframe(
            shock_df.sort_values("Shock (σ)", key=abs, ascending=False)
                .style.background_gradient(
                    subset=["Shock (σ)"], cmap="RdBu_r", vmin=-2, vmax=2,
                )
                .format({"Shock (σ)": "{:+.3f}σ"}),
            width="stretch",
            hide_index=True,
        )

    st.divider()

    # ── 6. Portfolio Impact ────────────────────────────────────────
    st.markdown("### Portfolio Impact — Asset-Class Scenario Returns")
    ar_left, ar_right = st.columns([2, 1])

    ar_rows = []
    for ar in sorted(asset_returns, key=lambda a: a.scenario_return):
        label = ASSET_CLASS_LABELS.get(ar.asset_class, ar.asset_class)
        ar_rows.append({
            "Asset Class": label,
            "Expected Return": ar.scenario_return,
            "CI Low (95%)": ar.ci_lower,
            "CI High (95%)": ar.ci_upper,
        })
    ar_df = pd.DataFrame(ar_rows)

    with ar_left:
        chart_df = ar_df.set_index("Asset Class")[["Expected Return"]]
        st.bar_chart(chart_df, horizontal=True)

    with ar_right:
        st.dataframe(
            ar_df.style
                .background_gradient(
                    subset=["Expected Return"], cmap="RdYlGn", vmin=-0.1, vmax=0.1,
                )
                .format({
                    "Expected Return": "{:+.2%}",
                    "CI Low (95%)": "{:+.2%}",
                    "CI High (95%)": "{:+.2%}",
                }),
            width="stretch",
            hide_index=True,
        )

    # ── 7. Positioning Signals ─────────────────────────────────────
    if summary and hasattr(summary, "positions"):
        ow = [p for p in summary.positions if p.signal.value == "OVERWEIGHT"]
        uw = [p for p in summary.positions if p.signal.value == "UNDERWEIGHT"]
        if ow or uw:
            st.divider()
            st.markdown("### Positioning Signals")
            s1, s2 = st.columns(2)
            if ow:
                s1.success(
                    "**Overweight:** "
                    + ", ".join(
                        ASSET_CLASS_LABELS.get(p.asset_class, p.asset_class)
                        for p in ow
                    )
                )
            if uw:
                s2.error(
                    "**Underweight:** "
                    + ", ".join(
                        ASSET_CLASS_LABELS.get(p.asset_class, p.asset_class)
                        for p in uw
                    )
                )


# =====================================================================
# Page: Theme Dashboard
# =====================================================================
def render_themes(results: dict):
    st.markdown("## 📋 Active Themes")

    themes = results["themes"]

    # Summary table
    rows = []
    for t in sorted(themes, key=lambda x: x.strength, reverse=True):
        rows.append({
            "Theme": t.name,
            "Category": t.category.value,
            "Direction": f"{DIRECTION_ICONS.get(t.direction.value, '')} {t.direction.value}",
            "Horizon": t.horizon.value,
            "Likelihood": t.likelihood,
            "Strength": t.strength,
            "Confirmation": t.confirmation_score,
            "Evidence": len(t.evidence),
            "Status": t.status.value,
        })
    df = pd.DataFrame(rows)

    st.dataframe(
        df.style.background_gradient(subset=["Strength"], cmap="RdYlGn", vmin=0, vmax=1)
               .background_gradient(subset=["Confirmation"], cmap="RdBu", vmin=-1, vmax=1)
               .format({"Likelihood": "{:.2f}", "Strength": "{:.3f}", "Confirmation": "{:+.3f}"}),
        width="stretch",
        hide_index=True,
    )

    # Strength bar chart
    chart_df = pd.DataFrame({
        "Theme": [t.name for t in themes],
        "Strength": [t.strength for t in themes],
        "Category": [t.category.value for t in themes],
    }).sort_values("Strength", ascending=True)

    st.bar_chart(chart_df.set_index("Theme")["Strength"], horizontal=True)

    # Expandable detail for each theme
    st.markdown("### Theme Details")
    for t in sorted(themes, key=lambda x: x.strength, reverse=True):
        icon = DIRECTION_ICONS.get(t.direction.value, "")
        with st.expander(f"{icon} {t.name}  —  {t.category.value}  |  Strength: {t.strength:.3f}"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Likelihood", f"{t.likelihood:.0%}")
            c2.metric("Confidence", f"{t.confidence:.3f}")
            c3.metric("Confirmation", f"{t.confirmation_score:+.3f}")

            st.markdown(f"**Narrative:** {t.narrative}")
            if t.historical_analogue:
                st.markdown(f"**Historical Analogue:** {t.historical_analogue}")

            if t.evidence:
                st.markdown(f"**Evidence ({len(t.evidence)} items):**")
                ev_df = pd.DataFrame([
                    {
                        "Source": e.source,
                        "Title": e.title[:80],
                        "Usefulness": e.usefulness_score,
                        "Credibility": e.credibility_score,
                        "Date": e.date.isoformat(),
                    }
                    for e in sorted(t.evidence, key=lambda x: x.usefulness_score, reverse=True)
                ])
                st.dataframe(ev_df, width="stretch", hide_index=True)


# =====================================================================
# Page: Market Indicators
# =====================================================================
def render_indicators(results: dict):
    st.markdown("## 📈 Market Indicators")

    composites = results["composites"]
    latest_z = results["latest_z"]

    # Family composites
    cols = st.columns(len(composites))
    for i, fc in enumerate(composites):
        label, css = regime_label(fc.composite_z, fc.percentile)
        with cols[i]:
            st.metric(
                fc.family.value.replace("_", " ").title(),
                f"{fc.composite_z:+.2f}σ",
                delta=f"{fc.percentile:.0%} pctile",
            )
            st.markdown(f"<span class='{css}'>{label}</span> ({fc.n_indicators} indicators)",
                        unsafe_allow_html=True)

    st.divider()

    # Individual indicator z-scores
    st.markdown("### Individual Indicator Z-Scores")
    if latest_z:
        z_df = pd.DataFrame([
            {"Indicator": k, "Z-Score": v, "Abs Z": abs(v)}
            for k, v in latest_z.items()
        ]).sort_values("Abs Z", ascending=False)

        # Color-coded bar chart
        z_df_chart = z_df.set_index("Indicator")["Z-Score"].sort_values()
        st.bar_chart(z_df_chart, horizontal=True)

        # Table
        st.dataframe(
            z_df[["Indicator", "Z-Score"]].style
                .background_gradient(subset=["Z-Score"], cmap="RdBu_r", vmin=-3, vmax=3)
                .format({"Z-Score": "{:+.3f}"}),
            width="stretch",
            hide_index=True,
        )


# =====================================================================
# Page: Scenario Cards
# =====================================================================
def render_scenarios(results: dict):
    st.markdown("## 🎯 Scenario Cards")

    cards = results["scenario_cards"]

    for card in cards:
        t = card.theme
        icon = DIRECTION_ICONS.get(t.direction.value, "")
        cat_color = CATEGORY_COLORS.get(t.category.value, "#888")

        with st.expander(
            f"{icon} {t.name}  |  Strength: {t.strength:.3f}  |  "
            f"Confirmation: {t.confirmation_score:+.3f}",
            expanded=True,
        ):
            # Header metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Category", t.category.value)
            m2.metric("Direction", t.direction.value)
            m3.metric("Horizon", t.horizon.value)
            m4.metric("Likelihood", f"{t.likelihood:.0%}")

            st.markdown(f"> {t.narrative}")

            # Shock vector
            st.markdown("**Macro Driver Shocks**")
            shock_data = {
                DRIVER_LABELS.get(d, d): v
                for d, v in card.shock_vector.items()
                if abs(v) > 0.001
            }
            if shock_data:
                shock_df = pd.DataFrame({
                    "Driver": list(shock_data.keys()),
                    "Shock (σ)": list(shock_data.values()),
                })
                st.bar_chart(shock_df.set_index("Driver"), horizontal=True)

            # Asset returns
            st.markdown("**Asset-Class Return Estimates**")
            ar_rows = []
            for ar in card.asset_returns:
                ar_rows.append({
                    "Asset Class": ASSET_CLASS_LABELS.get(ar.asset_class, ar.asset_class),
                    "Expected Return": ar.scenario_return,
                    "CI Low (95%)": ar.ci_lower,
                    "CI High (95%)": ar.ci_upper,
                })
            if ar_rows:
                ar_df = pd.DataFrame(ar_rows)
                st.dataframe(
                    ar_df.style
                        .background_gradient(subset=["Expected Return"], cmap="RdYlGn", vmin=-0.1, vmax=0.1)
                        .format({
                            "Expected Return": "{:+.2%}",
                            "CI Low (95%)": "{:+.2%}",
                            "CI High (95%)": "{:+.2%}",
                        }),
                    width="stretch",
                    hide_index=True,
                )

            # Indicator confirmation
            if card.indicator_confirmations:
                st.markdown("**Indicator Confirmation**")
                ic_df = pd.DataFrame(card.indicator_confirmations)
                ic_df["consistent"] = ic_df["consistent"].map({True: "✅", False: "❌"})
                ic_df.columns = ["Family", "Composite Z", "Percentile", "Consistent"]
                st.dataframe(ic_df, width="stretch", hide_index=True)

            # Key risks
            if card.key_risks:
                st.markdown("**Key Risks**")
                for r in card.key_risks:
                    st.markdown(f"- {r}")


# =====================================================================
# Page: Portfolio Positioning
# =====================================================================
def render_portfolio(results: dict):
    st.markdown("## 💼 Portfolio Positioning")

    summary: PortfolioSummary = results["summary"]

    # Top-level metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Portfolio Return", f"{summary.portfolio_return:+.2%}")
    c2.metric("Portfolio Risk", f"{summary.portfolio_risk:.2%}")
    sharpe = summary.portfolio_return / summary.portfolio_risk if summary.portfolio_risk > 0 else 0
    c3.metric("Sharpe Ratio", f"{sharpe:.3f}")

    # Positions
    st.markdown("### Position Recommendations")
    pos_rows = []
    for p in summary.positions:
        pos_rows.append({
            "Asset Class": ASSET_CLASS_LABELS.get(p.asset_class, p.asset_class),
            "Weight": p.current_weight,
            "Scenario Return": p.scenario_return,
            "Scenario Risk": p.scenario_risk,
            "Risk-Adj Return": p.risk_adj_return,
            "MCTR": p.mctr,
            "Signal": p.signal.value,
            "Key Driver": p.key_theme_driver,
        })
    pos_df = pd.DataFrame(pos_rows)

    def color_signal(val):
        color = SIGNAL_COLORS.get(val, "#fff")
        return f"background-color: {color}; color: white; font-weight: bold;"

    st.dataframe(
        pos_df.style
            .map(color_signal, subset=["Signal"])
            .format({
                "Weight": "{:.1%}",
                "Scenario Return": "{:+.2%}",
                "Scenario Risk": "{:.2%}",
                "Risk-Adj Return": "{:+.3f}",
                "MCTR": "{:.4f}",
            }),
        width="stretch",
        hide_index=True,
    )

    # Signal summary
    ow = [p for p in summary.positions if p.signal.value == "OVERWEIGHT"]
    uw = [p for p in summary.positions if p.signal.value == "UNDERWEIGHT"]

    if ow or uw:
        st.markdown("### Signal Summary")
        s1, s2 = st.columns(2)
        if ow:
            s1.success("**Overweight:** " + ", ".join(
                ASSET_CLASS_LABELS.get(p.asset_class, p.asset_class) for p in ow
            ))
        if uw:
            s2.error("**Underweight:** " + ", ".join(
                ASSET_CLASS_LABELS.get(p.asset_class, p.asset_class) for p in uw
            ))

    # Return chart
    st.markdown("### Scenario Return by Asset Class")
    ret_df = pd.DataFrame({
        "Asset Class": [ASSET_CLASS_LABELS.get(p.asset_class, p.asset_class) for p in summary.positions],
        "Return": [p.scenario_return for p in summary.positions],
    }).sort_values("Return", ascending=True)
    st.bar_chart(ret_df.set_index("Asset Class"), horizontal=True)

    # Hedges
    if summary.hedges:
        st.markdown("### 🛡️ Hedge Recommendations")
        for h in summary.hedges:
            with st.container(border=True):
                hc1, hc2, hc3 = st.columns(3)
                hc1.metric("Theme", h.theme_name)
                hc2.metric("Confirmation", f"{h.confirmation_score:+.3f}")
                hc3.metric("Portfolio Impact", f"{h.portfolio_impact:+.2%}")
                st.markdown(f"**Instruments:** {', '.join(h.suggested_instruments)}")
                st.markdown(f"**Rationale:** {h.rationale}")

    # VFMC actual portfolio composition from BNY Data Vault
    _render_vfmc_composition()


@st.cache_data(ttl=600, show_spinner=False)
def _cached_load_portfolio():
    from theme_engine.vfmc_portfolio import load_portfolio
    return load_portfolio()


@st.cache_data(ttl=600, show_spinner=False)
def _cached_load_client_positions():
    from theme_engine.vfmc_portfolio import load_client_positions
    return load_client_positions()


def _render_vfmc_composition():
    """Show the actual VFMC portfolio composition from BNY Data Vault.
    
    This requires VFMCDataLayer (VFMC-internal package).
    Non-VFMC users will see a helpful info message instead.
    """
    st.markdown("### 🏛️ VFMC Portfolio Composition (BNY Data Vault)")

    try:
        from theme_engine.vfmc_portfolio import load_portfolio, load_client_positions
        positions = _cached_load_portfolio()
        client_positions = _cached_load_client_positions()
    except ImportError as e:
        if "VFMCDataLayer" in str(e):
            st.info(
                "**BNY Data Vault Integration** — Not available in this environment.\n\n"
                "This section requires `VFMCDataLayer`, which is an internal VFMC package. "
                "It's only available within VFMC's internal network.\n\n"
                "- **VFMC Users:** Contact IT to enable package access on your machine.\n"
                "- **External Users:** This is expected; the dashboard still works with live news themes and indicators."
            )
        else:
            st.info(f"BNY Data Vault not available: {e}")
        return
    except Exception as exc:
        st.info(f"BNY Data Vault error: {exc}")
        return

    total_exp = sum(p.exposure_aud for p in positions)
    total_mv = sum(p.market_value_aud for p in positions)

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Exposure", f"A${total_exp / 1e9:,.2f}B")
    c2.metric("Total Market Value", f"A${total_mv / 1e9:,.2f}B")
    c3.metric("Asset Classes", str(len(positions)))
    clients = sorted(set(p.client for p in client_positions))
    c4.metric("Clients", str(len(clients)))

    # Main table with SAA
    rows = []
    for p in positions:
        rows.append({
            "Asset Class": ASSET_CLASS_LABELS.get(p.asset_class, p.bny_name),
            "Category": p.category,
            "Exposure (A$M)": p.exposure_aud / 1e6,
            "Market Value (A$M)": p.market_value_aud / 1e6,
            "Weight (%)": p.weight_pct,
            "SAA (%)": p.saa_weight * 100,
            "Active (%)": round(p.weight_pct - p.saa_weight * 100, 2),
        })
    comp_df = pd.DataFrame(rows)
    st.dataframe(
        comp_df.style
            .background_gradient(subset=["Weight (%)"], cmap="Blues")
            .background_gradient(subset=["Active (%)"], cmap="RdYlGn", vmin=-5, vmax=5)
            .format({
                "Exposure (A$M)": "{:,.0f}",
                "Market Value (A$M)": "{:,.0f}",
                "Weight (%)": "{:.2f}%",
                "SAA (%)": "{:.2f}%",
                "Active (%)": "{:+.2f}%",
            }),
        width="stretch",
        hide_index=True,
    )

    # Category breakdown chart
    cat_df = comp_df.groupby("Category")["Weight (%)"].sum().sort_values(ascending=False)
    st.markdown("**Allocation by Category**")
    st.bar_chart(cat_df)

    # Per-client breakdown
    with st.expander("📊 Per-Client Breakdown"):
        client_rows = []
        for p in client_positions:
            client_rows.append({
                "Client": p.client,
                "Asset Class": ASSET_CLASS_LABELS.get(p.asset_class, p.bny_name),
                "Category": p.category,
                "Exposure (A$M)": p.exposure_aud / 1e6,
                "Weight (%)": round(p.exposure_pct * 100, 2),
                "SAA (%)": p.saa_weight * 100,
            })
        client_df = pd.DataFrame(client_rows)
        st.dataframe(
            client_df.style.format({
                "Exposure (A$M)": "{:,.0f}",
                "Weight (%)": "{:.2f}%",
                "SAA (%)": "{:.2f}%",
            }),
            width="stretch",
            hide_index=True,
        )


# =====================================================================
# Page: Aggregate Shock Vector
# =====================================================================
def render_aggregate_shocks(results: dict):
    st.markdown("## ⚡ Aggregate Shock Vector")

    agg = results["agg_adjusted"]

    shock_df = pd.DataFrame({
        "Driver": [DRIVER_LABELS.get(d, d) for d in MACRO_DRIVERS],
        "Shock (σ)": [float(v) for v in agg],
    }).sort_values("Shock (σ)")

    st.bar_chart(shock_df.set_index("Driver"), horizontal=True)

    st.dataframe(
        shock_df.style.background_gradient(subset=["Shock (σ)"], cmap="RdBu_r", vmin=-2, vmax=2)
                .format({"Shock (σ)": "{:+.3f}σ"}),
        width="stretch",
        hide_index=True,
    )


# =====================================================================
# Page: News Evidence
# =====================================================================
def render_news(results: dict):
    st.markdown("## 📰 News Evidence")

    themes = results["themes"]
    total = sum(len(t.evidence) for t in themes)

    if total == 0:
        st.info("No news evidence ingested. Check that news_monitor DB exists.")
        return

    st.metric("Total Evidence Items", total)

    for t in sorted(themes, key=lambda x: len(x.evidence), reverse=True):
        if not t.evidence:
            continue
        with st.expander(f"{t.name} — {len(t.evidence)} items"):
            ev_df = pd.DataFrame([
                {
                    "Source": e.source,
                    "Title": e.title,
                    "Usefulness": e.usefulness_score,
                    "Credibility": e.credibility_score,
                    "Date": e.date.isoformat(),
                    "URL": e.url or "",
                }
                for e in sorted(t.evidence, key=lambda x: x.usefulness_score, reverse=True)
            ])
            st.dataframe(ev_df, width="stretch", hide_index=True)


# =====================================================================
# Main
# =====================================================================
def main():
    config = sidebar_config()

    if "results" not in st.session_state:
        st.session_state.results = None

    if config["run"]:
        results = run_pipeline(
            data_source=config["data_source"],
            bbg_host=config["bbg_host"],
            bbg_port=config["bbg_port"],
            csv_path=config["csv_path"],
            theme_source=config["theme_source"],
            themes_file=config["themes_file"],
            news_lookback=config["news_lookback"],
            refresh_news=config["refresh_news"],
        )
        st.session_state.results = results

    results = st.session_state.results

    if results is None:
        st.markdown("# 📊 Theme-Driven Macro Scenario Engine")
        st.markdown("---")
        st.markdown(
            "Configure the data source in the sidebar, then click "
            "**🚀 Run Pipeline** to generate the dashboard."
        )
        st.info(
            "**Theme sources:**\n"
            "- **Live News (bottom-up)** — auto-discovers themes from "
            "GDELT news articles in real time\n"
            "- **JSON file** — loads predefined themes from a file\n\n"
            "**Indicator sources:**\n"
            "- **Bloomberg** — live indicator data via BLPAPI\n"
            "- **CSV** — load from a local CSV file"
        )
        return

    # ── Render all sections ──
    render_header(results)
    st.divider()

    tabs = st.tabs([
        "� Emerging Risks",
        "📋 Themes",
        "📈 Indicators",
        "🎯 Scenarios",
        "💼 Portfolio",
        "⚡ Shocks",
        "📰 News",
    ])

    with tabs[0]:
        render_emerging_risks(results)
    with tabs[1]:
        render_themes(results)
    with tabs[2]:
        render_indicators(results)
    with tabs[3]:
        render_scenarios(results)
    with tabs[4]:
        render_portfolio(results)
    with tabs[5]:
        render_aggregate_shocks(results)
    with tabs[6]:
        render_news(results)


if __name__ == "__main__":
    main()
