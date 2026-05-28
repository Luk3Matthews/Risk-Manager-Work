"""
Streamlit UI — VFMC News & Macro Risk Monitor
Run with: streamlit run news_monitor/app.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src import db
from src.bloomberg import get_key_market_movers, get_market_cache
from src.risk_mapper import build_risk_summary, get_affected_asset_classes
from src.scheduler import NewsScheduler, load_config
from src.taxonomy import get_taxonomy

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="VFMC News & Macro Risk Monitor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Init DB
DB_PATH = Path(__file__).parent / "news_monitor.db"
db.init_db(DB_PATH)

# Load taxonomy
taxonomy = get_taxonomy()

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

st.sidebar.title("📡 News Monitor")
st.sidebar.markdown("---")

# Time filter
hours_back = st.sidebar.slider("Lookback (hours)", 1, 168, 24)

# Theme filter
all_themes = list(taxonomy.themes.keys())
selected_themes = st.sidebar.multiselect("Filter by Theme", all_themes, default=[])

# Factor filter
all_factors = list(taxonomy.macro_factors.keys())
selected_factors = st.sidebar.multiselect("Filter by Macro Factor", all_factors, default=[])

# Source filter
source_filter = st.sidebar.selectbox("Source", ["All", "bing", "gdelt"])

# Manual poll button
if st.sidebar.button("🔄 Poll Now"):
    try:
        config = load_config()
        scheduler = NewsScheduler(config, db_path=DB_PATH)
        scheduler.poll_once()
        st.sidebar.success("Poll complete!")
    except Exception as e:
        st.sidebar.error(f"Poll failed: {e}")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**DB:** `{DB_PATH.name}`")
st.sidebar.markdown(f"**Last refresh:** {datetime.now(tz=__import__('datetime').timezone.utc).strftime('%H:%M:%S UTC')}")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📰 Live News Feed",
    "🗂️ Theme Summaries",
    "🌡️ Macro Factor Heatmap",
    "📊 Key Market Moves",
    "🔍 What Changed",
])

# ─── TAB 1: LIVE NEWS FEED ───────────────────────────────────────────────────
with tab1:
    st.header("Live News Feed")

    # Fetch articles
    theme_filter = selected_themes[0] if len(selected_themes) == 1 else None
    factor_filter = selected_factors[0] if len(selected_factors) == 1 else None
    src = source_filter if source_filter != "All" else None

    articles = db.get_recent_articles(
        hours=hours_back,
        source=src,
        theme=theme_filter,
        macro_factor=factor_filter,
        limit=100,
        db_path=DB_PATH,
    )

    if not articles:
        st.info("No articles found. Try adjusting filters or run a poll.")
    else:
        st.caption(f"Showing {len(articles)} articles from last {hours_back}h")

        for article in articles:
            tags = article.get("tags", [])

            # Filter by selected themes/factors if multiple selected
            if selected_themes:
                if not any(t["theme"] in selected_themes for t in tags):
                    continue
            if selected_factors:
                if not any(t["macro_factor"] in selected_factors for t in tags):
                    continue

            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**[{article['title']}]({article['url']})**")
                    if article.get("snippet"):
                        st.caption(article["snippet"][:200])
                with col2:
                    st.caption(f"Source: {article['source']}")
                    st.caption(f"Published: {article.get('published_at', 'N/A')[:16]}")

                # Tags
                if tags:
                    tag_cols = st.columns(min(len(tags), 4))
                    for i, tag in enumerate(tags[:4]):
                        with tag_cols[i]:
                            direction_emoji = {"up": "🔺", "down": "🔻", "neutral": "➡️"}.get(
                                tag.get("direction", ""), "➡️"
                            )
                            st.markdown(
                                f"`{tag['theme']}` → **{tag['macro_factor']}** "
                                f"{direction_emoji} _{tag.get('confidence', 0):.0%}_"
                            )

                    # Bloomberg tickers linked
                    all_tickers = []
                    for tag in tags:
                        for t in tag.get("bloomberg_tickers", []):
                            if isinstance(t, dict):
                                all_tickers.append(t)
                            else:
                                try:
                                    all_tickers.append(json.loads(t) if isinstance(t, str) else {})
                                except (json.JSONDecodeError, TypeError):
                                    pass

                    if all_tickers:
                        ticker_strs = [
                            f"`{t.get('ticker', '')}` ({t.get('name', '')})"
                            for t in all_tickers[:6]
                        ]
                        st.markdown(f"**Linked tickers:** {' • '.join(ticker_strs)}")

                st.markdown("---")

# ─── TAB 2: THEME SUMMARIES ──────────────────────────────────────────────────
with tab2:
    st.header("Theme Summaries")
    st.caption(f"Summarising signals from the last {hours_back} hours")

    theme_data = db.get_theme_summary(hours=hours_back, db_path=DB_PATH)

    if not theme_data:
        st.info("No tagged articles yet. Run a poll to ingest news.")
    else:
        # Theme descriptions from taxonomy
        theme_descs = {name: t.get("description", "") for name, t in taxonomy.themes.items()}

        for theme_name in taxonomy.themes:
            data = theme_data.get(theme_name)
            if not data or data["count"] == 0:
                with st.expander(f"**{theme_name}** — _No signals_"):
                    st.caption(theme_descs.get(theme_name, ""))
                    st.info("No articles matched this theme in the selected period.")
                continue

            # Compute dominant direction
            dirs = data["directions"]
            up = dirs.get("up", 0)
            down = dirs.get("down", 0)
            neutral = dirs.get("neutral", 0)

            if up > down and up > neutral:
                dominant = "Upward pressure"
                dir_icon = "🔺"
            elif down > up and down > neutral:
                dominant = "Downward pressure"
                dir_icon = "🔻"
            else:
                dominant = "Mixed / Neutral"
                dir_icon = "➡️"

            # Build the narrative summary
            factor_list = ", ".join(data["factors"]) if data["factors"] else "N/A"
            entity_list = ", ".join(data["entities"][:10]) if data["entities"] else "none detected"
            keyword_sample = ", ".join(data["keywords"][:12]) if data["keywords"] else ""

            summary_lines = []
            summary_lines.append(
                f"**{data['count']}** signals detected with **{data['avg_confidence']:.0%}** "
                f"average confidence. The dominant direction is **{dominant.lower()}** "
                f"({dir_icon} ↑{up} ↓{down} →{neutral})."
            )
            summary_lines.append(
                f"Linked macro factors: **{factor_list}**."
            )
            if entity_list != "none detected":
                summary_lines.append(f"Key entities mentioned: {entity_list}.")
            if keyword_sample:
                summary_lines.append(f"Top keywords: _{keyword_sample}_.")

            header_label = f"**{theme_name}** — {data['count']} signals {dir_icon} {dominant}"
            with st.expander(header_label, expanded=(data["count"] >= 5)):
                st.caption(theme_descs.get(theme_name, ""))
                st.markdown("---")

                # Narrative summary
                st.markdown("### Summary")
                for line in summary_lines:
                    st.markdown(line)

                # Direction gauge
                st.markdown("---")
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("🔺 Upward", up)
                col_b.metric("🔻 Downward", down)
                col_c.metric("➡️ Neutral", neutral)

                # Top articles
                if data["articles"]:
                    st.markdown("---")
                    st.markdown("### Top Articles")
                    for i, art in enumerate(data["articles"], 1):
                        dir_emoji = {"up": "🔺", "down": "🔻", "neutral": "➡️"}.get(
                            art["direction"], "➡️"
                        )
                        st.markdown(
                            f"{i}. {dir_emoji} [{art['title']}]({art['url']})  \n"
                            f"   _Source: {art['source']} | Factor: {art['factor']} | "
                            f"Confidence: {art['confidence']:.0%}_"
                        )

# ─── TAB 3: MACRO FACTOR HEATMAP ─────────────────────────────────────────────
with tab3:
    st.header("Macro Risk Factor Heatmap")
    st.caption(f"Aggregation over last {hours_back} hours")

    heatmap = db.get_factor_heatmap(hours=hours_back, db_path=DB_PATH)

    if not heatmap:
        st.info("No tagged articles in the selected period.")
    else:
        # Build summary grid
        cols = st.columns(3)
        for i, (factor, data) in enumerate(sorted(heatmap.items(), key=lambda x: x[1]["count"], reverse=True)):
            col = cols[i % 3]
            with col:
                directions = data.get("directions", {})
                up_count = directions.get("up", 0)
                down_count = directions.get("down", 0)
                neutral_count = directions.get("neutral", 0)

                if up_count > down_count:
                    color = "🟥" if up_count >= 3 else "🟧"
                elif down_count > up_count:
                    color = "🟦" if down_count >= 3 else "🟩"
                else:
                    color = "⬜"

                st.metric(
                    label=f"{color} {factor}",
                    value=f"{data['count']} signals",
                    delta=f"Conf: {data['avg_confidence']:.0%}",
                )
                st.caption(f"↑{up_count}  ↓{down_count}  →{neutral_count}")

                # Show linked tickers
                tickers = taxonomy.get_tickers_for_factor(factor)
                if tickers:
                    ticker_str = ", ".join(t["ticker"] for t in tickers[:4])
                    st.caption(f"Tickers: {ticker_str}")

        # Detailed table
        st.markdown("---")
        st.subheader("Factor → Asset Class Impact")
        # Get all tags for the period
        all_articles = db.get_recent_articles(hours=hours_back, limit=500, db_path=DB_PATH)
        all_tags = []
        for a in all_articles:
            all_tags.extend(a.get("tags", []))

        if all_tags:
            summary = build_risk_summary(all_tags)
            for factor, data in sorted(summary.items(), key=lambda x: x[1]["count"], reverse=True):
                with st.expander(f"**{factor}** — {data['count']} signals, "
                                 f"direction: {data.get('dominant_direction', '?')}"):
                    # Asset class impacts
                    impacts = data.get("asset_classes", [])
                    if impacts:
                        for ac in impacts:
                            emoji = {"positive": "✅", "negative": "❌", "mixed": "⚠️"}.get(
                                ac["impact"], "❓"
                            )
                            st.markdown(f"  {emoji} **{ac['asset_class']}** — {ac['note']}")

                    # Tickers
                    if data.get("tickers"):
                        st.markdown("**Bloomberg Tickers:**")
                        for t in data["tickers"][:5]:
                            st.markdown(f"  • `{t['ticker']}` — {t.get('name', '')}")

# ─── TAB 4: KEY MARKET MOVES ─────────────────────────────────────────────────
with tab4:
    st.header("Key Market Moves & Signal Intensity")

    import pandas as pd

    movers = get_key_market_movers(hours=1, threshold_pct=0.3)

    # ── Signal Intensity Time Series ──────────────────────────────────────────
    st.subheader("📈 Factor Signal Intensity Over Time")
    chart_hours = st.selectbox("Chart period", [6, 12, 24, 48, 168], index=2,
                               format_func=lambda x: f"Last {x}h", key="chart_hours")

    timeseries = db.get_factor_timeseries(hours=chart_hours, bucket_minutes=60, db_path=DB_PATH)

    if timeseries:
        df_ts = pd.DataFrame(timeseries)
        df_ts["bucket"] = pd.to_datetime(df_ts["bucket"], errors="coerce")
        df_ts = df_ts.dropna(subset=["bucket"])

        # Aggregate: net signal = up_count - down_count per factor per bucket
        df_up = df_ts[df_ts["direction"] == "up"].groupby(["bucket", "macro_factor"])["cnt"].sum().reset_index()
        df_up.columns = ["bucket", "macro_factor", "up_count"]
        df_down = df_ts[df_ts["direction"] == "down"].groupby(["bucket", "macro_factor"])["cnt"].sum().reset_index()
        df_down.columns = ["bucket", "macro_factor", "down_count"]
        df_net = pd.merge(df_up, df_down, on=["bucket", "macro_factor"], how="outer").fillna(0)
        df_net["net_signal"] = df_net["up_count"] - df_net["down_count"]

        # Total signal count per bucket per factor
        df_total = df_ts.groupby(["bucket", "macro_factor"])["cnt"].sum().reset_index()
        df_total.columns = ["bucket", "macro_factor", "total_signals"]

        # Chart 1: Stacked area of total signals by factor
        if not df_total.empty:
            pivot_total = df_total.pivot(index="bucket", columns="macro_factor", values="total_signals").fillna(0)
            st.markdown("**Signal Count by Factor** (stacked area)")
            st.area_chart(pivot_total, use_container_width=True)

        # Chart 2: Net direction (up minus down) per factor - line chart
        if not df_net.empty:
            pivot_net = df_net.pivot(index="bucket", columns="macro_factor", values="net_signal").fillna(0)
            st.markdown("**Net Direction** (positive = upward pressure, negative = downward)")
            st.line_chart(pivot_net, use_container_width=True)

        # Chart 3: Confidence-weighted signal by factor
        df_ts["weighted"] = df_ts["cnt"] * df_ts["avg_conf"]
        df_weighted = df_ts.groupby(["bucket", "macro_factor"])["weighted"].sum().reset_index()
        if not df_weighted.empty:
            pivot_w = df_weighted.pivot(index="bucket", columns="macro_factor", values="weighted").fillna(0)
            st.markdown("**Confidence-Weighted Signal Intensity**")
            st.line_chart(pivot_w, use_container_width=True)
    else:
        st.info("No signal data yet for the selected period. Run a poll to ingest articles.")

    st.markdown("---")

    # ── News Event Timeline ───────────────────────────────────────────────────
    st.subheader("🕒 News Signal Timeline")
    timeline = db.get_signal_timeline(hours=chart_hours, db_path=DB_PATH)

    if timeline:
        df_tl = pd.DataFrame(timeline)
        df_tl["created_at"] = pd.to_datetime(df_tl["created_at"], errors="coerce")
        df_tl = df_tl.dropna(subset=["created_at"])

        # Bar chart: signals per hour by theme
        df_tl["hour"] = df_tl["created_at"].dt.floor("h")
        theme_hourly = df_tl.groupby(["hour", "theme"]).size().reset_index(name="count")
        if not theme_hourly.empty:
            pivot_theme = theme_hourly.pivot(index="hour", columns="theme", values="count").fillna(0)
            st.markdown("**Signals per Hour by Theme**")
            st.bar_chart(pivot_theme, use_container_width=True)

        # Direction distribution chart
        dir_hourly = df_tl.groupby(["hour", "direction"]).size().reset_index(name="count")
        if not dir_hourly.empty:
            pivot_dir = dir_hourly.pivot(index="hour", columns="direction", values="count").fillna(0)
            st.markdown("**Directional Signal Distribution**")
            st.bar_chart(pivot_dir, use_container_width=True)
    else:
        st.info("No timeline data available.")

    st.markdown("---")

    # ── Bloomberg Live Movers (if available) ──────────────────────────────────
    st.subheader("💹 Bloomberg Live Movers")
    if not movers:
        st.info(
            "No significant market moves detected. "
            "Bloomberg streaming must be active for real-time data."
        )
        st.markdown("---")
        st.subheader("Configured Tickers (from taxonomy)")
        all_tickers = taxonomy.get_all_tickers_flat()
        for category, tickers in taxonomy.bloomberg_tickers.items():
            with st.expander(f"**{category.title()}** ({len(tickers)} tickers)"):
                for t in tickers:
                    themes_str = ", ".join(t.get("themes", []))
                    st.markdown(
                        f"• `{t['ticker']}` — {t['name']}  \n"
                        f"  Factor: {t.get('factor', 'N/A')} | Themes: {themes_str}"
                    )
    else:
        st.caption("Tickers moving > 0.3% in the last hour")
        for mover in movers[:20]:
            emoji = "🔺" if mover["change_pct"] > 0 else "🔻"
            st.markdown(
                f"{emoji} **`{mover['ticker']}`** ({mover['name']}) — "
                f"**{mover['change_pct']:+.2f}%**  |  "
                f"Factor: {mover.get('factor', 'N/A')} | "
                f"Themes: {', '.join(mover.get('themes', []))}"
            )

# ─── TAB 5: WHAT CHANGED ─────────────────────────────────────────────────────
with tab5:
    st.header("What Changed")
    change_hours = st.selectbox("Since...", [1, 4, 8, 24], index=0, format_func=lambda x: f"Last {x}h")

    audit_entries = db.get_audit_since(hours=change_hours, db_path=DB_PATH)

    if not audit_entries:
        st.info(f"No activity in the last {change_hours} hour(s).")
    else:
        for entry in audit_entries[:50]:
            ts = entry.get("timestamp", "")[:19]
            action = entry.get("action", "")
            details = entry.get("details_json", "{}")
            try:
                details_dict = json.loads(details) if isinstance(details, str) else details
            except (json.JSONDecodeError, TypeError):
                details_dict = {}

            icon = {
                "poll_complete": "✅",
                "poll_error": "❌",
                "db_init": "🗄️",
            }.get(action, "📝")

            st.markdown(f"{icon} **{ts}** — `{action}`")
            if details_dict:
                st.json(details_dict)
