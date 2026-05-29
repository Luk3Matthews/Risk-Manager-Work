# Emerging Risks Information Funnel

A **theme-driven macro scenario engine** powered by live news, Bloomberg market signals, and rigorous quantitative transmission channels. Auto-discovers emerging risks from news articles, maps them to four risk factors (Conflict, Supply Chain, Health, Inflation), and quantifies portfolio impact.

**Goal:** Transform a high-volume news/signals stream into an **actionable investment committee report** showing which macro risks are emerging, how they transmit through markets, and what portfolio hedges make sense.

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.11+** (tested on 3.12)
- **VS Code** (recommended)
- Optional: **Bloomberg Terminal** running locally for live indicator data

### Installation

1. **Clone the repo:**
   ```bash
   git clone https://github.com/Luk3Matthews/Risk-Manager-Work.git
   cd Risk-Manager-Work
   ```

2. **Create a Python virtual environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\Activate.ps1   # Windows PowerShell
   # or: source .venv/bin/activate  # macOS/Linux
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   **For Bloomberg live data** (optional):
   ```bash
   pip install xbbg
   # Requires Bloomberg Terminal running on localhost:8194
   ```

4. **Run the dashboard:**
   ```bash
   python -m streamlit run theme_engine/web_dashboard.py
   ```
   Dashboard opens at `http://localhost:8501`

5. **Click "🚀 Run Pipeline"** in the sidebar to generate themes from news and compute portfolio impact.

---

## 📊 Architecture

### Data Flow

```
News Articles (GDELT, Bing)
       ↓
[news_monitor] → Auto-tag & ingest into SQLite DB
       ↓
[theme_engine] → Create themes, compute shock vectors
       ↓
[Bloomberg] → Pull live market indicators (composites)
       ↓
[Scenario] → Aggregate shocks, compute asset returns
       ↓
[Dashboard] → Group by risk factor, show portfolio impact
```

### Key Modules

| Module | Purpose |
|--------|---------|
| **theme_engine/web_dashboard.py** | Streamlit UI; renders Emerging Risks tab grouped by risk factors |
| **theme_engine/indicators.py** | Bloomberg indicator pipeline; computes z-scores, composites |
| **theme_engine/scenario.py** | Scenario shocks; maps themes → macro drivers via transmission matrix |
| **theme_engine/factor_model.py** | Portfolio impact; confidence intervals on asset-class returns |
| **news_monitor/** | News ingestion & tagging; GDELT / Bing integration |

---

## 🎯 How It Works

1. **News → Themes** (bottom-up)
   - Auto-discovers themes from news articles using keyword matching + category tagging
   - Assigns strength (0–1) based on volume, recency, cross-source confirmation
   - Each theme has evidence (articles), narrative (summary), horizon (short/medium/long)

2. **Themes → Risk Factors** (aggregation)
   - Themes automatically grouped into 4 **Emerging Risk Factors:**
     - **Conflict** ⚔️ — GEOPOLITICAL themes
     - **Supply Chain** 🔗 — Supply/commodity/energy disruption
     - **Health** 🏥 — CONTAGION (pandemics, epidemics)
     - **Inflation** 📈 — Price pressure + policy response
   - Mapping is theme-driven: if news mentions "war" or "Strait," it matches multiple factors

3. **Risk Factors → Macro Shocks** (quantification)
   - Each theme's shock vector summed by risk factor
   - Transmission matrix maps theme categories → 9 macro drivers:
     - `expected_growth`, `expected_inflation`, `real_rates`, `equity_risk_premium`, `credit_premium`, `liquidity`, `commodity_supply`, `fx_risk_appetite`, `policy_uncertainty`
   - Price-signal confirmation: Bloomberg composites validate or dampen theme shocks

4. **Shocks → Portfolio Impact** (asset returns)
   - Factor model computes scenario return for each asset class
   - 95% confidence intervals account for shock uncertainty
   - Signals: Overweight / Underweight based on risk-adjusted returns

---

## 📁 Key Files & Configs

### Data Files (in repo)

- **theme_engine/data/config/**
  - `bloomberg_tickers.yaml` — Maps 24+ indicator mnemonics (CRB, VIX, 10Y, etc.)
  - `transmission_matrix.yaml` — Theme category → macro driver shocks
  - `parameters.yaml` — Diversification, confirmation thresholds
  - `exposure_matrix.yaml` — Asset class × macro driver sensitivity

- **theme_engine/data/themes/**
  - `example_themes.json` — Static themes for testing (if live news unavailable)

### News Database

- **news_monitor/news_monitor.db** — SQLite DB of articles (NOT in git; generated at runtime)
- Populated via GDELT (historical) or Bing News (real-time)
- Max 168-hour lookback; auto-aged out after 7 days

### Outputs (NOT in git)

- **outputs/** — Runtime-generated CSVs/JSONs (hedges, positions, scenarios, etc.)
- **theme_engine/data/cache/** — Parquet files of Bloomberg indicator pulls (24-hour TTL)

---

## 🖥️ Dashboard Tabs

1. **🔍 Emerging Risks** (landing page)
   - Risk Dashboard metrics (stress, active themes, exposures)
   - Indicator regimes (z-scores, percentiles)
   - Risk factor sections grouped by **Conflict / Supply Chain / Health / Inflation**
     - Contributing themes with strength & confirmation
     - Key articles with links
     - Macro transmission chart (shock by driver)
   - Aggregate shocks & portfolio impact
   - Positioning signals (Overweight/Underweight)

2. **📋 Themes** — All auto-discovered themes; detailed narratives & evidence
3. **📈 Indicators** — Bloomberg composite z-scores & individual indicator pulls
4. **🎯 Scenarios** — Per-theme scenario cards (shocks, asset returns, key risks)
5. **💼 Portfolio** — Position recommendations & VFMC portfolio composition (BNY Data Vault)
6. **⚡ Shocks** — Aggregate macro driver shock vector
7. **📰 News** — Full evidence inventory with clickable URLs

---

## ⚙️ Configuration

### Environment Variables (optional)

Create a `.env` file in the root directory:

```bash
# Bloomberg
BLPAPI_ROOT=/path/to/blpapi    # Usually auto-detected
BLOOMBERG_TIMEOUT=30           # seconds

# News ingestion
GDELT_MAX_RECORDS=100          # per query
BING_MAX_RECORDS=50

# Cache
CACHE_TTL_HOURS=24
```

### Sidebar Controls

- **Data Source:** Bloomberg (live) vs. CSV (offline)
- **Theme Source:** Live News (bottom-up) vs. JSON file (top-down)
- **News Lookback:** 24–720 hours
- **Refresh News:** Fetch fresh articles before run (adds ~30s)

---

## 🔒 Security & VFMC-Internal Features

### External Access

This repo is **public on GitHub** and contains no confidential data:
- ✅ Example themes, config YAMLs, source code
- ❌ NOT included: VFMC client data, Bloomberg data snapshots, news DB

### VFMC-Internal Only

The following features **require VFMC internal network:**

1. **BNY Data Vault** (`theme_engine/vfmc_portfolio.py`)
   - Queries `ENTERPRISE_LOOKTHROUGH_VDM` for live portfolio positions
   - Needs `VFMCDataLayer` package (internal only)
   - Non-VFMC users will see warning but dashboard still runs

2. **Bloomberg Terminal** (optional)
   - If Terminal not running, falls back to synthetic data or cached snapshots
   - `theme_engine/bloomberg_loader.py` gracefully degrades

---

## 🧪 Testing

Run the test suite:

```bash
pytest theme_engine/tests/
```

Tests cover:
- Indicator pipeline (z-scores, composites)
- Scenario shocks (transmission matrix)
- Portfolio summaries (signal generation)
- Factor model (confidence intervals)

---

## 🚀 Deployment / Transfer to VFMC

### To Push to VFMC Azure DevOps

1. Get repo URL from VFMC IT (e.g., `https://dev.azure.com/VFMC/Project/_git/RiskManagerWork`)
2. Add remote:
   ```bash
   git remote add vfmc <VFMC_REPO_URL>
   git push vfmc main
   ```
3. VFMC DevOps will handle CI/CD (Docker, tests, deployment)

### Docker (optional)

Create a `Dockerfile` (not in repo yet):

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "theme_engine/web_dashboard.py", "--server.port=8501"]
```

Build & run:
```bash
docker build -t emerging-risks-engine .
docker run -p 8501:8501 emerging-risks-engine
```

---

## 📞 Support

- **Dashboard issues?** Check sidebar logs or reload the page (Streamlit caches pipeline results)
- **Bloomberg connection failing?** Ensure Terminal is running on localhost:8194
- **News not ingesting?** Check `news_monitor/news_monitor.db` exists; try "Fetch fresh articles" in sidebar
- **Questions about transmission matrix?** See `theme_engine/data/config/transmission_matrix.yaml` for shock mappings

---

## 📝 License & Attribution

Internal VFMC tool. Source code is available for team collaboration and auditing.

---

**Last updated:** May 2026  
**Python Version:** 3.12  
**Streamlit:** 1.40+  
**Status:** Active development
