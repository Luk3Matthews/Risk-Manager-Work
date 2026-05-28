"""Theme Engine — Synthetic Data Generator.

Generates realistic synthetic data for end-to-end testing:
  - 5 example themes (geopolitical, growth, inflation, liquidity, valuation)
  - Synthetic indicator time series (5 years daily)
  - Example portfolio weights
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from .models import (
    Direction,
    EvidenceItem,
    Horizon,
    IndicatorFamily,
    Magnitude,
    RiskSignal,
    SignalDirection,
    Theme,
    ThemeCategory,
    ThemeStatus,
)
from .indicators import INDICATOR_REGISTRY, get_family_indicators


# ---------------------------------------------------------------------------
# Example themes
# ---------------------------------------------------------------------------

def create_example_themes() -> list[Theme]:
    """Create 5 example themes covering different categories."""
    today = date.today()

    themes = [
        # 1. GEOPOLITICAL
        Theme(
            theme_id="geo_hormuz_01",
            name="Strait of Hormuz escalation",
            category=ThemeCategory.GEOPOLITICAL,
            narrative=(
                "Rising tensions in the Persian Gulf with increased naval "
                "posturing near the Strait of Hormuz. Iran has threatened "
                "to restrict tanker passage in response to renewed sanctions. "
                "Oil supply disruption risk elevated."
            ),
            direction=Direction.BEARISH,
            horizon=Horizon.NEAR_TERM,
            likelihood=0.35,
            historical_analogue="2019 Strait of Hormuz tanker attacks",
            first_observed=today - timedelta(days=45),
            last_updated=today - timedelta(days=2),
            status=ThemeStatus.ACTIVE,
            evidence=[
                EvidenceItem(
                    source="reuters", date=today - timedelta(days=2),
                    title="Iran warns of Hormuz response to fresh US sanctions",
                    url="https://example.com/reuters/1",
                    credibility_score=0.9, timeliness_score=0.95,
                    corroboration_count=4,
                ),
                EvidenceItem(
                    source="ft", date=today - timedelta(days=5),
                    title="US Navy deploys additional carrier group to Gulf",
                    url="https://example.com/ft/1",
                    credibility_score=0.85, timeliness_score=0.8,
                    corroboration_count=3,
                ),
                EvidenceItem(
                    source="bloomberg", date=today - timedelta(days=10),
                    title="Oil tanker insurance premiums surge for Gulf routes",
                    url="https://example.com/bbg/1",
                    credibility_score=0.95, timeliness_score=0.7,
                    corroboration_count=2,
                ),
            ],
        ),

        # 2. GROWTH
        Theme(
            theme_id="growth_soft_01",
            name="US soft landing materialises",
            category=ThemeCategory.GROWTH,
            narrative=(
                "US economic data continues to show resilience with labour "
                "market gradually cooling without significant job losses. "
                "PMIs stabilising above 50, consumer spending holding up. "
                "Goldilocks scenario gaining traction."
            ),
            direction=Direction.BULLISH,
            horizon=Horizon.MEDIUM_TERM,
            likelihood=0.55,
            historical_analogue="1995 Fed soft landing",
            first_observed=today - timedelta(days=90),
            last_updated=today - timedelta(days=1),
            status=ThemeStatus.ACTIVE,
            evidence=[
                EvidenceItem(
                    source="wsj", date=today - timedelta(days=1),
                    title="US jobs report shows gradual cooling, unemployment at 4.1%",
                    url="https://example.com/wsj/1",
                    credibility_score=0.9, timeliness_score=1.0,
                    corroboration_count=5,
                ),
                EvidenceItem(
                    source="fed", date=today - timedelta(days=7),
                    title="Fed Beige Book signals moderate growth across districts",
                    url="https://example.com/fed/1",
                    credibility_score=0.95, timeliness_score=0.85,
                    corroboration_count=1,
                ),
            ],
        ),

        # 3. INFLATION
        Theme(
            theme_id="infl_persist_01",
            name="Inflation persistence — services sticky",
            category=ThemeCategory.INFLATION,
            narrative=(
                "Core services inflation remains stubbornly above target. "
                "Shelter costs, healthcare, and insurance keeping core PCE "
                "elevated at 3.5%. Risk of inflation expectations de-anchoring "
                "if no progress by Q3."
            ),
            direction=Direction.BEARISH,
            horizon=Horizon.MEDIUM_TERM,
            likelihood=0.45,
            historical_analogue="1970s inflation persistence",
            first_observed=today - timedelta(days=120),
            last_updated=today - timedelta(days=3),
            status=ThemeStatus.ACTIVE,
            evidence=[
                EvidenceItem(
                    source="bls", date=today - timedelta(days=3),
                    title="CPI report: core services inflation accelerates to 5.2% YoY",
                    url="https://example.com/bls/1",
                    credibility_score=0.95, timeliness_score=0.9,
                    corroboration_count=6,
                ),
                EvidenceItem(
                    source="umich", date=today - timedelta(days=8),
                    title="Michigan 5y inflation expectations rise to 3.3%",
                    credibility_score=0.85, timeliness_score=0.75,
                    corroboration_count=2,
                ),
            ],
        ),

        # 4. LIQUIDITY
        Theme(
            theme_id="liq_tighten_01",
            name="Liquidity tightening — QT acceleration",
            category=ThemeCategory.LIQUIDITY,
            narrative=(
                "Federal Reserve balance sheet runoff accelerating, reserves "
                "approaching scarcity threshold. Repo rate spikes becoming "
                "more frequent. Risk of a 2019-style repo crisis if reserves "
                "drop below $2.5T."
            ),
            direction=Direction.BEARISH,
            horizon=Horizon.NEAR_TERM,
            likelihood=0.30,
            historical_analogue="Sep 2019 repo crisis",
            first_observed=today - timedelta(days=60),
            last_updated=today - timedelta(days=5),
            status=ThemeStatus.MONITORING,
            evidence=[
                EvidenceItem(
                    source="ny_fed", date=today - timedelta(days=5),
                    title="Fed balance sheet falls below $7T, reserves at $2.8T",
                    url="https://example.com/nyfed/1",
                    credibility_score=0.95, timeliness_score=0.85,
                    corroboration_count=2,
                ),
                EvidenceItem(
                    source="bbg", date=today - timedelta(days=12),
                    title="Overnight repo rates spike 30bp above SOFR",
                    credibility_score=0.9, timeliness_score=0.6,
                    corroboration_count=3,
                ),
            ],
        ),

        # 5. VALUATION
        Theme(
            theme_id="val_rich_01",
            name="US equity valuation stretch — AI exuberance",
            category=ThemeCategory.VALUATION,
            narrative=(
                "S&P 500 forward PE at 22x, CAPE at 35x — both in the "
                "top decile historically. Concentration in Mag-7 tech names "
                "driving index-level overvaluation. If AI revenue growth "
                "disappoints, multiple compression risk is material."
            ),
            direction=Direction.BEARISH,
            horizon=Horizon.LONG_TERM,
            likelihood=0.40,
            historical_analogue="2000 dot-com bubble",
            first_observed=today - timedelta(days=180),
            last_updated=today - timedelta(days=7),
            status=ThemeStatus.ACTIVE,
            evidence=[
                EvidenceItem(
                    source="gs", date=today - timedelta(days=7),
                    title="Goldman: S&P 500 CAPE at 35x implies negative 10Y real returns",
                    credibility_score=0.85, timeliness_score=0.8,
                    corroboration_count=3,
                ),
                EvidenceItem(
                    source="ft", date=today - timedelta(days=14),
                    title="Mag-7 now 35% of S&P 500 market cap — record concentration",
                    credibility_score=0.85, timeliness_score=0.65,
                    corroboration_count=4,
                ),
            ],
        ),
    ]

    return themes


# ---------------------------------------------------------------------------
# Synthetic indicator time series
# ---------------------------------------------------------------------------

def _generate_mean_reverting(
    n: int,
    mu: float = 0.0,
    sigma: float = 1.0,
    theta: float = 0.02,
    dt: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """Ornstein-Uhlenbeck process: mean-reverting with specified volatility."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    x[0] = mu + rng.normal(0, sigma)
    for t in range(1, n):
        x[t] = x[t - 1] + theta * (mu - x[t - 1]) * dt + sigma * np.sqrt(dt) * rng.normal()
    return x


def _generate_trending(
    n: int,
    mu: float = 0.0,
    trend: float = 0.001,
    sigma: float = 0.5,
    seed: int | None = None,
) -> np.ndarray:
    """Random walk with drift."""
    rng = np.random.default_rng(seed)
    shocks = rng.normal(trend, sigma, n)
    return mu + np.cumsum(shocks)


# Realistic statistical properties for each indicator
_INDICATOR_PARAMS: dict[str, dict] = {
    # MARKET RISK
    "VIX": {"mu": 18, "sigma": 5, "theta": 0.05},
    "MOVE": {"mu": 100, "sigma": 20, "theta": 0.03},
    "CVIX": {"mu": 9, "sigma": 2, "theta": 0.04},
    "CDX_IG": {"mu": 60, "sigma": 15, "theta": 0.03},
    "CDX_HY": {"mu": 350, "sigma": 80, "theta": 0.02},
    "HY_minus_IG": {"mu": 290, "sigma": 70, "theta": 0.02},
    "Turbulence": {"mu": 1.0, "sigma": 0.5, "theta": 0.1},
    "Systemic_Risk": {"mu": 0.5, "sigma": 0.3, "theta": 0.05},
    "FinStress": {"mu": 0, "sigma": 1.0, "theta": 0.03},
    # GEOPOLITICAL
    "GPRD": {"mu": 100, "sigma": 30, "theta": 0.02},
    "GPRD_MA30": {"mu": 100, "sigma": 15, "theta": 0.01},
    "Brent_Crude": {"mu": 75, "sigma": 12, "theta": 0.01},
    "Gold": {"mu": 1900, "sigma": 150, "theta": 0.005},
    "Oil_ImpliedVol": {"mu": 35, "sigma": 8, "theta": 0.04},
    "Gold_ImpliedVol": {"mu": 15, "sigma": 4, "theta": 0.04},
    "TPU": {"mu": 100, "sigma": 40, "theta": 0.015},
    # EXPECTED DIRECTION
    "CBOE_Skew": {"mu": 130, "sigma": 10, "theta": 0.03},
    "SPX_PutCall": {"mu": 0.9, "sigma": 0.2, "theta": 0.05},
    "AAII_BearBull": {"mu": 0, "sigma": 15, "theta": 0.1},
    # MARKET STRUCTURE
    "Yield_2s10s": {"mu": 0.5, "sigma": 0.5, "theta": 0.01},
    "Yield_3m10y": {"mu": 0.8, "sigma": 0.6, "theta": 0.01},
    "ERP_Level": {"mu": 4.5, "sigma": 1.0, "theta": 0.02},
    "VIX_TermStructure": {"mu": -2, "sigma": 3, "theta": 0.05},
    "Implied_Correlation": {"mu": 50, "sigma": 10, "theta": 0.03},
    # EQUITY VALUATION
    "PE_Ratio": {"mu": 20, "sigma": 3, "theta": 0.005},
    "EV_EBITDA": {"mu": 14, "sigma": 2, "theta": 0.005},
    "PB_Ratio": {"mu": 3.5, "sigma": 0.8, "theta": 0.005},
    "PCF_Ratio": {"mu": 15, "sigma": 3, "theta": 0.005},
    "PS_Ratio": {"mu": 2.5, "sigma": 0.6, "theta": 0.005},
    "CAPE": {"mu": 28, "sigma": 5, "theta": 0.003},
    "Tobin_Q": {"mu": 1.2, "sigma": 0.3, "theta": 0.005},
    "Buffett_Indicator": {"mu": 150, "sigma": 25, "theta": 0.003},
}


def generate_synthetic_indicators(
    n_years: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic daily indicator time series.

    Returns DataFrame with DatetimeIndex and one column per indicator.
    """
    n_days = n_years * 252  # trading days
    end_date = pd.Timestamp.today()
    dates = pd.bdate_range(end=end_date, periods=n_days)

    data = {}
    for i, meta in enumerate(INDICATOR_REGISTRY):
        params = _INDICATOR_PARAMS.get(meta.name, {"mu": 0, "sigma": 1, "theta": 0.03})
        series = _generate_mean_reverting(
            n_days,
            mu=params["mu"],
            sigma=params["sigma"],
            theta=params["theta"],
            seed=seed + i,
        )
        data[meta.name] = series

    return pd.DataFrame(data, index=dates)


def save_synthetic_indicators(
    output_dir: str = "data/indicators",
    n_years: int = 5,
    seed: int = 42,
) -> None:
    """Generate and save synthetic indicators as individual CSVs."""
    from pathlib import Path

    df = generate_synthetic_indicators(n_years, seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for col in df.columns:
        csv_df = pd.DataFrame({"date": df.index, "value": df[col].values})
        csv_df.to_csv(out / f"{col}.csv", index=False)


# ---------------------------------------------------------------------------
# Example risk signals (linked to example themes)
# ---------------------------------------------------------------------------

def create_example_signals() -> list[RiskSignal]:
    """Create example risk signals for the example themes."""
    return [
        RiskSignal(
            signal_id="sig_hormuz_oil",
            parent_theme_id="geo_hormuz_01",
            factor="Oil supply disruption",
            direction=SignalDirection.UP,
            magnitude=Magnitude.LARGE,
            horizon=Horizon.NEAR_TERM,
            confidence=0.7,
            macro_driver_shocks={
                "commodity_supply": 1.5,
                "expected_inflation": 0.5,
                "equity_risk_premium": 0.8,
                "policy_uncertainty": 0.6,
            },
            linked_indicators=["Brent_Crude", "Oil_ImpliedVol", "VIX", "GPRD"],
        ),
        RiskSignal(
            signal_id="sig_soft_landing",
            parent_theme_id="growth_soft_01",
            factor="Growth resilience",
            direction=SignalDirection.UP,
            magnitude=Magnitude.MODERATE,
            horizon=Horizon.MEDIUM_TERM,
            confidence=0.6,
            macro_driver_shocks={
                "expected_growth": 0.8,
                "equity_risk_premium": -0.5,
                "credit_premium": -0.3,
                "liquidity": 0.3,
            },
            linked_indicators=["VIX", "CDX_HY", "Yield_2s10s", "PE_Ratio"],
        ),
        RiskSignal(
            signal_id="sig_inflation_sticky",
            parent_theme_id="infl_persist_01",
            factor="Services inflation persistence",
            direction=SignalDirection.UP,
            magnitude=Magnitude.MODERATE,
            horizon=Horizon.MEDIUM_TERM,
            confidence=0.65,
            macro_driver_shocks={
                "expected_inflation": 1.0,
                "real_rates": -0.3,
                "equity_risk_premium": 0.3,
                "policy_uncertainty": 0.2,
            },
            linked_indicators=["Gold", "CAPE", "Yield_2s10s"],
        ),
        RiskSignal(
            signal_id="sig_qt_squeeze",
            parent_theme_id="liq_tighten_01",
            factor="Repo market stress",
            direction=SignalDirection.DOWN,
            magnitude=Magnitude.MODERATE,
            horizon=Horizon.NEAR_TERM,
            confidence=0.5,
            macro_driver_shocks={
                "liquidity": -1.0,
                "credit_premium": 0.4,
                "equity_risk_premium": 0.3,
            },
            linked_indicators=["FinStress", "CDX_IG", "VIX", "Systemic_Risk"],
        ),
        RiskSignal(
            signal_id="sig_valuation_stretch",
            parent_theme_id="val_rich_01",
            factor="Multiple compression risk",
            direction=SignalDirection.DOWN,
            magnitude=Magnitude.LARGE,
            horizon=Horizon.LONG_TERM,
            confidence=0.55,
            macro_driver_shocks={
                "equity_risk_premium": 1.0,
                "expected_growth": -0.3,
            },
            linked_indicators=["PE_Ratio", "CAPE", "Buffett_Indicator", "CBOE_Skew"],
        ),
    ]


# ---------------------------------------------------------------------------
# Example news articles for sifter testing
# ---------------------------------------------------------------------------

def create_example_articles() -> list[dict]:
    """Create synthetic news articles for testing the news sifter."""
    today = date.today()
    return [
        {
            "title": "Iran threatens Strait of Hormuz blockade amid sanctions escalation",
            "snippet": "Iranian military officials warned of potential restrictions on tanker "
                       "passage through the Strait of Hormuz following new US sanctions.",
            "source": "reuters",
            "published_at": (today - timedelta(days=1)).isoformat(),
            "url": "https://example.com/article/1",
        },
        {
            "title": "US economy adds 180K jobs, unemployment holds at 4.1%",
            "snippet": "The US labour market showed continued resilience with moderate job gains "
                       "and stable unemployment, supporting soft landing narrative.",
            "source": "wsj",
            "published_at": (today - timedelta(days=2)).isoformat(),
            "url": "https://example.com/article/2",
        },
        {
            "title": "Core CPI surprises to upside at 3.8% YoY",
            "snippet": "US core inflation accelerated more than expected, driven by sticky "
                       "services costs including shelter and insurance.",
            "source": "bloomberg",
            "published_at": today.isoformat(),
            "url": "https://example.com/article/3",
        },
        {
            "title": "Fed repo facility usage surges to $500B",
            "snippet": "Overnight repo facility demand jumped sharply as reserves tightened, "
                       "raising concerns about liquidity conditions.",
            "source": "ft",
            "published_at": (today - timedelta(days=3)).isoformat(),
            "url": "https://example.com/article/4",
        },
        {
            "title": "S&P 500 forward PE hits 22x as AI stocks rally",
            "snippet": "Equity valuations stretched further with the S&P 500 forward PE ratio "
                       "reaching 22x, driven by AI sector exuberance.",
            "source": "barrons",
            "published_at": (today - timedelta(days=1)).isoformat(),
            "url": "https://example.com/article/5",
        },
        {
            "title": "Saudi Arabia extends oil production cuts through Q2",
            "snippet": "OPEC+ leader Saudi Arabia announced extension of 1M bpd voluntary "
                       "cuts, tightening global oil supply balance.",
            "source": "reuters",
            "published_at": (today - timedelta(days=4)).isoformat(),
            "url": "https://example.com/article/6",
        },
        {
            "title": "China GDP growth slows to 4.2%, property sector drags",
            "snippet": "Chinese economic growth disappointed expectations as property crisis "
                       "continued to weigh on domestic demand and investment.",
            "source": "bloomberg",
            "published_at": (today - timedelta(days=2)).isoformat(),
            "url": "https://example.com/article/7",
        },
        {
            "title": "ECB signals pause as eurozone inflation returns to target",
            "snippet": "European Central Bank officials indicated rates are likely at peak "
                       "as headline inflation dropped to 2.1%.",
            "source": "ecb",
            "published_at": (today - timedelta(days=5)).isoformat(),
            "url": "https://example.com/article/8",
        },
        {
            "title": "VIX spikes to 28 on Middle East escalation fears",
            "snippet": "Equity volatility surged as geopolitical tensions in the Middle East "
                       "intensified, with oil prices jumping 5%.",
            "source": "cboe",
            "published_at": today.isoformat(),
            "url": "https://example.com/article/9",
        },
        {
            "title": "Gold breaks $2,200 as central banks boost reserves",
            "snippet": "Gold prices hit record highs as central bank purchases accelerated "
                       "and geopolitical uncertainty drove safe-haven demand.",
            "source": "reuters",
            "published_at": (today - timedelta(days=1)).isoformat(),
            "url": "https://example.com/article/10",
        },
    ]
