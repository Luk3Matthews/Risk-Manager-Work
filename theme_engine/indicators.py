"""Theme Engine — Module 4: Market Indicator Families.

Implements the 5 indicator families (Market Risk, Geopolitical Risk,
Expected Direction, Market Structure, Equity Valuation), composite
construction via PCA / equal / inverse-variance weighting, and
theme confirmation / divergence detection.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .config import EngineConfig, get_config
from .models import (
    MACRO_DRIVERS,
    CompositeMethod,
    FamilyComposite,
    IndicatorFamily,
    IndicatorMeta,
    IndicatorReading,
)
from .utils import cosine_similarity, pca_weights, rolling_percentile, rolling_z_score


# ---------------------------------------------------------------------------
# Indicator registry — canonical definitions
# ---------------------------------------------------------------------------

INDICATOR_REGISTRY: list[IndicatorMeta] = [
    # FAMILY 1: MARKET RISK
    IndicatorMeta(name="VIX", family=IndicatorFamily.MARKET_RISK, risk_on=True,
                  mapped_drivers=["equity_risk_premium", "liquidity"]),
    IndicatorMeta(name="MOVE", family=IndicatorFamily.MARKET_RISK, risk_on=True,
                  mapped_drivers=["equity_risk_premium", "real_rates"]),
    IndicatorMeta(name="CVIX", family=IndicatorFamily.MARKET_RISK, risk_on=True,
                  mapped_drivers=["fx_risk_appetite"]),
    IndicatorMeta(name="CDX_IG", family=IndicatorFamily.MARKET_RISK, risk_on=True,
                  mapped_drivers=["credit_premium"]),
    IndicatorMeta(name="CDX_HY", family=IndicatorFamily.MARKET_RISK, risk_on=True,
                  mapped_drivers=["credit_premium", "equity_risk_premium"]),
    IndicatorMeta(name="HY_minus_IG", family=IndicatorFamily.MARKET_RISK, risk_on=True,
                  mapped_drivers=["credit_premium"]),
    IndicatorMeta(name="Turbulence", family=IndicatorFamily.MARKET_RISK, risk_on=True,
                  mapped_drivers=["equity_risk_premium", "liquidity"]),
    IndicatorMeta(name="Systemic_Risk", family=IndicatorFamily.MARKET_RISK, risk_on=True,
                  mapped_drivers=["liquidity", "credit_premium"]),
    IndicatorMeta(name="FinStress", family=IndicatorFamily.MARKET_RISK, risk_on=True,
                  mapped_drivers=["liquidity", "credit_premium"]),

    # FAMILY 2: GEOPOLITICAL RISK
    IndicatorMeta(name="GPRD", family=IndicatorFamily.GEOPOLITICAL_RISK, risk_on=True,
                  mapped_drivers=["policy_uncertainty", "equity_risk_premium"]),
    IndicatorMeta(name="GPRD_MA30", family=IndicatorFamily.GEOPOLITICAL_RISK, risk_on=True,
                  mapped_drivers=["policy_uncertainty"]),
    IndicatorMeta(name="Brent_Crude", family=IndicatorFamily.GEOPOLITICAL_RISK, risk_on=True,
                  mapped_drivers=["commodity_supply", "expected_inflation"]),
    IndicatorMeta(name="Gold", family=IndicatorFamily.GEOPOLITICAL_RISK, risk_on=True,
                  mapped_drivers=["policy_uncertainty", "expected_inflation"]),
    IndicatorMeta(name="Oil_ImpliedVol", family=IndicatorFamily.GEOPOLITICAL_RISK, risk_on=True,
                  mapped_drivers=["commodity_supply"]),
    IndicatorMeta(name="Gold_ImpliedVol", family=IndicatorFamily.GEOPOLITICAL_RISK, risk_on=True,
                  mapped_drivers=["policy_uncertainty"]),
    IndicatorMeta(name="TPU", family=IndicatorFamily.GEOPOLITICAL_RISK, risk_on=True,
                  mapped_drivers=["policy_uncertainty"]),

    # FAMILY 3: EXPECTED MARKET DIRECTION
    IndicatorMeta(name="CBOE_Skew", family=IndicatorFamily.EXPECTED_DIRECTION, risk_on=True,
                  mapped_drivers=["equity_risk_premium"]),
    IndicatorMeta(name="SPX_PutCall", family=IndicatorFamily.EXPECTED_DIRECTION, risk_on=True,
                  mapped_drivers=["equity_risk_premium", "expected_growth"]),
    IndicatorMeta(name="AAII_BearBull", family=IndicatorFamily.EXPECTED_DIRECTION, risk_on=True,
                  mapped_drivers=["expected_growth", "fx_risk_appetite"],
                  description="Inverted AAII Bull-Bear spread (higher = more bearish)"),

    # FAMILY 4: MARKET STRUCTURE
    IndicatorMeta(name="Yield_2s10s", family=IndicatorFamily.MARKET_STRUCTURE, risk_on=False,
                  mapped_drivers=["expected_growth", "real_rates"],
                  description="2y-10y yield curve slope; inverted = recession signal"),
    IndicatorMeta(name="Yield_3m10y", family=IndicatorFamily.MARKET_STRUCTURE, risk_on=False,
                  mapped_drivers=["expected_growth", "real_rates"]),
    IndicatorMeta(name="ERP_Level", family=IndicatorFamily.MARKET_STRUCTURE, risk_on=True,
                  mapped_drivers=["equity_risk_premium"]),
    IndicatorMeta(name="VIX_TermStructure", family=IndicatorFamily.MARKET_STRUCTURE, risk_on=True,
                  mapped_drivers=["equity_risk_premium", "liquidity"],
                  description="VIX - VIX3M; positive = backwardation = stress"),
    IndicatorMeta(name="Implied_Correlation", family=IndicatorFamily.MARKET_STRUCTURE, risk_on=True,
                  mapped_drivers=["equity_risk_premium", "liquidity"]),

    # FAMILY 5: EQUITY VALUATION
    IndicatorMeta(name="PE_Ratio", family=IndicatorFamily.EQUITY_VALUATION, risk_on=True,
                  mapped_drivers=["equity_risk_premium", "expected_growth"]),
    IndicatorMeta(name="EV_EBITDA", family=IndicatorFamily.EQUITY_VALUATION, risk_on=True,
                  mapped_drivers=["equity_risk_premium"]),
    IndicatorMeta(name="PB_Ratio", family=IndicatorFamily.EQUITY_VALUATION, risk_on=True,
                  mapped_drivers=["equity_risk_premium"]),
    IndicatorMeta(name="PCF_Ratio", family=IndicatorFamily.EQUITY_VALUATION, risk_on=True,
                  mapped_drivers=["equity_risk_premium"]),
    IndicatorMeta(name="PS_Ratio", family=IndicatorFamily.EQUITY_VALUATION, risk_on=True,
                  mapped_drivers=["equity_risk_premium"]),
    IndicatorMeta(name="CAPE", family=IndicatorFamily.EQUITY_VALUATION, risk_on=True,
                  mapped_drivers=["equity_risk_premium", "expected_growth", "real_rates"]),
    IndicatorMeta(name="Tobin_Q", family=IndicatorFamily.EQUITY_VALUATION, risk_on=True,
                  mapped_drivers=["equity_risk_premium"]),
    IndicatorMeta(name="Buffett_Indicator", family=IndicatorFamily.EQUITY_VALUATION, risk_on=True,
                  mapped_drivers=["equity_risk_premium", "expected_growth"]),
]

_REGISTRY_MAP: dict[str, IndicatorMeta] = {m.name: m for m in INDICATOR_REGISTRY}


def get_indicator_meta(name: str) -> IndicatorMeta | None:
    return _REGISTRY_MAP.get(name)


def get_family_indicators(family: IndicatorFamily) -> list[IndicatorMeta]:
    return [m for m in INDICATOR_REGISTRY if m.family == family]


# ---------------------------------------------------------------------------
# Data loaders (modular — swap in Bloomberg / yfinance / FRED / CSV)
# ---------------------------------------------------------------------------

class IndicatorDataLoader:
    """Abstract-ish base for loading indicator time series.

    Subclass and override ``load()`` for specific data sources.
    """

    def load(self, indicator_name: str, start: str | None = None,
             end: str | None = None) -> pd.Series:
        """Return a pd.Series (DatetimeIndex → float) for the indicator."""
        raise NotImplementedError

    def load_family(self, family: IndicatorFamily, **kw) -> pd.DataFrame:
        """Load all indicators in a family as a DataFrame (date × indicator)."""
        metas = get_family_indicators(family)
        frames = {}
        for m in metas:
            try:
                frames[m.name] = self.load(m.name, **kw)
            except (NotImplementedError, FileNotFoundError, KeyError):
                continue
        if not frames:
            return pd.DataFrame()
        return pd.DataFrame(frames)


class CSVDataLoader(IndicatorDataLoader):
    """Load indicator series from CSV files in a directory.

    Expected format: one CSV per indicator named ``{indicator_name}.csv``
    with columns ``date`` and ``value``.
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)

    def load(self, indicator_name: str, start: str | None = None,
             end: str | None = None) -> pd.Series:
        path = self.data_dir / f"{indicator_name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"No CSV for {indicator_name} at {path}")
        df = pd.read_csv(path, parse_dates=["date"], index_col="date")
        s = df["value"].sort_index()
        if start:
            s = s[s.index >= start]
        if end:
            s = s[s.index <= end]
        return s


class DataFrameLoader(IndicatorDataLoader):
    """Load indicators from an in-memory DataFrame (date × indicator)."""

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def load(self, indicator_name: str, **kw) -> pd.Series:
        if indicator_name not in self._df.columns:
            raise KeyError(indicator_name)
        return self._df[indicator_name].dropna()


# ---------------------------------------------------------------------------
# Indicator processing pipeline
# ---------------------------------------------------------------------------

def process_indicator(
    series: pd.Series,
    meta: IndicatorMeta,
    lookback: int = 1260,
) -> pd.DataFrame:
    """Process a raw indicator series into z-scores and percentiles.

    Returns DataFrame with columns: raw, z_score, percentile, stress_z
    (stress_z is normalised so higher = more stress for all indicators).
    """
    df = pd.DataFrame({"raw": series})
    df["z_score"] = rolling_z_score(series, lookback=lookback)
    df["percentile"] = rolling_percentile(series, lookback=lookback)

    # Normalise: higher z = more stress
    if meta.risk_on:
        df["stress_z"] = df["z_score"]
    else:
        df["stress_z"] = -df["z_score"]

    return df


def compute_family_composite(
    family: IndicatorFamily,
    loader: IndicatorDataLoader,
    method: CompositeMethod = CompositeMethod.PCA,
    lookback: int = 1260,
    cfg: EngineConfig | None = None,
) -> tuple[FamilyComposite, pd.DataFrame]:
    """Compute the composite z-score for an indicator family.

    Parameters
    ----------
    family : IndicatorFamily
    loader : IndicatorDataLoader
    method : CompositeMethod (pca, equal, inverse_variance)
    lookback : int — rolling window

    Returns
    -------
    (FamilyComposite, DataFrame of processed indicators)
    """
    cfg = cfg or get_config()
    metas = get_family_indicators(family)

    processed: dict[str, pd.DataFrame] = {}
    for m in metas:
        try:
            raw = loader.load(m.name)
            proc = process_indicator(raw, m, lookback=lookback)
            processed[m.name] = proc
        except (FileNotFoundError, KeyError, NotImplementedError):
            continue

    if not processed:
        return FamilyComposite(family=family), pd.DataFrame()

    # Align all stress_z series
    stress_df = pd.DataFrame({
        name: p["stress_z"] for name, p in processed.items()
    }).dropna()

    if stress_df.empty:
        return FamilyComposite(family=family), pd.DataFrame()

    k = stress_df.shape[1]
    names = list(stress_df.columns)

    # Compute weights
    if method == CompositeMethod.PCA and stress_df.shape[0] > k + 1:
        w = pca_weights(stress_df.values)
    elif method == CompositeMethod.INVERSE_VARIANCE:
        variances = stress_df.var()
        inv_var = 1.0 / variances.replace(0, np.nan)
        w = (inv_var / inv_var.sum()).values
    else:
        w = np.ones(k) / k

    weight_dict = {names[i]: float(w[i]) for i in range(k)}

    # Composite
    composite = (stress_df.values @ w)
    composite_series = pd.Series(composite, index=stress_df.index, name=f"{family.value}_composite")

    # Latest values
    latest_z = float(composite_series.iloc[-1]) if len(composite_series) > 0 else 0.0
    latest_pct = float(
        (composite_series.rank(pct=True).iloc[-1]) if len(composite_series) > 0 else 0.0
    )

    fc = FamilyComposite(
        family=family,
        composite_z=latest_z,
        percentile=latest_pct,
        n_indicators=k,
        weights=weight_dict,
    )

    return fc, stress_df


def compute_overall_market_stress(
    composites: list[FamilyComposite],
    family_weights: dict[IndicatorFamily, float] | None = None,
) -> float:
    """Compute Overall Market Stress score across families.

    OMS = Σ_F (w_F · Composite_F)
    """
    if not composites:
        return 0.0

    if family_weights is None:
        n = len(composites)
        family_weights = {c.family: 1.0 / n for c in composites}

    oms = sum(
        family_weights.get(c.family, 0.0) * c.composite_z
        for c in composites
    )
    return oms


# ---------------------------------------------------------------------------
# Theme confirmation / divergence
# ---------------------------------------------------------------------------

def compute_theme_confirmation(
    shock_vector: np.ndarray,
    indicator_z_scores: dict[str, float],
) -> float:
    """Compute market confirmation score for a theme.

    Confirmation = cosine_similarity(expected_indicator_directions, actual_z)

    Parameters
    ----------
    shock_vector : (D,) array — the theme's macro driver shocks
    indicator_z_scores : dict {indicator_name: current stress z-score}

    Returns
    -------
    float in [-1, +1]
        +1 = market fully prices the theme
         0 = market agnostic
        -1 = market prices the opposite
    """
    # Build expected and actual vectors over indicators that have data
    expected = []
    actual = []

    for meta in INDICATOR_REGISTRY:
        if meta.name not in indicator_z_scores:
            continue

        # Expected direction: weighted average of shock on mapped drivers
        exp_dir = 0.0
        for driver in meta.mapped_drivers:
            if driver in MACRO_DRIVERS:
                idx = MACRO_DRIVERS.index(driver)
                exp_dir += shock_vector[idx]
        if not meta.risk_on:
            exp_dir = -exp_dir  # flip for risk-off indicators

        expected.append(exp_dir)
        actual.append(indicator_z_scores[meta.name])

    if not expected:
        return 0.0

    return cosine_similarity(np.array(expected), np.array(actual))


def adjust_shock_for_confirmation(
    shock_vector: np.ndarray,
    confirmation: float,
    beta: float | None = None,
    cfg: EngineConfig | None = None,
) -> np.ndarray:
    """Dampen shock vector based on market confirmation.

    δ_adjusted = δ · (1 - β · Confirmation)

    High confirmation (already priced) → smaller shock.
    Negative confirmation (opposite) → amplified shock.
    """
    cfg = cfg or get_config()
    b = beta if beta is not None else cfg.indicators["confirmation_beta"]
    return shock_vector * (1.0 - b * confirmation)


# ---------------------------------------------------------------------------
# Convenience: full indicator pipeline
# ---------------------------------------------------------------------------

def run_indicator_pipeline(
    loader: IndicatorDataLoader,
    cfg: EngineConfig | None = None,
) -> tuple[list[FamilyComposite], dict[str, float], float]:
    """Run the full indicator pipeline across all families.

    Returns
    -------
    (composites, latest_z_scores, overall_market_stress)
    """
    cfg = cfg or get_config()
    method = CompositeMethod(cfg.indicators["composite_method"])
    lookback = cfg.indicators["lookback_trading_days"]

    composites = []
    all_z: dict[str, float] = {}

    for family in IndicatorFamily:
        fc, stress_df = compute_family_composite(
            family, loader, method=method, lookback=lookback, cfg=cfg
        )
        composites.append(fc)
        if not stress_df.empty:
            for col in stress_df.columns:
                all_z[col] = float(stress_df[col].iloc[-1])

    oms = compute_overall_market_stress(composites)
    return composites, all_z, oms
