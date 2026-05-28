"""Theme Engine — Module 6: Portfolio Impact & Positioning.

Computes portfolio-level returns, stressed covariance, MCTR,
CVaR, positioning signals, and hedge recommendations.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from .config import EngineConfig, get_config
from .models import (
    ASSET_CLASSES,
    MACRO_DRIVERS,
    AssetReturn,
    HedgeRecommendation,
    PortfolioPosition,
    PortfolioSummary,
    PositionSignal,
    ScenarioCard,
    Theme,
)
from .utils import cornish_fisher_cvar, stressed_covariance


def portfolio_return(
    weights: np.ndarray,
    returns: np.ndarray,
) -> float:
    """R_portfolio = w' · R."""
    return float(weights @ returns)


def portfolio_risk(
    weights: np.ndarray,
    cov: np.ndarray,
) -> float:
    """σ_portfolio = sqrt(w' · Σ · w)."""
    return float(np.sqrt(weights @ cov @ weights))


def marginal_contribution_to_risk(
    weights: np.ndarray,
    cov: np.ndarray,
) -> np.ndarray:
    """MCTR_a = (Σ · w)_a / σ_portfolio."""
    sigma_p = portfolio_risk(weights, cov)
    if sigma_p < 1e-12:
        return np.zeros_like(weights)
    return (cov @ weights) / sigma_p


def compute_stressed_covariance(
    delta: np.ndarray,
    cfg: EngineConfig | None = None,
) -> np.ndarray:
    """Build scenario-stressed covariance matrix for asset classes.

    Σ_stressed = B · Σ_drivers_stressed · B' + diag(σ²_residual)
    """
    cfg = cfg or get_config()
    return stressed_covariance(
        B=cfg.exposure_matrix,
        sigma_drivers=cfg.driver_volatilities,
        rho_drivers=cfg.driver_covariance,
        delta=delta,
        stress_mult=cfg.portfolio["stress_multiplier"],
        delta_rho=cfg.portfolio["delta_rho_stress"],
        sigma_residual=cfg.residual_vol_vector(),
    )


def positioning_signal(
    sharpe: float,
    mctr: float,
    avg_mctr: float,
    cfg: EngineConfig | None = None,
) -> PositionSignal:
    """Determine OW / UW / N based on risk-adjusted return and MCTR.

    - OVERWEIGHT  if Sharpe > threshold_high AND MCTR < avg
    - UNDERWEIGHT if Sharpe < threshold_low  OR  MCTR > avg
    - NEUTRAL     otherwise
    """
    cfg = cfg or get_config()
    pos = cfg.positioning
    if sharpe > pos["overweight_sharpe"] and mctr < avg_mctr:
        return PositionSignal.OVERWEIGHT
    elif sharpe < pos["underweight_sharpe"] or mctr > avg_mctr * 1.2:
        return PositionSignal.UNDERWEIGHT
    return PositionSignal.NEUTRAL


def compute_portfolio_positions(
    delta: np.ndarray,
    asset_returns: list[AssetReturn],
    weights: np.ndarray | None = None,
    theme_drivers: dict[str, str] | None = None,
    cfg: EngineConfig | None = None,
) -> list[PortfolioPosition]:
    """Compute full positioning for each asset class.

    Parameters
    ----------
    delta : (D,) — calibrated shock vector
    asset_returns : list[AssetReturn] from factor_model
    weights : (A,) portfolio weights (default from config)
    theme_drivers : {asset_class: dominant_theme_name}
    cfg : EngineConfig
    """
    cfg = cfg or get_config()
    if weights is None:
        weights = cfg.default_weights_vector()
    theme_drivers = theme_drivers or {}

    cov_stressed = compute_stressed_covariance(delta, cfg)
    mctr = marginal_contribution_to_risk(weights, cov_stressed)
    avg_mctr = float(np.mean(mctr))

    positions = []
    for i, ar in enumerate(asset_returns):
        vol = float(np.sqrt(cov_stressed[i, i]))
        sharpe = ar.scenario_return / vol if vol > 1e-12 else 0.0
        signal = positioning_signal(sharpe, mctr[i], avg_mctr, cfg)

        positions.append(PortfolioPosition(
            asset_class=ar.asset_class,
            current_weight=float(weights[i]),
            scenario_return=ar.scenario_return,
            scenario_risk=vol,
            risk_adj_return=sharpe,
            mctr=float(mctr[i]),
            signal=signal,
            key_theme_driver=theme_drivers.get(ar.asset_class, ""),
        ))

    return positions


def identify_hedges(
    themes: list[Theme],
    shock_vectors: list[np.ndarray],
    confirmations: list[float],
    portfolio_impacts: list[float],
    cfg: EngineConfig | None = None,
) -> list[HedgeRecommendation]:
    """Flag unpriced themes with large negative portfolio impact.

    Hedge candidate if:
      - confirmation < threshold (market hasn't priced it)
      - portfolio impact < negative threshold
    """
    cfg = cfg or get_config()
    pos = cfg.positioning
    conf_thresh = pos["hedge_confirmation_threshold"]
    impact_thresh = pos["hedge_impact_threshold"]

    hedges = []
    for i, theme in enumerate(themes):
        conf = confirmations[i] if i < len(confirmations) else 0.0
        impact = portfolio_impacts[i] if i < len(portfolio_impacts) else 0.0

        if conf < conf_thresh and impact < impact_thresh:
            # Suggest instruments based on theme category
            instruments = _suggest_hedge_instruments(theme, shock_vectors[i])
            hedges.append(HedgeRecommendation(
                theme_name=theme.name,
                confirmation_score=conf,
                portfolio_impact=impact,
                suggested_instruments=instruments,
                rationale=(
                    f"Theme '{theme.name}' is poorly priced (confirmation={conf:.2f}) "
                    f"with significant negative portfolio impact ({impact:.2%}). "
                    f"Consider protective positioning."
                ),
            ))

    return hedges


def _suggest_hedge_instruments(
    theme: Theme,
    shock: np.ndarray,
) -> list[str]:
    """Heuristic hedge instrument suggestions based on shock profile."""
    instruments = []
    driver_map = dict(zip(MACRO_DRIVERS, shock))

    # ERP shock → equity puts / VIX calls
    if driver_map.get("equity_risk_premium", 0) > 0.3:
        instruments.extend(["SPX Put Options", "VIX Call Options"])

    # Credit shock → CDS protection
    if driver_map.get("credit_premium", 0) > 0.3:
        instruments.extend(["CDX HY Protection", "CDX IG Protection"])

    # Rates shock → duration hedge
    if abs(driver_map.get("real_rates", 0)) > 0.3:
        instruments.append("Treasury Futures (duration hedge)")

    # Commodity shock → commodity options
    if driver_map.get("commodity_supply", 0) > 0.3:
        instruments.append("Crude Oil Call Options")

    # FX shock → USD hedge
    if driver_map.get("fx_risk_appetite", 0) < -0.3:
        instruments.extend(["DXY Long", "EM FX Puts"])

    # Policy shock → gold / vol
    if driver_map.get("policy_uncertainty", 0) > 0.3:
        instruments.extend(["Gold", "Long Volatility Strategy"])

    return instruments if instruments else ["Diversified Tail Risk Hedge"]


def build_portfolio_summary(
    themes: list[Theme],
    individual_shocks: list[np.ndarray],
    aggregate_shock: np.ndarray,
    asset_returns: list[AssetReturn],
    confirmations: list[float],
    scenario_cards: list[ScenarioCard],
    weights: np.ndarray | None = None,
    cfg: EngineConfig | None = None,
) -> PortfolioSummary:
    """Build complete portfolio summary output.

    Ties together all modules into the final portfolio-level output.
    """
    cfg = cfg or get_config()
    if weights is None:
        weights = cfg.default_weights_vector()

    # Determine dominant theme for each asset class
    theme_drivers: dict[str, str] = {}
    if themes and individual_shocks:
        B = cfg.exposure_matrix
        for a_idx, ac in enumerate(ASSET_CLASSES):
            max_impact = 0.0
            driver = ""
            for t_idx, theme in enumerate(themes):
                impact = float(np.abs(B[a_idx] @ individual_shocks[t_idx]) * theme.strength)
                if impact > max_impact:
                    max_impact = impact
                    driver = theme.name
            theme_drivers[ac] = driver

    positions = compute_portfolio_positions(
        aggregate_shock, asset_returns, weights, theme_drivers, cfg
    )

    # Portfolio-level metrics
    ret_vec = np.array([ar.scenario_return for ar in asset_returns])
    cov_stressed = compute_stressed_covariance(aggregate_shock, cfg)
    p_ret = portfolio_return(weights, ret_vec)
    p_risk = portfolio_risk(weights, cov_stressed)

    # Portfolio impacts per theme
    portfolio_impacts = []
    for shock in individual_shocks:
        theme_ret = cfg.baseline_returns_vector() + cfg.exposure_matrix @ shock
        portfolio_impacts.append(float(weights @ theme_ret) - float(weights @ cfg.baseline_returns_vector()))

    hedges = identify_hedges(themes, individual_shocks, confirmations, portfolio_impacts, cfg)

    return PortfolioSummary(
        portfolio_return=p_ret,
        portfolio_risk=p_risk,
        positions=positions,
        hedges=hedges,
        scenario_cards=scenario_cards,
    )
