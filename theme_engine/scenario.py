"""Theme Engine — Module 3: Theme → Macro Driver Shock Vectors.

Translates active themes into calibrated macro driver shock vectors
and aggregates across multiple themes with diversification adjustment.
"""
from __future__ import annotations

import numpy as np

from .config import EngineConfig, get_config
from .models import (
    MACRO_DRIVERS,
    Direction,
    Magnitude,
    RiskSignal,
    SignalDirection,
    Theme,
    ThemeCategory,
)
from .utils import diversification_scalar


def _direction_sign(d: Direction | SignalDirection) -> float:
    """Convert direction enum to numeric sign.

    The transmission matrix T already encodes the TYPICAL directional
    relationship when a theme fires (e.g., geopolitical → growth down).

    - BEARISH / DOWN → +1.0  (affirm T: the risk scenario plays out)
    - BULLISH / UP   → -1.0  (reverse T: opposite of typical, e.g., peace dividend)
    - AMBIGUOUS / NEUTRAL → 0.0
    """
    if d in (Direction.BEARISH, SignalDirection.DOWN):
        return 1.0
    elif d in (Direction.BULLISH, SignalDirection.UP):
        return -1.0
    return 0.0


def compute_shock_vector(
    theme: Theme,
    overrides: dict[str, float] | None = None,
    cfg: EngineConfig | None = None,
) -> np.ndarray:
    """Compute the macro driver shock vector for a single theme.

    δ_j = magnitude_σ · T[:, category] · direction_sign

    Parameters
    ----------
    theme : Theme
    overrides : dict, optional
        Manual overrides {driver_name: shock_value} that replace the
        default transmission-matrix entry.
    cfg : EngineConfig, optional

    Returns
    -------
    (D,) numpy array — shock vector in standardised σ units.
    """
    cfg = cfg or get_config()

    # Direction sign
    sign = _direction_sign(theme.direction)

    # Magnitude multiplier (use MODERATE as default)
    # Themes don't have magnitude directly; derive from evidence strength
    # Map strength to magnitude: >0.8 EXTREME, >0.6 LARGE, >0.3 MODERATE, else SMALL
    if theme.strength >= 0.8:
        mag = Magnitude.EXTREME
    elif theme.strength >= 0.6:
        mag = Magnitude.LARGE
    elif theme.strength >= 0.3:
        mag = Magnitude.MODERATE
    else:
        mag = Magnitude.SMALL
    mag_sigma = cfg.magnitude_sigma(mag)

    # Transmission column for this theme's category
    t_col = cfg.transmission_column(theme.category)

    # Base shock vector
    delta = mag_sigma * t_col * sign

    # Apply overrides
    if overrides:
        for driver, val in overrides.items():
            if driver in MACRO_DRIVERS:
                idx = MACRO_DRIVERS.index(driver)
                delta[idx] = val

    return delta


def compute_signal_shock_vector(
    signal: RiskSignal,
    theme: Theme | None = None,
    cfg: EngineConfig | None = None,
) -> np.ndarray:
    """Compute shock vector from a RiskSignal directly.

    If the signal has explicit macro_driver_shocks, use those.
    Otherwise fall back to the theme's transmission matrix.
    """
    cfg = cfg or get_config()

    if signal.macro_driver_shocks:
        delta = np.zeros(len(MACRO_DRIVERS), dtype=np.float64)
        for driver, val in signal.macro_driver_shocks.items():
            if driver in MACRO_DRIVERS:
                idx = MACRO_DRIVERS.index(driver)
                delta[idx] = val
        return delta

    if theme is None:
        return np.zeros(len(MACRO_DRIVERS), dtype=np.float64)

    # Use theme's transmission with signal's magnitude
    sign = _direction_sign(signal.direction)
    mag_sigma = cfg.magnitude_sigma(signal.magnitude)
    t_col = cfg.transmission_column(theme.category)
    return mag_sigma * t_col * sign


def build_theme_correlation_matrix(
    themes: list[Theme],
    cfg: EngineConfig | None = None,
) -> np.ndarray:
    """Build correlation matrix between themes based on category proximity.

    Same category → ρ = same_category_correlation (default 0.6)
    Different category → ρ = cross_category_correlation (default 0.2)
    """
    cfg = cfg or get_config()
    n = len(themes)
    rho_same = cfg.scenario["same_category_correlation"]
    rho_cross = cfg.scenario["cross_category_correlation"]

    corr = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            if themes[i].category == themes[j].category:
                corr[i, j] = rho_same
            else:
                corr[i, j] = rho_cross
            corr[j, i] = corr[i, j]
    return corr


def aggregate_shocks(
    themes: list[Theme],
    shock_vectors: list[np.ndarray],
    cfg: EngineConfig | None = None,
) -> np.ndarray:
    """Aggregate shock vectors across multiple themes with diversification.

    δ_total = Σ_j (S_j · δ_j) · diversification_scalar

    Parameters
    ----------
    themes : list[Theme]       — active themes (must have .strength computed)
    shock_vectors : list[ndarray] — corresponding shock vectors (D,)

    Returns
    -------
    (D,) aggregate shock vector.
    """
    cfg = cfg or get_config()
    n = len(themes)
    if n == 0:
        return np.zeros(len(MACRO_DRIVERS), dtype=np.float64)

    strengths = np.array([t.strength for t in themes], dtype=np.float64)
    corr = build_theme_correlation_matrix(themes, cfg)
    div_scalar = diversification_scalar(strengths, corr)

    # Weighted sum
    delta_total = np.zeros(len(MACRO_DRIVERS), dtype=np.float64)
    for j in range(n):
        delta_total += strengths[j] * shock_vectors[j]

    return delta_total * div_scalar


def run_scenario_shocks(
    themes: list[Theme],
    overrides: dict[str, dict[str, float]] | None = None,
    cfg: EngineConfig | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Full pipeline: compute individual + aggregated shock vectors.

    Parameters
    ----------
    themes : list[Theme] — active themes with strength computed
    overrides : dict, optional — {theme_id: {driver: value}} overrides

    Returns
    -------
    (individual_shocks, aggregate_shock)
        individual_shocks : list of (D,) arrays
        aggregate_shock   : (D,) array
    """
    cfg = cfg or get_config()
    overrides = overrides or {}

    individual = []
    for theme in themes:
        ovr = overrides.get(theme.theme_id)
        delta = compute_shock_vector(theme, overrides=ovr, cfg=cfg)
        individual.append(delta)

    agg = aggregate_shocks(themes, individual, cfg)
    return individual, agg
