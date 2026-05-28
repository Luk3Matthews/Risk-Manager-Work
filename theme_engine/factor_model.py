"""Theme Engine — Module 5: Macro Drivers → Asset-Class Returns.

Factor model:  R_a = α_a + Σ_d (β_{a,d} · δ_d) + ε_a

Supports both structural (default) and regression-based β estimation.
"""
from __future__ import annotations

import numpy as np

from .config import EngineConfig, get_config
from .models import ASSET_CLASSES, MACRO_DRIVERS, AssetReturn


def scenario_returns(
    delta: np.ndarray,
    cfg: EngineConfig | None = None,
) -> np.ndarray:
    """Compute asset-class scenario returns.

    R_scenario = α + B · δ

    Parameters
    ----------
    delta : (D,) array — macro driver shock vector (after calibration)
    cfg : EngineConfig

    Returns
    -------
    (A,) array — expected scenario returns for each asset class.
    """
    cfg = cfg or get_config()
    alpha = cfg.baseline_returns_vector()      # (A,)
    B = cfg.exposure_matrix                     # (A, D)
    return alpha + B @ delta


def scenario_return_variance(
    delta: np.ndarray,
    cfg: EngineConfig | None = None,
    shock_uncertainty: float = 0.25,
) -> np.ndarray:
    """Approximate variance of scenario returns.

    Var(R_a) ≈ Σ_d β²_{a,d} · Var(δ_d) + σ²_ε_a

    Parameters
    ----------
    delta : (D,) — shock vector (used to scale Var(δ_d))
    cfg : EngineConfig
    shock_uncertainty : float — fractional uncertainty on shock magnitude
        e.g., MODERATE = 1σ ± 0.25σ → shock_uncertainty = 0.25

    Returns
    -------
    (A,) array — variance of scenario return for each asset class.
    """
    cfg = cfg or get_config()
    B = cfg.exposure_matrix                     # (A, D)
    sigma_res = cfg.residual_vol_vector()       # (A,)

    # Var(δ_d) ≈ (shock_uncertainty · |δ_d|)² + small floor
    var_delta = (shock_uncertainty * np.abs(delta)) ** 2 + 1e-6

    # Var(R_a) = Σ_d β²_{a,d} · Var(δ_d) + σ²_ε
    var_r = (B ** 2) @ var_delta + sigma_res ** 2
    return var_r


def scenario_confidence_interval(
    delta: np.ndarray,
    z: float = 1.96,
    cfg: EngineConfig | None = None,
    shock_uncertainty: float = 0.25,
) -> list[AssetReturn]:
    """Full scenario return estimates with 95% CI.

    Parameters
    ----------
    delta : (D,) — calibrated shock vector
    z : float — z-score for CI (default 1.96 = 95%)
    cfg : EngineConfig
    shock_uncertainty : float

    Returns
    -------
    list[AssetReturn] — one per asset class with return, CI, vol.
    """
    cfg = cfg or get_config()
    alpha = cfg.baseline_returns_vector()
    R = scenario_returns(delta, cfg)
    Var = scenario_return_variance(delta, cfg, shock_uncertainty)
    Sigma = np.sqrt(Var)

    results = []
    for i, ac in enumerate(ASSET_CLASSES):
        results.append(AssetReturn(
            asset_class=ac,
            baseline_return=float(alpha[i]),
            scenario_return=float(R[i]),
            ci_lower=float(R[i] - z * Sigma[i]),
            ci_upper=float(R[i] + z * Sigma[i]),
            scenario_vol=float(Sigma[i]),
        ))
    return results
