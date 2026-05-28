"""Theme Engine — Utility functions.

Rolling statistics, z-scores, percentile ranks, cosine similarity,
Cornish-Fisher CVaR, and other reusable helpers.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Rolling statistics
# ---------------------------------------------------------------------------

def rolling_z_score(
    series: pd.Series,
    lookback: int = 1260,
    min_periods: int = 252,
) -> pd.Series:
    """Compute rolling z-score: z(t) = (x(t) - μ(t,L)) / σ(t,L).

    Parameters
    ----------
    series : pd.Series
        Raw time series (daily).
    lookback : int
        Rolling window length in trading days (default 5y = 1260).
    min_periods : int
        Minimum observations before producing a value.

    Returns
    -------
    pd.Series of z-scores (same index as input).
    """
    mu = series.rolling(window=lookback, min_periods=min_periods).mean()
    sigma = series.rolling(window=lookback, min_periods=min_periods).std()
    return (series - mu) / sigma.replace(0, np.nan)


def rolling_percentile(
    series: pd.Series,
    lookback: int = 1260,
    min_periods: int = 252,
) -> pd.Series:
    """Compute rolling percentile rank within the lookback window.

    Returns values in [0, 1].
    """
    def _pctile(window: np.ndarray) -> float:
        if len(window) < 2:
            return np.nan
        return sp_stats.percentileofscore(window, window[-1], kind="rank") / 100.0

    return series.rolling(window=lookback, min_periods=min_periods).apply(
        _pctile, raw=True
    )


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Returns 0 if either is zero."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Recency decay
# ---------------------------------------------------------------------------

def recency_weight(days_ago: float, half_life_days: float = 30.0) -> float:
    """Exponential recency weight: w = exp(-λ * Δt), λ = ln(2)/half_life."""
    lam = math.log(2) / half_life_days
    return math.exp(-lam * max(days_ago, 0.0))


# ---------------------------------------------------------------------------
# Cornish-Fisher CVaR
# ---------------------------------------------------------------------------

def cornish_fisher_cvar(
    mu: float,
    sigma: float,
    skewness: float = 0.0,
    alpha: float = 0.05,
) -> float:
    """Cornish-Fisher expansion CVaR approximation.

    CVaR ≈ μ + σ · [φ(z_α) / (1-α)] · [1 + (γ/6)(z²_α - 1)]

    Parameters
    ----------
    mu : float      – expected return
    sigma : float   – volatility
    skewness : float – skewness (γ)
    alpha : float   – tail probability (default 5%)

    Returns
    -------
    float – CVaR (negative = loss)
    """
    z_alpha = sp_stats.norm.ppf(alpha)
    phi_z = sp_stats.norm.pdf(z_alpha)
    cf_adj = 1.0 + (skewness / 6.0) * (z_alpha ** 2 - 1.0)
    return mu + sigma * (phi_z / (1.0 - alpha)) * cf_adj


# ---------------------------------------------------------------------------
# Diversification scalar
# ---------------------------------------------------------------------------

def diversification_scalar(
    strengths: np.ndarray,
    correlations: np.ndarray,
) -> float:
    """Compute the diversification adjustment scalar.

    scalar = sqrt(Σ_j S_j² + 2·Σ_{i<j} S_i·S_j·ρ_{ij}) / Σ_j S_j

    Parameters
    ----------
    strengths : (N,) array of theme strengths
    correlations : (N, N) correlation matrix between themes

    Returns
    -------
    float – scalar in (0, 1] (1 = no diversification benefit)
    """
    n = len(strengths)
    if n == 0:
        return 1.0
    total_s = strengths.sum()
    if total_s < 1e-12:
        return 1.0

    variance = 0.0
    for i in range(n):
        variance += strengths[i] ** 2
        for j in range(i + 1, n):
            variance += 2 * strengths[i] * strengths[j] * correlations[i, j]

    return math.sqrt(max(variance, 0.0)) / total_s


# ---------------------------------------------------------------------------
# PCA composite weights
# ---------------------------------------------------------------------------

def pca_weights(data: np.ndarray) -> np.ndarray:
    """Compute weights proportional to first-PC loadings.

    Parameters
    ----------
    data : (T, K) array — T observations of K indicators

    Returns
    -------
    (K,) normalised weight vector (sums to 1, all non-negative via abs).
    """
    if data.shape[0] < data.shape[1] + 1:
        # Not enough data — fall back to equal weight
        k = data.shape[1]
        return np.ones(k) / k

    # Centre
    data_c = data - data.mean(axis=0)
    cov = np.cov(data_c, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # First PC is the one with the largest eigenvalue (last in eigh output)
    pc1 = np.abs(eigvecs[:, -1])
    return pc1 / pc1.sum()


# ---------------------------------------------------------------------------
# Stressed covariance
# ---------------------------------------------------------------------------

def stressed_covariance(
    B: np.ndarray,
    sigma_drivers: np.ndarray,
    rho_drivers: np.ndarray,
    delta: np.ndarray,
    stress_mult: float = 0.5,
    delta_rho: float = 0.1,
    sigma_residual: np.ndarray | None = None,
) -> np.ndarray:
    """Compute scenario-stressed covariance matrix for asset classes.

    Σ_stressed = B · Σ_drivers_stressed · B' + diag(σ²_residual)

    Stress adjustments:
      σ_d_stressed = σ_d · (1 + |δ_d| · stress_mult)
      ρ_stressed = min(ρ + Δρ, 1)  where Δρ = delta_rho · max(|δ_d1|, |δ_d2|)
    """
    n_d = len(sigma_drivers)
    n_a = B.shape[0]

    # Stressed volatilities
    sig_stressed = sigma_drivers * (1.0 + np.abs(delta) * stress_mult)

    # Stressed correlations
    rho_stressed = rho_drivers.copy()
    for i in range(n_d):
        for j in range(i + 1, n_d):
            adj = delta_rho * max(abs(delta[i]), abs(delta[j]))
            rho_stressed[i, j] = min(rho_drivers[i, j] + adj, 1.0)
            rho_stressed[j, i] = rho_stressed[i, j]

    # Stressed covariance of drivers
    D = np.diag(sig_stressed)
    Sigma_d = D @ rho_stressed @ D

    # Asset covariance
    Sigma_a = B @ Sigma_d @ B.T

    # Add residual
    if sigma_residual is not None:
        Sigma_a += np.diag(sigma_residual ** 2)

    return Sigma_a
