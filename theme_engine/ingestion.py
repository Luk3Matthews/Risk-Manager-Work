"""Theme Engine — Module 2: Theme Ingestion & Evidence Scoring.

Handles:
  (a) Scoring evidence items for usefulness
  (b) Aggregating evidence into theme confidence
  (c) Computing theme strength
  (d) Managing the active theme ledger
"""
from __future__ import annotations

import math
from datetime import date
from typing import Sequence

from .config import EngineConfig, get_config
from .models import (
    EvidenceItem,
    Theme,
    ThemeStatus,
)
from .utils import recency_weight


def score_evidence(
    item: EvidenceItem,
    cfg: EngineConfig | None = None,
) -> float:
    """Score a single evidence item for usefulness.

    U = w_cred·credibility + w_time·timeliness + w_corr·min(corr/k, 1)

    Returns the computed usefulness score (also stored on the item).
    """
    cfg = cfg or get_config()
    ev = cfg.evidence
    return item.compute_usefulness(
        w_cred=ev["w_credibility"],
        w_time=ev["w_timeliness"],
        w_corr=ev["w_corroboration"],
        k=ev["corroboration_saturation_k"],
    )


def score_all_evidence(
    items: Sequence[EvidenceItem],
    cfg: EngineConfig | None = None,
) -> list[float]:
    """Score a batch of evidence items. Returns list of usefulness scores."""
    return [score_evidence(item, cfg) for item in items]


def compute_theme_confidence(
    theme: Theme,
    reference_date: date | None = None,
    cfg: EngineConfig | None = None,
) -> float:
    """Aggregate evidence into theme-level confidence.

    C = 1 - ∏_{i=1..N} (1 - U_i · w_i)

    where w_i = exp(-λ · Δt_i) is the recency decay weight.

    Parameters
    ----------
    theme : Theme
        The theme whose confidence to compute.
    reference_date : date, optional
        Date to compute recency against (default: today).
    cfg : EngineConfig, optional

    Returns
    -------
    float — confidence in [0, 1]. Also sets theme.confidence.
    """
    cfg = cfg or get_config()
    ref = reference_date or date.today()
    half_life = cfg.evidence["half_life_days"]

    if not theme.evidence:
        theme.confidence = 0.0
        return 0.0

    # Ensure all evidence items are scored
    for item in theme.evidence:
        if item.usefulness_score <= 0.0:
            score_evidence(item, cfg)

    # Product term
    product = 1.0
    for item in theme.evidence:
        days_ago = (ref - item.date).days
        w = recency_weight(days_ago, half_life)
        product *= (1.0 - item.usefulness_score * w)

    theme.confidence = 1.0 - product
    return theme.confidence


def compute_theme_strength(
    theme: Theme,
    alpha: float | None = None,
    cfg: EngineConfig | None = None,
) -> float:
    """Compute theme strength as a blend of likelihood and confidence.

    S = likelihood^α · confidence^(1-α)

    Parameters
    ----------
    theme : Theme
    alpha : float, optional — blend parameter (default from config: 0.5)

    Returns
    -------
    float — strength in [0, 1]. Also sets theme.strength.
    """
    cfg = cfg or get_config()
    a = alpha if alpha is not None else cfg.evidence["theme_strength_alpha"]

    # Avoid 0^x domain issues
    lik = max(theme.likelihood, 1e-12)
    conf = max(theme.confidence, 1e-12)

    theme.strength = (lik ** a) * (conf ** (1.0 - a))
    return theme.strength


def refresh_theme(
    theme: Theme,
    reference_date: date | None = None,
    cfg: EngineConfig | None = None,
) -> Theme:
    """Re-score evidence, recompute confidence and strength for a theme."""
    cfg = cfg or get_config()
    score_all_evidence(theme.evidence, cfg)
    compute_theme_confidence(theme, reference_date, cfg)
    compute_theme_strength(theme, cfg=cfg)
    theme.last_updated = date.today()
    return theme


def build_active_ledger(
    themes: Sequence[Theme],
    max_themes: int = 25,
    min_strength: float = 0.0,
    reference_date: date | None = None,
    cfg: EngineConfig | None = None,
) -> list[Theme]:
    """Build the active theme ledger.

    1. Refresh all themes (score evidence, confidence, strength).
    2. Filter to ACTIVE / MONITORING status.
    3. Filter by minimum strength.
    4. Sort by strength descending.
    5. Return top N.
    """
    cfg = cfg or get_config()
    active = []
    for t in themes:
        refresh_theme(t, reference_date, cfg)
        if t.status in (ThemeStatus.ACTIVE, ThemeStatus.MONITORING):
            if t.strength >= min_strength:
                active.append(t)

    active.sort(key=lambda t: t.strength, reverse=True)
    return active[:max_themes]


def add_evidence_to_theme(
    theme: Theme,
    items: Sequence[EvidenceItem],
    cfg: EngineConfig | None = None,
) -> Theme:
    """Add evidence items to a theme and refresh scores."""
    cfg = cfg or get_config()
    for item in items:
        score_evidence(item, cfg)
        theme.evidence.append(item)
    compute_theme_confidence(theme, cfg=cfg)
    compute_theme_strength(theme, cfg=cfg)
    theme.last_updated = date.today()
    return theme
