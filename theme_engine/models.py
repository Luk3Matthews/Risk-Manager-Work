"""Theme-Driven Macro Scenario Engine — Pydantic Data Models.

Defines all core data structures: Theme, EvidenceItem, RiskSignal,
IndicatorReading, PortfolioPosition, ScenarioCard, etc.
"""
from __future__ import annotations

import uuid
from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ThemeCategory(str, Enum):
    GEOPOLITICAL = "GEOPOLITICAL"
    GROWTH = "GROWTH"
    INFLATION = "INFLATION"
    LIQUIDITY = "LIQUIDITY"
    STRUCTURAL = "STRUCTURAL"
    POLICY = "POLICY"
    VALUATION = "VALUATION"
    CONTAGION = "CONTAGION"


class Direction(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    AMBIGUOUS = "AMBIGUOUS"


class SignalDirection(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    NEUTRAL = "NEUTRAL"


class Horizon(str, Enum):
    NEAR_TERM = "NEAR_TERM"      # < 3 months
    MEDIUM_TERM = "MEDIUM_TERM"  # 3-12 months
    LONG_TERM = "LONG_TERM"      # > 12 months


class Magnitude(str, Enum):
    SMALL = "SMALL"       # 0.5σ
    MODERATE = "MODERATE"  # 1.0σ
    LARGE = "LARGE"       # 1.5σ
    EXTREME = "EXTREME"   # 2.5σ


class ThemeStatus(str, Enum):
    DRAFT = "DRAFT"
    ACTIVE = "ACTIVE"
    MONITORING = "MONITORING"
    RETIRED = "RETIRED"


class SignalStatus(str, Enum):
    ACTIVE = "ACTIVE"
    MONITORING = "MONITORING"
    RETIRED = "RETIRED"


class PositionSignal(str, Enum):
    OVERWEIGHT = "OVERWEIGHT"
    UNDERWEIGHT = "UNDERWEIGHT"
    NEUTRAL = "NEUTRAL"


class CompositeMethod(str, Enum):
    PCA = "pca"
    EQUAL = "equal"
    INVERSE_VARIANCE = "inverse_variance"


# ---------------------------------------------------------------------------
# Macro Drivers & Asset Classes (canonical ordering)
# ---------------------------------------------------------------------------

MACRO_DRIVERS: list[str] = [
    "expected_growth",
    "expected_inflation",
    "real_rates",
    "equity_risk_premium",
    "credit_premium",
    "liquidity",
    "commodity_supply",
    "fx_risk_appetite",
    "policy_uncertainty",
]

ASSET_CLASSES: list[str] = [
    "international_equities",
    "australian_equities",
    "hedge_funds",
    "infrastructure",
    "property",
    "private_credit",
    "inflation_linked_bonds",
    "cash",
    "australian_bonds",
    "us_bonds",
    "aaa_overlay",
    "other_strategies",
]

ASSET_CLASS_LABELS: dict[str, str] = {
    "international_equities": "International Equities",
    "australian_equities": "Australian Equities",
    "hedge_funds": "Hedge Funds",
    "infrastructure": "Infrastructure",
    "property": "Property",
    "private_credit": "Private Credit",
    "inflation_linked_bonds": "Inflation Linked Bonds",
    "cash": "Cash",
    "australian_bonds": "Australian Bonds",
    "us_bonds": "US Bonds",
    "aaa_overlay": "AAA Overlay and WoP",
    "other_strategies": "Other Strategies",
}

# Mapping from BNY Data Vault *VFMC_Asset Class names to internal keys
VFMC_ASSET_CLASS_MAP: dict[str, str] = {
    "International Equities": "international_equities",
    "Australian Equities": "australian_equities",
    "Hedge Funds": "hedge_funds",
    "Infrastructure": "infrastructure",
    "Property": "property",
    "Private Credit": "private_credit",
    "Inflation Linked Bonds": "inflation_linked_bonds",
    "Cash": "cash",
    "Australian Bonds": "australian_bonds",
    "US Bonds": "us_bonds",
    "AAA Overlay and WoP": "aaa_overlay",
    "Other Strategies": "other_strategies",
}

# VFMC asset class category groupings (matching BNY Data Vault names)
ASSET_CLASS_CATEGORIES: dict[str, str] = {
    "international_equities": "Equity",
    "australian_equities": "Equity",
    "hedge_funds": "Alternative Strategies",
    "infrastructure": "Real Assets",
    "property": "Real Assets",
    "private_credit": "Credit",
    "inflation_linked_bonds": "Cash and Fixed Interest",
    "cash": "Cash and Fixed Interest",
    "australian_bonds": "Cash and Fixed Interest",
    "us_bonds": "Cash and Fixed Interest",
    "aaa_overlay": "Other",
    "other_strategies": "Other",
}

DRIVER_LABELS: dict[str, str] = {
    "expected_growth": "Expected Growth",
    "expected_inflation": "Expected Inflation",
    "real_rates": "Real Rates",
    "equity_risk_premium": "Equity Risk Premium",
    "credit_premium": "Credit Premium",
    "liquidity": "Liquidity",
    "commodity_supply": "Commodity Supply",
    "fx_risk_appetite": "FX Risk Appetite",
    "policy_uncertainty": "Policy Uncertainty",
}


# ---------------------------------------------------------------------------
# Evidence & Change-tracking
# ---------------------------------------------------------------------------

def _uid() -> str:
    return uuid.uuid4().hex[:12]


class EvidenceItem(BaseModel):
    """A single piece of evidence (news article, research note, data point)."""
    source: str
    date: date
    title: str
    url: str | None = None
    credibility_score: float = Field(ge=0.0, le=1.0, default=0.5)
    timeliness_score: float = Field(ge=0.0, le=1.0, default=0.5)
    corroboration_count: int = Field(ge=0, default=0)
    usefulness_score: float = Field(ge=0.0, le=1.0, default=0.0)

    def compute_usefulness(
        self,
        w_cred: float = 0.4,
        w_time: float = 0.3,
        w_corr: float = 0.3,
        k: int = 3,
    ) -> float:
        """U = w_cred*cred + w_time*time + w_corr*min(corr/k, 1)."""
        self.usefulness_score = (
            w_cred * self.credibility_score
            + w_time * self.timeliness_score
            + w_corr * min(self.corroboration_count / k, 1.0)
        )
        return self.usefulness_score


class ChangeEntry(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    field: str
    old_value: str
    new_value: str
    reason: str = ""


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

class Theme(BaseModel):
    """The atomic unit of the scenario engine."""
    theme_id: str = Field(default_factory=_uid)
    name: str
    category: ThemeCategory
    narrative: str = ""
    direction: Direction = Direction.AMBIGUOUS
    horizon: Horizon = Horizon.MEDIUM_TERM
    likelihood: float = Field(ge=0.0, le=1.0, default=0.5)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    historical_analogue: str = ""
    first_observed: date = Field(default_factory=date.today)
    last_updated: date = Field(default_factory=date.today)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    status: ThemeStatus = ThemeStatus.DRAFT
    change_log: list[ChangeEntry] = Field(default_factory=list)

    # Computed fields (set by engine)
    strength: float = 0.0
    confirmation_score: float = 0.0


# ---------------------------------------------------------------------------
# Risk Signal
# ---------------------------------------------------------------------------

class RiskSignal(BaseModel):
    """A structured risk signal derived from a theme."""
    signal_id: str = Field(default_factory=_uid)
    parent_theme_id: str
    factor: str
    direction: SignalDirection = SignalDirection.NEUTRAL
    magnitude: Magnitude = Magnitude.MODERATE
    horizon: Horizon = Horizon.MEDIUM_TERM
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    macro_driver_shocks: dict[str, float] = Field(default_factory=dict)
    linked_indicators: list[str] = Field(default_factory=list)
    status: SignalStatus = SignalStatus.ACTIVE


# ---------------------------------------------------------------------------
# Indicator Family
# ---------------------------------------------------------------------------

class IndicatorFamily(str, Enum):
    MARKET_RISK = "MARKET_RISK"
    GEOPOLITICAL_RISK = "GEOPOLITICAL_RISK"
    EXPECTED_DIRECTION = "EXPECTED_DIRECTION"
    MARKET_STRUCTURE = "MARKET_STRUCTURE"
    EQUITY_VALUATION = "EQUITY_VALUATION"


class IndicatorMeta(BaseModel):
    """Metadata for a single market indicator."""
    name: str
    family: IndicatorFamily
    risk_on: bool = True  # True => higher value = more stress
    mapped_drivers: list[str] = Field(default_factory=list)
    description: str = ""


class IndicatorReading(BaseModel):
    """Processed indicator value at a point in time."""
    name: str
    date: date
    raw_value: float
    z_score: float = 0.0
    percentile: float = 0.0
    signal_direction: float = 0.0  # normalised: positive = stress


# ---------------------------------------------------------------------------
# Family Composite
# ---------------------------------------------------------------------------

class FamilyComposite(BaseModel):
    """Composite z-score for an indicator family."""
    family: IndicatorFamily
    composite_z: float = 0.0
    percentile: float = 0.0
    n_indicators: int = 0
    weights: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Scenario & Portfolio Output
# ---------------------------------------------------------------------------

class AssetReturn(BaseModel):
    """Expected return for one asset class under a scenario."""
    asset_class: str
    baseline_return: float = 0.0
    scenario_return: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    scenario_vol: float = 0.0


class PortfolioPosition(BaseModel):
    """Positioning recommendation for one asset class."""
    asset_class: str
    current_weight: float = 0.0
    scenario_return: float = 0.0
    scenario_risk: float = 0.0
    risk_adj_return: float = 0.0
    mctr: float = 0.0
    signal: PositionSignal = PositionSignal.NEUTRAL
    key_theme_driver: str = ""


class HedgeRecommendation(BaseModel):
    """A suggested hedge for an unpriced theme."""
    theme_name: str
    confirmation_score: float
    portfolio_impact: float
    suggested_instruments: list[str] = Field(default_factory=list)
    rationale: str = ""


class ScenarioCard(BaseModel):
    """Complete scenario output for one theme."""
    theme: Theme
    shock_vector: dict[str, float] = Field(default_factory=dict)
    indicator_confirmations: list[dict[str, Any]] = Field(default_factory=list)
    asset_returns: list[AssetReturn] = Field(default_factory=list)
    key_risks: list[str] = Field(default_factory=list)


class PortfolioSummary(BaseModel):
    """Aggregate portfolio output."""
    portfolio_return: float = 0.0
    portfolio_risk: float = 0.0
    positions: list[PortfolioPosition] = Field(default_factory=list)
    hedges: list[HedgeRecommendation] = Field(default_factory=list)
    scenario_cards: list[ScenarioCard] = Field(default_factory=list)
