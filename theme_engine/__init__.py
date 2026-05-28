"""Theme-Driven Macro Scenario Engine.

A complete pipeline from themes → scenarios → signals → indicators →
asset-class outcomes → portfolio positioning.
"""
from .models import (
    ASSET_CLASSES,
    MACRO_DRIVERS,
    AssetReturn,
    Direction,
    EvidenceItem,
    FamilyComposite,
    HedgeRecommendation,
    Horizon,
    IndicatorFamily,
    Magnitude,
    PortfolioPosition,
    PortfolioSummary,
    PositionSignal,
    RiskSignal,
    ScenarioCard,
    SignalDirection,
    Theme,
    ThemeCategory,
    ThemeStatus,
)
from .config import EngineConfig, get_config
from .bloomberg_loader import BloombergDataLoader, create_bloomberg_loader

__version__ = "0.1.0"
