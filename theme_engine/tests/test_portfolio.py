"""Tests for Module 6: Portfolio — impact, positioning, hedging."""
import numpy as np
import pytest

from theme_engine.config import get_config
from theme_engine.factor_model import scenario_confidence_interval
from theme_engine.models import (
    ASSET_CLASSES,
    MACRO_DRIVERS,
    AssetReturn,
    Direction,
    EvidenceItem,
    Horizon,
    PositionSignal,
    Theme,
    ThemeCategory,
    ThemeStatus,
)
from theme_engine.portfolio import (
    build_portfolio_summary,
    compute_portfolio_positions,
    compute_stressed_covariance,
    identify_hedges,
    marginal_contribution_to_risk,
    portfolio_return,
    portfolio_risk,
    positioning_signal,
)
from theme_engine.scenario import compute_shock_vector
from datetime import date


@pytest.fixture
def cfg():
    return get_config()


class TestPortfolioReturn:
    def test_equal_weight_equal_return(self):
        n = 5
        w = np.ones(n) / n
        r = np.ones(n) * 0.10
        assert abs(portfolio_return(w, r) - 0.10) < 1e-10

    def test_concentrated_portfolio(self):
        w = np.array([1.0, 0.0, 0.0])
        r = np.array([0.15, 0.05, 0.10])
        assert abs(portfolio_return(w, r) - 0.15) < 1e-10


class TestPortfolioRisk:
    def test_single_asset(self):
        w = np.array([1.0])
        cov = np.array([[0.04]])  # σ = 0.2
        assert abs(portfolio_risk(w, cov) - 0.2) < 1e-10

    def test_diversification_reduces_risk(self):
        cov = np.array([[0.04, 0.01], [0.01, 0.04]])
        w_conc = np.array([1.0, 0.0])
        w_div = np.array([0.5, 0.5])
        assert portfolio_risk(w_div, cov) < portfolio_risk(w_conc, cov)


class TestMCTR:
    def test_sums_to_portfolio_risk(self):
        """Sum of weight × MCTR should equal portfolio variance / σ."""
        n = 3
        w = np.array([0.4, 0.3, 0.3])
        cov = np.array([[0.04, 0.01, 0.005],
                        [0.01, 0.03, 0.008],
                        [0.005, 0.008, 0.025]])
        mctr = marginal_contribution_to_risk(w, cov)
        p_risk = portfolio_risk(w, cov)
        # w' · MCTR = σ_p (Euler decomposition)
        assert abs(w @ mctr - p_risk) < 1e-10


class TestStressedCovariance:
    def test_positive_definite(self, cfg):
        delta = np.ones(len(MACRO_DRIVERS)) * 0.5
        cov = compute_stressed_covariance(delta, cfg)
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues > -1e-10)

    def test_stress_increases_variance(self, cfg):
        delta_zero = np.zeros(len(MACRO_DRIVERS))
        delta_stress = np.ones(len(MACRO_DRIVERS)) * 2.0

        cov_zero = compute_stressed_covariance(delta_zero, cfg)
        cov_stress = compute_stressed_covariance(delta_stress, cfg)

        # Stressed should have higher diagonal (variances)
        assert np.all(np.diag(cov_stress) >= np.diag(cov_zero) - 1e-10)

    def test_shape(self, cfg):
        delta = np.zeros(len(MACRO_DRIVERS))
        cov = compute_stressed_covariance(delta, cfg)
        assert cov.shape == (len(ASSET_CLASSES), len(ASSET_CLASSES))


class TestPositioningSignal:
    def test_overweight(self, cfg):
        sig = positioning_signal(sharpe=0.8, mctr=0.1, avg_mctr=0.2, cfg=cfg)
        assert sig == PositionSignal.OVERWEIGHT

    def test_underweight_low_sharpe(self, cfg):
        sig = positioning_signal(sharpe=-0.5, mctr=0.1, avg_mctr=0.2, cfg=cfg)
        assert sig == PositionSignal.UNDERWEIGHT

    def test_neutral(self, cfg):
        sig = positioning_signal(sharpe=0.2, mctr=0.2, avg_mctr=0.2, cfg=cfg)
        assert sig == PositionSignal.NEUTRAL


class TestPositions:
    def test_returns_all_asset_classes(self, cfg):
        delta = np.zeros(len(MACRO_DRIVERS))
        asset_returns = scenario_confidence_interval(delta, cfg=cfg)
        positions = compute_portfolio_positions(delta, asset_returns, cfg=cfg)
        assert len(positions) == len(ASSET_CLASSES)

    def test_weights_sum_to_one(self, cfg):
        delta = np.zeros(len(MACRO_DRIVERS))
        asset_returns = scenario_confidence_interval(delta, cfg=cfg)
        positions = compute_portfolio_positions(delta, asset_returns, cfg=cfg)
        total_weight = sum(p.current_weight for p in positions)
        assert abs(total_weight - 1.0) < 1e-10


class TestHedges:
    def test_identifies_unpriced_negative_themes(self, cfg):
        theme = Theme(
            name="Unpriced Risk",
            category=ThemeCategory.GEOPOLITICAL,
            direction=Direction.BEARISH,
            status=ThemeStatus.ACTIVE,
            strength=0.7,
            evidence=[EvidenceItem(
                source="x", date=date.today(), title="X",
                credibility_score=0.8, timeliness_score=0.8,
                corroboration_count=2,
            )],
        )
        shock = compute_shock_vector(theme, cfg=cfg)
        hedges = identify_hedges(
            [theme], [shock],
            confirmations=[0.1],      # low confirmation = unpriced
            portfolio_impacts=[-0.05],  # large negative impact
            cfg=cfg,
        )
        assert len(hedges) >= 1
        assert hedges[0].theme_name == "Unpriced Risk"
        assert len(hedges[0].suggested_instruments) > 0

    def test_no_hedge_for_priced_theme(self, cfg):
        theme = Theme(
            name="Priced Risk",
            category=ThemeCategory.GEOPOLITICAL,
            direction=Direction.BEARISH,
            status=ThemeStatus.ACTIVE,
        )
        shock = compute_shock_vector(theme, cfg=cfg)
        hedges = identify_hedges(
            [theme], [shock],
            confirmations=[0.9],     # high confirmation = already priced
            portfolio_impacts=[-0.01],
            cfg=cfg,
        )
        assert len(hedges) == 0


class TestPortfolioSummary:
    def test_full_summary(self, cfg):
        theme = Theme(
            name="Test", category=ThemeCategory.GROWTH,
            direction=Direction.BULLISH, status=ThemeStatus.ACTIVE,
            strength=0.5, likelihood=0.5,
            evidence=[EvidenceItem(
                source="x", date=date.today(), title="X",
                credibility_score=0.7, timeliness_score=0.7,
                corroboration_count=1,
            )],
        )
        shock = compute_shock_vector(theme, cfg=cfg)
        asset_returns = scenario_confidence_interval(shock, cfg=cfg)

        summary = build_portfolio_summary(
            themes=[theme],
            individual_shocks=[shock],
            aggregate_shock=shock,
            asset_returns=asset_returns,
            confirmations=[0.2],
            scenario_cards=[],
            cfg=cfg,
        )

        assert len(summary.positions) == len(ASSET_CLASSES)
        assert isinstance(summary.portfolio_return, float)
        assert isinstance(summary.portfolio_risk, float)
