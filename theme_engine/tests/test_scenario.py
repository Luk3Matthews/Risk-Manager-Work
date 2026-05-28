"""Tests for Module 3: Scenario — theme → macro driver shock vectors."""
from datetime import date

import numpy as np
import pytest

from theme_engine.config import get_config
from theme_engine.models import (
    MACRO_DRIVERS,
    Direction,
    EvidenceItem,
    Horizon,
    Magnitude,
    Theme,
    ThemeCategory,
    ThemeStatus,
)
from theme_engine.scenario import (
    aggregate_shocks,
    build_theme_correlation_matrix,
    compute_shock_vector,
    run_scenario_shocks,
)


@pytest.fixture
def cfg():
    return get_config()


def _make_theme(
    name: str = "Test",
    category: ThemeCategory = ThemeCategory.GEOPOLITICAL,
    direction: Direction = Direction.BEARISH,
    strength: float = 0.5,
    likelihood: float = 0.5,
) -> Theme:
    return Theme(
        name=name, category=category, direction=direction,
        likelihood=likelihood, status=ThemeStatus.ACTIVE,
        strength=strength,
        evidence=[EvidenceItem(
            source="x", date=date.today(), title="X",
            credibility_score=0.7, timeliness_score=0.7,
            corroboration_count=2,
        )],
    )


class TestShockVector:
    def test_shape(self, cfg):
        theme = _make_theme()
        delta = compute_shock_vector(theme, cfg=cfg)
        assert delta.shape == (len(MACRO_DRIVERS),)

    def test_direction_sign(self, cfg):
        """Bearish geopolitical → growth down (T affirmed); bullish → growth up (T reversed)."""
        t_bearish = _make_theme(direction=Direction.BEARISH)
        t_bullish = _make_theme(direction=Direction.BULLISH)

        d_bear = compute_shock_vector(t_bearish, cfg=cfg)
        d_bull = compute_shock_vector(t_bullish, cfg=cfg)

        # Growth index = 0; T[growth][GEO] = -0.5
        assert d_bear[0] < 0  # bearish geopolitical → growth down (T kept)
        assert d_bull[0] > 0  # bullish geopolitical → growth up (T flipped)

    def test_ambiguous_direction_zero(self, cfg):
        theme = _make_theme(direction=Direction.AMBIGUOUS)
        delta = compute_shock_vector(theme, cfg=cfg)
        assert np.allclose(delta, 0.0)

    def test_magnitude_scaling(self, cfg):
        """Higher strength themes should produce larger shocks."""
        t_weak = _make_theme(strength=0.1)   # SMALL
        t_strong = _make_theme(strength=0.9)  # EXTREME

        d_weak = compute_shock_vector(t_weak, cfg=cfg)
        d_strong = compute_shock_vector(t_strong, cfg=cfg)

        # Strong should have larger absolute shocks
        assert np.linalg.norm(d_strong) > np.linalg.norm(d_weak)

    def test_overrides(self, cfg):
        theme = _make_theme()
        overrides = {"expected_growth": 99.0}
        delta = compute_shock_vector(theme, overrides=overrides, cfg=cfg)
        growth_idx = MACRO_DRIVERS.index("expected_growth")
        assert delta[growth_idx] == 99.0


class TestCorrelationMatrix:
    def test_shape(self, cfg):
        themes = [_make_theme(name=f"T{i}") for i in range(3)]
        corr = build_theme_correlation_matrix(themes, cfg)
        assert corr.shape == (3, 3)

    def test_diagonal_ones(self, cfg):
        themes = [_make_theme(name=f"T{i}") for i in range(3)]
        corr = build_theme_correlation_matrix(themes, cfg)
        np.testing.assert_array_almost_equal(np.diag(corr), 1.0)

    def test_same_category_higher(self, cfg):
        t1 = _make_theme(name="Geo1", category=ThemeCategory.GEOPOLITICAL)
        t2 = _make_theme(name="Geo2", category=ThemeCategory.GEOPOLITICAL)
        t3 = _make_theme(name="Growth", category=ThemeCategory.GROWTH)

        corr = build_theme_correlation_matrix([t1, t2, t3], cfg)
        assert corr[0, 1] > corr[0, 2]  # same category > cross category


class TestAggregation:
    def test_single_theme(self, cfg):
        theme = _make_theme(strength=1.0)
        shock = compute_shock_vector(theme, cfg=cfg)
        agg = aggregate_shocks([theme], [shock], cfg)
        # With single theme, diversification scalar = 1.0
        np.testing.assert_array_almost_equal(agg, 1.0 * shock)

    def test_aggregation_reduces_with_diversification(self, cfg):
        """Two themes in different categories should diversify."""
        t1 = _make_theme(name="Geo", category=ThemeCategory.GEOPOLITICAL, strength=0.5)
        t2 = _make_theme(name="Growth", category=ThemeCategory.GROWTH,
                         direction=Direction.BULLISH, strength=0.5)

        s1 = compute_shock_vector(t1, cfg=cfg)
        s2 = compute_shock_vector(t2, cfg=cfg)

        agg = aggregate_shocks([t1, t2], [s1, s2], cfg)
        naive = 0.5 * s1 + 0.5 * s2

        # Aggregate should be <= naive (due to diversification scalar)
        assert np.linalg.norm(agg) <= np.linalg.norm(naive) + 1e-10

    def test_empty_themes(self, cfg):
        agg = aggregate_shocks([], [], cfg)
        assert np.allclose(agg, 0.0)


class TestRunScenarioShocks:
    def test_returns_correct_counts(self, cfg):
        themes = [
            _make_theme(name="T1", strength=0.5),
            _make_theme(name="T2", category=ThemeCategory.GROWTH,
                        direction=Direction.BULLISH, strength=0.6),
        ]
        individual, agg = run_scenario_shocks(themes, cfg=cfg)
        assert len(individual) == 2
        assert agg.shape == (len(MACRO_DRIVERS),)
