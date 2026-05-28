"""Tests for Module 4: Indicators — market indicator processing."""
import numpy as np
import pandas as pd
import pytest

from theme_engine.config import get_config
from theme_engine.indicators import (
    INDICATOR_REGISTRY,
    DataFrameLoader,
    IndicatorFamily,
    adjust_shock_for_confirmation,
    compute_family_composite,
    compute_overall_market_stress,
    compute_theme_confirmation,
    get_family_indicators,
    process_indicator,
    run_indicator_pipeline,
)
from theme_engine.models import MACRO_DRIVERS, CompositeMethod, FamilyComposite, IndicatorMeta
from theme_engine.synthetic_data import generate_synthetic_indicators


@pytest.fixture
def cfg():
    return get_config()


@pytest.fixture
def synthetic_df():
    return generate_synthetic_indicators(n_years=5, seed=42)


@pytest.fixture
def loader(synthetic_df):
    return DataFrameLoader(synthetic_df)


class TestRegistry:
    def test_all_families_covered(self):
        families = {m.family for m in INDICATOR_REGISTRY}
        for f in IndicatorFamily:
            assert f in families, f"Missing family: {f}"

    def test_all_have_mapped_drivers(self):
        for m in INDICATOR_REGISTRY:
            assert len(m.mapped_drivers) > 0, f"{m.name} has no mapped drivers"

    def test_mapped_drivers_valid(self):
        for m in INDICATOR_REGISTRY:
            for d in m.mapped_drivers:
                assert d in MACRO_DRIVERS, f"{m.name} maps to unknown driver {d}"

    def test_get_family_indicators(self):
        metas = get_family_indicators(IndicatorFamily.MARKET_RISK)
        assert len(metas) > 0
        assert all(m.family == IndicatorFamily.MARKET_RISK for m in metas)


class TestProcessIndicator:
    def test_output_columns(self, synthetic_df):
        meta = IndicatorMeta(name="VIX", family=IndicatorFamily.MARKET_RISK,
                             risk_on=True, mapped_drivers=["equity_risk_premium"])
        series = synthetic_df["VIX"]
        result = process_indicator(series, meta, lookback=252)
        assert "raw" in result.columns
        assert "z_score" in result.columns
        assert "percentile" in result.columns
        assert "stress_z" in result.columns

    def test_risk_on_stress_same_sign(self, synthetic_df):
        """For risk-on indicators, stress_z should equal z_score."""
        meta = IndicatorMeta(name="VIX", family=IndicatorFamily.MARKET_RISK,
                             risk_on=True, mapped_drivers=["equity_risk_premium"])
        result = process_indicator(synthetic_df["VIX"], meta, lookback=252)
        valid = result.dropna()
        np.testing.assert_array_almost_equal(
            valid["stress_z"].values, valid["z_score"].values
        )

    def test_risk_off_stress_flipped(self, synthetic_df):
        """For risk-off indicators, stress_z should be -z_score."""
        meta = IndicatorMeta(name="Yield_2s10s", family=IndicatorFamily.MARKET_STRUCTURE,
                             risk_on=False, mapped_drivers=["expected_growth"])
        result = process_indicator(synthetic_df["Yield_2s10s"], meta, lookback=252)
        valid = result.dropna()
        np.testing.assert_array_almost_equal(
            valid["stress_z"].values, -valid["z_score"].values
        )


class TestFamilyComposite:
    def test_composite_pca(self, loader, cfg):
        fc, stress_df = compute_family_composite(
            IndicatorFamily.MARKET_RISK, loader,
            method=CompositeMethod.PCA, lookback=252, cfg=cfg,
        )
        assert fc.family == IndicatorFamily.MARKET_RISK
        assert fc.n_indicators > 0
        assert len(fc.weights) == fc.n_indicators
        assert abs(sum(fc.weights.values()) - 1.0) < 1e-6

    def test_composite_equal(self, loader, cfg):
        fc, _ = compute_family_composite(
            IndicatorFamily.MARKET_RISK, loader,
            method=CompositeMethod.EQUAL, lookback=252, cfg=cfg,
        )
        n = fc.n_indicators
        for w in fc.weights.values():
            assert abs(w - 1.0 / n) < 1e-6

    def test_all_families_produce_composites(self, loader, cfg):
        for family in IndicatorFamily:
            fc, _ = compute_family_composite(
                family, loader, method=CompositeMethod.EQUAL,
                lookback=252, cfg=cfg,
            )
            assert fc.n_indicators > 0


class TestOverallMarketStress:
    def test_equal_weight(self):
        composites = [
            FamilyComposite(family=IndicatorFamily.MARKET_RISK, composite_z=1.0),
            FamilyComposite(family=IndicatorFamily.GEOPOLITICAL_RISK, composite_z=-1.0),
        ]
        oms = compute_overall_market_stress(composites)
        assert abs(oms - 0.0) < 1e-10  # average of 1 and -1

    def test_empty(self):
        assert compute_overall_market_stress([]) == 0.0


class TestThemeConfirmation:
    def test_perfect_alignment(self):
        shock = np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0])
        # Only indicator that maps to expected_growth
        z_scores = {"SPX_PutCall": 1.0}  # maps to expected_growth
        conf = compute_theme_confirmation(shock, z_scores)
        # Should be positive (same direction)
        assert conf > 0

    def test_no_data_returns_zero(self):
        shock = np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0])
        conf = compute_theme_confirmation(shock, {})
        assert conf == 0.0


class TestShockAdjustment:
    def test_full_confirmation_dampens(self, cfg):
        shock = np.array([1.0, 0.5, -0.3, 0.8, 0, 0, 0, 0, 0])
        adj = adjust_shock_for_confirmation(shock, confirmation=1.0, beta=0.5, cfg=cfg)
        # δ_adj = δ * (1 - 0.5 * 1.0) = δ * 0.5
        np.testing.assert_array_almost_equal(adj, shock * 0.5)

    def test_no_confirmation_unchanged(self, cfg):
        shock = np.array([1.0, 0.5, -0.3, 0.8, 0, 0, 0, 0, 0])
        adj = adjust_shock_for_confirmation(shock, confirmation=0.0, cfg=cfg)
        np.testing.assert_array_almost_equal(adj, shock)

    def test_negative_confirmation_amplifies(self, cfg):
        shock = np.array([1.0, 0.5, -0.3, 0.8, 0, 0, 0, 0, 0])
        adj = adjust_shock_for_confirmation(shock, confirmation=-1.0, beta=0.5, cfg=cfg)
        # δ_adj = δ * (1 - 0.5 * (-1)) = δ * 1.5
        np.testing.assert_array_almost_equal(adj, shock * 1.5)


class TestFullPipeline:
    def test_pipeline_returns_all(self, loader, cfg):
        composites, z_scores, oms = run_indicator_pipeline(loader, cfg)
        assert len(composites) == len(IndicatorFamily)
        assert isinstance(z_scores, dict)
        assert isinstance(oms, float)
