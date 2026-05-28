"""Tests for Module 5: Factor Model — macro drivers → asset-class returns."""
import numpy as np
import pytest

from theme_engine.config import get_config
from theme_engine.factor_model import (
    scenario_confidence_interval,
    scenario_return_variance,
    scenario_returns,
)
from theme_engine.models import ASSET_CLASSES, MACRO_DRIVERS


@pytest.fixture
def cfg():
    return get_config()


class TestScenarioReturns:
    def test_zero_shock_returns_baseline(self, cfg):
        """No shock → returns equal baseline."""
        delta = np.zeros(len(MACRO_DRIVERS))
        R = scenario_returns(delta, cfg)
        baseline = cfg.baseline_returns_vector()
        np.testing.assert_array_almost_equal(R, baseline)

    def test_shape(self, cfg):
        delta = np.ones(len(MACRO_DRIVERS)) * 0.5
        R = scenario_returns(delta, cfg)
        assert R.shape == (len(ASSET_CLASSES),)

    def test_positive_growth_shock_helps_equities(self, cfg):
        """Positive growth shock should increase equity returns."""
        delta = np.zeros(len(MACRO_DRIVERS))
        growth_idx = MACRO_DRIVERS.index("expected_growth")
        delta[growth_idx] = 1.0  # 1σ growth shock

        R = scenario_returns(delta, cfg)
        baseline = cfg.baseline_returns_vector()
        eq_idx = ASSET_CLASSES.index("global_equities_dm")

        assert R[eq_idx] > baseline[eq_idx]

    def test_erp_shock_hurts_equities(self, cfg):
        """Positive ERP shock should decrease equity returns (β < 0 for ERP on equities)."""
        delta = np.zeros(len(MACRO_DRIVERS))
        erp_idx = MACRO_DRIVERS.index("equity_risk_premium")
        delta[erp_idx] = 1.0

        R = scenario_returns(delta, cfg)
        baseline = cfg.baseline_returns_vector()
        eq_idx = ASSET_CLASSES.index("global_equities_dm")

        assert R[eq_idx] < baseline[eq_idx]

    def test_inflation_shock_hurts_bonds(self, cfg):
        """Positive inflation shock should decrease bond returns."""
        delta = np.zeros(len(MACRO_DRIVERS))
        infl_idx = MACRO_DRIVERS.index("expected_inflation")
        delta[infl_idx] = 1.0

        R = scenario_returns(delta, cfg)
        baseline = cfg.baseline_returns_vector()
        bond_idx = ASSET_CLASSES.index("sovereign_bonds")

        assert R[bond_idx] < baseline[bond_idx]

    def test_commodity_supply_shock_helps_commodities(self, cfg):
        """Positive commodity supply shock should increase commodity returns."""
        delta = np.zeros(len(MACRO_DRIVERS))
        comm_idx = MACRO_DRIVERS.index("commodity_supply")
        delta[comm_idx] = 1.0

        R = scenario_returns(delta, cfg)
        baseline = cfg.baseline_returns_vector()
        comm_ac_idx = ASSET_CLASSES.index("commodities")

        assert R[comm_ac_idx] > baseline[comm_ac_idx]


class TestVariance:
    def test_zero_shock_has_residual_only(self, cfg):
        """With zero shock, variance should be mainly residual."""
        delta = np.zeros(len(MACRO_DRIVERS))
        var = scenario_return_variance(delta, cfg)
        residual_var = cfg.residual_vol_vector() ** 2
        # Should be close to residual + small floor
        for i in range(len(ASSET_CLASSES)):
            assert var[i] >= residual_var[i] - 1e-6

    def test_larger_shock_more_variance(self, cfg):
        """Larger shocks should produce more variance."""
        delta_small = np.ones(len(MACRO_DRIVERS)) * 0.5
        delta_large = np.ones(len(MACRO_DRIVERS)) * 2.0

        var_small = scenario_return_variance(delta_small, cfg)
        var_large = scenario_return_variance(delta_large, cfg)

        assert np.all(var_large >= var_small)


class TestConfidenceInterval:
    def test_returns_all_asset_classes(self, cfg):
        delta = np.ones(len(MACRO_DRIVERS)) * 0.5
        results = scenario_confidence_interval(delta, cfg=cfg)
        assert len(results) == len(ASSET_CLASSES)

    def test_ci_contains_point_estimate(self, cfg):
        delta = np.ones(len(MACRO_DRIVERS)) * 0.5
        results = scenario_confidence_interval(delta, cfg=cfg)
        for ar in results:
            assert ar.ci_lower <= ar.scenario_return <= ar.ci_upper

    def test_wider_ci_with_larger_shock(self, cfg):
        delta_small = np.ones(len(MACRO_DRIVERS)) * 0.1
        delta_large = np.ones(len(MACRO_DRIVERS)) * 2.0

        ci_small = scenario_confidence_interval(delta_small, cfg=cfg)
        ci_large = scenario_confidence_interval(delta_large, cfg=cfg)

        for i in range(len(ASSET_CLASSES)):
            width_small = ci_small[i].ci_upper - ci_small[i].ci_lower
            width_large = ci_large[i].ci_upper - ci_large[i].ci_lower
            assert width_large >= width_small
