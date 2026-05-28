"""Theme Engine — Configuration loader.

Loads YAML configs for transmission matrix, exposure matrix, and parameters.
Provides typed access to all configuration values.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .models import ASSET_CLASSES, MACRO_DRIVERS, Magnitude, ThemeCategory

_CONFIG_DIR = Path(__file__).parent / "data" / "config"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class EngineConfig:
    """Central configuration holder — loads once, reused across modules."""

    def __init__(self, config_dir: Path | str | None = None):
        self._dir = Path(config_dir) if config_dir else _CONFIG_DIR
        self._params: dict[str, Any] = _load_yaml(self._dir / "parameters.yaml")
        self._trans_raw: dict[str, Any] = _load_yaml(self._dir / "transmission_matrix.yaml")
        self._expo_raw: dict[str, Any] = _load_yaml(self._dir / "exposure_matrix.yaml")

        # Build numpy matrices
        self.transmission_matrix = self._build_transmission_matrix()
        self.exposure_matrix = self._build_exposure_matrix()
        self.driver_covariance = self._build_driver_covariance()
        self.driver_volatilities = self._build_driver_volatilities()

    # ----- parameters shortcuts -----
    @property
    def evidence(self) -> dict[str, Any]:
        return self._params["evidence"]

    @property
    def scenario(self) -> dict[str, Any]:
        return self._params["scenario"]

    @property
    def indicators(self) -> dict[str, Any]:
        return self._params["indicators"]

    @property
    def factor_model(self) -> dict[str, Any]:
        return self._params["factor_model"]

    @property
    def portfolio(self) -> dict[str, Any]:
        return self._params["portfolio"]

    @property
    def positioning(self) -> dict[str, Any]:
        return self._params["positioning"]

    @property
    def news_sifter(self) -> dict[str, Any]:
        return self._params["news_sifter"]

    @property
    def dashboard(self) -> dict[str, Any]:
        return self._params["dashboard"]

    # ----- magnitude helpers -----
    def magnitude_sigma(self, mag: Magnitude) -> float:
        return float(self.scenario["magnitude_sigma"][mag.value])

    # ----- baseline returns -----
    def baseline_returns_vector(self) -> np.ndarray:
        br = self.factor_model["baseline_returns"]
        return np.array([br[ac] for ac in ASSET_CLASSES], dtype=np.float64)

    def residual_vol_vector(self) -> np.ndarray:
        rv = self.factor_model["residual_volatility"]
        return np.array([rv[ac] for ac in ASSET_CLASSES], dtype=np.float64)

    def default_weights_vector(self) -> np.ndarray:
        pw = self.portfolio["default_weights"]
        return np.array([pw[ac] for ac in ASSET_CLASSES], dtype=np.float64)

    # ----- transmission matrix T (D × C) -----
    def _build_transmission_matrix(self) -> np.ndarray:
        """Build T as (n_drivers × n_categories) numpy array."""
        categories = [c.value for c in ThemeCategory]
        mat = self._trans_raw["matrix"]
        T = np.zeros((len(MACRO_DRIVERS), len(categories)), dtype=np.float64)
        for i, driver in enumerate(MACRO_DRIVERS):
            for j, cat in enumerate(categories):
                T[i, j] = float(mat.get(driver, {}).get(cat, 0.0))
        return T

    def transmission_column(self, category: ThemeCategory) -> np.ndarray:
        """Return the shock direction vector for a theme category (D×1)."""
        cat_idx = list(ThemeCategory).index(category)
        return self.transmission_matrix[:, cat_idx].copy()

    # ----- exposure matrix B (A × D) -----
    def _build_exposure_matrix(self) -> np.ndarray:
        """Build B as (n_assets × n_drivers) numpy array."""
        mat = self._expo_raw["matrix"]
        B = np.zeros((len(ASSET_CLASSES), len(MACRO_DRIVERS)), dtype=np.float64)
        for i, ac in enumerate(ASSET_CLASSES):
            for j, driver in enumerate(MACRO_DRIVERS):
                B[i, j] = float(mat.get(ac, {}).get(driver, 0.0))
        return B

    # ----- driver covariance Σ_drivers (D × D) -----
    def _build_driver_covariance(self) -> np.ndarray:
        """Build symmetric covariance matrix from upper-triangle YAML."""
        raw = self._expo_raw.get("driver_covariance", {})
        n = len(MACRO_DRIVERS)
        Sigma = np.eye(n, dtype=np.float64)
        for i, d1 in enumerate(MACRO_DRIVERS):
            if d1 not in raw:
                continue
            for j, d2 in enumerate(MACRO_DRIVERS):
                if d2 in raw[d1]:
                    val = float(raw[d1][d2])
                    Sigma[i, j] = val
                    Sigma[j, i] = val
        return Sigma

    def _build_driver_volatilities(self) -> np.ndarray:
        raw = self._expo_raw.get("driver_volatilities", {})
        return np.array(
            [float(raw.get(d, 1.0)) for d in MACRO_DRIVERS], dtype=np.float64
        )


# Module-level singleton (lazily created)
_singleton: EngineConfig | None = None


def get_config(config_dir: Path | str | None = None) -> EngineConfig:
    """Return (and cache) the global EngineConfig."""
    global _singleton
    if _singleton is None or config_dir is not None:
        _singleton = EngineConfig(config_dir)
    return _singleton
