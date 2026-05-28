"""Theme Engine -- Bloomberg Data Loader.

Pulls historical indicator time-series via xbbg (//blp/refdata)
and maps them to the engine's canonical indicator names.

Supports:
  - Simple single-ticker pulls (VIX, MOVE, Brent, Gold, etc.)
  - Computed indicators (HY-IG spread, yield curve slope, ERP, VIX term-structure)
  - Graceful fallback tickers when primary is unavailable
  - Local parquet cache to avoid redundant Bloomberg hits
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .indicators import IndicatorDataLoader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# xbbg import
# ---------------------------------------------------------------------------
try:
    import xbbg
    BLPAPI_AVAILABLE = True
except ImportError:
    xbbg = None  # type: ignore[assignment]
    BLPAPI_AVAILABLE = False
    logger.warning("xbbg not installed -- BloombergDataLoader will not function")

# ---------------------------------------------------------------------------
# Ticker config loader
# ---------------------------------------------------------------------------
_TICKER_CFG_PATH = Path(__file__).parent / "data" / "config" / "bloomberg_tickers.yaml"


def _load_ticker_config(path: Path | None = None) -> dict[str, Any]:
    p = path or _TICKER_CFG_PATH
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Low-level xbbg helpers
# ---------------------------------------------------------------------------

class _BloombergSession:
    """Thin wrapper around xbbg for historical refdata queries."""

    def __init__(self, host: str = "localhost", port: int = 8194):
        if not BLPAPI_AVAILABLE:
            raise RuntimeError(
                "xbbg is not installed.  "
                "Install it with: pip install xbbg"
            )
        self._started = False

    def ensure_started(self) -> None:
        if self._started:
            return
        # xbbg connects lazily on first bdh/bdp call — just mark ready
        self._started = True

    def historical_data(
        self,
        tickers: list[str],
        fields: list[str],
        start: str,
        end: str,
        periodicity: str = "DAILY",
        non_trading_day_fill: str = "PREVIOUS_VALUE",
        currency: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Request historical data and return {ticker: DataFrame}."""
        self.ensure_started()

        # xbbg.bdh expects dates as YYYY-MM-DD strings
        start_fmt = _yyyymmdd_to_iso(start)
        end_fmt = _yyyymmdd_to_iso(end)

        result: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            try:
                nw_df = xbbg.bdh(ticker, fields, start_fmt, end_fmt)
                pdf = nw_df.to_pandas()
                if pdf.empty:
                    logger.warning("Bloomberg returned no data for %s", ticker)
                    continue
                # xbbg bdh returns columns: ticker, date, field, value
                # Pivot into date-indexed DataFrame with field columns
                if "date" in pdf.columns and "field" in pdf.columns:
                    pivoted = pdf.pivot_table(
                        index="date", columns="field", values="value",
                    )
                    pivoted.index = pd.to_datetime(pivoted.index)
                    pivoted.index.name = "date"
                    result[ticker] = pivoted
                else:
                    # Fallback: already in wide format
                    pdf.index = pd.to_datetime(pdf.index)
                    result[ticker] = pdf
            except Exception as exc:
                logger.warning("Bloomberg error for %s: %s", ticker, exc)

        return result

    def close(self) -> None:
        if self._started:
            try:
                xbbg.shutdown()
            except Exception:
                pass
            self._started = False


def _yyyymmdd_to_iso(d: str) -> str:
    """Convert '20250101' to '2025-01-01'."""
    if len(d) == 8 and d.isdigit():
        return f"{d[:4]}-{d[4:6]}-{d[6:8]}"
    return d


# ---------------------------------------------------------------------------
# Bloomberg Data Loader
# ---------------------------------------------------------------------------

class BloombergDataLoader(IndicatorDataLoader):
    """Load indicator time-series from Bloomberg via BLPAPI.

    Features:
      - Reads ticker mappings from bloomberg_tickers.yaml
      - Handles simple pulls, spreads, diffs, and computed indicators
      - Caches to local parquet files to minimise Bloomberg API calls
      - Falls back to secondary tickers if primary fails

    Parameters
    ----------
    host : str       Bloomberg terminal host (default localhost)
    port : int       Bloomberg terminal port (default 8194)
    cache_dir : Path Local cache directory for parquet files
    lookback_years : int  How many years of history to pull
    ticker_config_path : Path  Custom path to bloomberg_tickers.yaml
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8194,
        cache_dir: str | Path | None = None,
        lookback_years: int = 7,
        ticker_config_path: Path | None = None,
    ):
        self._host = host
        self._port = port
        self._cache_dir = Path(cache_dir) if cache_dir else (
            Path(__file__).parent / "data" / "cache"
        )
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._lookback_years = lookback_years
        self._cfg = _load_ticker_config(ticker_config_path)
        self._indicator_cfg: dict[str, dict] = self._cfg.get("indicators", {})
        self._defaults: dict[str, Any] = self._cfg.get("defaults", {})
        self._session: _BloombergSession | None = None
        self._raw_cache: dict[str, pd.DataFrame] = {}
        self._failed_tickers: set[str] = set()

    def _get_session(self) -> _BloombergSession:
        if self._session is None:
            self._session = _BloombergSession(self._host, self._port)
        return self._session

    def _date_range(self) -> tuple[str, str]:
        end = datetime.today()
        start = end - timedelta(days=self._lookback_years * 365)
        return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")

    # ----- cache helpers -----
    def _cache_path(self, key: str) -> Path:
        safe = key.replace(" ", "_").replace("/", "_")
        return self._cache_dir / f"{safe}.parquet"

    def _read_cache(self, key: str) -> pd.Series | None:
        p = self._cache_path(key)
        if p.exists():
            age = datetime.now().timestamp() - p.stat().st_mtime
            if age < 18 * 3600:  # cache valid for 18 hours
                try:
                    df = pd.read_parquet(p)
                    return df.iloc[:, 0]
                except Exception:
                    pass
        return None

    def _write_cache(self, key: str, series: pd.Series) -> None:
        try:
            df = series.to_frame(name="value")
            df.to_parquet(self._cache_path(key))
        except Exception as e:
            logger.debug("Cache write failed for %s: %s", key, e)

    # ----- raw Bloomberg pull -----
    def _pull_ticker(
        self,
        ticker: str,
        field: str = "PX_LAST",
    ) -> pd.Series:
        """Pull a single ticker/field from Bloomberg (with cache)."""
        cache_key = f"{ticker}__{field}"
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached

        # Skip tickers that already failed this session or previously
        if cache_key in self._failed_tickers:
            raise KeyError(f"Bloomberg ticker {ticker}/{field} previously failed")
        neg_path = self._cache_path(cache_key + "__FAILED")
        if neg_path.exists():
            age = datetime.now().timestamp() - neg_path.stat().st_mtime
            if age < 6 * 3600:  # negative cache valid for 6 hours
                self._failed_tickers.add(cache_key)
                raise KeyError(f"Bloomberg ticker {ticker}/{field} previously failed")

        # Check in-memory raw cache
        if cache_key in self._raw_cache:
            df = self._raw_cache[cache_key]
            if field in df.columns:
                s = df[field].dropna()
                self._write_cache(cache_key, s)
                return s

        start, end = self._date_range()
        session = self._get_session()
        result = session.historical_data(
            [ticker], [field], start, end,
            periodicity=self._defaults.get("periodicity", "DAILY"),
            non_trading_day_fill=self._defaults.get(
                "non_trading_day_fill", "PREVIOUS_VALUE"
            ),
        )

        if ticker not in result or field not in result[ticker].columns:
            self._failed_tickers.add(cache_key)
            # Write negative cache marker to disk
            try:
                self._cache_path(cache_key + "__FAILED").write_text("")
            except Exception:
                pass
            raise KeyError(
                f"Bloomberg returned no data for {ticker} / {field}"
            )

        df = result[ticker]
        self._raw_cache[cache_key] = df
        s = df[field].dropna()
        self._write_cache(cache_key, s)
        return s

    def _pull_with_fallback(
        self,
        cfg: dict[str, Any],
        ticker_key: str = "ticker",
        field_key: str = "field",
    ) -> pd.Series:
        """Try primary ticker, fall back to fallback_ticker if it fails."""
        primary = cfg.get(ticker_key, "")
        field = cfg.get(field_key, "PX_LAST")

        try:
            return self._pull_ticker(primary, field)
        except (KeyError, ConnectionError) as e:
            fb_ticker = cfg.get(f"fallback_{ticker_key}", "")
            fb_field = cfg.get(f"fallback_{field_key}", field)
            if fb_ticker:
                logger.info(
                    "Primary %s failed (%s), trying fallback %s",
                    primary, e, fb_ticker,
                )
                return self._pull_ticker(fb_ticker, fb_field)
            raise

    # ----- computed indicators -----
    def _compute_spread(self, cfg: dict) -> pd.Series:
        """HY - IG spread or similar two-ticker difference."""
        field = cfg.get("field", "PX_LAST")
        try:
            long_s = self._pull_ticker(cfg["ticker_long"], field)
        except (KeyError, ConnectionError):
            long_s = self._pull_ticker(cfg["fallback_ticker_long"], field)
        try:
            short_s = self._pull_ticker(cfg["ticker_short"], field)
        except (KeyError, ConnectionError):
            short_s = self._pull_ticker(cfg["fallback_ticker_short"], field)

        combined = pd.concat([long_s, short_s], axis=1).dropna()
        return combined.iloc[:, 0] - combined.iloc[:, 1]

    def _compute_diff(self, cfg: dict) -> pd.Series:
        """Simple difference of two tickers (e.g., VIX - VIX3M)."""
        field = cfg.get("field", "PX_LAST")
        long_s = self._pull_ticker(cfg["ticker_long"], field)
        short_s = self._pull_ticker(cfg["ticker_short"], field)
        combined = pd.concat([long_s, short_s], axis=1).dropna()
        return combined.iloc[:, 0] - combined.iloc[:, 1]

    def _compute_ma30(self, cfg: dict) -> pd.Series:
        """30-day moving average of the primary ticker."""
        s = self._pull_with_fallback(cfg)
        return s.rolling(window=30, min_periods=15).mean().dropna()

    def _compute_aaii_invert(self, cfg: dict) -> pd.Series:
        """Inverted AAII: Bear% - Bull% (higher = more bearish)."""
        bull = self._pull_ticker(cfg["ticker"], cfg.get("field", "PX_LAST"))
        bear = self._pull_ticker(
            cfg["secondary_tickers"][0], cfg.get("field", "PX_LAST")
        )
        combined = pd.concat([bear, bull], axis=1).dropna()
        result = combined.iloc[:, 0] - combined.iloc[:, 1]
        # Weekly series -- forward-fill to daily
        return result.resample("B").ffill()

    def _compute_erp(self, cfg: dict) -> pd.Series:
        """Equity Risk Premium = earnings yield - real yield."""
        pe = self._pull_ticker(
            cfg["ticker_earnings"], cfg.get("field_earnings", "BEST_PE_RATIO")
        )
        real_y = self._pull_ticker(
            cfg["ticker_real_yield"], cfg.get("field_real_yield", "PX_LAST")
        )
        combined = pd.concat([pe, real_y], axis=1).dropna()
        # ERP = (1/PE)*100 - real_yield
        earnings_yield = 100.0 / combined.iloc[:, 0]
        return earnings_yield - combined.iloc[:, 1]

    def _compute_buffett(self, cfg: dict) -> pd.Series:
        """Buffett Indicator = Wilshire 5000 / GDP * 100."""
        mktcap = self._pull_ticker(
            cfg["ticker_mktcap"], cfg.get("field_mktcap", "PX_LAST")
        )
        gdp = self._pull_ticker(
            cfg["ticker_gdp"], cfg.get("field_gdp", "PX_LAST")
        )
        # GDP is quarterly -- forward-fill
        gdp_daily = gdp.resample("B").ffill()
        combined = pd.concat([mktcap, gdp_daily], axis=1).dropna()
        ratio = combined.iloc[:, 0] / combined.iloc[:, 1] * 100
        return ratio

    def _compute_turbulence_proxy(self, cfg: dict) -> pd.Series:
        """Turbulence proxy: normalised rolling cross-asset vol.

        Uses VIX directly as a turbulence proxy (simplification).
        For a true Mahalanobis distance, one would compute from
        multi-asset daily returns.
        """
        # Simple proxy: just use VIX normalised
        return self._pull_with_fallback(cfg)

    def _compute_tobin_proxy(self, cfg: dict) -> pd.Series:
        """Tobin's Q proxy via P/B ratio."""
        return self._pull_with_fallback(cfg)

    # ----- dispatcher -----
    _TRANSFORM_MAP: dict[str, str] = {
        "spread": "_compute_spread",
        "diff": "_compute_diff",
        "ma30": "_compute_ma30",
        "aaii_invert": "_compute_aaii_invert",
        "erp": "_compute_erp",
        "buffett": "_compute_buffett",
        "turbulence_proxy": "_compute_turbulence_proxy",
        "tobin_proxy": "_compute_tobin_proxy",
    }

    def load(
        self,
        indicator_name: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.Series:
        """Load a single indicator time-series from Bloomberg.

        Parameters
        ----------
        indicator_name : str
            Must match a key in bloomberg_tickers.yaml (and INDICATOR_REGISTRY).
        start, end : str, optional
            Date filters (ISO format). Applied after data pull.

        Returns
        -------
        pd.Series with DatetimeIndex.
        """
        if indicator_name not in self._indicator_cfg:
            raise KeyError(
                f"No Bloomberg ticker mapping for indicator '{indicator_name}'. "
                f"Add it to bloomberg_tickers.yaml."
            )

        cfg = self._indicator_cfg[indicator_name]
        transform = cfg.get("transform")

        if transform and transform in self._TRANSFORM_MAP:
            method = getattr(self, self._TRANSFORM_MAP[transform])
            series = method(cfg)
        else:
            series = self._pull_with_fallback(cfg)

        # Apply date filters
        if start:
            series = series[series.index >= pd.Timestamp(start)]
        if end:
            series = series[series.index <= pd.Timestamp(end)]

        series.name = indicator_name
        return series

    def load_all(
        self,
        indicators: list[str] | None = None,
    ) -> pd.DataFrame:
        """Load all (or specified) indicators into a single DataFrame.

        Returns DataFrame with DatetimeIndex and one column per indicator.
        Indicators that fail to load are skipped with a warning.
        """
        names = indicators or list(self._indicator_cfg.keys())
        frames: dict[str, pd.Series] = {}

        for name in names:
            try:
                frames[name] = self.load(name)
                logger.info("Loaded %s (%d points)", name, len(frames[name]))
            except Exception as e:
                logger.warning("Failed to load %s: %s", name, e)

        if not frames:
            return pd.DataFrame()
        return pd.DataFrame(frames)

    def close(self) -> None:
        """Close the Bloomberg session."""
        if self._session:
            self._session.close()
            self._session = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_bloomberg_loader(
    host: str = "localhost",
    port: int = 8194,
    cache_dir: str | Path | None = None,
    lookback_years: int = 7,
) -> BloombergDataLoader:
    """Create a BloombergDataLoader with default settings."""
    return BloombergDataLoader(
        host=host,
        port=port,
        cache_dir=cache_dir,
        lookback_years=lookback_years,
    )
