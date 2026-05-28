"""Theme Engine — VFMC Portfolio Loader.

Queries BNY Data Vault (ENTERPRISE_LOOKTHROUGH_VDM) for live portfolio
positions across all VFMC clients.  Returns asset class allocations,
market values, exposures, and SAA weights.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from .models import VFMC_ASSET_CLASS_MAP

logger = logging.getLogger(__name__)

CLIENTS = ["VWA", "TAC", "VMI", "SSF", "ESSS_DB", "VIF"]

# Module-level cache for BNY Data Vault query results
_bny_cache: dict[str, pd.DataFrame] = {}
_bny_cache_date: str | None = None


@dataclass
class VFMCAssetClassPosition:
    """One asset-class row from the BNY Data Vault."""
    asset_class: str          # internal key (e.g. "international_equities")
    bny_name: str             # BNY name (e.g. "International Equities")
    category: str             # BNY category (e.g. "Equity")
    exposure_aud: float       # Base Effective Exposure (AUD)
    market_value_aud: float   # Base Market Value (AUD)
    weight_pct: float         # % of total portfolio exposure
    saa_weight: float         # Strategic Asset Allocation weight


@dataclass
class ClientAssetClassPosition:
    """Per-client asset-class row."""
    client: str
    asset_class: str
    bny_name: str
    category: str
    exposure_aud: float
    market_value_aud: float
    exposure_pct: float
    saa_weight: float


def _build_query(client: str, effective_date: str) -> str:
    """Build the BNY Data Vault SQL query for a single client."""
    return (
        f'SELECT X.CLIENT, X.ASSET_CLASS_CAT, X.ASSET_CLASS, X.EXPOSURE_BASE, '
        f'X.EXPOSURE_BASE / SUM(X.EXPOSURE_BASE) OVER() AS EXPOSURE_PCT, '
        f'X.MKT_VAL_BASE, Y.SUM_SAA '
        f'FROM '
        f'( '
        f'    SELECT "Input -Portfolio/Composite Name" AS CLIENT, '
        f"    COALESCE(\"Asset Class Long Name\",'NULL') AS ASSET_CLASS, "
        f'    "Asset Category Long Name" AS ASSET_CLASS_CAT, '
        f'    SUM(Base_Effective_Exposure) AS "EXPOSURE_BASE", '
        f'    SUM(Base_Market_Value) AS "MKT_VAL_BASE" '
        f'    FROM ENTERPRISE_LOOKTHROUGH_VDM('
        f"FROMEFFECTIVEDATE = '{effective_date}', "
        f"TOEFFECTIVEDATE = '{effective_date}', "
        f"LOOKTHRU_TYPE='DEFAULT') "
        f"    WHERE \"Lookthru Flag\" IN ('No', 'N', 'NONE') "
        f"    AND \"Input -Portfolio/Composite Name\"='{client}' "
        f'    GROUP BY "Input -Portfolio/Composite Name", '
        f'    "Asset Class Long Name", "Asset Category Long Name" '
        f') X '
        f'LEFT JOIN '
        f'( '
        f'    SELECT A.CLIENT, A.CLASS, SUM(A.SAA) AS SUM_SAA '
        f'    FROM '
        f'    ( '
        f'        SELECT DISTINCT "Input -Portfolio/Composite Name" AS CLIENT, '
        f"        COALESCE(\"Asset Class Long Name\",'NULL') AS CLASS, SAA "
        f'        FROM ENTERPRISE_LOOKTHROUGH_VDM('
        f"FROMEFFECTIVEDATE = '{effective_date}', "
        f"TOEFFECTIVEDATE = '{effective_date}', "
        f"LOOKTHRU_TYPE='DEFAULT') "
        f"        WHERE \"Lookthru Flag\" IN ('No', 'N', 'NONE') "
        f"        AND \"Input -Portfolio/Composite Name\"='{client}' "
        f'    ) A '
        f'    GROUP BY A.CLIENT, A.CLASS '
        f') Y ON X.CLIENT=Y.CLIENT AND X.ASSET_CLASS=Y.CLASS '
        f'ORDER BY X.ASSET_CLASS_CAT, X.ASSET_CLASS'
    )


def _find_effective_date() -> str:
    """Return the most recent business day (T-1) as YYYY-MM-DD.

    BNY Data Vault typically has T-1 data available.
    """
    today = date.today()
    d = today - timedelta(days=1)
    # Skip weekends
    while d.weekday() >= 5:  # Saturday=5, Sunday=6
        d -= timedelta(days=1)
    return d.strftime("%Y-%m-%d")


def _query_all_clients(
    effective_date: str | None = None,
    clients: list[str] | None = None,
) -> pd.DataFrame:
    """Query BNY Data Vault for all clients and return combined DataFrame.

    Results are cached per effective_date so that load_portfolio() and
    load_client_positions() don't duplicate the same 6 SQL queries.
    """
    global _bny_cache, _bny_cache_date

    from VFMCDataLayer import Environment, BNYDataVault

    if effective_date is None:
        effective_date = _find_effective_date()
    if clients is None:
        clients = CLIENTS

    # Return cached result if same date
    cache_key = effective_date
    if cache_key == _bny_cache_date and cache_key in _bny_cache:
        logger.info("Using cached BNY data for %s", effective_date)
        return _bny_cache[cache_key]

    datasource = BNYDataVault(environment=Environment.PROD)
    all_dfs: list[pd.DataFrame] = []

    for client in clients:
        logger.info("Querying BNY Data Vault for %s (date=%s)", client, effective_date)
        datasource.query = _build_query(client, effective_date)
        df = datasource.get_data_frame()
        if not df.empty:
            all_dfs.append(df)
        else:
            logger.warning("No data returned for %s on %s", client, effective_date)

    if not all_dfs:
        raise RuntimeError(
            f"BNY Data Vault returned no data for any client on {effective_date}"
        )

    result = pd.concat(all_dfs, ignore_index=True)

    # Cache for this date
    _bny_cache[cache_key] = result
    _bny_cache_date = cache_key

    return result


def load_portfolio(
    effective_date: str | None = None,
) -> list[VFMCAssetClassPosition]:
    """Load aggregated VFMC portfolio positions from BNY Data Vault.

    Parameters
    ----------
    effective_date : YYYY-MM-DD string.  Defaults to T-1 business day.

    Returns
    -------
    list of VFMCAssetClassPosition, one per mapped asset class,
    sorted by weight descending.
    """
    combined = _query_all_clients(effective_date)

    # Aggregate across all clients
    agg = (
        combined.groupby(["ASSET_CLASS_CAT", "ASSET_CLASS"])
        .agg(
            EXPOSURE_BASE=("EXPOSURE_BASE", "sum"),
            MKT_VAL_BASE=("MKT_VAL_BASE", "sum"),
            SUM_SAA=("SUM_SAA", "mean"),
        )
        .reset_index()
    )
    total_exp = agg["EXPOSURE_BASE"].sum()

    positions: list[VFMCAssetClassPosition] = []
    for _, row in agg.iterrows():
        bny_name = row["ASSET_CLASS"]
        internal_key = VFMC_ASSET_CLASS_MAP.get(bny_name)
        if internal_key is None:
            continue  # skip unmapped classes (e.g. Enhanced Income with 0 exposure)

        weight = (row["EXPOSURE_BASE"] / total_exp * 100) if total_exp else 0.0
        saa = row["SUM_SAA"] if pd.notna(row["SUM_SAA"]) else 0.0

        positions.append(
            VFMCAssetClassPosition(
                asset_class=internal_key,
                bny_name=bny_name,
                category=row["ASSET_CLASS_CAT"],
                exposure_aud=row["EXPOSURE_BASE"],
                market_value_aud=row["MKT_VAL_BASE"],
                weight_pct=round(weight, 2),
                saa_weight=round(saa, 4),
            )
        )

    positions.sort(key=lambda p: p.weight_pct, reverse=True)
    return positions


def load_client_positions(
    effective_date: str | None = None,
) -> list[ClientAssetClassPosition]:
    """Load per-client asset class positions from BNY Data Vault."""
    combined = _query_all_clients(effective_date)

    positions: list[ClientAssetClassPosition] = []
    for _, row in combined.iterrows():
        bny_name = row["ASSET_CLASS"]
        internal_key = VFMC_ASSET_CLASS_MAP.get(bny_name)
        if internal_key is None:
            continue

        saa = row["SUM_SAA"] if pd.notna(row["SUM_SAA"]) else 0.0

        positions.append(
            ClientAssetClassPosition(
                client=row["CLIENT"],
                asset_class=internal_key,
                bny_name=bny_name,
                category=row["ASSET_CLASS_CAT"],
                exposure_aud=row["EXPOSURE_BASE"],
                market_value_aud=row["MKT_VAL_BASE"],
                exposure_pct=row["EXPOSURE_PCT"],
                saa_weight=round(saa, 4),
            )
        )

    return positions


def portfolio_weights_dict(
    effective_date: str | None = None,
) -> dict[str, float]:
    """Return {asset_class_key: weight_fraction} for use in the factor model."""
    positions = load_portfolio(effective_date)
    return {p.asset_class: round(p.weight_pct / 100, 4) for p in positions}
