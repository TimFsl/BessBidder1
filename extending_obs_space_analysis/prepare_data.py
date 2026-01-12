# feature_pipeline.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import pandas as pd


# -----------------------------
# Config / Paths
# -----------------------------
@dataclass(frozen=True)
class PipelinePaths:
    drl_csv: Path
    ri_profit_2019_csv: Path
    ri_profit_2023_csv: Path
    da_milp_exaa_csv: Path
    da_milp_epex_csv: Path


DEFAULT_RENAME_MAP = {
    "epex_spot_60min_de_lu_eur_per_mwh": "epex_price",
    "exaa_15min_de_lu_eur_per_mwh": "exaa_price",
    "load_forecast_d_minus_1_1000_total_de_lu_mw": "load_forecast",
    "pv_forecast_d_minus_1_1000_de_lu_mw": "pv_forecast",
    "wind_offshore_forecast_d_minus_1_1000_de_lu_mw": "wind_offshore_forecast",
    "wind_onshore_forecast_d_minus_1_1000_de_lu_mw": "wind_onshore_forecast",
}


# -----------------------------
# Helpers
# -----------------------------
def _ensure_utc_datetime_index(df: pd.DataFrame, index_name: str = "time") -> pd.DataFrame:
    """Ensure DatetimeIndex and UTC."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex.")
    out = df.copy()
    out.index = pd.to_datetime(out.index, utc=True)
    out.index.name = index_name
    return out


def _read_csv_datetime_index(path: Path, time_col: str, drop_cols: Iterable[str] = ()) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[time_col]).set_index(time_col)
    df = _ensure_utc_datetime_index(df, index_name=time_col)
    if drop_cols:
        df = df.drop(columns=list(drop_cols), errors="ignore")
    return df


# Load data
def load_drl_data(
    drl_csv: Path,
    rename_map: dict = DEFAULT_RENAME_MAP,
    drop_cols: Tuple[str, ...] = ("id_full_qh",),
) -> pd.DataFrame:
    """
    Loads hourly DRL input data, enforces UTC index, drops columns, renames to canonical names,
    and computes residual load forecast.
    """
    drl = _read_csv_datetime_index(drl_csv, time_col="time", drop_cols=drop_cols)

    # Rename columns (keep unknown cols untouched)
    drl = drl.rename(columns=rename_map)

    # Residual load forecast
    needed = {"load_forecast", "pv_forecast", "wind_offshore_forecast", "wind_onshore_forecast"}
    missing = needed - set(drl.columns)
    if missing:
        raise KeyError(f"Missing columns in drl_data for residual load: {sorted(missing)}")

    drl["residual_load_forecast"] = (
        drl["load_forecast"]
        - drl["pv_forecast"]
        - drl["wind_offshore_forecast"]
        - drl["wind_onshore_forecast"]
    )

    return drl


def load_profit_data(
    ri_profit_2019_csv: Path,
    ri_profit_2023_csv: Path,
    da_milp_exaa_csv: Path,
    da_milp_epex_csv: Path,
    date_slice: Tuple[str, str] = ("2019-01-01", "2023-12-31"),
) -> pd.DataFrame:
    """
    Loads and harmonizes:
    - rolling intrinsic profit (two files concatenated)
    - day-ahead MILP profit (EXAA forecast + EPEX perfect forecast)
    Returns daily indexed profit dataframe in UTC.
    """

    # --- rolling intrinsic profit
    ri1 = pd.read_csv(ri_profit_2019_csv)
    ri2 = pd.read_csv(ri_profit_2023_csv)
    ri = pd.concat([ri1, ri2], ignore_index=True).set_index("day")

    # Your original index parsing: take first 19 chars -> to_datetime -> tz_localize UTC
    idx = pd.to_datetime(ri.index.astype(str).str.slice(0, 19)).tz_localize("UTC")
    ri.index = idx
    ri.index.name = "date"
    ri = ri.drop(columns=["cycles"], errors="ignore").rename(columns={"profit": "ri_profit"})

    # --- DA MILP EXAA
    da_exaa = pd.read_csv(da_milp_exaa_csv)
    da_exaa["time"] = pd.to_datetime(da_exaa["time"], utc=True)
    da_exaa = da_exaa.set_index("time").resample("D").sum()
    da_exaa["da_profit_exaa"] = da_exaa["discharge_revenues"] + da_exaa["charge_costs"]
    da_exaa = da_exaa[["da_profit_exaa"]]
    da_exaa.index.name = "date"

    # --- DA MILP EPEX
    da_epex = pd.read_csv(da_milp_epex_csv)
    da_epex["time"] = pd.to_datetime(da_epex["time"], utc=True)
    da_epex = da_epex.set_index("time").resample("D").sum()
    da_epex["da_profit_epex"] = da_epex["discharge_revenues"] + da_epex["charge_costs"]
    da_epex = da_epex[["da_profit_epex"]]
    da_epex.index.name = "date"

    profit = pd.concat([ri, da_exaa, da_epex], axis=1)
    profit["da_idc_profit_diff"] = profit["da_profit_epex"] - profit["ri_profit"]

    start, end = date_slice
    profit = profit.loc[start:end]

    return profit


# Ad features
def add_daily_delta_features(
    df: pd.DataFrame,
    column: str,
    prefix: str | None = None,
) -> pd.DataFrame:
    """
    For an intra-day time series column:
    - daily sum of absolute deltas
    - daily max/min of absolute deltas
    Output is daily indexed.
    """
    if prefix is None:
        prefix = column

    if column not in df.columns:
        raise KeyError(f"Column '{column}' not in dataframe.")

    delta = df[column].groupby(df.index.date).diff().abs()

    daily = pd.DataFrame(
        {
            f"{prefix}_delta_sum": delta.resample("D").sum(),
            f"{prefix}_delta_max": delta.resample("D").max(),
            f"{prefix}_delta_min": delta.resample("D").min(),
        }
    )
    daily.index.name = "date"
    return daily


def add_rolling_features(
    daily_df: pd.DataFrame,
    base_col: str,
    windows: Sequence[int] = (7, 14, 30),
    stats: Sequence[str] = ("mean",),
    prefix: str | None = None,
    shift_days: int = 1,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """
    Creates rolling features on daily data.

    shift_days:
      - 1 => leakage-safe (use only past)
      - 0 => includes current day (as in your exaa std/spread case)
    """
    if base_col not in daily_df.columns:
        raise KeyError(f"base_col '{base_col}' not in daily_df.")

    if prefix is None:
        prefix = base_col

    out = daily_df.copy()

    for w in windows:
        mp = w if min_periods is None else min_periods
        s = out[base_col].shift(shift_days).rolling(window=w, min_periods=mp)

        if "mean" in stats:
            out[f"{prefix}_roll{w}d_mean_lag{shift_days}"] = s.mean()
        if "std" in stats:
            out[f"{prefix}_roll{w}d_std_lag{shift_days}"] = s.std()
        if "min" in stats:
            out[f"{prefix}_roll{w}d_min_lag{shift_days}"] = s.min()
        if "max" in stats:
            out[f"{prefix}_roll{w}d_max_lag{shift_days}"] = s.max()

    return out


def build_daily_features(
    drl_hourly: pd.DataFrame,
) -> pd.DataFrame:
    """
    Builds daily aggregated features from hourly DRL data, incl. spread, calendar vars,
    delta features and rolling features (matching your notebook logic).
    """
    drl = _ensure_utc_datetime_index(drl_hourly, index_name="time")

    agg_features = [
        "epex_price",
        "exaa_price",
        "load_forecast",
        "pv_forecast",
        "wind_offshore_forecast",
        "wind_onshore_forecast",
        "residual_load_forecast",
        "id_full_h",
    ]
    missing = set(agg_features) - set(drl.columns)
    if missing:
        raise KeyError(f"Missing columns in drl_data for aggregation: {sorted(missing)}")

    agg_dict = {col: ["mean", "std", "min", "max"] for col in agg_features}

    daily = drl.resample("D").agg(agg_dict)
    daily.columns = [f"{col}_{stat}" for col, stat in daily.columns.to_flat_index()]
    daily.index.name = "date"

    # spreads
    daily["epex_price_spread"] = daily["epex_price_max"] - daily["epex_price_min"]
    daily["id_full_h_spread"] = daily["id_full_h_max"] - daily["id_full_h_min"]
    daily["exaa_price_spread"] = daily["exaa_price_max"] - daily["exaa_price_min"]
    daily["residual_load_forecast_spread"] = (
        daily["residual_load_forecast_max"] - daily["residual_load_forecast_min"]
    )

    # calendar
    daily["day_of_week"] = daily.index.dayofweek
    daily["month"] = daily.index.month
    daily["year"] = daily.index.year

    # delta features
    delta_blocks = [
        add_daily_delta_features(drl, "epex_price", prefix="epex_price"),
        add_daily_delta_features(drl, "exaa_price", prefix="exaa_price"),
        add_daily_delta_features(drl, "pv_forecast", prefix="pv"),
        add_daily_delta_features(drl, "wind_onshore_forecast", prefix="wind_onshore"),
        add_daily_delta_features(drl, "wind_offshore_forecast", prefix="wind_offshore"),
        add_daily_delta_features(drl, "residual_load_forecast", prefix="residual_load"),
    ]
    daily = daily.join(pd.concat(delta_blocks, axis=1), how="left")

    # rolling features (dein Setup)
    daily = add_rolling_features(
        daily,
        base_col="exaa_price_std",
        windows=(1, 3, 7, 14, 30, 90),
        stats=("mean",),
        prefix="exaa_std",
        shift_days=0,
    )

    daily = add_rolling_features(
        daily,
        base_col="exaa_price_spread",
        windows=(3, 7, 14, 30, 90),
        stats=("mean",),
        prefix="exaa_spread",
        shift_days=0,
    )

    daily = add_rolling_features(
        daily,
        base_col="epex_price_spread",
        windows=(3, 7, 14, 30, 90),
        stats=("mean",),
        prefix="epex_spread",
        shift_days=1,
    )

    daily = add_rolling_features(
        daily,
        base_col="id_full_h_std",
        windows=(3, 7),
        stats=("mean",),
        prefix="id_full_h_std",
        shift_days=1,
    )

    daily = add_rolling_features(
        daily,
        base_col="id_full_h_spread",
        windows=(3, 7, 14, 30, 90),
        stats=("mean",),
        prefix="id_full_h_spread",
        shift_days=1,
    )

    daily = add_rolling_features(
        daily,
        base_col="wind_onshore_forecast_std",
        windows=(3, 7),
        stats=("mean",),
        prefix="wind_onshore_std",
        shift_days=0,
    )

    daily = add_rolling_features(
        daily,
        base_col="wind_offshore_forecast_std",
        windows=(3, 7),
        stats=("mean",),
        prefix="wind_offshore_std",
        shift_days=0,
    )

    daily = add_rolling_features(
        daily,
        base_col="residual_load_forecast_std",
        windows=(3, 7, 14, 30),
        stats=("mean",),
        prefix="residual_load_std",
        shift_days=0,
    )

    daily = add_rolling_features(
        daily,
        base_col="pv_forecast_std",
        windows=(3, 7),
        stats=("mean",),
        prefix="pv_std",
        shift_days=0,
    )

    return daily


# -----------------------------
# 3) High-level pipeline function
# -----------------------------
def build_training_dataframe(
    paths: PipelinePaths,
    date_slice: Tuple[str, str] = ("2019-01-01", "2023-12-31"),
) -> pd.DataFrame:
    """
    Full pipeline:
    - load hourly drl data
    - build daily features
    - load profit targets
    - join to df_daily (inner join)
    """
    drl = load_drl_data(paths.drl_csv)
    daily_features = build_daily_features(drl)

    profit = load_profit_data(
        paths.ri_profit_2019_csv,
        paths.ri_profit_2023_csv,
        paths.da_milp_exaa_csv,
        paths.da_milp_epex_csv,
        date_slice=date_slice,
    )

    df_daily = daily_features.join(profit, how="inner")
    return df_daily
