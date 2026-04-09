from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd


HoldoutUnit = Literal["day", "week"]


@dataclass(frozen=True)
class HoldoutSplit:
    df_train: pd.DataFrame
    df_val: pd.DataFrame
    # For traceability / debugging / thesis writeup
    held_out_days: tuple[pd.Timestamp, ...]  # tz-aware (Europe/Berlin), normalized to midnight


_DEFAULT_REQUIRED_DAY_COLS: tuple[str, ...] = (
    "epex_spot_60min_de_lu_eur_per_mwh",
    "exaa_15min_de_lu_eur_per_mwh",
    "load_forecast_d_minus_1_1000_total_de_lu_mw",
    "pv_forecast_d_minus_1_1000_de_lu_mw",
    "wind_offshore_forecast_d_minus_1_1000_de_lu_mw",
    "wind_onshore_forecast_d_minus_1_1000_de_lu_mw",
    "date_month",
    "day_of_week",
    "wind_forecast_daily_mean",
    "wind_forecast_daily_std",
    "spread_id_full_da_qh_mean",
    "spread_id_full_da_qh_std",
    "spread_id_full_da_qh_min",
    "spread_id_full_da_qh_max",
    "exaa_pf_daily_mean",
    "exaa_pf_daily_std",
    "exaa_pf_daily_min",
    "exaa_pf_daily_max",
    "exaa_pf_daily_spread",
    "exaa_pf_daily_diff_sum",
    "exaa_pf_daily_diff_max",
)


def _ensure_berlin_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")
    if df.index.tz is None:
        df = df.copy()
        df.index = df.index.tz_localize("utc").tz_convert("Europe/Berlin")
        return df
    if str(df.index.tz) != "Europe/Berlin":
        df = df.copy()
        df.index = df.index.tz_convert("Europe/Berlin")
    return df


def _day_start(ts: pd.Timestamp) -> pd.Timestamp:
    # Normalize to Berlin midnight (tz-aware)
    if ts.tz is None:
        raise ValueError("Timestamp must be tz-aware.")
    return ts.tz_convert("Europe/Berlin").normalize()


def _valid_24h_days(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    required_cols: Iterable[str] = _DEFAULT_REQUIRED_DAY_COLS,
) -> list[pd.Timestamp]:
    """
    Return Berlin-calendar days within [start, end) that have exactly 24 rows
    and no NaNs in required columns.

    This intentionally excludes DST 23/25-hour days to keep v1/v2 behaviour
    (which expects 24 rows per day) stable and comparable across versions.
    """
    df = _ensure_berlin_index(df)
    start = start.tz_convert("Europe/Berlin")
    end = end.tz_convert("Europe/Berlin")

    if start >= end:
        return []

    cols = list(required_cols)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for day eligibility: {missing}")

    in_range = df.loc[(df.index >= start) & (df.index < end), cols].copy()
    if len(in_range) == 0:
        return []

    day_index = in_range.index.normalize()
    in_range["_day"] = day_index

    days: list[pd.Timestamp] = []
    for day, g in in_range.groupby("_day", sort=True):
        if len(g) != 24:
            continue
        if g.drop(columns=["_day"]).isna().any().any():
            continue
        ts = pd.Timestamp(day)
        # groupby key keeps tz info; handle both tz-naive and tz-aware robustly
        if ts.tz is None:
            ts = ts.tz_localize("Europe/Berlin")
        else:
            ts = ts.tz_convert("Europe/Berlin")
        days.append(ts.normalize())
    return days


def random_holdout_from_train(
    *,
    df: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    holdout_unit: HoldoutUnit,
    holdout_n: int,
    seed: int,
    required_cols: Iterable[str] = _DEFAULT_REQUIRED_DAY_COLS,
) -> HoldoutSplit:
    """
    Split the training period into (train_remainder, val_holdout) by randomly
    selecting calendar days or ISO weeks from within the training window.
    """
    if holdout_n <= 0:
        raise ValueError("holdout_n must be > 0 for random holdout split.")

    df = _ensure_berlin_index(df)
    train_start = train_start.tz_convert("Europe/Berlin")
    train_end = train_end.tz_convert("Europe/Berlin")

    eligible_days = _valid_24h_days(
        df,
        start=train_start,
        end=train_end,
        required_cols=required_cols,
    )
    if len(eligible_days) == 0:
        raise ValueError("No eligible 24h days found in training window.")

    rng = np.random.default_rng(int(seed))

    if holdout_unit == "day":
        if holdout_n > len(eligible_days):
            raise ValueError(
                f"holdout_n={holdout_n} exceeds eligible_days={len(eligible_days)}."
            )
        held_out_days = sorted(rng.choice(eligible_days, size=holdout_n, replace=False))
    elif holdout_unit == "week":
        # Group eligible days by ISO week; sample weeks; then hold out all days in those weeks.
        iso = pd.DataFrame({"day": eligible_days})
        iso["iso_year"] = iso["day"].dt.isocalendar().year.astype(int)
        iso["iso_week"] = iso["day"].dt.isocalendar().week.astype(int)
        week_keys = iso[["iso_year", "iso_week"]].drop_duplicates().to_records(index=False)
        week_keys = list(week_keys)

        if holdout_n > len(week_keys):
            raise ValueError(
                f"holdout_n={holdout_n} exceeds eligible_weeks={len(week_keys)}."
            )
        chosen = set(rng.choice(week_keys, size=holdout_n, replace=False))
        mask = iso.apply(lambda r: (int(r["iso_year"]), int(r["iso_week"])) in chosen, axis=1)
        held_out_days = sorted(iso.loc[mask, "day"].tolist())
    else:
        raise ValueError(f"Unknown holdout_unit: {holdout_unit!r}")

    held_out_days = tuple(pd.Timestamp(d).tz_convert("Europe/Berlin").normalize() for d in held_out_days)
    held_out_day_set = set(held_out_days)

    # Build val/train by Berlin-calendar day membership.
    day_starts = df.index.normalize()
    is_holdout = day_starts.isin(list(held_out_day_set))

    in_train_window = (df.index >= train_start) & (df.index < train_end)
    df_train_window = df.loc[in_train_window].copy()

    df_val = df_train_window.loc[is_holdout[in_train_window]].copy()
    df_train = df_train_window.loc[~is_holdout[in_train_window]].copy()

    return HoldoutSplit(df_train=df_train, df_val=df_val, held_out_days=held_out_days)

