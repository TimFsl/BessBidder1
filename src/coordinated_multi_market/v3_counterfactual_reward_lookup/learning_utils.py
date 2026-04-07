import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from src.shared.config import (
    DATA_START,
    DATA_END,
    TRAIN_START,
    TRAIN_END,
    VAL_START,
    VAL_END,
    TEST_START,
    TEST_END,
    DATA_PATH,
    PROBLEMATIC_DATES,
    DA_PRICE_FORECAST_COLUMN,
)

if isinstance(DA_PRICE_FORECAST_COLUMN, tuple):
    if len(DA_PRICE_FORECAST_COLUMN) != 1 or not isinstance(
        DA_PRICE_FORECAST_COLUMN[0], str
    ):
        raise ValueError(
            "DA_PRICE_FORECAST_COLUMN must be a string column name "
            "(or single-item tuple due to accidental trailing comma)."
        )
    DA_PRICE_FORECAST_COLUMN = DA_PRICE_FORECAST_COLUMN[0]


train_start = DATA_START.date().isoformat()
train_end = DATA_END.date().isoformat()
path = f"df_spot_train_{train_start}_{train_end}_with_features_utc.csv"

# Columns required for each day row in prepare_input_data (shape (24, len)).
_PREPARE_DAY_COLS = [
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
]


def _berlin_day_bounds(day) -> tuple[pd.Timestamp, pd.Timestamp]:
    d = pd.Timestamp(day).date()
    start = pd.Timestamp(d).tz_localize("Europe/Berlin")
    end = start + pd.Timedelta(days=1)
    return start, end


def _interpolate_day_to_24(
    sub: pd.DataFrame, full_idx: pd.DatetimeIndex
) -> pd.DataFrame | None:
    """Reindex to 24 hourly stamps; fill gaps (DST / gaps) with time/interpolation."""
    out = sub.reindex(full_idx)
    # Time-aware interpolation along the hour index
    num_cols = out.select_dtypes(include=[np.number]).columns
    if len(num_cols):
        out[num_cols] = out[num_cols].interpolate(
            method="time", limit_direction="both"
        )
    out = out.ffill().bfill()
    if out[_PREPARE_DAY_COLS].isna().any().any():
        out[_PREPARE_DAY_COLS] = out[_PREPARE_DAY_COLS].ffill(axis=0).bfill(axis=0)
    if out[_PREPARE_DAY_COLS].isna().any().any():
        return None
    return out


def _berlin_calendar_day_slice(df: pd.DataFrame, day) -> pd.DataFrame | None:
    """
    One calendar day in Europe/Berlin as 24 hourly rows for the battery env.

    Uses ``[start, start+1d)`` in local time (not ``df.loc[iso]``) so the last hour
    before midnight is included consistently. Short DST days (23 rows) are reindexed
    to 24 hourly stamps (``periods=24, freq='1h'``) with forward/backward fill.
    """
    start, end = _berlin_day_bounds(day)
    full_idx = pd.date_range(start, periods=24, freq="1h", tz="Europe/Berlin")
    sub = df.loc[(df.index >= start) & (df.index < end)].copy()
    if len(sub) == 0:
        return None
    sub.index = pd.to_datetime(sub.index).tz_convert("Europe/Berlin")
    sub = sub[~sub.index.duplicated(keep="last")]
    if len(sub) == 24:
        return sub
    if len(sub) == 23:
        return _interpolate_day_to_24(sub, full_idx)
    if len(sub) == 25:
        sub = sub[~sub.index.duplicated(keep="first")]
        if len(sub) == 24:
            return sub
        return _interpolate_day_to_24(sub, full_idx)
    # Irregular length (e.g. gaps): align to 24 delivery hours
    return _interpolate_day_to_24(sub, full_idx)


def split_df_by_date(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
   
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame-Index muss ein DatetimeIndex sein.")

    if df.index.tz is None:
        df.index = df.index.tz_localize("utc").tz_convert("Europe/Berlin")
    else:
        df.index = df.index.tz_convert("Europe/Berlin")

    train_mask = (df.index >= TRAIN_START) & (df.index < TRAIN_END)
    val_mask   = (df.index >= VAL_START)   & (df.index < VAL_END)
    test_mask  = (df.index >= TEST_START)  & (df.index < TEST_END)

    df_train = df[train_mask].copy()
    df_val   = df[val_mask].copy()
    df_test  = df[test_mask].copy()

    return df_train, df_val, df_test



def load_input_data(write_test: bool = False):
    """
    Lädt den vollständigen Spot-Datensatz und splittet ihn deterministisch
    in Train-, Val- und Test-Sets nach den Zeiträumen in config.py.

    Rückgabe:
        df_spot_train, df_spot_val, df_spot_test
    """

    # Load dataset
    #df = pd.read_csv(SPOT_DATA_CSV_PATH, index_col="time", parse_dates=True)
    df = pd.read_csv(
        os.path.join(
            DATA_PATH,
            path,
        ),
        index_col=0,
        parse_dates=True,
    )
    # Remove problematic dates
    if PROBLEMATIC_DATES:
        mask_bad = df.index.date.astype("O")
        df = df[~pd.Series(mask_bad, index=df.index).isin(PROBLEMATIC_DATES)]


    # Apply time periods
    df_spot_train, df_spot_val, df_spot_test = split_df_by_date(df)

    # Optional: Write test data to CSV (e.g., for further analysis)
    if write_test:
        df_spot_test.to_csv("spot_test_period.csv")
    
    print("Training start:", df_spot_train.index.min(), "Training end:", df_spot_train.index.max())
    print("Validation start:", df_spot_val.index.min(), "Validation end:", df_spot_val.index.max())
    print("Test start:", df_spot_test.index.min(), "Test end:", df_spot_test.index.max())

    return df_spot_train, df_spot_val, df_spot_test


def prepare_input_data(
    df: pd.DataFrame, versioned_scaler_path: str, fit_scaler: bool = False
) -> dict[str, dict[str, np.array]]:
    scalable_features = df[
        [
            "load_forecast_d_minus_1_1000_total_de_lu_mw",
            "pv_forecast_d_minus_1_1000_de_lu_mw",
            "wind_offshore_forecast_d_minus_1_1000_de_lu_mw",
            "wind_onshore_forecast_d_minus_1_1000_de_lu_mw",
            #"date_month",
            #"day_of_week",
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
        ]
    ].copy()

    scaler_file = os.path.join(versioned_scaler_path, "scaler.pkl")

    if fit_scaler:
        # Training phase: fit scaler and save
        scaler = MinMaxScaler()
        scaler.fit(scalable_features)
        os.makedirs(versioned_scaler_path, exist_ok=True)
        joblib.dump(scaler, scaler_file)
    else:
        # Test/Inference: load existing scaler
        if not os.path.exists(scaler_file):
            raise FileNotFoundError(
                f"Scaler file not found at {scaler_file}. "
                "Make sure to run prepare_input_data with fit_scaler=True on the train set first."
            )
        scaler = joblib.load(scaler_file)

    # In both cases: transform
    features_scaled = scaler.transform(scalable_features)
    df_scaled = pd.DataFrame(
        features_scaled, columns=scalable_features.columns, index=df.index
    )

    # Prices remain unscaled, appended again
    df = pd.concat(
        [
            df_scaled,
            df[
                [
                    "epex_spot_60min_de_lu_eur_per_mwh",
                    "exaa_15min_de_lu_eur_per_mwh",
                    "date_month",
                    "day_of_week",
                ]
            ],
        ],
        axis=1,
    )

    input_dict = {}
    days = np.unique(df.index.date)
    skipped: list[tuple[str, str]] = []
    for day in days:
        start, end = _berlin_day_bounds(day)
        raw_n = int(len(df.loc[(df.index >= start) & (df.index < end)]))
        if raw_n == 0:
            skipped.append((str(day), "no_rows_in_berlin_day"))
            continue
        sub = _berlin_calendar_day_slice(df, day)
        if sub is None:
            skipped.append((str(day), f"could_not_align_24h_raw_n={raw_n}"))
            continue
        feat = sub[_PREPARE_DAY_COLS]
        if feat.isna().any().any() or feat.shape != (24, 21):
            nan_cols = (
                feat.columns[feat.isna().any()].tolist() if feat.isna().any().any() else []
            )
            skipped.append(
                (
                    str(day),
                    f"nan_or_bad_shape_shape={feat.shape}_nan_cols={nan_cols}",
                )
            )
            continue

        day_key = day.isoformat() if hasattr(day, "isoformat") else str(day)
        input_dict.update(
            {
                day_key: {
                    "price_forecast": np.array(
                        sub[DA_PRICE_FORECAST_COLUMN].astype(np.float32).values
                    ),
                    "price_realized": np.array(
                        sub["epex_spot_60min_de_lu_eur_per_mwh"].astype(np.float32).values
                    ),
                    "pv_forecast_d_minus_1_1000_de_lu_mw": np.array(
                        sub["pv_forecast_d_minus_1_1000_de_lu_mw"].astype(np.float32).values
                    ),
                    "wind_onshore_forecast_d_minus_1_1000_de_lu_mw": np.array(
                        sub["wind_onshore_forecast_d_minus_1_1000_de_lu_mw"].astype(
                            np.float32
                        ).values
                    ),
                    "wind_offshore_forecast_d_minus_1_1000_de_lu_mw": np.array(
                        sub["wind_offshore_forecast_d_minus_1_1000_de_lu_mw"].astype(
                            np.float32
                        ).values
                    ),
                    "load_forecast_d_minus_1_1000_total_de_lu_mw": np.array(
                        sub["load_forecast_d_minus_1_1000_total_de_lu_mw"].astype(
                            np.float32
                        ).values
                    ),
                    "date_month": np.array(sub["date_month"].astype(np.float32).values),
                    "day_of_week": np.array(sub["day_of_week"].astype(np.float32).values),
                    "wind_forecast_daily_mean": np.array(
                        sub["wind_forecast_daily_mean"].astype(np.float32).values
                    ),
                    "wind_forecast_daily_std": np.array(
                        sub["wind_forecast_daily_std"].astype(np.float32).values
                    ),
                    "spread_id_full_da_mean": np.array(
                        sub["spread_id_full_da_qh_mean"].astype(np.float32).values
                    ),
                    "spread_id_full_da_std": np.array(
                        sub["spread_id_full_da_qh_std"].astype(np.float32).values
                    ),
                    "spread_id_full_da_min": np.array(
                        sub["spread_id_full_da_qh_min"].astype(np.float32).values
                    ),
                    "spread_id_full_da_max": np.array(
                        sub["spread_id_full_da_qh_max"].astype(np.float32).values
                    ),
                    "exaa_pf_daily_mean": np.array(
                        sub["exaa_pf_daily_mean"].astype(np.float32).values
                    ),
                    "exaa_pf_daily_std": np.array(
                        sub["exaa_pf_daily_std"].astype(np.float32).values
                    ),
                    "exaa_pf_daily_min": np.array(
                        sub["exaa_pf_daily_min"].astype(np.float32).values
                    ),
                    "exaa_pf_daily_max": np.array(
                        sub["exaa_pf_daily_max"].astype(np.float32).values
                    ),
                    "exaa_pf_daily_spread": np.array(
                        sub["exaa_pf_daily_spread"].astype(np.float32).values
                    ),
                    "exaa_pf_daily_diff_sum": np.array(
                        sub["exaa_pf_daily_diff_sum"].astype(np.float32).values
                    ),
                    "exaa_pf_daily_diff_max": np.array(
                        sub["exaa_pf_daily_diff_max"].astype(np.float32).values
                    ),
                    "timestamps": np.array(sub.index.values),
                }
            }
        )
    if skipped:
        skip_path = os.path.join(
            versioned_scaler_path, "prepare_input_data_skipped_days.txt"
        )
        try:
            os.makedirs(versioned_scaler_path, exist_ok=True)
            with open(skip_path, "w", encoding="utf-8") as sf:
                for d, reason in skipped:
                    sf.write(f"{d}\t{reason}\n")
            print(
                f"[prepare_input_data] skipped {len(skipped)} day(s); "
                f"wrote {skip_path}"
            )
        except OSError as e:
            print(f"[prepare_input_data] could not write skipped-days file: {e}")
    return input_dict



# Define a linear learning rate schedule
def linear_schedule(initial_value):
    """
    Returns a function that computes the learning rate linearly decaying
    from `initial_value` to 0 based on progress remaining.
    """

    def schedule(progress_remaining):
        return progress_remaining * initial_value

    return schedule


def orthogonal_weight_init(module):
    """
    Custom weight initialization using orthogonal initialization.
    Applies orthogonal initialization to linear layers and zeros to biases.
    """
    if isinstance(module, nn.Linear):  # Apply only to Linear layers
        nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.zeros_(module.bias)  # Initialize biases to 0
