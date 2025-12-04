import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple

from src.shared.config import DATA_START, DATA_END, TRAIN_START, TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END, DATA_PATH, PROBLEMATIC_DATES


train_start = DATA_START.date().isoformat()
train_end = DATA_END.date().isoformat()
path = f"df_spot_train_{train_start}_{train_end}_with_features_utc.csv"

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
    for day in days:
        if df.loc[day.isoformat()][
            [
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
            ]
        ].isna().any().any() or df.loc[day.isoformat()][
            [
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
            ]
        ].shape != (
            24,
            14,
        ):
            continue

        input_dict.update(
            {
                day.isoformat(): {
                    "price_forecast": np.array(
                        df.loc[day.isoformat()]["exaa_15min_de_lu_eur_per_mwh"]
                        .astype(np.float32)
                        .values
                    ),
                    "price_realized": np.array(
                        df.loc[day.isoformat()]["epex_spot_60min_de_lu_eur_per_mwh"]
                        .astype(np.float32)
                        .values
                    ),
                    "pv_forecast_d_minus_1_1000_de_lu_mw": np.array(
                        df.loc[day.isoformat()]["pv_forecast_d_minus_1_1000_de_lu_mw"]
                        .astype(np.float32)
                        .values
                    ),
                    "wind_onshore_forecast_d_minus_1_1000_de_lu_mw": np.array(
                        df.loc[day.isoformat()][
                            "wind_onshore_forecast_d_minus_1_1000_de_lu_mw"
                        ]
                        .astype(np.float32)
                        .values
                    ),
                    "wind_offshore_forecast_d_minus_1_1000_de_lu_mw": np.array(
                        df.loc[day.isoformat()][
                            "wind_offshore_forecast_d_minus_1_1000_de_lu_mw"
                        ]
                        .astype(np.float32)
                        .values
                    ),
                    "load_forecast_d_minus_1_1000_total_de_lu_mw": np.array(
                        df.loc[day.isoformat()][
                            "load_forecast_d_minus_1_1000_total_de_lu_mw"
                        ]
                        .astype(np.float32)
                        .values
                    ),
                    "date_month": np.array(
                        df.loc[day.isoformat()]["date_month"].astype(np.float32).values
                    ),
                    "day_of_week": np.array(
                        df.loc[day.isoformat()]["day_of_week"].astype(np.float32).values
                    ),
                    "wind_forecast_daily_mean": np.array(
                        df.loc[day.isoformat()]["wind_forecast_daily_mean"]
                        .astype(np.float32)
                        .values
                    ),
                    "wind_forecast_daily_std": np.array(
                        df.loc[day.isoformat()]["wind_forecast_daily_std"]
                        .astype(np.float32)
                        .values
                    ),
                    "spread_id_full_da_mean": np.array(
                        df.loc[day.isoformat()]["spread_id_full_da_qh_mean"]
                        .astype(np.float32)
                        .values
                    ),
                    "spread_id_full_da_std": np.array(
                        df.loc[day.isoformat()]["spread_id_full_da_qh_std"]
                        .astype(np.float32)
                        .values
                    ),
                    "spread_id_full_da_min": np.array(
                        df.loc[day.isoformat()]["spread_id_full_da_qh_min"]
                        .astype(np.float32)
                        .values
                    ),
                    "spread_id_full_da_max": np.array(
                        df.loc[day.isoformat()]["spread_id_full_da_qh_max"]
                        .astype(np.float32)
                        .values
                    ),
                    "timestamps": np.array(df.loc[day.isoformat()].index.values),
                }
            }
        )
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
