import os
import pandas as pd
from pathlib import Path

# GENERAL TECHNICAL CONFIGURATION

# Config file for parameters of the case study
C_RATE = 1
RTE = 0.86
MAX_CYCLES_PER_YEAR = 365
MAX_CYCLES_PER_DAY = 1


# Rolling Intrinsic specific
BUCKET_SIZE = 15
MIN_TRADES = 10

# Data timeframe configuration (importnant for naming csv files, etc.)
DATA_START = pd.Timestamp(year=2019, month=1, day=1, tz="Europe/Berlin")
DATA_END   = pd.Timestamp(year=2025, month=9, day=30, tz="Europe/Berlin")

# Train data timeframe
TRAIN_START = pd.Timestamp(year=2019, month=1, day=1, tz="Europe/Berlin")
TRAIN_END   = pd.Timestamp(year=2024, month=9, day=30, tz="Europe/Berlin") + pd.Timedelta(days=1)

# Validation timeframe
VAL_START = pd.Timestamp(year=2021, month=4, day=1, tz="Europe/Berlin")
VAL_END   = pd.Timestamp(year=2021, month=9, day=30, tz="Europe/Berlin") + pd.Timedelta(days=1)

# ------------------------------------------------------------
# Validation split mode for DRL (coordinated multi-market)
#
# - "contiguous": use VAL_START / VAL_END as before
# - "random_holdout_from_train": draw random day/week units from TRAIN range,
#   remove them from training, and use them as validation.
#
# Important: keep this deterministic for thesis comparability.
VAL_SPLIT_MODE = "contiguous"  # "contiguous" or "random_holdout_from_train"
VAL_HOLDOUT_UNIT = "day"       # "day" or "week"
VAL_HOLDOUT_N = 180             # number of days or ISO weeks held out from training
VAL_HOLDOUT_SEED = 42        # default: reuse global seed # 42

# Test timeframe
TEST_START = pd.Timestamp(year=2024, month=10, day=1, tz="Europe/Berlin")
TEST_END   = pd.Timestamp(year=2025, month=9, day=30, tz="Europe/Berlin") + pd.Timedelta(days=1)

# For single market day ahead optimizer, rolling intrinsic and myopic market
START = pd.Timestamp(year=2019, month=1, day=1, tz="Europe/Berlin")
END   = pd.Timestamp(year=2025, month=9, day=30, tz="Europe/Berlin")  + pd.Timedelta(days=1)


# Problematic dates that need to be removed from the data for the rolling intrinsic algorithm to work
PROBLEMATIC_DATES = [
    pd.Timestamp("2020-11-15").date(),
    pd.Timestamp("2020-12-27").date(),
    pd.Timestamp("2020-12-31").date(),
    pd.Timestamp("2022-07-25").date(),
    pd.Timestamp("2024-02-27").date(),
    pd.Timestamp("2024-12-31").date(),
    
]

# ----------------------------------------------
# MODELLING CONFIGURATIONS

# 02 SINGLE MARKET CONFIGURATION

OUTPUT_DIR_DA = os.path.join("output", "myopic_multi_market", "day_ahead_milp")
FILENAME_DA = "da_milp_results_exaa.csv"
DATA_PATH_DA = Path("data", "data_2019-01-01_2025-09-30_hourly.csv")


# 03 MYOPIC MULTI-MARKET CONFIGURATION

# Should naturally be the same as OUTPUT_DIR_DA, but is separated in case different results should be used
INPUT_DIR_DA = os.path.join(OUTPUT_DIR_DA, FILENAME_DA)
LOGGING_PATH_MYOPIC = Path("output/myopic_multi_market/")

# 04 COORDINATED MULTI-MARKET CONFIGURATION
# Column used as DA "forecast" in the env (for ablation: epex = realized, exaa = EXAA forecast)
DA_PRICE_FORECAST_COLUMN = "exaa_15min_de_lu_eur_per_mwh" #"epex_spot_60min_de_lu_eur_per_mwh"  # or "exaa_15min_de_lu_eur_per_mwh"
SEED = 42
TRAINING_STEPS_INTELLIGENT = 10_000_000
TRAINING_STEPS_BASIC = 10_000_000
START_IDC_STEPS = 0

DATA_PATH = Path("data", "simplified_data_jan_with_exaa_and_id_full")

PRECOMPUTED_VWAP_PATH = os.path.join( "data/precomputed_vwaps")

# Precomputed DA+RI combined profit tables (v3 lookup training): summary_YYYY-MM-DD.csv
PRECOMPUTED_DA_RI_SUMMARY_DIR = os.path.join(
    "coordinated_market_upper_bound_analysis", "results_merged"
)

COORDINATED_MODEL_NAME_QH = "model_intelligent_quarterhourly_products"
TRAIN_CSV_NAME = "basic_battery_dam_train_log_v3.csv"
TEST_CSV_NAME = "basic_battery_dam_test_log_v3.csv"

COORDINATED_STACKED_RI_QH_TRAINING_OUTPUT_CSV = (
    "output_ri_qh_intelligent_stacking_training.csv"
)
COORDINATED_STACKED_RI_H_TRAINING_OUTPUT_CSV = (
    "output_ri_h_intelligent_stacking_training.csv"
)


LOGGING_PATH_COORDINATED = os.path.join("output", "coordinated_multi_market", "logging")
TENSORBOARD_PATH_INTELLIGENT = os.path.join(
    "output", "coordinated_multi_market", "tensorboard"
)
MODEL_OUTPUT_PATH_COORDINATED = os.path.join(
    "output", "coordinated_multi_market", "models"
)
SCALER_OUTPUT_PATH_COORDINATED = os.path.join(
    "output", "coordinated_multi_market", "scalers"
)
