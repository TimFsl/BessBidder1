"""
Local test configuration for src/Archive/old_setup.

Keeps old_setup experiments self-contained so shared config does not need edits.
Most values are forwarded from src.shared.config.
"""

from src.shared.config import (
    START,
    END,
    DATA_PATH,
    BUCKET_SIZE,
    C_RATE,
    MAX_CYCLES_PER_DAY,
    MIN_TRADES,
    RTE,
    SEED,
    TRAINING_STEPS_INTELLIGENT,
)

# Store old-setup runs in a dedicated output subtree.
LOGGING_PATH_COORDINATED = "output/coordinated_multi_market_old_setup/logging"
MODEL_OUTPUT_PATH_COORDINATED = "output/coordinated_multi_market_old_setup/models"
SCALER_OUTPUT_PATH_COORDINATED = "output/coordinated_multi_market_old_setup/scalers"
TENSORBOARD_PATH_INTELLIGENT = "output/coordinated_multi_market_old_setup/tensorboard"

COORDINATED_MODEL_NAME_QH = "model_intelligent_quarterhourly_products_old_setup"
TRAIN_CSV_NAME = "basic_battery_dam_train_log_old_setup.csv"
TEST_CSV_NAME = "basic_battery_dam_test_log_old_setup.csv"

# Hard-coded warmup: only DA rewards before this step, then add IDC reward.
IDC_REWARD_START_STEP = 200_000
