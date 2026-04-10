"""
Train PPO Agent — v3: precomputed DA+RI lookup (no live Gurobi RI during training).

Uses Discrete(3) DA actions (idle / full buy / full sell). Requires
``summary_YYYY-MM-DD.csv`` files under ``PRECOMPUTED_DA_RI_SUMMARY_DIR`` (see config).

This script:
- Loads and preprocesses spot market data.
- Sets up ``BasicBatteryDAM`` and ``CustomPPO`` with lookup-based combined profit.
- Saves the model, logs, and scaler using versioned output folders.
"""

import os
import warnings

import shutil
from sklearn.externals.array_api_compat.numpy import False_
import torch
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.coordinated_multi_market.v3_counterfactual_reward_lookup.basic_battery_dam_env import (
    BasicBatteryDAM,
)
from src.coordinated_multi_market.v3_counterfactual_reward_lookup.custom_ppo import CustomPPO
from src.coordinated_multi_market.v3_counterfactual_reward_lookup.learning_utils import (
    load_input_data,
    prepare_input_data,
    linear_schedule,
)

from src.shared.folder_versioning import create_new_dir_version
from src.shared.config import (
    COORDINATED_MODEL_NAME_QH,
    LOGGING_PATH_COORDINATED,
    MODEL_OUTPUT_PATH_COORDINATED,
    PRECOMPUTED_DA_RI_SUMMARY_DIR,
    RTE,
    SCALER_OUTPUT_PATH_COORDINATED,
    SEED,
    TENSORBOARD_PATH_INTELLIGENT,
    TRAIN_CSV_NAME,
    TRAINING_STEPS_INTELLIGENT,
)

warnings.simplefilter(action="ignore", category=FutureWarning)

RESUME_TRAINING = True
# If True together with RESUME_TRAINING, initialize a fresh model with the
# hyperparameters below and load policy weights from SOURCE_MODEL_CHECKPOINT.
RESUME_WITH_NEW_HYPERPARAMS = True
# For resume-with-new-hparams: set whether SB3 timestep counter is reset.
RESET_TIMESTEPS_ON_RESTART = False

# Source checkpoint (where to load pretrained weights from)
SOURCE_MODEL_NUMBER = "4"
SOURCE_MODEL_CHECKPOINT = "ppo_stacked_checkpoint_1000000_steps"
# Target run folder (where to write logs/models/scaler for this run)
TARGET_MODEL_NUMBER = "5"

# Override if summaries live elsewhere (default: config.PRECOMPUTED_DA_RI_SUMMARY_DIR)
PRECOMPUTED_SUMMARY_DIR = None  # e.g. os.path.join("coordinated_market_upper_bound_analysis", "results")


if __name__ == "__main__":
    os.makedirs(LOGGING_PATH_COORDINATED, exist_ok=True)
    os.makedirs(MODEL_OUTPUT_PATH_COORDINATED, exist_ok=True)
    os.makedirs(SCALER_OUTPUT_PATH_COORDINATED, exist_ok=True)

    if RESUME_TRAINING:
        versioned_log_path = os.path.join(LOGGING_PATH_COORDINATED, TARGET_MODEL_NUMBER)
        versioned_model_path = os.path.join(
            MODEL_OUTPUT_PATH_COORDINATED, TARGET_MODEL_NUMBER
        )
        versioned_scaler_path = os.path.join(
            SCALER_OUTPUT_PATH_COORDINATED, TARGET_MODEL_NUMBER
        )
        source_model_path = os.path.join(
            MODEL_OUTPUT_PATH_COORDINATED, SOURCE_MODEL_NUMBER
        )
        source_scaler_path = os.path.join(
            SCALER_OUTPUT_PATH_COORDINATED, SOURCE_MODEL_NUMBER
        )
    else:
        versioned_log_path = create_new_dir_version(LOGGING_PATH_COORDINATED)
        versioned_model_path = create_new_dir_version(MODEL_OUTPUT_PATH_COORDINATED)
        versioned_scaler_path = create_new_dir_version(SCALER_OUTPUT_PATH_COORDINATED)
        source_model_path = None
        source_scaler_path = None

    os.makedirs(versioned_log_path, exist_ok=True)
    os.makedirs(versioned_model_path, exist_ok=True)
    os.makedirs(versioned_scaler_path, exist_ok=True)

    train_log_path = os.path.join(versioned_log_path, TRAIN_CSV_NAME)
    lookup_debug_log_path = os.path.join(versioned_log_path, "lookup_debug.csv")
    if RESUME_TRAINING:
        tensorboard_run_name = f"PPO_{TARGET_MODEL_NUMBER}"
    else:
        tensorboard_run_name = f"PPO_{os.path.basename(versioned_model_path)}"
    print(f"[Train Script] Train log CSV: {train_log_path}")
    print(f"[Train Script] Lookup debug CSV: {lookup_debug_log_path}")
    print(f"[Train Script] TensorBoard run name: {tensorboard_run_name}")
    _summary_dir = PRECOMPUTED_SUMMARY_DIR or PRECOMPUTED_DA_RI_SUMMARY_DIR
    print(f"[Train Script] Precomputed summary CSV dir: {_summary_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df_spot_train, _df_val, _df_test = load_input_data(write_test=False)
    scaler_input_path = (
        source_scaler_path if RESUME_TRAINING and source_scaler_path else versioned_scaler_path
    )
    # If we resume using a pretrained scaler, copy it into the target run folder
    # so validation/test can always find versioned_scaler_path/scaler.pkl.
    if RESUME_TRAINING and source_scaler_path and source_scaler_path != versioned_scaler_path:
        src_scaler = os.path.join(source_scaler_path, "scaler.pkl")
        dst_scaler = os.path.join(versioned_scaler_path, "scaler.pkl")
        if os.path.exists(src_scaler) and not os.path.exists(dst_scaler):
            shutil.copy2(src_scaler, dst_scaler)
        # also copy the optional skipped-days debug file if present
        src_skipped = os.path.join(source_scaler_path, "prepare_input_data_skipped_days.txt")
        dst_skipped = os.path.join(versioned_scaler_path, "prepare_input_data_skipped_days.txt")
        if os.path.exists(src_skipped) and not os.path.exists(dst_skipped):
            shutil.copy2(src_skipped, dst_skipped)
    input_data_train = prepare_input_data(
        df_spot_train,
        scaler_input_path,
        fit_scaler=not RESUME_TRAINING,
    )

    env = BasicBatteryDAM(
        modus="train",
        logging_path=versioned_log_path,
        input_data=input_data_train,
        round_trip_efficiency=RTE,
    )

    check_env(env)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=versioned_model_path,
        name_prefix="ppo_stacked_checkpoint",
    )

    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[64, 64, 64, 64], vf=[64, 64, 64, 64]),
        log_std_init=-0.5,
    )

    model_kwargs = dict(
        verbose=0,
        tensorboard_log=TENSORBOARD_PATH_INTELLIGENT,
        device=device,
        seed=SEED,
        intraday_product_type="QH",
        train_log_path=train_log_path,
        policy_kwargs=policy_kwargs,
        ent_coef=0.05,
        n_steps=1920,
        clip_range=0.4,
        batch_size=480,
        vf_coef=0.4,
        learning_rate=linear_schedule(1e-4),
        gamma=0.999,
        counterfactual_idle_action=0,
        counterfactual_active_mode="first_buy_first_sell",
        precomputed_summary_dir=_summary_dir,
        lookup_debug_log_path=lookup_debug_log_path,
        da_market_reward_weight=1.0,
        cf_reward_weight=1.0,
        # Example: set to START_IDC_STEPS to switch to CF-only (+ keep invalid penalties)
        # after initial DA-focused phase.
        cf_only_after_steps=0,
        # "contextual" (default): marginal contribution in full schedule context
        # "buy_only_isolated": for buy-step CF, neutralize sell legs for attribution
        buy_cf_attribution_mode="buy_only_isolated",
        # Reward normalization across years/regimes:
        # - "none": keep current scaling
        # - "daily_iqr_forecast": divide DA-market and CF rewards by IQR of daily forecast prices
        reward_normalization_mode="none",
        reward_normalization_min_scale=1.0,
        # Multiply the IQR divisor by this factor. Values < 1.0 increase rewards.
        # Example: 0.1 => ~10x larger rewards; 0.02 => ~50x larger rewards.
        reward_normalization_scale_multiplier=0.01,
    )

    if RESUME_TRAINING:
        load_path = os.path.join(source_model_path, SOURCE_MODEL_CHECKPOINT + ".zip")
        print(f"Resuming training from: {load_path}")
        if RESUME_WITH_NEW_HYPERPARAMS:
            print("Resume mode: loading checkpoint weights into fresh model config.")
            model = CustomPPO("MlpPolicy", env, **model_kwargs)
            # Load policy/value (and optimizer) state from checkpoint into the
            # fresh instance. exact_match=False allows changed algorithm attrs.
            model.set_parameters(load_path, exact_match=False, device=device)
            reset_num_timesteps = RESET_TIMESTEPS_ON_RESTART
        else:
            model = CustomPPO.load(load_path, device=device)
            model.set_env(env)
            model.train_log_path = train_log_path
            # Keep original run counter when continuing exactly from checkpoint.
            reset_num_timesteps = False
    else:
        print("Starting training from scratch (v3 lookup).")
        model = CustomPPO("MlpPolicy", env, **model_kwargs)
        reset_num_timesteps = True

    model.learn(
        total_timesteps=TRAINING_STEPS_INTELLIGENT,
        callback=checkpoint_callback,
        reset_num_timesteps=reset_num_timesteps,
        tb_log_name=tensorboard_run_name,
    )

    model.save(os.path.join(versioned_model_path, COORDINATED_MODEL_NAME_QH))
    print("Finished training!")
