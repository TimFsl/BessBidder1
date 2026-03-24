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

RESUME_TRAINING = False

MODEL_NUMBER = "3"
MODEL_CHECKPOINT = "ppo_stacked_checkpoint_280000_steps"

# Override if summaries live elsewhere (default: config.PRECOMPUTED_DA_RI_SUMMARY_DIR)
PRECOMPUTED_SUMMARY_DIR = None  # e.g. os.path.join("coordinated_market_upper_bound_analysis", "results")


if __name__ == "__main__":
    os.makedirs(LOGGING_PATH_COORDINATED, exist_ok=True)
    os.makedirs(MODEL_OUTPUT_PATH_COORDINATED, exist_ok=True)
    os.makedirs(SCALER_OUTPUT_PATH_COORDINATED, exist_ok=True)

    if RESUME_TRAINING:
        versioned_log_path = os.path.join(LOGGING_PATH_COORDINATED, MODEL_NUMBER)
        versioned_model_path = os.path.join(MODEL_OUTPUT_PATH_COORDINATED, MODEL_NUMBER)
        versioned_scaler_path = os.path.join(SCALER_OUTPUT_PATH_COORDINATED, MODEL_NUMBER)
    else:
        versioned_log_path = create_new_dir_version(LOGGING_PATH_COORDINATED)
        versioned_model_path = create_new_dir_version(MODEL_OUTPUT_PATH_COORDINATED)
        versioned_scaler_path = create_new_dir_version(SCALER_OUTPUT_PATH_COORDINATED)

    train_log_path = os.path.join(versioned_log_path, TRAIN_CSV_NAME)
    lookup_debug_log_path = os.path.join(versioned_log_path, "lookup_debug.csv")
    print(f"[Train Script] Train log CSV: {train_log_path}")
    print(f"[Train Script] Lookup debug CSV: {lookup_debug_log_path}")
    _summary_dir = PRECOMPUTED_SUMMARY_DIR or PRECOMPUTED_DA_RI_SUMMARY_DIR
    print(f"[Train Script] Precomputed summary CSV dir: {_summary_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df_spot_train, _df_val, _df_test = load_input_data(write_test=False)
    input_data_train = prepare_input_data(df_spot_train, versioned_scaler_path, fit_scaler=True)

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
        save_freq=10_000,
        save_path=versioned_model_path,
        name_prefix="ppo_stacked_checkpoint",
    )

    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[64, 64, 64, 64], vf=[64, 64, 64, 64]),
        log_std_init=-0.5,
    )

    if RESUME_TRAINING:
        load_path = os.path.join(versioned_model_path, MODEL_CHECKPOINT + ".zip")
        print(f"Resuming training from: {load_path}")
        model = CustomPPO.load(load_path, device=device)
        model.set_env(env)
        model.train_log_path = train_log_path
        reset_num_timesteps = False
    else:
        print("Starting training from scratch (v3 lookup).")
        model = CustomPPO(
            "MlpPolicy",
            env,
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
            cf_only_after_steps= 900_000,
        )
        reset_num_timesteps = True

    model.learn(
        total_timesteps=TRAINING_STEPS_INTELLIGENT,
        callback=checkpoint_callback,
        reset_num_timesteps=reset_num_timesteps,
    )

    model.save(os.path.join(versioned_model_path, COORDINATED_MODEL_NAME_QH))
    print("Finished training!")
