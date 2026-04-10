"""
Train PPO Agent for Coordinated Multi-Market Battery Dispatch

Uses Discrete(3) DA actions (idle / full buy / full sell). Old checkpoints from
7-action training cannot be loaded on this policy.

This script:
- Loads and preprocesses spot market data.
- Sets up a custom Stable-Baselines3 PPO environment (`BasicBatteryDAM`).
- Trains the agent using PPO with custom architecture and logging.
- Saves the model, logs, and scaler using versioned output folders.

Requires:
- Training data via `load_input_data()`
- `BasicBatteryDAM` environment for DRL coordination
- Stable-Baselines3 and PyTorch
"""

import os
import pandas as pd
import torch
import warnings

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


# Imports from this version package (counterfactual RI reward; shared rolling_intrinsic)
from src.coordinated_multi_market.v2_counterfactual_reward.basic_battery_dam_env import (
    BasicBatteryDAM,
)
from src.coordinated_multi_market.v2_counterfactual_reward.custom_ppo import CustomPPO
from src.coordinated_multi_market.v2_counterfactual_reward.learning_utils import (
    load_input_data,
    prepare_input_data,
    linear_schedule,
    orthogonal_weight_init,
    # CustomPPO,
)

from src.shared.folder_versioning import create_new_dir_version
from src.shared.config import (
    COORDINATED_MODEL_NAME_QH,
    LOGGING_PATH_COORDINATED,
    MODEL_OUTPUT_PATH_COORDINATED,
    RTE,
    SCALER_OUTPUT_PATH_COORDINATED,
    SEED,
    TENSORBOARD_PATH_INTELLIGENT,
    TRAIN_CSV_NAME,
    TRAINING_STEPS_INTELLIGENT,
)
warnings.simplefilter(action="ignore", category=FutureWarning)



RESUME_TRAINING = False      # set to TRUE, if training should be continued from a checkpoint

# Only relevant if RESUME_TRAINING = True
MODEL_NUMBER = "3"  
MODEL_CHECKPOINT = "ppo_stacked_checkpoint_280000_steps"


if __name__ == "__main__":
    
    # Ensure output folders exist
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
    print(f"[Train Script] Train log CSV: {train_log_path}")

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and prepare input data
    df_spot_train, df_spot_val, df_spot_test = load_input_data(write_test=False)

    # Fürs Training nur df_spot_train verwenden
    input_data_train = prepare_input_data(df_spot_train, versioned_scaler_path, fit_scaler=True)

    # Initialize training environment
    env = BasicBatteryDAM(
        modus="train",
        logging_path=versioned_log_path,
        input_data=input_data_train,
        round_trip_efficiency=RTE,
    )

    # Validate environment
    check_env(env)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    # Callback to save intermediate model checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=versioned_model_path,
        name_prefix="ppo_stacked_checkpoint",
    )

    # Define custom policy architecture
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[64, 64, 64, 64], vf=[64, 64, 64, 64]),
        log_std_init=-0.5,
        # init_fn=orthogonal_weight_init,  # Optional: Orthogonal weight init
    )

    if RESUME_TRAINING:
        load_path = os.path.join(
            versioned_model_path,
            MODEL_CHECKPOINT + ".zip",
        )
        print(f"Resuming training from: {load_path}")

        model = CustomPPO.load(load_path, device=device)
        model.set_env(env)
        model.train_log_path = train_log_path

        reset_num_timesteps = False
    else:
        print("Starting training from scratch.")

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
            n_steps=480,
            clip_range=0.4,
            batch_size=120,
            vf_coef=0.4,
            learning_rate=linear_schedule(1e-4),
            gamma=0.999,
            # 3-action env: idle=0; at most 2 CF replays/day (first buy + first sell hour)
            counterfactual_idle_action=0,
            counterfactual_active_mode="first_buy_first_sell",
        )
        reset_num_timesteps = True

    # Train the model
    model.learn(
        total_timesteps=TRAINING_STEPS_INTELLIGENT,
        callback=checkpoint_callback,
        reset_num_timesteps=reset_num_timesteps
    )

    # Save the final model
    model.save(os.path.join(versioned_model_path, COORDINATED_MODEL_NAME_QH))

    print("Finished training!")
