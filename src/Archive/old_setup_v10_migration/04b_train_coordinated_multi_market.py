"""
Train PPO Agent — old_setup_v10_migration.

Uses the v10 training flow while keeping outputs in the old-setup migration
directory tree so behavior can be compared step by step.
"""

import os
import shutil
import warnings

import torch

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.Archive.old_setup_v10_migration.basic_battery_dam_env import BasicBatteryDAM
from src.Archive.old_setup_v10_migration.custom_ppo import CustomPPO
from src.Archive.old_setup_v10_migration.learning_utils import (
    load_input_data,
    prepare_input_data,
    linear_schedule,
)

from src.shared.folder_versioning import create_new_dir_version
from src.Archive.old_setup_v10_migration.local_config import (
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
SOURCE_MODEL_NUMBER = "13"
MODEL_CHECKPOINT = "ppo_stacked_checkpoint_1000000_steps"
TARGET_MODEL_NUMBER = "19"
RESUME_WITH_NEW_HYPERPARAMS = True
RESET_TIMESTEPS_ON_RESTART = False
PRECOMPUTED_SUMMARY_DIR = None

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
    if RESUME_TRAINING and source_scaler_path and source_scaler_path != versioned_scaler_path:
        src_scaler = os.path.join(source_scaler_path, "scaler.pkl")
        dst_scaler = os.path.join(versioned_scaler_path, "scaler.pkl")
        if os.path.exists(src_scaler) and not os.path.exists(dst_scaler):
            shutil.copy2(src_scaler, dst_scaler)
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
        verbose=1,
        tensorboard_log=TENSORBOARD_PATH_INTELLIGENT,
        device=device,
        seed=SEED,
        intraday_product_type="QH",
        train_log_path=train_log_path,
        precomputed_summary_dir=_summary_dir,
        lookup_debug_log_path=lookup_debug_log_path,
        policy_kwargs=policy_kwargs,
        ent_coef=0.05,
        n_steps=512,
        clip_range=0.4,
        batch_size=128,
        vf_coef=0.4,
        learning_rate=linear_schedule(1e-4),
    )

    if RESUME_TRAINING:
        load_path = os.path.join(
            source_model_path,
            MODEL_CHECKPOINT + ".zip",
        )
        print(f"Resuming training from: {load_path}")
        if RESUME_WITH_NEW_HYPERPARAMS:
            print("Resume mode: loading checkpoint weights into fresh model config.")
            model = CustomPPO("MlpPolicy", env, **model_kwargs)
            model.set_parameters(load_path, exact_match=False, device=device)
            reset_num_timesteps = RESET_TIMESTEPS_ON_RESTART
        else:
            model = CustomPPO.load(load_path, device=device)
            model.set_env(env)
            model.train_log_path = train_log_path
            model.lookup_debug_log_path = lookup_debug_log_path
            reset_num_timesteps = False
    else:
        print("Starting training from scratch (old_setup_v10_migration precomputed RI lookup).")
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