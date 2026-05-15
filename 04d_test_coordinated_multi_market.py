"""
Test a trained coordinated multi-market PPO model on train + test periods,
then run rolling intrinsic on the produced DA bid CSVs.

Set ``MODEL_VERSION`` to the same package you used for training
(e.g. ``v3_counterfactual_reward_lookup``) so ``learning_utils``, ``BasicBatteryDAM``,
and ``CustomPPO`` stay aligned with the checkpoint.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pandas as pd
import torch
from loguru import logger
from stable_baselines3.common.vec_env import DummyVecEnv

from src.coordinated_multi_market.rolling_intrinsic.testing_rolling_intrinsic_qh_intelligent_stacking_open_pos import (
    simulate_days_stacked_quarterhourly_products,
)
from src.shared.config import (
    BUCKET_SIZE,
    C_RATE,
    LOGGING_PATH_COORDINATED,
    MAX_CYCLES_PER_YEAR,
    MIN_TRADES,
    MODEL_OUTPUT_PATH_COORDINATED,
    RTE,
    SCALER_OUTPUT_PATH_COORDINATED,
    TEST_CSV_NAME,
)

# ---------------------------------------------------------------------------
# Pipeline selection: must match the module you trained with.
# Keys are package suffixes under ``src.coordinated_multi_market``.
# Use "legacy" for the top-level ``src.coordinated_multi_market.*`` (no v-prefixed subpackage).
# ---------------------------------------------------------------------------
MODEL_VERSION = "v13_baseline_counterfactual_reward"
# Examples: "legacy", "v1_baseline_without_extensions", "v2_baseline_extended_obs_space",
#           "v3_counterfactual_reward_lookup", "v4_counterfactual_reward_calculating"

_PIPELINE_PACKAGES: dict[str, str] = {
    "legacy": "src.coordinated_multi_market",
    "v1_baseline_without_extensions": (
        "src.coordinated_multi_market.v1_baseline_without_extensions"
    ),
    "v10_baseline_without_extensions_reward_lookup": (
        "src.coordinated_multi_market.v10_baseline_without_extensions_reward_lookup"
    ),
    "v11_baseline_extended_observation_space": (
        "src.coordinated_multi_market.v11_baseline_extended_observation_space"
    ),
    "v12_baseline_open_positions": (
        "src.coordinated_multi_market.v12_baseline_open_positions"
    ),
    "v13_baseline_counterfactual_reward": (
        "src.coordinated_multi_market.v13_baseline_counterfactual_reward"
    ),

    "v2_baseline_extended_obs_space": (
        "src.coordinated_multi_market.v2_baseline_extended_obs_space"
    ),
    "v3_counterfactual_reward_lookup": (
        "src.coordinated_multi_market.v3_counterfactual_reward_lookup"
    ),
    "v4_counterfactual_reward_calculating": (
        "src.coordinated_multi_market.v4_counterfactual_reward_calculating"
    ),
}


def _load_pipeline(version_key: str):
    if version_key not in _PIPELINE_PACKAGES:
        raise ValueError(
            f"Unknown MODEL_VERSION {version_key!r}. "
            f"Choose one of: {sorted(_PIPELINE_PACKAGES)}"
        )
    pkg = _PIPELINE_PACKAGES[version_key]
    lu = importlib.import_module(f"{pkg}.learning_utils")
    env_mod = importlib.import_module(f"{pkg}.basic_battery_dam_env")
    ppo_mod = importlib.import_module(f"{pkg}.custom_ppo")
    return lu.load_input_data, lu.prepare_input_data, env_mod.BasicBatteryDAM, ppo_mod.CustomPPO


def _vec_reset(env):
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out


def _write_drl_da_bids_csv(
    path: str,
    behaviour_rows: list[dict],
) -> None:
    """
    CSV required by ``simulate_days_stacked_quarterhourly_products``:
    index ``time`` (hourly Berlin timestamps), columns at least
    ``capacity_trade`` and ``epex_spot_60min_de_lu_eur_per_mwh``.
    The v3 env no longer appends this file in test mode; we build it here.
    """
    if not behaviour_rows:
        raise ValueError(f"No behaviour rows to write; cannot create {path}")
    df = pd.DataFrame(behaviour_rows).sort_values("time")
    df = df.set_index("time")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path)
    logger.info("Wrote DRL DA bids CSV for rolling intrinsic: %s (%s rows)", path, len(df))


def _collect_behaviour_rows_one_day(
    model,
    BasicBatteryDAM,
    day_key: str,
    day_value: dict,
    logging_path: str,
) -> list[dict]:
    base_env = BasicBatteryDAM(
        modus="test",
        logging_path=logging_path,
        input_data={day_key: day_value},
        round_trip_efficiency=RTE,
    )
    env = DummyVecEnv([lambda b=base_env: b])
    obs = _vec_reset(env)
    rows: list[dict] = []
    for _ in range(24):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        step_info = info[0]
        rows.append(
            {
                "time": pd.to_datetime(
                    step_info["timestamp"], utc=True
                ).tz_convert("Europe/Berlin"),
                "capacity_trade": float(step_info["position"]),
                "epex_spot_60min_de_lu_eur_per_mwh": float(
                    step_info["clearing_price"]
                ),
                "reward": float(reward[0]),
            }
        )
        if done[0]:
            break
    env.close()
    return rows


if __name__ == "__main__":
    load_input_data, prepare_input_data, BasicBatteryDAM, CustomPPO = _load_pipeline(
        MODEL_VERSION
    )
    logger.info("Test pipeline MODEL_VERSION=%s", MODEL_VERSION)

    # Specify to be analysed model
    model_number = "18"
    model_checkpoint = "ppo_stacked_checkpoint_5000000_steps"

    versioned_log_path = os.path.join(LOGGING_PATH_COORDINATED, model_number)
    versioned_model_path = os.path.join(MODEL_OUTPUT_PATH_COORDINATED, model_number)
    versioned_scaler_path = os.path.join(SCALER_OUTPUT_PATH_COORDINATED, model_number)
    Path(versioned_log_path).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s" % device)
    df_spot_train, _df_spot_val, df_spot_test = load_input_data(write_test=True)

    input_data_test = prepare_input_data(
        df_spot_test, versioned_scaler_path, fit_scaler=False
    )

    model = CustomPPO.load(
        path=os.path.join(versioned_model_path, model_checkpoint + ".zip"),
        device=device,
    )

    # ---------- TEST-SET: RL-Läufe + DA-Bids-CSV für Rolling Intrinsic ----------
    test_behaviour_rows: list[dict] = []
    for key, value in input_data_test.items():
        test_behaviour_rows.extend(
            _collect_behaviour_rows_one_day(
                model, BasicBatteryDAM, key, value, versioned_log_path
            )
        )

    test_da_bids_path = os.path.join(versioned_log_path, TEST_CSV_NAME)
    _write_drl_da_bids_csv(test_da_bids_path, test_behaviour_rows)

    logger.info(
        "Finished testing for period: df_spot_test (length %s)" % (len(df_spot_test))
    )

    logger.info("Finished creating report of model behaviour")

    # ---------- TEST-SET: Rolling Intrinsic ----------
    ri_qh_output_path_test = os.path.join(
        versioned_log_path,
        "rolling_intrinsic_intelligently_stacked_on_day_ahead_qh",
        "bs"
        + str(BUCKET_SIZE)
        + "cr"
        + str(C_RATE)
        + "rto"
        + str(RTE)
        + "mc"
        + str(MAX_CYCLES_PER_YEAR)
        + "mt"
        + str(MIN_TRADES),
    )

    start_day_test = (
        df_spot_test.index.min().tz_convert("Europe/Berlin").normalize()
    )
    end_day_test = (
        df_spot_test.index.max().tz_convert("Europe/Berlin").normalize()
        + pd.Timedelta(days=1)
    )

    simulate_days_stacked_quarterhourly_products(
        da_bids_path=test_da_bids_path,
        output_path=ri_qh_output_path_test,
        start_day=start_day_test,
        end_day=end_day_test,
        discount_rate=0,
        c_rate=C_RATE,
        roundtrip_eff=RTE,
        max_cycles=MAX_CYCLES_PER_YEAR,
        min_trades=MIN_TRADES,
    )

    logger.info(
        "Finished calculating intelligently stacked rolling intrinsic revenues with quarterhourly products (TEST)."
    )

    test_advanced_drl_ri_qh = pd.read_csv(
        os.path.join(ri_qh_output_path_test, "profit.csv")
    )

    mean_test_profit = (
        test_advanced_drl_ri_qh.sort_values(by="day").reset_index()["profit"].mean()
    )

    print("Test: ", mean_test_profit)

    # ---------- TRAIN-SET: RL-Läufe ----------
    input_data_train = prepare_input_data(
        df_spot_train, versioned_scaler_path, fit_scaler=False
    )

    versioned_log_path_train = os.path.join(versioned_log_path, "train")
    Path(versioned_log_path_train).mkdir(parents=True, exist_ok=True)

    train_behaviour_rows: list[dict] = []
    for key, value in input_data_train.items():
        train_behaviour_rows.extend(
            _collect_behaviour_rows_one_day(
                model, BasicBatteryDAM, key, value, versioned_log_path_train
            )
        )

    train_da_bids_path = os.path.join(versioned_log_path_train, TEST_CSV_NAME)
    _write_drl_da_bids_csv(train_da_bids_path, train_behaviour_rows)

    logger.info(
        "Finished testing for period: df_spot_train (length %s)"
        % (len(df_spot_train))
    )

    logger.info("Finished creating report of model behaviour (TRAIN)")

    # ---------- TRAIN-SET: Rolling Intrinsic ----------
    ri_qh_output_path_train = os.path.join(
        versioned_log_path_train,
        "rolling_intrinsic_intelligently_stacked_on_day_ahead_qh",
        "bs"
        + str(BUCKET_SIZE)
        + "cr"
        + str(C_RATE)
        + "rto"
        + str(RTE)
        + "mc"
        + str(MAX_CYCLES_PER_YEAR)
        + "mt"
        + str(MIN_TRADES),
    )

    start_day_train = (
        df_spot_train.index.min().tz_convert("Europe/Berlin").normalize()
    )
    end_day_train = (
        df_spot_train.index.max().tz_convert("Europe/Berlin").normalize()
        + pd.Timedelta(days=1)
    )

    simulate_days_stacked_quarterhourly_products(
        da_bids_path=train_da_bids_path,
        output_path=ri_qh_output_path_train,
        start_day=start_day_train,
        end_day=end_day_train,
        discount_rate=0,
        c_rate=C_RATE,
        roundtrip_eff=RTE,
        max_cycles=MAX_CYCLES_PER_YEAR,
        min_trades=MIN_TRADES,
    )

    logger.info(
        "Finished calculating intelligently stacked rolling intrinsic revenues with quarterhourly products (TRAIN)."
    )

    train_advanced_drl_ri_qh = pd.read_csv(
        os.path.join(ri_qh_output_path_train, "profit.csv")
    )

    mean_train_profit = (
        train_advanced_drl_ri_qh.sort_values(by="day").reset_index()["profit"].mean()
    )

    print("Average Profit of Test or Train set")
    print("Test: ", mean_test_profit)
    print("Train: ", mean_train_profit)

    if mean_train_profit > 1.1 * mean_test_profit:
        raise ValueError(
            f"Train profit {mean_train_profit} is more than 10% higher than test profit {mean_test_profit}, this points towards overfitting - Check!"
        )
