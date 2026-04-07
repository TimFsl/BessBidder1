"""
Validate coordinated multi-market PPO checkpoints using **precomputed DA+RI summaries**
(same lookup as v3 training: ``summary_YYYY-MM-DD.csv``).

This mirrors ``04c_validate_coordinated_multi_market.py`` for RL rollout and
``validation_summary.csv`` columns, but **does not** run the live Gurobi rolling-intrinsic
simulation — profits come from ``PrecomputedSummaryLookup`` keyed by realized
``capacity_trade`` per day.

Requires a precomputed results directory covering all validation delivery days.

Per checkpoint, besides ``TEST_CSV_NAME`` behaviour log, writes
``da_schedules_validation.csv``: one row per day with realized DA schedule
(``schedule_kind``, CSV hours, timesteps 0–23, volumes).
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from stable_baselines3.common.vec_env import DummyVecEnv

from src.coordinated_multi_market.v3_counterfactual_reward_lookup.basic_battery_dam_env import (
    BasicBatteryDAM,
)
from src.coordinated_multi_market.v3_counterfactual_reward_lookup.custom_ppo import CustomPPO
from src.coordinated_multi_market.v3_counterfactual_reward_lookup.learning_utils import (
    load_input_data,
    prepare_input_data,
)
from src.coordinated_multi_market.v3_counterfactual_reward_lookup.precomputed_summary_lookup import (
    PrecomputedSummaryLookup,
    realized_volumes_to_schedule,
)
from src.shared.config import (
    LOGGING_PATH_COORDINATED,
    MODEL_OUTPUT_PATH_COORDINATED,
    PRECOMPUTED_DA_RI_SUMMARY_DIR,
    SCALER_OUTPUT_PATH_COORDINATED,
    TEST_CSV_NAME,
)

# Override if summaries live elsewhere (default: config.PRECOMPUTED_DA_RI_SUMMARY_DIR)
PRECOMPUTED_SUMMARY_DIR = None


def _compute_series_stats(series: pd.Series):
    series = series.dropna()
    if len(series) == 0:
        return (None, None, None, None, None)

    total = series.sum()
    mean = series.mean()
    median = series.median()
    std = series.std(ddof=0)
    q95 = series.quantile(0.95) if len(series) > 1 else series.iloc[0]

    return total, mean, median, std, q95


def _profits_from_precomputed_lookup(
    df_behaviour: pd.DataFrame,
    lookup: PrecomputedSummaryLookup,
) -> tuple[pd.Series, pd.Series, pd.Series, int]:
    """
    Per delivery day: combined, DA, IDC profit (EUR) from lookup.
    Uses realized volumes (capacity_trade) → schedule key, same as v3 training.
    Returns (daily_total, daily_dam, daily_idc, lookup_misses_increment).
    """
    df = df_behaviour.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"])
    df["delivery_day"] = df["time"].dt.tz_convert("Europe/Berlin").dt.date

    combined: dict = {}
    dam: dict = {}
    misses = 0

    for day, g in df.groupby("delivery_day"):
        g = g.sort_values("time")
        vol = g["capacity_trade"].to_numpy(dtype=np.float64)
        if len(vol) != 24:
            logger.warning(
                f"Day {day}: expected 24 hourly rows, got {len(vol)}. Skipping."
            )
            misses += 1
            continue
        day_iso = str(day)
        k, bh, sh = realized_volumes_to_schedule(vol)
        if k == "invalid":
            misses += 1
            combined[day] = 0.0
            dam[day] = 0.0
            continue
        row = lookup.row_for_schedule(day_iso, k, bh, sh)
        if row is None:
            misses += 1
            combined[day] = 0.0
            dam[day] = 0.0
            continue
        p = float(row["profit"])
        dv = row.get("da_profit", np.nan)
        d = 0.0 if pd.isna(dv) else float(dv)
        combined[day] = p
        dam[day] = d

    daily_total = pd.Series(combined, dtype=float).sort_index()
    daily_dam = pd.Series(dam, dtype=float).reindex(daily_total.index, fill_value=0.0)
    daily_idc = daily_total - daily_dam

    return daily_total, daily_dam, daily_idc, misses


def _build_da_schedule_table(df_behaviour: pd.DataFrame) -> pd.DataFrame:
    """
    One row per delivery day: realized DA schedule derived from ``capacity_trade``
    (same keying as precomputed lookup / v3 training).
    """
    df = df_behaviour.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"])
    df["delivery_day"] = df["time"].dt.tz_convert("Europe/Berlin").dt.date

    rows: list[dict] = []
    for day, g in df.groupby("delivery_day"):
        g = g.sort_values("time")
        vol = g["capacity_trade"].to_numpy(dtype=np.float64)
        if len(vol) != 24:
            rows.append(
                {
                    "delivery_day": str(day),
                    "n_hours": len(vol),
                    "schedule_kind": "incomplete_day",
                    "buy_hour_csv": np.nan,
                    "sell_hour_csv": np.nan,
                    "buy_timestep_0_23": np.nan,
                    "sell_timestep_0_23": np.nan,
                    "volume_buy_abs_sum": np.nan,
                    "volume_sell_abs_sum": np.nan,
                    "summary": "incomplete_day",
                }
            )
            continue

        k, bh, sh = realized_volumes_to_schedule(vol)
        neg_idx = np.where(vol < -1e-9)[0]
        pos_idx = np.where(vol > 1e-9)[0]
        v_buy = float(np.sum(-vol[neg_idx])) if len(neg_idx) else 0.0
        v_sell = float(np.sum(vol[pos_idx])) if len(pos_idx) else 0.0

        ib = int(neg_idx[0]) if len(neg_idx) else np.nan
        is_ = int(pos_idx[0]) if len(pos_idx) else np.nan

        if k == "no_da":
            summary = "no_da"
        elif k == "buy_only":
            summary = f"buy_only buy_hour={float(bh)}"
        elif k == "buy_sell":
            summary = f"buy_sell buy_hour={float(bh)} sell_hour={float(sh)}"
        else:
            summary = "invalid (e.g. sell before buy or sell-only)"

        rows.append(
            {
                "delivery_day": str(day),
                "n_hours": 24,
                "schedule_kind": k,
                "buy_hour_csv": bh,
                "sell_hour_csv": sh,
                "buy_timestep_0_23": ib,
                "sell_timestep_0_23": is_,
                "volume_buy_abs_sum": v_buy,
                "volume_sell_abs_sum": v_sell,
                "summary": summary,
            }
        )

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("delivery_day").reset_index(drop=True)
    return out


if __name__ == "__main__":
    # -------- Same knobs as 04c_validate_coordinated_multi_market.py --------
    model_number = "3"
    start_checkpoint = "ppo_stacked_checkpoint_100000_steps"
    STEP_INCREMENT = 100_000

    summary_dir = PRECOMPUTED_SUMMARY_DIR or PRECOMPUTED_DA_RI_SUMMARY_DIR
    logger.info(f"Precomputed summary dir: {summary_dir}")

    versioned_log_base_path = os.path.join(LOGGING_PATH_COORDINATED, model_number)
    versioned_model_path = os.path.join(MODEL_OUTPUT_PATH_COORDINATED, model_number)
    versioned_scaler_path = os.path.join(SCALER_OUTPUT_PATH_COORDINATED, model_number)

    validation_root_path = os.path.join(versioned_log_base_path, "validation_precomputed")
    Path(validation_root_path).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s" % device)

    df_spot_train, df_spot_val, _df_test = load_input_data(write_test=False)
    logger.info(
        "Validation period: %s -> %s (len=%s)"
        % (df_spot_val.index.min(), df_spot_val.index.max(), len(df_spot_val))
    )

    parts = start_checkpoint.split("_")
    try:
        start_steps = int(parts[-2])
    except (ValueError, IndexError):
        raise ValueError(
            f"Cannot parse steps from checkpoint name '{start_checkpoint}'. "
            "Expected: <prefix>_<steps>_steps"
        )
    checkpoint_prefix = "_".join(parts[:-2])

    summary_path = os.path.join(validation_root_path, "validation_summary.csv")

    validation_summary = pd.DataFrame(
        columns=[
            "checkpoint_name",
            "steps",
            "mean_rl_reward",
            "total_rl_reward",
            "num_days_ri",
            "total_profit",
            "mean_profit",
            "median_profit",
            "std_profit",
            "q95_profit",
            "idc_dam_profit_ratio",
            "total_dam_profit",
            "mean_dam_profit",
            "total_idc_profit",
            "mean_idc_profit",
            "precomputed_lookup_misses",
        ]
    )

    lookup = PrecomputedSummaryLookup(summary_dir)

    current_steps = start_steps
    while True:
        checkpoint_name = f"{checkpoint_prefix}_{current_steps}_steps"
        checkpoint_file = os.path.join(versioned_model_path, checkpoint_name + ".zip")

        if not os.path.exists(checkpoint_file):
            logger.info(f"Checkpoint {checkpoint_file} does not exist. Stopping.")
            break

        logger.info(f"Validate (precomputed RI): {checkpoint_name}")

        ckpt_log_path = os.path.join(validation_root_path, checkpoint_name)
        Path(ckpt_log_path).mkdir(parents=True, exist_ok=True)

        model = CustomPPO.load(path=checkpoint_file, device=device)

        input_data_val = prepare_input_data(
            df_spot_val, versioned_scaler_path, fit_scaler=False
        )
        behaviour_rows = []

        for _key, value in input_data_val.items():
            env = BasicBatteryDAM(
                modus="test",
                logging_path=ckpt_log_path,
                input_data={_key: value},
            )
            env = DummyVecEnv([lambda: env])
            obs = env.reset()

            for _ in range(24):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                step_info = info[0]
                behaviour_rows.append(
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
                if bool(done[0]):
                    obs = env.reset()
                    break
            env.close()

        behaviour_path = os.path.join(ckpt_log_path, TEST_CSV_NAME)
        if behaviour_rows:
            df_behaviour = pd.DataFrame(behaviour_rows).sort_values("time")
            df_behaviour.to_csv(behaviour_path, index=False)
            da_sched_path = os.path.join(ckpt_log_path, "da_schedules_validation.csv")
            _build_da_schedule_table(df_behaviour).to_csv(da_sched_path, index=False)
            logger.info(f"Saved DA schedule summary: {da_sched_path}")
        else:
            df_behaviour = pd.DataFrame()

        mean_rl_reward = None
        total_rl_reward = None
        if os.path.exists(behaviour_path):
            df_b = pd.read_csv(behaviour_path)
            if "reward" in df_b.columns:
                total_rl_reward = df_b["reward"].sum()
                mean_rl_reward = df_b["reward"].mean()

        # ---- Precomputed DA+RI profit (no live RI run) ----
        num_days_ri = 0
        total_profit = mean_profit = median_profit = std_profit = q95_profit = None
        total_dam_profit = mean_dam_profit = None
        total_idc_profit = mean_idc_profit = None
        idc_dam_profit_ratio = None
        lookup_misses_total = 0

        if len(df_behaviour) > 0:
            daily_total, daily_dam, daily_idc, lookup_misses_total = (
                _profits_from_precomputed_lookup(df_behaviour, lookup)
            )
            num_days_ri = len(daily_total)

            (
                total_profit,
                mean_profit,
                median_profit,
                std_profit,
                q95_profit,
            ) = _compute_series_stats(daily_total)

            (
                total_dam_profit,
                mean_dam_profit,
                _,
                _,
                _,
            ) = _compute_series_stats(daily_dam)

            (
                total_idc_profit,
                mean_idc_profit,
                _,
                _,
                _,
            ) = _compute_series_stats(daily_idc)

            if total_idc_profit not in (None, 0):
                idc_dam_profit_ratio = total_dam_profit / total_idc_profit
            else:
                idc_dam_profit_ratio = None
        else:
            logger.warning("No behaviour rows; skipping precomputed profit.")

        validation_summary = pd.concat(
            [
                validation_summary,
                pd.DataFrame(
                    [
                        [
                            checkpoint_name,
                            current_steps,
                            mean_rl_reward,
                            total_rl_reward,
                            num_days_ri,
                            total_profit,
                            mean_profit,
                            median_profit,
                            std_profit,
                            q95_profit,
                            idc_dam_profit_ratio,
                            total_dam_profit,
                            mean_dam_profit,
                            total_idc_profit,
                            mean_idc_profit,
                            lookup_misses_total,
                        ]
                    ],
                    columns=validation_summary.columns,
                ),
            ],
            ignore_index=True,
        )

        validation_summary.to_csv(summary_path, index=False)
        logger.info(f"Saved validation summary: {summary_path}")

        current_steps += STEP_INCREMENT

    logger.info("Done (precomputed RI validation).")
    print(validation_summary)
