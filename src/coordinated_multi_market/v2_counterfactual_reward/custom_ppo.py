from __future__ import annotations

import concurrent.futures
import os
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.ppo import PPO


from src.coordinated_multi_market.rolling_intrinsic.training_rolling_intrinsic_qh_intelligent_stacking_open_pos import (
    simulate_days_stacked_quarterhourly_products,
)


from src.shared.config import BUCKET_SIZE, C_RATE, MAX_CYCLES_PER_DAY, MIN_TRADES, RTE, START_IDC_STEPS

if TYPE_CHECKING:
    from .basic_battery_dam_env import BasicBatteryDAM

# Matches v1: total RI reward over episode = ri_eur / RI_REWARD_SCALE
RI_REWARD_SCALE = 10.0
# With Discrete(3): 0 = idle, 1 = full buy, 2 = full sell
DEFAULT_IDLE_DA_ACTION = 0


# Curriculum: (step_threshold, max_cycles). Steps < threshold get that max_cycles.
CURRICULUM_MAX_CYCLES = [
    (300_000, 2.0),
    (600_000, 1.0),
    #(900_000, 1.0),
    (float("inf"), 1.0),
]


class CustomPPO(PPO):
    def __init__(
        self,
        *args,
        intraday_product_type: str = None,
        reward_log_path: str | None = None,
        train_log_path: str | None = None,
        use_counterfactual_ri_reward: bool = True,
        counterfactual_idle_action: int = DEFAULT_IDLE_DA_ACTION,
        counterfactual_min_abs_volume: float = 1e-9,
        max_counterfactual_steps_per_episode: Optional[int] = None,
        counterfactual_active_mode: str = "first_buy_first_sell",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.intraday_product_type = intraday_product_type
        self.current_step = 0
        self.lambda_val = 0.5
        self._last_ri_reward_per_euro = 0
        self.reward_log_path = reward_log_path
        self.train_log_path = train_log_path
        self.use_counterfactual_ri_reward = use_counterfactual_ri_reward
        self.counterfactual_idle_action = int(counterfactual_idle_action)
        self.counterfactual_min_abs_volume = float(counterfactual_min_abs_volume)
        self.max_counterfactual_steps_per_episode = max_counterfactual_steps_per_episode
        self.counterfactual_active_mode = str(counterfactual_active_mode)

    def _update_cycle_curriculum(self, env: VecEnv) -> None:
        """Set max_cycles in env from current step (0–200k: 3, 200k–400k: 2, 400k+: 1)."""
        steps = self.num_timesteps
        max_cycles = 1.0
        for threshold, cycles in CURRICULUM_MAX_CYCLES:
            if steps < threshold:
                max_cycles = cycles
                break
        try:
            env.env_method("set_max_cycles", max_cycles)
        except AttributeError:
            pass
        self.logger.record("curriculum/max_cycles", max_cycles)

    @staticmethod
    def _unwrap_to_basic_battery_dam(env: VecEnv):
        """Unwrap Monitor / VecEnv to the inner ``BasicBatteryDAM``."""
        e = env.envs[0]
        while hasattr(e, "env"):
            e = e.env
        return e

    @staticmethod
    def _replay_episode_replacing_action_at(
        template_env: BasicBatteryDAM,
        day: str,
        actions: np.ndarray,
        horizon: int,
        replace_with_idle_at: int | None,
        idle_action: int,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Physics-consistent replay: same discrete action sequence as the rollout,
        except at ``replace_with_idle_at`` the action is replaced by ``idle_action``
        (default: hold / 0 MW desired quantity).
        """
        from .basic_battery_dam_env import BasicBatteryDAM

        rte = float(template_env._efficiency**2)
        replay = BasicBatteryDAM(
            modus=template_env._modus,
            logging_path=template_env._logging_path,
            input_data=template_env._input_data,
            power=template_env._power,
            capacity=template_env._capacity,
            round_trip_efficiency=np.float32(rte),
            start_end_soc=template_env._start_end_soc,
            max_cycles=float(template_env.max_cycles),
        )
        replay.reset(options={"day": day})
        volumes: list[float] = []
        clearings: list[float] = []
        total_da_reward = 0.0
        for t in range(horizon):
            a = int(np.asarray(actions[t]).item())
            if replace_with_idle_at is not None and t == replace_with_idle_at:
                a = idle_action
            _obs, r, _term, _trunc, info = replay.step(a)
            total_da_reward += float(r)
            volumes.append(float(info["position"]))
            clearings.append(float(info["clearing_price"]))
        return (
            np.asarray(volumes, dtype=np.float64),
            np.asarray(clearings, dtype=np.float64),
            total_da_reward,
        )

    def _ri_total_profit_qh(self, period_timestamps, da_trades: pd.DataFrame) -> float:
        res = self.run_simulations_quarterhourly_products_in_parallel(
            period_timestamps, da_trades
        )
        return float(res["total_profit"].sum())

    @staticmethod
    def _counterfactual_indices_first_buy_first_sell(
        actions_period: np.ndarray,
    ) -> np.ndarray:
        """
        At most 2 counterfactuals per day: first hour with action==1 (full buy),
        first hour with action==2 (full sell). Reduces compute vs one CF per trade.
        """
        a = np.asarray(actions_period).flatten().astype(np.int64, copy=False)
        idxs: list[int] = []
        buys = np.where(a == 1)[0]
        if len(buys) > 0:
            idxs.append(int(buys[0]))
        sells = np.where(a == 2)[0]
        if len(sells) > 0:
            idxs.append(int(sells[0]))
        return np.array(sorted(set(idxs)), dtype=np.int64)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)
        n_steps = 0
        rollout_buffer.reset()
        timestamp_buffer = np.zeros(n_rollout_steps)
        position_buffer = np.zeros(n_rollout_steps)
        clearing_price_buffer = np.zeros(n_rollout_steps)
        log_episode_id_buffer = np.zeros(n_rollout_steps, dtype=np.int64)
        log_dam_price_forecast_buffer = np.zeros(n_rollout_steps, dtype=np.float64)
        log_capacity_bid_buffer = np.zeros(n_rollout_steps, dtype=np.float64)
        log_delta_soc_buffer = np.zeros(n_rollout_steps, dtype=np.float64)
        log_remaining_cycles_buffer = np.zeros(n_rollout_steps, dtype=np.float64)
        log_profit_buffer = np.zeros(n_rollout_steps, dtype=np.float64)
        log_soc_buffer = np.zeros(n_rollout_steps, dtype=np.float64)

        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()
        self._update_cycle_curriculum(env)

        while n_steps < n_rollout_steps:
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and n_steps % self.sde_sample_freq == 0
            ):
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(
                        actions, self.action_space.low, self.action_space.high
                    )

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"]
                    )[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones
            if n_steps > 0:
                idx = n_steps - 1
                timestamp_buffer[idx] = infos[0]["timestamp"]
                position_buffer[idx] = infos[0]["position"]
                clearing_price_buffer[idx] = infos[0]["clearing_price"]
                scaling_max_price = infos[0]["scaling_max_price"]
                scaling_min_price = infos[0]["scaling_min_price"]
                if "log_episode_id" in infos[0]:
                    log_episode_id_buffer[idx] = infos[0]["log_episode_id"]
                    log_dam_price_forecast_buffer[idx] = infos[0]["log_dam_price_forecast"]
                    log_capacity_bid_buffer[idx] = infos[0]["log_capacity_bid"]
                    log_delta_soc_buffer[idx] = infos[0]["log_delta_soc"]
                    log_remaining_cycles_buffer[idx] = infos[0]["log_remaining_cycles"]
                    log_profit_buffer[idx] = infos[0]["log_profit"]
                    log_soc_buffer[idx] = infos[0]["log_soc"]

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        complete_periods = self._derive_period_lenghts_from_episode_starts_array(
            rollout_buffer.episode_starts.flatten()
        )

        ep_profit_combined = []
        ep_profit_da = []
        ep_profit_idc = []
        ep_reward_combined = []
        ep_reward_da = []
        ep_reward_advantage = []
        # Counterfactual diagnostics (per completed episode in this rollout)
        ep_cf_mean_margins: list[float] = []
        ep_cf_sum_margins_per_episode: list[float] = []
        ep_cf_n_active_hours: list[int] = []
        cf_episodes_strict = 0
        cf_episodes_uniform_fallback = 0

        for row_start, num_rows in complete_periods.items():
            if num_rows <= 0:
                continue

            period_timestamps = pd.to_datetime(
                timestamp_buffer[row_start : row_start + num_rows], utc=True
            ).tz_convert("Europe/Berlin")
            period_volumes = position_buffer[row_start : row_start + num_rows]
            period_clearing_prices = clearing_price_buffer[
                row_start : row_start + num_rows
            ]

            da_rewards = rollout_buffer.rewards[row_start : row_start + num_rows].flatten()
            da_trades = self._derive_day_ahead_trades(
                timestamps=period_timestamps,
                volumes=period_volumes,
                clearing_prices=period_clearing_prices,
                intraday_product_type=self.intraday_product_type,
            )
            da_profit = float(da_trades.profit.sum()) if len(da_trades) > 0 else 0.0

            ri_stacked_profit = 0.0
            baseline_profit = 0.0
            combined_rewards = da_rewards.copy()
            rolling_intrinsic_rewards = np.zeros(num_rows, dtype=np.float64)

            run_ri = (
                self.num_timesteps >= START_IDC_STEPS
                and self._check_if_complete_cycle(period_volumes)
            )
            if run_ri:
                if self.intraday_product_type == "H":
                    (
                        rolling_intrinsic_results_stacked,
                        _,
                    ) = self.run_simulations_hourly_products_in_parallel(
                        period_timestamps, da_trades
                    )
                elif self.intraday_product_type == "QH":
                    rolling_intrinsic_results_stacked = (
                        self.run_simulations_quarterhourly_products_in_parallel(
                            period_timestamps, da_trades
                        )
                    )
                else:
                    raise ValueError(
                        f"Unsupported intraday product type {self.intraday_product_type}. Only QH or H supported"
                    )
                ri_stacked_profit = float(
                    rolling_intrinsic_results_stacked["total_profit"].sum()
                )
                da_rewards = rollout_buffer.rewards[
                    row_start : row_start + num_rows
                ].flatten()
                actions_period = rollout_buffer.actions[
                    row_start : row_start + num_rows
                ]
                if actions_period.ndim > 1:
                    actions_period = actions_period.flatten()

                # Total return scaling matches v1: sum of RI-shaped reward = ri_eur / RI_REWARD_SCALE
                ri_scaled_full = ri_stacked_profit / RI_REWARD_SCALE
                R_full_total = float(np.sum(da_rewards)) + ri_scaled_full

                use_cf = (
                    self.use_counterfactual_ri_reward
                    and self.intraday_product_type == "QH"
                )
                if use_cf:
                    base_env = self._unwrap_to_basic_battery_dam(env)
                    day_str = str(period_timestamps[0].date().isoformat())
                    rolling_intrinsic_rewards = np.zeros(num_rows, dtype=np.float64)
                    combined_rewards = da_rewards.astype(np.float64).copy()

                    if self.counterfactual_active_mode == "first_buy_first_sell":
                        active = self._counterfactual_indices_first_buy_first_sell(
                            actions_period
                        )
                    elif self.counterfactual_active_mode == "volume_nonzero":
                        active = np.where(
                            np.abs(period_volumes) > self.counterfactual_min_abs_volume
                        )[0]
                        if (
                            self.max_counterfactual_steps_per_episode is not None
                            and len(active) > self.max_counterfactual_steps_per_episode
                        ):
                            order = np.argsort(-np.abs(period_volumes[active]))
                            active = active[
                                order[: self.max_counterfactual_steps_per_episode]
                            ]
                    else:
                        raise ValueError(
                            "counterfactual_active_mode must be "
                            "'first_buy_first_sell' or 'volume_nonzero', got "
                            f"{self.counterfactual_active_mode!r}"
                        )

                    margins: list[float] = []
                    if len(active) == 0:
                        # No realized trades: fall back to uniform RI spread (same as v1)
                        cf_episodes_uniform_fallback += 1
                        ri_reward_per_step = ri_stacked_profit / (
                            RI_REWARD_SCALE * num_rows
                        )
                        rolling_intrinsic_rewards = np.full(
                            num_rows, ri_reward_per_step, dtype=np.float64
                        )
                        combined_rewards = da_rewards + rolling_intrinsic_rewards
                    else:
                        cf_episodes_strict += 1
                        for t in active:
                            _vol_cf, _clr_cf, da_sum_cf = (
                                self._replay_episode_replacing_action_at(
                                    base_env,
                                    day_str,
                                    actions_period,
                                    num_rows,
                                    replace_with_idle_at=int(t),
                                    idle_action=self.counterfactual_idle_action,
                                )
                            )
                            da_trades_cf = self._derive_day_ahead_trades(
                                timestamps=period_timestamps,
                                volumes=_vol_cf,
                                clearing_prices=_clr_cf,
                                intraday_product_type=self.intraday_product_type,
                            )
                            ri_cf = self._ri_total_profit_qh(
                                period_timestamps, da_trades_cf
                            )
                            ri_scaled_cf = ri_cf / RI_REWARD_SCALE
                            R_cf_total = float(da_sum_cf) + ri_scaled_cf
                            margin_t = R_full_total - R_cf_total
                            margins.append(margin_t)
                            ti = int(t)
                            rolling_intrinsic_rewards[ti] = margin_t
                            combined_rewards[ti] = float(da_rewards[ti]) + margin_t
                        if margins:
                            ep_cf_mean_margins.append(float(np.mean(margins)))
                            ep_cf_sum_margins_per_episode.append(float(np.sum(margins)))
                            ep_cf_n_active_hours.append(len(margins))

                    rollout_buffer.rewards[row_start : row_start + num_rows] = (
                        combined_rewards.reshape(-1, 1)
                    )
                else:
                    # Legacy: DA + equally scaled IDC profit per step (or hourly product path)
                    ri_reward_per_step = ri_stacked_profit / (
                        RI_REWARD_SCALE * num_rows
                    )
                    rolling_intrinsic_rewards = np.full(
                        num_rows, ri_reward_per_step, dtype=np.float64
                    )
                    combined_rewards = da_rewards + rolling_intrinsic_rewards
                    rollout_buffer.rewards[row_start : row_start + num_rows] = (
                        combined_rewards.reshape(-1, 1)
                    )

            advantage_reward_per_step = rolling_intrinsic_rewards

            ep_profit_combined.append(float(da_profit + ri_stacked_profit))
            ep_profit_da.append(float(da_profit))
            ep_profit_idc.append(float(ri_stacked_profit))
            ep_reward_combined.append(float(np.sum(combined_rewards)))
            ep_reward_da.append(float(np.sum(da_rewards)))
            ep_reward_advantage.append(float(np.sum(advantage_reward_per_step)))

            if self.train_log_path is not None:
                self._write_train_log_period(
                    row_start=row_start,
                    num_rows=num_rows,
                    period_timestamps=period_timestamps,
                    period_volumes=period_volumes,
                    period_clearing_prices=period_clearing_prices,
                    log_episode_id_buffer=log_episode_id_buffer,
                    log_dam_price_forecast_buffer=log_dam_price_forecast_buffer,
                    log_capacity_bid_buffer=log_capacity_bid_buffer,
                    log_delta_soc_buffer=log_delta_soc_buffer,
                    log_remaining_cycles_buffer=log_remaining_cycles_buffer,
                    log_profit_buffer=log_profit_buffer,
                    log_soc_buffer=log_soc_buffer,
                    rollout_buffer=rollout_buffer,
                    da_rewards=da_rewards,
                    combined_rewards=combined_rewards,
                    da_profit=da_profit,
                    ri_stacked_profit=ri_stacked_profit,
                    baseline_profit=baseline_profit,
                    advantage_reward_per_step=advantage_reward_per_step,
                )

        if ep_profit_combined:
            self.logger.record(
                "episode_profit/combined_eur", float(np.mean(ep_profit_combined))
            )
            self.logger.record(
                "episode_profit/day_ahead_eur", float(np.mean(ep_profit_da))
            )
            self.logger.record(
                "episode_profit/intraday_eur", float(np.mean(ep_profit_idc))
            )
            self.logger.record(
                "episode_reward/combined_sum", float(np.mean(ep_reward_combined))
            )
            self.logger.record(
                "episode_reward/day_ahead_sum", float(np.mean(ep_reward_da))
            )
            # Sum of per-step intrinsic / CF term (NOT PPO GAE advantage). CSV column "advantage_reward".
            mean_intrinsic = float(np.mean(ep_reward_advantage))
            self.logger.record(
                "episode_reward/intrinsic_step_sum_mean", mean_intrinsic
            )
            self.logger.record(
                "episode_reward/advantage_sum",
                mean_intrinsic,
            )
            self.logger.record(
                "episode_reward/n_episodes_in_rollout", len(ep_profit_combined)
            )

        # Counterfactual-specific TensorBoard metrics (strict CF path only)
        if cf_episodes_strict > 0 and ep_cf_mean_margins:
            self.logger.record(
                "counterfactual/mean_margin_within_episode",
                float(np.mean(ep_cf_mean_margins)),
            )
            self.logger.record(
                "counterfactual/sum_margins_per_episode_mean",
                float(np.mean(ep_cf_sum_margins_per_episode)),
            )
            self.logger.record(
                "counterfactual/active_hours_per_episode_mean",
                float(np.mean(ep_cf_n_active_hours)),
            )
            self.logger.record(
                "counterfactual/episodes_with_strict_cf", float(cf_episodes_strict)
            )
        if cf_episodes_uniform_fallback > 0:
            self.logger.record(
                "counterfactual/episodes_uniform_ri_fallback",
                float(cf_episodes_uniform_fallback),
            )
        # Backwards-compatible alias
        if ep_cf_mean_margins:
            self.logger.record(
                "episode_reward/cf_mean_margin",
                float(np.mean(ep_cf_mean_margins)),
            )

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    @staticmethod
    def run_simulations_quarterhourly_products_in_parallel(
        period_timestamps, da_trades
    ):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_stacked = executor.submit(
                #simulate_period_quarterhourly_products,
                simulate_days_stacked_quarterhourly_products,
                start_day=period_timestamps[0],
                end_day=period_timestamps[0] + pd.Timedelta(days=1),
                #threshold=0,
                # threshold_abs_min=0,
                discount_rate=0,
                #bucket_size=BUCKET_SIZE,
                c_rate=C_RATE,
                roundtrip_eff=RTE,
                max_cycles=MAX_CYCLES_PER_DAY,
                min_trades=MIN_TRADES,
                #day_ahead_trades_drl=da_trades,
                drl_output=da_trades
            )


            rolling_intrinsic_results_stacked = future_stacked.result()


        return rolling_intrinsic_results_stacked

    @staticmethod
    def run_simulations_hourly_products_in_parallel(period_timestamps, da_trades):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_stacked = executor.submit(
                simulate_period_hourly_products,
                start_day=period_timestamps[0],
                end_day=period_timestamps[0] + pd.Timedelta(days=1),
                threshold=0,
                threshold_abs_min=0,
                discount_rate=0,
                bucket_size=BUCKET_SIZE,
                c_rate=C_RATE,
                roundtrip_eff=RTE,
                max_cycles=MAX_CYCLES_PER_DAY,
                min_trades=MIN_TRADES,
                day_ahead_trades_drl=da_trades,
            )

            future_non_stacked = executor.submit(
                simulate_period_hourly_products,
                start_day=period_timestamps[0],
                end_day=period_timestamps[0] + pd.Timedelta(days=1),
                threshold=0,
                threshold_abs_min=0,
                discount_rate=0,
                bucket_size=BUCKET_SIZE,
                c_rate=C_RATE,
                roundtrip_eff=RTE,
                max_cycles=MAX_CYCLES_PER_DAY,
                min_trades=MIN_TRADES,
            )

            rolling_intrinsic_results_stacked = future_stacked.result()
            rolling_intrinsic_results_non_stacked = future_non_stacked.result()

        return rolling_intrinsic_results_stacked, rolling_intrinsic_results_non_stacked

    def _write_train_log_period(
        self,
        row_start: int,
        num_rows: int,
        period_timestamps: pd.DatetimeIndex,
        period_volumes: np.ndarray,
        period_clearing_prices: np.ndarray,
        log_episode_id_buffer: np.ndarray,
        log_dam_price_forecast_buffer: np.ndarray,
        log_capacity_bid_buffer: np.ndarray,
        log_delta_soc_buffer: np.ndarray,
        log_remaining_cycles_buffer: np.ndarray,
        log_profit_buffer: np.ndarray,
        log_soc_buffer: np.ndarray,
        rollout_buffer: RolloutBuffer,
        da_rewards: np.ndarray,
        combined_rewards: np.ndarray,
        da_profit: float,
        ri_stacked_profit: float,
        baseline_profit: float,
        advantage_reward_per_step: np.ndarray,
    ) -> None:
        """Append one row per step of this period to the single train log CSV."""
        actions = rollout_buffer.actions[row_start : row_start + num_rows]
        if actions.ndim > 1:
            actions = actions.flatten()
        ep_ids = log_episode_id_buffer[row_start : row_start + num_rows]
        dam_pf = log_dam_price_forecast_buffer[row_start : row_start + num_rows]
        cap_bid = log_capacity_bid_buffer[row_start : row_start + num_rows]
        delta_soc = log_delta_soc_buffer[row_start : row_start + num_rows]
        rem_cycles = log_remaining_cycles_buffer[row_start : row_start + num_rows]
        profit_step = log_profit_buffer[row_start : row_start + num_rows]
        soc = log_soc_buffer[row_start : row_start + num_rows]
        times = period_timestamps
        if hasattr(times, "tolist"):
            time_strs = [pd.Timestamp(t).isoformat() for t in times]
        else:
            time_strs = [pd.Timestamp(times[i]).isoformat() for i in range(num_rows)]
        log_df = pd.DataFrame(
            {
                "time": time_strs,
                "episode_id": ep_ids,
                "timestep": np.arange(num_rows),
                "dam_price_forecast": dam_pf,
                "epex_spot_60min_de_lu_eur_per_mwh": period_clearing_prices,
                "action_1": actions,
                "reward": combined_rewards,
                "da_reward": da_rewards,
                "price_bid": np.nan,
                "capacity_bid": cap_bid,
                "capacity_trade": period_volumes,
                "obs: soc_t": soc,
                "delta_soc": delta_soc,
                "remaining_cycles": rem_cycles,
                "profit": profit_step,
                "da_profit_episode": da_profit,
                "idc_profit_episode": ri_stacked_profit,
                "baseline_ri": baseline_profit,
                "advantage_reward": advantage_reward_per_step,
            }
        )
        write_header = not os.path.isfile(self.train_log_path)
        log_df.to_csv(
            self.train_log_path,
            mode="a",
            header=write_header,
            index=False,
        )

    @staticmethod
    def _derive_period_lenghts_from_episode_starts_array(episode_starts: np.ndarray):
        # Find the indices where episodes start
        episode_indices = np.where(episode_starts == 1)[0]
        # Calculate episode lengths by finding the difference between consecutive start indices
        episode_lengths = np.diff(episode_indices)
        # Add the length of the last episode (from the last start to the end of the array)
        final_length = len(episode_starts)  - episode_indices[-1]
        episode_lengths = np.append(episode_lengths, final_length)
        episode_dict = {
            index: length
            for index, length in zip(episode_indices, episode_lengths)
                if length > 0
            #if length == 24
        }
        return episode_dict

    @staticmethod
    def _check_if_complete_cycle(period_volumes, capacity: float = 1.0,
                             min_cycle_fraction: float = 0.0):
        traded_volume = abs(period_volumes).sum()
        cycle_fraction = traded_volume / (2 * capacity)
        return cycle_fraction >= min_cycle_fraction

    


    @staticmethod
    def _derive_day_ahead_trades(
        timestamps, volumes, clearing_prices, intraday_product_type: str
    ):
        day_ahead_trades = {}
        for idx in range(len(timestamps)):
            if volumes[idx] == 0:
                continue

            side = "buy" if volumes[idx] < 0 else "sell"
            net_volume = abs(volumes[idx])
            price = clearing_prices[idx]
            profit = volumes[idx] * clearing_prices[idx]

            day_ahead_market_clearing = (timestamps[0] - pd.Timedelta(days=1)).replace(
                hour=13
            )

            if intraday_product_type == "H":
                day_ahead_trades.update(
                    {
                        timestamps[idx]: {
                            "execution_time": day_ahead_market_clearing,
                            "side": side,
                            "quantity": net_volume,
                            "price": price,
                            "product": timestamps[idx],
                            "profit": profit,
                        }
                    }
                )
            elif intraday_product_type == "QH":
                product_indexes = pd.date_range(
                    timestamps[idx], periods=4, freq="15min"
                )
                for product_index in product_indexes:
                    day_ahead_trades.update(
                        {
                            product_index: {
                                "execution_time": day_ahead_market_clearing,
                                "side": side,
                                "quantity": net_volume,
                                "price": price,
                                #"product": timestamps[idx],
                                "product": product_index,
                                "profit": profit / 4,
                            }
                        }
                    )
            else:
                raise ValueError(
                    "Wrong intraday product type %s. Only QH or H allowed."
                    % intraday_product_type
                )
        
        # If no trading activity, return empty DataFrame
        if not day_ahead_trades:
            return pd.DataFrame(
                columns=["execution_time", "side", "quantity", "price", "product", "profit"]
            )

        return pd.DataFrame(day_ahead_trades).T.reset_index(drop=True)