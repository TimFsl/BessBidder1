from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.ppo import PPO

from src.shared.config import (
    MIN_CYCLE_FRACTION_FOR_IDC_REWARD,
    PRECOMPUTED_DA_RI_SUMMARY_DIR,
    START_IDC_STEPS,
)

from .precomputed_summary_lookup import (
    PrecomputedSummaryLookup,
    realized_volumes_to_schedule,
)


# Curriculum: (step_threshold, max_cycles). Steps < threshold get that max_cycles.
CURRICULUM_MAX_CYCLES = [
    (300_000, 3.0),
    (600_000, 2.0),
    (1_000_000, 1.0),
    (float("inf"), 1.0),
]


class CustomPPO(PPO):
    def __init__(
        self,
        *args,
        intraday_product_type: str = None,
        reward_log_path: str | None = None,
        train_log_path: str | None = None,
        precomputed_summary_dir: Optional[str] = None,
        lookup_debug_log_path: str | None = None,
        da_reward_weight: float = 0.5,
        idc_reward_weight: float = 0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.intraday_product_type = intraday_product_type
        self.current_step = 0
        self.reward_log_path = reward_log_path
        self.train_log_path = train_log_path
        _dir = precomputed_summary_dir or PRECOMPUTED_DA_RI_SUMMARY_DIR
        self._precomputed_lookup = PrecomputedSummaryLookup(_dir)
        self.lookup_debug_log_path = lookup_debug_log_path
        self.da_reward_weight = float(da_reward_weight)
        self.idc_reward_weight = float(idc_reward_weight)

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
        self._precomputed_lookup.reset_miss_count()

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

        # TensorBoard: did intraday reward actually get written into rollout_buffer.rewards?
        ri_diag_total_episodes = 0
        ri_diag_applied_episodes = 0
        ri_diag_cycle_ok_episodes = 0
        ri_diag_episode_ri_sum_reward_units: list[float] = []

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

            ri_diag_total_episodes += 1
            start_idc_ok = self.num_timesteps >= START_IDC_STEPS
            cycle_ok = self.check_if_complete_cycle(
                period_volumes,
                min_cycle_fraction=MIN_CYCLE_FRACTION_FOR_IDC_REWARD,
            )
            if cycle_ok:
                ri_diag_cycle_ok_episodes += 1
            run_ri = start_idc_ok and cycle_ok
            if run_ri:
                ri_diag_applied_episodes += 1
                if self.intraday_product_type != "QH":
                    raise NotImplementedError(
                        "v11_baseline_extended_observation_space only supports "
                        "intraday_product_type='QH' (precomputed summary tables)."
                    )
                actions_period = rollout_buffer.actions[
                    row_start : row_start + num_rows
                ]
                if actions_period.ndim > 1:
                    actions_period = actions_period.flatten()

                day_str = str(period_timestamps[0].date().isoformat())
                k_realized, bh_realized, sh_realized = realized_volumes_to_schedule(
                    period_volumes
                )
                row_realized = self._precomputed_lookup.row_for_schedule(
                    day_str, k_realized, bh_realized, sh_realized
                )
                if row_realized is None or k_realized == "invalid":
                    self._append_lookup_debug_row(
                        source="full_realized",
                        day_str=day_str,
                        kind=str(k_realized),
                        buy_hour=bh_realized,
                        sell_hour=sh_realized,
                        replace_at=None,
                        actions=actions_period,
                        volumes=period_volumes,
                    )
                profit_full_eur = self._precomputed_lookup.profit_eur_for_key(
                    day_str, k_realized, bh_realized, sh_realized, warn=True
                )
                da_profit_lookup = self._precomputed_lookup.da_profit_eur_for_key(
                    day_str, k_realized, bh_realized, sh_realized, warn=True
                )
                da_profit = float(da_profit_lookup)
                ri_stacked_profit = float(profit_full_eur - da_profit_lookup)

                da_rewards = rollout_buffer.rewards[
                    row_start : row_start + num_rows
                ].flatten()
                ri_reward_per_step = ri_stacked_profit / (10.0 * num_rows)
                rolling_intrinsic_rewards = np.full(
                    num_rows, ri_reward_per_step, dtype=np.float64
                )
                combined_rewards = (
                    da_rewards + rolling_intrinsic_rewards
                )
                rollout_buffer.rewards[row_start : row_start + num_rows] = (
                    combined_rewards.reshape(-1, 1)
                )

            ri_diag_episode_ri_sum_reward_units.append(
                float(np.sum(rolling_intrinsic_rewards))
            )

            ep_profit_combined.append(float(da_profit + ri_stacked_profit))
            ep_profit_da.append(float(da_profit))
            ep_profit_idc.append(float(ri_stacked_profit))
            ep_reward_combined.append(float(np.sum(combined_rewards)))
            ep_reward_da.append(float(np.sum(da_rewards)))

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
            self.logger.record(
                "episode_reward/n_episodes_in_rollout", len(ep_profit_combined)
            )

        if ri_diag_total_episodes > 0:
            te = float(ri_diag_total_episodes)
            self.logger.record(
                "ri_injection/applied_episode_fraction",
                float(ri_diag_applied_episodes) / te,
            )
            self.logger.record(
                "ri_gate/fraction_cycle_ok",
                float(ri_diag_cycle_ok_episodes) / te,
            )
            self.logger.record(
                "ri_gate/start_idc_satisfied",
                1.0 if self.num_timesteps >= START_IDC_STEPS else 0.0,
            )
            if ri_diag_episode_ri_sum_reward_units:
                self.logger.record(
                    "episode_reward/intraday_sum_reward_units_mean",
                    float(np.mean(ri_diag_episode_ri_sum_reward_units)),
                )
                self.logger.record(
                    "episode_reward/intraday_sum_reward_units_max",
                    float(np.max(ri_diag_episode_ri_sum_reward_units)),
                )

        self.logger.record(
            "lookup/miss_count_rollout",
            float(self._precomputed_lookup.lookup_misses),
        )

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def _append_lookup_debug_row(
        self,
        *,
        source: str,
        day_str: str,
        kind: str,
        buy_hour: float | None,
        sell_hour: float | None,
        replace_at: int | None,
        actions: np.ndarray,
        volumes: np.ndarray,
    ) -> None:
        """Append one compact debug row per missing/invalid lookup key."""
        if not self.lookup_debug_log_path:
            return
        a = np.asarray(actions).flatten().astype(np.int64, copy=False)
        v = np.asarray(volumes, dtype=np.float64).ravel()
        neg_hours = (np.where(v < -1e-9)[0] + 1).tolist()
        pos_hours = (np.where(v > 1e-9)[0] + 1).tolist()
        df = pd.DataFrame(
            [
                {
                    "source": source,
                    "day": day_str,
                    "kind": kind,
                    "buy_hour": buy_hour,
                    "sell_hour": sell_hour,
                    "replace_at_step": replace_at,
                    "actions_0_23": "|".join(map(str, a.tolist())),
                    "volumes_0_23": "|".join(f"{x:.6f}" for x in v.tolist()),
                    "neg_hours_1_24": "|".join(map(str, neg_hours)),
                    "pos_hours_1_24": "|".join(map(str, pos_hours)),
                }
            ]
        )
        write_header = not os.path.isfile(self.lookup_debug_log_path)
        df.to_csv(
            self.lookup_debug_log_path,
            mode="a",
            header=write_header,
            index=False,
        )

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
    def check_if_complete_cycle(
        period_volumes,
        capacity: float = 1.0,
        min_cycle_fraction: float = 0.0,
    ) -> bool:
        """Return True if ``sum(|hourly DA volume|) >= min_cycle_fraction * 2 * capacity``."""
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