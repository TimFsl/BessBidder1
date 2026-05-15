from __future__ import annotations

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


from src.shared.config import (
    MIN_CYCLE_FRACTION_FOR_IDC_REWARD,
    PRECOMPUTED_DA_RI_SUMMARY_DIR,
    START_IDC_STEPS,
)

from .precomputed_summary_lookup import (
    PrecomputedSummaryLookup,
    realized_volumes_to_schedule,
)

if TYPE_CHECKING:
    from .basic_battery_dam_env import BasicBatteryDAM

# Matches v1: total RI reward over episode = ri_eur / RI_REWARD_SCALE
RI_REWARD_SCALE = 10.0
# With Discrete(3): 0 = idle, 1 = full buy, 2 = full sell
DEFAULT_IDLE_DA_ACTION = 0


# Curriculum: (step_threshold, max_cycles). Steps < threshold get that max_cycles.
CURRICULUM_MAX_CYCLES = [
  #  (300_000, 3.0),
   #  (600_000, 2.0),
  #  (1_000_000, 1.0),
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
        precomputed_summary_dir: Optional[str] = None,
        lookup_debug_log_path: str | None = None,
        da_market_reward_weight: float = 1.0,
        cf_reward_weight: float = 1.0,
        cf_only_after_steps: Optional[int] = None,
        buy_cf_attribution_mode: str = "contextual",
        reward_normalization_mode: str = "none",
        reward_normalization_min_scale: float = 1.0,
        reward_normalization_scale_multiplier: float = 1.0,
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
        _dir = precomputed_summary_dir or PRECOMPUTED_DA_RI_SUMMARY_DIR
        self._precomputed_lookup = PrecomputedSummaryLookup(_dir)
        self.lookup_debug_log_path = lookup_debug_log_path
        self.da_market_reward_weight = float(da_market_reward_weight)
        self.cf_reward_weight = float(cf_reward_weight)
        self.cf_only_after_steps = cf_only_after_steps
        self.buy_cf_attribution_mode = str(buy_cf_attribution_mode)
        self.reward_normalization_mode = str(reward_normalization_mode)
        self.reward_normalization_min_scale = float(reward_normalization_min_scale)
        self.reward_normalization_scale_multiplier = float(
            reward_normalization_scale_multiplier
        )

    def _daily_reward_scale(
        self, *, dam_price_forecast: np.ndarray
    ) -> float:
        """
        Return a positive scaling factor for this episode's rewards.
        Uses only forecast prices (observable ex-ante) to avoid leakage.
        """
        mode = self.reward_normalization_mode
        if mode in ("none", "", None):
            return 1.0
        if mode == "daily_iqr_forecast":
            x = np.asarray(dam_price_forecast, dtype=np.float64).ravel()
            if x.size == 0:
                return 1.0
            q75 = float(np.nanquantile(x, 0.75))
            q25 = float(np.nanquantile(x, 0.25))
            s = q75 - q25
            if not np.isfinite(s):
                s = 1.0
            s = float(max(s, self.reward_normalization_min_scale))
            m = float(self.reward_normalization_scale_multiplier)
            if not np.isfinite(m) or m <= 0:
                raise ValueError(
                    "reward_normalization_scale_multiplier must be finite and > 0, "
                    f"got {m!r}"
                )
            return float(s * m)
        raise ValueError(
            "reward_normalization_mode must be 'none' or 'daily_iqr_forecast', got "
            f"{mode!r}"
        )

    @staticmethod
    def _unwrap_to_basic_battery_dam(env: VecEnv):
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
        """Physics replay for CF keys (same as v2)."""
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

    @staticmethod
    def _replay_episode_with_actions(
        template_env: BasicBatteryDAM,
        day: str,
        actions: np.ndarray,
        horizon: int,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Physics replay for an explicit action sequence."""
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
        a_in = np.asarray(actions).flatten().astype(np.int64, copy=False)
        for t in range(horizon):
            a = int(a_in[t])
            _obs, r, _term, _trunc, info = replay.step(a)
            total_da_reward += float(r)
            volumes.append(float(info["position"]))
            clearings.append(float(info["clearing_price"]))
        return (
            np.asarray(volumes, dtype=np.float64),
            np.asarray(clearings, dtype=np.float64),
            total_da_reward,
        )

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
        log_da_reward_market_buffer = np.zeros(n_rollout_steps, dtype=np.float64)
        log_invalid_penalty_buffer = np.zeros(n_rollout_steps, dtype=np.float64)

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
                    log_da_reward_market_buffer[idx] = infos[0]["log_da_reward_market"]
                    log_invalid_penalty_buffer[idx] = -float(
                        infos[0]["log_invalid_penalty"]
                    )

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
        # Counterfactual diagnostics (minimal)
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
            da_market_rewards = log_da_reward_market_buffer[
                row_start : row_start + num_rows
            ].astype(np.float64, copy=False)
            invalid_penalties = log_invalid_penalty_buffer[
                row_start : row_start + num_rows
            ].astype(np.float64, copy=False)
            dam_price_forecast = log_dam_price_forecast_buffer[
                row_start : row_start + num_rows
            ].astype(np.float64, copy=False)
            reward_scale = self._daily_reward_scale(dam_price_forecast=dam_price_forecast)
            da_trades = self._derive_day_ahead_trades(
                timestamps=period_timestamps,
                volumes=period_volumes,
                clearing_prices=period_clearing_prices,
                intraday_product_type=self.intraday_product_type,
            )
            da_profit = float(da_trades.profit.sum()) if len(da_trades) > 0 else 0.0

            ri_stacked_profit = 0.0
            combined_rewards = da_rewards.copy()
            rolling_intrinsic_rewards = np.zeros(num_rows, dtype=np.float64)

            run_ri = (
                self.num_timesteps >= START_IDC_STEPS
                and self._check_if_complete_cycle(
                    period_volumes,
                    min_cycle_fraction=MIN_CYCLE_FRACTION_FOR_IDC_REWARD,
                )
            )
            if run_ri:
                if self.intraday_product_type != "QH":
                    raise NotImplementedError(
                        "v13_baseline_counterfactual_reward only supports "
                        "intraday_product_type='QH' (precomputed summary tables)."
                    )
                da_rewards = rollout_buffer.rewards[
                    row_start : row_start + num_rows
                ].flatten()
                actions_period = rollout_buffer.actions[
                    row_start : row_start + num_rows
                ]
                if actions_period.ndim > 1:
                    actions_period = actions_period.flatten()

                day_str = str(period_timestamps[0].date().isoformat())
                # Use realized (post-clipping) DA volumes as schedule key source.
                # This ensures lookup is aligned with the actual schedule forwarded
                # to intraday optimization ("capacity_trade"), not raw actions.
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
                        kind=k_realized,
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

                # Total episode return in reward units (combined € / scale)
                R_full_total = profit_full_eur / RI_REWARD_SCALE

                use_cf = self.use_counterfactual_ri_reward
                if use_cf:
                    base_env = self._unwrap_to_basic_battery_dam(env)
                    rolling_intrinsic_rewards = np.zeros(num_rows, dtype=np.float64)
                    effective_da_market_weight = self.da_market_reward_weight
                    if (
                        self.cf_only_after_steps is not None
                        and self.num_timesteps >= int(self.cf_only_after_steps)
                    ):
                        effective_da_market_weight = 0.0
                    base_rewards = (
                        effective_da_market_weight * (da_market_rewards / reward_scale)
                    ) + invalid_penalties
                    combined_rewards = base_rewards.astype(np.float64).copy()

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
                        cf_episodes_uniform_fallback += 1
                        # No selected CF steps => no CF shaping on this episode.
                        # Keep intrinsic term at zero (no uniform fallback).
                        rolling_intrinsic_rewards = np.zeros(num_rows, dtype=np.float64)
                        combined_rewards = base_rewards.astype(np.float64).copy()
                    else:
                        for t in active:
                            ti = int(t)
                            a_t = int(np.asarray(actions_period[ti]).item())

                            if (
                                self.buy_cf_attribution_mode == "buy_only_isolated"
                                and a_t == 1
                            ):
                                actions_ref = (
                                    np.asarray(actions_period)
                                    .flatten()
                                    .astype(np.int64, copy=True)
                                )
                                # Isolate buy-leg attribution by neutralizing sell legs.
                                actions_ref[actions_ref == 2] = self.counterfactual_idle_action
                                vol_ref, _clr_ref, _da_sum_ref = (
                                    self._replay_episode_with_actions(
                                        base_env,
                                        day_str,
                                        actions_ref,
                                        num_rows,
                                    )
                                )
                                k_ref, bh_ref, sh_ref = realized_volumes_to_schedule(vol_ref)
                                profit_ref_eur = self._precomputed_lookup.profit_eur_for_key(
                                    day_str, k_ref, bh_ref, sh_ref, warn=True
                                )

                                actions_cf = actions_ref.copy()
                                actions_cf[ti] = self.counterfactual_idle_action
                                vol_cf, _clr_cf, _da_sum_cf = (
                                    self._replay_episode_with_actions(
                                        base_env,
                                        day_str,
                                        actions_cf,
                                        num_rows,
                                    )
                                )
                                k_cf, bh_cf, sh_cf = realized_volumes_to_schedule(vol_cf)
                                row_cf = self._precomputed_lookup.row_for_schedule(
                                    day_str, k_cf, bh_cf, sh_cf
                                )
                                if row_cf is None or k_cf == "invalid":
                                    self._append_lookup_debug_row(
                                        source="cf_replay",
                                        day_str=day_str,
                                        kind=k_cf,
                                        buy_hour=bh_cf,
                                        sell_hour=sh_cf,
                                        replace_at=ti,
                                        actions=actions_cf,
                                        volumes=vol_cf,
                                    )
                                profit_cf_eur = self._precomputed_lookup.profit_eur_for_key(
                                    day_str, k_cf, bh_cf, sh_cf, warn=True
                                )
                                margin_t = (
                                    (profit_ref_eur - profit_cf_eur) / RI_REWARD_SCALE
                                ) / reward_scale
                            else:
                                vol_cf, _clr_cf, _da_sum_cf = (
                                    self._replay_episode_replacing_action_at(
                                        base_env,
                                        day_str,
                                        actions_period,
                                        num_rows,
                                        replace_with_idle_at=ti,
                                        idle_action=self.counterfactual_idle_action,
                                    )
                                )
                                k_cf, bh_cf, sh_cf = realized_volumes_to_schedule(vol_cf)
                                row_cf = self._precomputed_lookup.row_for_schedule(
                                    day_str, k_cf, bh_cf, sh_cf
                                )
                                if row_cf is None or k_cf == "invalid":
                                    self._append_lookup_debug_row(
                                        source="cf_replay",
                                        day_str=day_str,
                                        kind=k_cf,
                                        buy_hour=bh_cf,
                                        sell_hour=sh_cf,
                                        replace_at=ti,
                                        actions=actions_period,
                                        volumes=vol_cf,
                                    )
                                profit_cf_eur = self._precomputed_lookup.profit_eur_for_key(
                                    day_str, k_cf, bh_cf, sh_cf, warn=True
                                )
                                R_cf_total = profit_cf_eur / RI_REWARD_SCALE
                                margin_t = (R_full_total - R_cf_total) / reward_scale

                            margins.append(margin_t)
                            rolling_intrinsic_rewards[ti] = margin_t
                            combined_rewards[ti] = (
                                float(base_rewards[ti])
                                + self.cf_reward_weight * margin_t
                            )

                    rollout_buffer.rewards[row_start : row_start + num_rows] = (
                        combined_rewards.reshape(-1, 1)
                    )
                else:
                    ri_reward_per_step = ri_stacked_profit / (
                        RI_REWARD_SCALE * num_rows
                    )
                    rolling_intrinsic_rewards = np.full(
                        num_rows, ri_reward_per_step, dtype=np.float64
                    )
                    combined_rewards = da_rewards + (
                        self.cf_reward_weight * rolling_intrinsic_rewards
                    )
                    rollout_buffer.rewards[row_start : row_start + num_rows] = (
                        combined_rewards.reshape(-1, 1)
                    )

            cf_reward_per_step = rolling_intrinsic_rewards

            ep_profit_combined.append(float(da_profit + ri_stacked_profit))
            ep_profit_da.append(float(da_profit))
            ep_profit_idc.append(float(ri_stacked_profit))
            ep_reward_combined.append(float(np.sum(combined_rewards)))
            ep_reward_da.append(float(np.sum(da_rewards)))
            ep_reward_advantage.append(float(np.sum(cf_reward_per_step)))

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
                    cf_reward_per_step=cf_reward_per_step,
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
            # Mean over episodes of (sum of per-step CF reward in each episode)
            mean_cf_reward_sum = float(np.mean(ep_reward_advantage))
            self.logger.record("episode_reward/cf_reward_sum_mean", mean_cf_reward_sum)
            self.logger.record(
                "episode_reward/n_episodes_in_rollout", len(ep_profit_combined)
            )

        # Track fallback usage only (no redundant opposite counter)
        self.logger.record(
            "counterfactual/episodes_uniform_ri_fallback",
            float(cf_episodes_uniform_fallback),
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
        cf_reward_per_step: np.ndarray,
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
                "cf_reward": cf_reward_per_step,
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