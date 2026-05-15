import os
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.core import ActType, ObsType


SEED = 42
np.random.seed(SEED)

PERIOD_LENGTH = 24
# EPEX_DA_MIN_PRICE = np.float32(-500.0)
# EPEX_DA_MAX_PRICE = np.float32(4000.0)
PENALTY_SOC_LIMITS = 0.5


class BasicBatteryDAM(gym.Env):
    def __init__(
        self,
        modus: str,
        logging_path: str,
        input_data: dict[str, dict[str, np.array]],
        power: np.float32 = 1.0,
        capacity: np.float32 = 1.0,
        round_trip_efficiency: np.float32 = 1.0,
        start_end_soc: np.float32 = 0.0,
        max_cycles: float = 1.0,
    ):
        self._episode_id = -1
        self._step_in_episode = 0
        self._global_step = 0

        self._modus = modus
        self._logging_path = logging_path
        self._input_data = input_data
        self._days_left = np.array(list(self._input_data.keys()), dtype=str)

        self._reinitialize_input_data_after_reset()

        self._power = power
        self._capacity = capacity
        self._efficiency = round_trip_efficiency**0.5
        # TODO: implement start end restriction for the storage
        self._start_end_soc = start_end_soc
        self._current_soc = self._start_end_soc

        self.max_cycles = float(max_cycles)
        self._remaining_cycles = self.max_cycles


       #self._remaining_cycles = 1
        # 0 = idle, 1 = full buy/charge (-1), 2 = full sell/discharge (+1) — same as v4
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-1, 1, shape=(50,), dtype=np.float32)
        self._current_time_step = 0
        self._realized_quantity_t_minus_1 = 0
        self._total_profit = 0.0

    def _get_obs(self):
        # Calculate sine and cosine of the current time step
        sin_time_step = np.sin(2 * np.pi * self._current_time_step / 24)
        cos_time_step = np.cos(2 * np.pi * self._current_time_step / 24)

        # Calculate sine and cosine of day of the week
        sin_day_of_week = np.sin(2 * np.pi * self._day_of_week / 7)
        cos_day_of_week = np.cos(2 * np.pi * self._day_of_week / 7)

        # Calculate sine and cosine of month
        sin_month = np.sin(2 * np.pi * self._date_month / 12)
        cos_month = np.cos(2 * np.pi * self._date_month / 12)

        obs = np.concatenate(
            (
                self._realized_quantity_t_minus_1,
                self._current_soc,
                self._remaining_cycles,
                # encoded time
                sin_time_step,
                cos_time_step,
                # hourly forecasts
                self._residual_load_forecast_scaled[self._current_time_step],
                self._forecasted_price_vector_scaled,
                # include avergae change per hour in forecasts
                self._delta_load_forecast[self._current_time_step],
                self._delta_pv_forecast_scaled[self._current_time_step],
                self._delta_wind_onshore_forecast_scaled[self._current_time_step],
                # encoded month and day of week
                sin_month,
                cos_month,
                sin_day_of_week,
                cos_day_of_week,
                # get daily RE statistics
                self._wind_forecast_daily_mean,
                self._wind_forecast_daily_std,
                self._spread_id_full_da_mean,
                self._spread_id_full_da_std,
                self._spread_id_full_da_min,
                self._spread_id_full_da_max,
                self._exaa_pf_daily_mean,
                self._exaa_pf_daily_std,
                self._exaa_pf_daily_min,
                self._exaa_pf_daily_max,
                self._exaa_pf_daily_spread,
                self._exaa_pf_daily_diff_sum,
                self._exaa_pf_daily_diff_max,
            ),
            axis=None,
            dtype=np.float32,
        )
        # Some components (e.g. residual load, deltas) are derived via subtraction/diffs
        # and can exceed [-1, 1] depending on the scaler fit range. The observation_space
        # is defined as Box(-1, 1), so we clip defensively to keep SB3 env checks stable.
        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        day: str | None = None
        if options is not None:
            day = options.get("day")
        self._reinitialize_input_data_after_reset(day=day)
        self._current_time_step = 0
        self._realized_quantity_t_minus_1 = 0
        self._current_soc = self._start_end_soc
        self._remaining_cycles = float(self.max_cycles)
        self._total_profit = 0.0
        
        self._episode_id += 1
        self._step_in_episode = 0

        observation = self._get_obs()
        return observation, {}  # empty info dict
    


    
    def set_max_cycles(self, value: float) -> None:
        """
        Setzt die maximal erlaubten Vollzyklen pro Tag (für Curriculum Learning).
        Greift ab dem nächsten reset().
        """
        self.max_cycles = float(value)


    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        # IDs für diesen Step einfrieren
        current_episode_id = self._episode_id
        current_step_in_episode = self._step_in_episode
        timestep_in_day = self._current_time_step  # 0..23

        # Map diskrete Action auf kontinuierliche Menge in [-1, 1]
        action_continuous = float(self._map_discrete_action_to_continuous(action))
        desired_quantity = float(np.clip(action_continuous, -1.0, 1.0))  # Laden (<0) / Entladen (>0)

        clearing_price = float(self._realized_price_vector[timestep_in_day])

        soc_before = float(self._current_soc)
        remaining_cycles_before = float(self._remaining_cycles)

        
        # Physically feasible quantity to charge/discharge
        overflow = 0.0

        if desired_quantity > 0.0:
            # Entladen
            max_by_soc = float(self._current_soc)
            max_by_cycles = float(2.0 * self._remaining_cycles)
            feasible_max = max(0.0, min(max_by_soc, max_by_cycles))

            realized_quantity = min(desired_quantity, feasible_max)
            overflow = max(0.0, desired_quantity - feasible_max)

        elif desired_quantity < 0.0:
            # Laden # Charging
            max_charge_by_soc = float(self._capacity - self._current_soc)
            max_charge_by_cycles = float(2.0 * self._remaining_cycles)
            feasible_charge = max(0.0, min(max_charge_by_soc, max_charge_by_cycles))

            desired_charge = -desired_quantity  # > 0
            realized_charge = min(desired_charge, feasible_charge)
            realized_quantity = -realized_charge

            overflow = max(0.0, desired_charge - feasible_charge)
        else:
            realized_quantity = 0.0
            overflow = 0.0

        delta_soc = -realized_quantity
        self._current_soc = float(self._current_soc + delta_soc)

        delta_cycles = abs(delta_soc) / (2.0 * self._capacity)
        self._remaining_cycles = float(self._remaining_cycles - delta_cycles)
        self._realized_quantity_t_minus_1 = float(realized_quantity)

        if realized_quantity < 0.0:
            # Laden
            profit = -clearing_price * (-realized_quantity) / self._efficiency
            reward = -clearing_price * (-realized_quantity) / (10 * float(self._capacity))
        elif realized_quantity > 0.0:
            # Entladen
            profit = clearing_price * (realized_quantity * self._efficiency)
            reward = clearing_price * (realized_quantity) / (10 *float(self._capacity))
        else:
            profit = 0.0
            reward = 0.0

        self._total_profit += profit

        """
        if realized_quantity < 0.0:
            energy_into_batt = -realized_quantity
            energy_from_grid = energy_into_batt / self._efficiency
            profit = -clearing_price * energy_from_grid
        
        elif realized_quantity > 0.0:
            energy_from_batt = realized_quantity
            energy_to_grid = energy_from_batt * self._efficiency
            profit = clearing_price * energy_to_grid
        else:
            profit = 0.0

        self._total_profit += profit

        reward = profit / (100.0 * float(self._capacity))

        """

        invalid_penalty_coef = 0.1
        invalid_penalty = 0.0
        if overflow > 1e-6:
            invalid_penalty = invalid_penalty_coef * overflow
            reward -= invalid_penalty

        game_over = round(self._remaining_cycles, 4) <= 0.0
        terminated = bool(timestep_in_day == PERIOD_LENGTH - 1 ) #or game_over)

        """
        if terminated and self._current_soc > 1e-3:
            soc_penalty_coef = 0.05
            reward -= soc_penalty_coef * float(self._current_soc)
        """

        timestamp = self._timestamps[timestep_in_day]

        self._log_episode_id = current_episode_id
        self._log_dam_price_forecast = float(
            self._forecasted_price_vector[timestep_in_day]
        )
        self._log_capacity_bid = desired_quantity
        self._log_delta_soc = delta_soc
        self._log_remaining_cycles = float(self._remaining_cycles)
        self._log_profit = profit
        self._log_soc = float(self._current_soc)
        self._log_da_reward_market = float(reward + invalid_penalty)
        self._log_invalid_penalty = float(invalid_penalty)
        self._log_da_reward_total = float(reward)

        info = self._get_info()
        observation = self._get_obs()
        reward = float(reward)

        self._step_in_episode += 1
        self._global_step += 1
        self._current_time_step += 1

        return observation, reward, terminated, False, info



    def _get_info(self):
        info = {
            "timestamp": self._timestamps[self._current_time_step].astype("int64"),
            "position": self._realized_quantity_t_minus_1,
            "clearing_price": self._realized_price_vector[self._current_time_step],
            "scaling_max_price": self._max_price_realized,
            "scaling_min_price": self._min_price_realized,
        }
        if hasattr(self, "_log_episode_id"):
            info["log_episode_id"] = self._log_episode_id
            info["log_dam_price_forecast"] = self._log_dam_price_forecast
            info["log_capacity_bid"] = self._log_capacity_bid
            info["log_delta_soc"] = self._log_delta_soc
            info["log_remaining_cycles"] = self._log_remaining_cycles
            info["log_profit"] = self._log_profit
            info["log_soc"] = self._log_soc
            info["log_da_reward_market"] = self._log_da_reward_market
            info["log_invalid_penalty"] = self._log_invalid_penalty
            info["log_da_reward_total"] = self._log_da_reward_total
        return info

    def close(self):
        pass

    def _map_discrete_action_to_continuous(self, action: int) -> float:
        """Map discrete action to desired normalized power in [-1, 1] (v4-style)."""
        a = int(action)
        if a == 0:
            return 0.0
        if a == 1:
            return -1.0
        if a == 2:
            return 1.0
        raise ValueError(f"Invalid action {action}; expected 0, 1, or 2.")

    @staticmethod
    def calculate_soc_penalty(current_soc: float) -> float:
        """
        Calculate penalty based on deviation from SOC limits (quadratic penalty).

        Parameters:
            current_soc (float): Current state of charge (SOC) of the battery.

        Returns:
            float: Penalty value.
        """
        if current_soc < 0.0:
            penalty = (-1) * current_soc
        elif current_soc > 1.0:
            penalty = current_soc - 1
        else:
            penalty = 0.0

        return penalty

    @staticmethod
    def _map_action_to_obs_space(
        action_normalized: np.float32, min_value: np.float32, max_value: np.float32
    ) -> np.float32:
        """
        Maps the agent's standardized action to a value within the specified range.

        Parameters:
            action_normalized (float): The standardized action taken by the agent, ranging from -1 to 1.
            min_value (float): The minimum value in the desired range.
            max_value (float): The maximum value in the desired range.

        Returns:
            float: The corresponding observation within the specified range.
        """
        return (action_normalized + 1) * (max_value - min_value) / 2 + min_value

    def _sample_random_day(self) -> str:
        return str(np.random.choice(self._days_left, 1, replace=False)[0])



    def _reinitialize_input_data_after_reset(self, day: str | None = None) -> None:
        """
        Load one delivery day. If ``day`` is given (YYYY-MM-DD), use that key from
        ``_input_data`` (for deterministic replay / counterfactuals). Otherwise
        sample a random day as before.
        """
        if day is None:
            self._random_day = self._sample_random_day()
        else:
            day_key = str(day)
            if day_key not in self._input_data:
                raise ValueError(
                    f"reset(day=...): day {day_key!r} not found in input_data keys."
                )
            self._random_day = day_key

        self._forecasted_price_vector = self._input_data[self._random_day]["price_forecast"]
        self._realized_price_vector = self._input_data[self._random_day]["price_realized"]

        self._min_price_realized = float(np.min(self._realized_price_vector))
        self._max_price_realized = float(np.max(self._realized_price_vector))
        self._min_price_forecasted = float(np.min(self._forecasted_price_vector))
        self._max_price_forecasted = float(np.max(self._forecasted_price_vector))

        # Daily scaling for forecast prices: 0..1 then map to -1..1 (Box(-1,1))
        daily_price_spread = self._max_price_forecasted - self._min_price_forecasted
        daily_price_spread = daily_price_spread if abs(daily_price_spread) > 1e-6 else 1.0
        pf01 = (self._forecasted_price_vector - self._min_price_forecasted) / daily_price_spread
        self._forecasted_price_vector_scaled = (pf01 * 2.0 - 1.0).astype(np.float32)

        # Scalar features (already globally scaled via MinMaxScaler in prepare_input_data)
        self._date_month = float(self._input_data[self._random_day]["date_month"][0])
        self._day_of_week = float(self._input_data[self._random_day]["day_of_week"][0])

        self._wind_forecast_daily_mean = float(self._input_data[self._random_day]["wind_forecast_daily_mean"][0])
        self._wind_forecast_daily_std = float(self._input_data[self._random_day]["wind_forecast_daily_std"][0])

        self._spread_id_full_da_mean = float(self._input_data[self._random_day]["spread_id_full_da_mean"][0])
        self._spread_id_full_da_std = float(self._input_data[self._random_day]["spread_id_full_da_std"][0])
        self._spread_id_full_da_min = float(self._input_data[self._random_day]["spread_id_full_da_min"][0])
        self._spread_id_full_da_max = float(self._input_data[self._random_day]["spread_id_full_da_max"][0])

        self._exaa_pf_daily_mean = float(self._input_data[self._random_day]["exaa_pf_daily_mean"][0])
        self._exaa_pf_daily_std = float(self._input_data[self._random_day]["exaa_pf_daily_std"][0])
        self._exaa_pf_daily_min = float(self._input_data[self._random_day]["exaa_pf_daily_min"][0])
        self._exaa_pf_daily_max = float(self._input_data[self._random_day]["exaa_pf_daily_max"][0])
        self._exaa_pf_daily_spread = float(self._input_data[self._random_day]["exaa_pf_daily_spread"][0])
        self._exaa_pf_daily_diff_sum = float(self._input_data[self._random_day]["exaa_pf_daily_diff_sum"][0])
        self._exaa_pf_daily_diff_max = float(self._input_data[self._random_day]["exaa_pf_daily_diff_max"][0])

        # Timestamps
        self._timestamps = self._input_data[self._random_day]["timestamps"]

        # Forecast time series (already globally scaled by scaler)
        self._pv_forecast = self._input_data[self._random_day]["pv_forecast_d_minus_1_1000_de_lu_mw"]
        self._wind_onshore_forecast = self._input_data[self._random_day]["wind_onshore_forecast_d_minus_1_1000_de_lu_mw"]
        self._wind_offshore_forecast = self._input_data[self._random_day]["wind_offshore_forecast_d_minus_1_1000_de_lu_mw"]
        self._load_forecast = self._input_data[self._random_day]["load_forecast_d_minus_1_1000_total_de_lu_mw"]

        # Residual load (still on globally scaled series)
        self._residual_load_forecast = (
            self._load_forecast
            - self._pv_forecast
            - self._wind_onshore_forecast
            - self._wind_offshore_forecast
        )

        # Use same arrays as "scaled" (because they already are)
        self._pv_forecast_scaled = self._pv_forecast
        self._wind_onshore_forecast_scaled = self._wind_onshore_forecast
        self._wind_offshore_forecast_scaled = self._wind_offshore_forecast
        self._load_forecast_scaled = self._load_forecast
        self._residual_load_forecast_scaled = self._residual_load_forecast

        # Gradients
        self._delta_load_forecast = np.append(np.diff(self._load_forecast_scaled), 0)
        self._delta_pv_forecast_scaled = np.append(np.diff(self._pv_forecast_scaled), 0)
        self._delta_wind_onshore_forecast_scaled = np.append(np.diff(self._wind_onshore_forecast_scaled), 0)