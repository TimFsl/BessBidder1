"""
Load precomputed DA+RI ``profit`` (EUR) from per-day ``summary_YYYY-MM-DD.csv`` files.

Convention (must match how summaries were generated):
- Env timestep 0..23 → CSV ``buy_hour`` / ``sell_hour`` = timestep + 1 (1..24).
- ``kind == 'no_da'``: no DA trades.
- ``kind == 'buy_only'``: one buy at ``buy_hour``, ``sell_hour`` is NaN.
- ``kind == 'buy_sell'``: buy at ``buy_hour``, sell at ``sell_hour`` (``sell_hour`` can be 24).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

ScheduleKind = Literal["no_da", "buy_only", "buy_sell", "invalid"]


def timestep_to_csv_hour(timestep: int) -> float:
    """Map env step index 0..23 to summary CSV hour column (1..24)."""
    return float(int(timestep) + 1)


def actions_to_schedule(
    actions: np.ndarray,
) -> tuple[ScheduleKind, Optional[float], Optional[float]]:
    """
    Map a length-24 discrete action vector (0=idle, 1=buy, 2=sell) to lookup keys.

    Returns (kind, buy_hour_csv, sell_hour_csv) with hours in 1..24 float, or None.
    """
    a = np.asarray(actions).flatten().astype(np.int64, copy=False)
    buys = np.where(a == 1)[0]
    sells = np.where(a == 2)[0]
    if len(buys) == 0 and len(sells) == 0:
        return "no_da", None, None
    if len(buys) == 0 and len(sells) > 0:
        return "invalid", None, None
    ib = int(buys[0])
    if len(sells) == 0:
        return "buy_only", timestep_to_csv_hour(ib), None
    is_ = int(sells[0])
    if is_ <= ib:
        return "invalid", None, None
    return "buy_sell", timestep_to_csv_hour(ib), timestep_to_csv_hour(is_)


def substitute_idle_at(
    actions: np.ndarray, t: int, idle_action: int = 0
) -> np.ndarray:
    out = np.asarray(actions).flatten().copy()
    out[int(t)] = idle_action
    return out


def realized_volumes_to_schedule(
    volumes: np.ndarray,
) -> tuple[ScheduleKind, Optional[float], Optional[float]]:
    """
    Map realized DA volumes (length 24, sign = buy/sell) to CSV keys.
    Uses first negative volume as buy hour and first positive as sell hour.
    """
    # Note: ndarray.flatten() does not accept dtype= (use asarray first).
    v = np.asarray(volumes, dtype=np.float64).ravel()
    neg = np.where(v < -1e-9)[0]
    pos = np.where(v > 1e-9)[0]
    if len(neg) == 0 and len(pos) == 0:
        return "no_da", None, None
    if len(neg) == 0 and len(pos) > 0:
        return "invalid", None, None
    ib = int(neg[0])
    if len(pos) == 0:
        return "buy_only", timestep_to_csv_hour(ib), None
    is_ = int(pos[0])
    if is_ <= ib:
        return "invalid", None, None
    return "buy_sell", timestep_to_csv_hour(ib), timestep_to_csv_hour(is_)


class PrecomputedSummaryLookup:
    """
    One CSV per delivery day: ``summary_{YYYY-MM-DD}.csv`` in ``results_dir``.
    """

    def __init__(self, results_dir: str | Path):
        self.results_dir = Path(results_dir)
        self._cache: dict[str, pd.DataFrame] = {}
        self.lookup_misses = 0

    def reset_miss_count(self) -> None:
        self.lookup_misses = 0

    def _path(self, day_iso: str) -> Path:
        return self.results_dir / f"summary_{day_iso}.csv"

    def _get_df(self, day_iso: str) -> pd.DataFrame:
        if day_iso not in self._cache:
            p = self._path(day_iso)
            if not p.is_file():
                raise FileNotFoundError(
                    f"Precomputed summary not found for day {day_iso}: {p}"
                )
            self._cache[day_iso] = pd.read_csv(p)
        return self._cache[day_iso]

    def row_for_schedule(
        self,
        day_iso: str,
        kind: ScheduleKind,
        buy_h: Optional[float],
        sell_h: Optional[float],
    ) -> Optional[pd.Series]:
        df = self._get_df(day_iso)
        if kind == "no_da":
            m = df["kind"] == "no_da"
        elif kind == "buy_only":
            m = (
                (df["kind"] == "buy_only")
                & (df["buy_hour"].astype(float) == float(buy_h))
                & (df["sell_hour"].isna())
            )
        elif kind == "buy_sell":
            m = (
                (df["kind"] == "buy_sell")
                & (df["buy_hour"].astype(float) == float(buy_h))
                & (df["sell_hour"].astype(float) == float(sell_h))
            )
        else:
            return None
        rows = df.loc[m]
        if len(rows) == 0:
            return None
        return rows.iloc[0]

    def profit_eur_for_key(
        self,
        day_iso: str,
        kind: ScheduleKind,
        buy_h: Optional[float],
        sell_h: Optional[float],
        *,
        warn: bool = True,
    ) -> float:
        """Combined profit (EUR) for an explicit schedule key."""
        if kind == "invalid":
            if warn:
                warnings.warn(
                    f"[PrecomputedSummaryLookup] Invalid schedule key for {day_iso}. "
                    "Using 0 EUR.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self.lookup_misses += 1
            return 0.0
        row = self.row_for_schedule(day_iso, kind, buy_h, sell_h)
        if row is None:
            self.lookup_misses += 1
            if warn:
                warnings.warn(
                    f"[PrecomputedSummaryLookup] No CSV row for {day_iso} "
                    f"kind={kind!r} buy_hour={buy_h!r} sell_hour={sell_h!r}. Using 0 EUR.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return 0.0
        return float(row["profit"])

    def da_profit_eur_for_key(
        self,
        day_iso: str,
        kind: ScheduleKind,
        buy_h: Optional[float],
        sell_h: Optional[float],
        *,
        warn: bool = True,
    ) -> float:
        """DA-only profit (EUR) for an explicit schedule key."""
        if kind == "invalid":
            if warn:
                warnings.warn(
                    f"[PrecomputedSummaryLookup] Invalid schedule key for {day_iso}. "
                    "Using 0 EUR for da_profit.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self.lookup_misses += 1
            return 0.0
        row = self.row_for_schedule(day_iso, kind, buy_h, sell_h)
        if row is None:
            self.lookup_misses += 1
            if warn:
                warnings.warn(
                    f"[PrecomputedSummaryLookup] No CSV row for {day_iso} "
                    f"kind={kind!r} buy_hour={buy_h!r} sell_hour={sell_h!r}. "
                    "Using 0 EUR for da_profit.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return 0.0
        v = row.get("da_profit", np.nan)
        if pd.isna(v):
            return 0.0
        return float(v)

    def profit_eur(
        self,
        day_iso: str,
        actions: np.ndarray,
        *,
        warn: bool = True,
    ) -> float:
        """Return combined DA+RI profit (EUR) for this action sequence, or 0.0 if missing."""
        kind, bh, sh = actions_to_schedule(actions)
        return self.profit_eur_for_key(day_iso, kind, bh, sh, warn=warn)

    def da_profit_eur(
        self,
        day_iso: str,
        actions: np.ndarray,
        *,
        warn: bool = True,
    ) -> float:
        """DA-only profit (EUR) from the same row as ``profit_eur``, or 0.0."""
        kind, bh, sh = actions_to_schedule(actions)
        return self.da_profit_eur_for_key(day_iso, kind, bh, sh, warn=warn)

    def intraday_profit_eur(
        self,
        day_iso: str,
        actions: np.ndarray,
        *,
        warn: bool = True,
    ) -> float:
        """RI part ≈ profit - da_profit from precomputed row."""
        p = self.profit_eur(day_iso, actions, warn=warn)
        d = self.da_profit_eur(day_iso, actions, warn=warn)
        return p - d
