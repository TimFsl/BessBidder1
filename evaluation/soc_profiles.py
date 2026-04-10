"""
State-of-charge (SoC) profiles from rolling-intrinsic ``trades_*.csv`` files.

SoC is integrated in **delivery-product time** (96 × 15-minute slots per day), using
signed power: buy → charge, sell → discharge. This matches the logic in your
reference plotting script.

- **Day-ahead SoC**: only trades whose ``execution_time`` is **13:00** in Europe/Berlin
  (hour == 13; DA auction).
- **Actual SoC**: all trades (DA + intraday) — the physically realised path over the day.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.shared.config import RTE  # round-trip efficiency


def trades_dir_next_to_profit(profit_csv: Path | str) -> Path:
    return Path(profit_csv).resolve().parent / "trades"


def prep_trades_for_soc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["execution_time"] = pd.to_datetime(df["execution_time"], utc=True).dt.tz_convert(
        "Europe/Berlin"
    )
    df["product"] = pd.to_datetime(df["product"], utc=True).dt.tz_convert("Europe/Berlin")

    df["prod_min"] = df["product"].dt.hour * 60 + df["product"].dt.minute
    df["prod_slot_15m"] = (df["prod_min"] // 15).astype(int)

    df["side"] = df["side"].astype(str).str.lower().str.strip()
    df = df[df["side"].isin(["buy", "sell"])]

    df["quantity"] = df["quantity"].astype(float)
    df.loc[df["quantity"].abs() < 1e-10, "quantity"] = 0.0
    return df


def is_da_execution(execution_time: pd.Series) -> pd.Series:
    """DA auction rows: execution at 13:xx Berlin (typically 13:00)."""
    return execution_time.dt.hour == 13


def soc_by_product_slots(
    df_run: pd.DataFrame,
    *,
    e_max_mwh: float = 1.0,
    soc0_mwh: float = 0.0,
    dt_hours: float = 0.25,
    eta_ch: float | None = None,
    eta_dis: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns ``(slot_index_0_95, soc_center_per_slot)`` each length 96.

    ``soc_center[k]`` is the midpoint SoC over delivery slot ``k`` (15 minutes).
    """
    if eta_ch is None or eta_dis is None:
        eta = float(RTE) ** 0.5
        eta_ch = eta
        eta_dis = eta

    if len(df_run) == 0:
        soc = np.full(97, float(soc0_mwh))
        soc = np.clip(soc, 0.0, e_max_mwh)
        soc_center = 0.5 * (soc[:-1] + soc[1:])
        return np.arange(0, 96, dtype=float), soc_center

    sign = df_run["side"].map({"buy": 1.0, "sell": -1.0}).astype(float)
    p_signed = sign * df_run["quantity"].astype(float)

    sched = (
        pd.DataFrame({"slot": df_run["prod_slot_15m"], "p_mw_signed": p_signed})
        .groupby("slot", as_index=False)["p_mw_signed"]
        .sum()
        .sort_values("slot")
    )

    full = pd.DataFrame({"slot": np.arange(0, 96)})
    sched = full.merge(sched, on="slot", how="left").fillna({"p_mw_signed": 0.0})

    e_slot = sched["p_mw_signed"].to_numpy(dtype=float) * dt_hours
    e_batt = np.where(e_slot >= 0, e_slot * eta_ch, e_slot / eta_dis)

    soc = np.empty(len(e_batt) + 1, dtype=float)
    soc[0] = float(soc0_mwh)
    for i in range(len(e_batt)):
        soc[i + 1] = np.clip(soc[i] + e_batt[i], 0.0, e_max_mwh)

    soc_center = 0.5 * (soc[:-1] + soc[1:])
    y_slots = np.arange(0, 96, dtype=float)
    return y_slots, soc_center


def delivery_hours_slot_centers() -> np.ndarray:
    """Hour-of-day (h) for each 15-min slot center, length 96."""
    return (np.arange(96) + 0.5) * 0.25


def soc_profile_for_trades_file(
    path: Path | str,
    *,
    mode: Literal["all", "da_only"] = "all",
    **soc_kw,
) -> np.ndarray:
    """One SoC center trajectory (length 96) for a single day's trades CSV."""
    df = pd.read_csv(path)
    df = prep_trades_for_soc(df)
    if mode == "da_only":
        df = df.loc[is_da_execution(df["execution_time"])].copy()
    _, soc_c = soc_by_product_slots(df, **soc_kw)
    return soc_c


def build_soc_profile_matrix(
    trades_dir: Path | str,
    delivery_days: pd.Series | list | np.ndarray,
    *,
    mode: Literal["all", "da_only"] = "all",
    on_missing: str = "warn",
    **soc_kw,
) -> pd.DataFrame:
    """
    Rows = delivery days, columns = ``slot_00`` … ``slot_95`` (SoC at slot center).

    ``delivery_days`` should be tz-aware or naive timestamps; normalized to Berlin date
    for the filename ``trades_YYYY-MM-DD.csv``.
    """
    td = Path(trades_dir)
    rows: list[np.ndarray] = []
    idx: list[pd.Timestamp] = []

    for d in delivery_days:
        dd = pd.Timestamp(d).tz_convert("Europe/Berlin").normalize()
        fp = td / f"trades_{dd.date().isoformat()}.csv"
        if not fp.is_file():
            msg = f"Missing trades file: {fp}"
            if on_missing == "raise":
                raise FileNotFoundError(msg)
            if on_missing == "warn":
                warnings.warn(msg, stacklevel=2)
            continue
        soc = soc_profile_for_trades_file(fp, mode=mode, **soc_kw)
        rows.append(soc)
        idx.append(dd)

    if not rows:
        cols = [f"slot_{i:02d}" for i in range(96)]
        return pd.DataFrame(columns=cols)

    mat = np.vstack(rows)
    cols = [f"slot_{i:02d}" for i in range(96)]
    return pd.DataFrame(mat, index=pd.DatetimeIndex(idx, name="delivery_day"), columns=cols)


def summarize_soc_across_days(
    mat: pd.DataFrame,
    *,
    q_low: float = 0.10,
    q_high: float = 0.90,
) -> pd.DataFrame:
    """Per-slot mean and quantiles across days (rows of ``mat``)."""
    h = delivery_hours_slot_centers()
    if mat.empty or mat.shape[0] == 0:
        nan96 = np.full(96, np.nan, dtype=float)
        return pd.DataFrame(
            {"hour": h, "mean": nan96, "q_low": nan96, "q_high": nan96}
        )
    x = mat.to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "hour": h,
            "mean": np.nanmean(x, axis=0),
            "q_low": np.nanquantile(x, q_low, axis=0),
            "q_high": np.nanquantile(x, q_high, axis=0),
        }
    )


def plot_mean_quantile_soc(
    summary_da: pd.DataFrame,
    summary_all: pd.DataFrame,
    *,
    title: str | None = None,
    q_low: float = 0.10,
    q_high: float = 0.90,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (10, 4.5),
):
    """
    One axes: DA-only vs actual (all trades) mean lines + quantile bands.

    Returns ``(figure_or_none, ax)`` — figure is ``None`` if ``ax`` was passed in.
    """
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    h = summary_da["hour"].to_numpy()

    ax.fill_between(
        h,
        summary_da["q_low"],
        summary_da["q_high"],
        alpha=0.25,
        color="C0",
        label=f"DA SoC q{q_low:.0%}–q{q_high:.0%}",
    )
    ax.plot(h, summary_da["mean"], color="C0", lw=2, label="DA SoC mean")

    ax.fill_between(
        h,
        summary_all["q_low"],
        summary_all["q_high"],
        alpha=0.25,
        color="C1",
        label=f"Actual SoC q{q_low:.0%}–q{q_high:.0%}",
    )
    ax.plot(h, summary_all["mean"], color="C1", lw=2, label="Actual SoC mean")

    ax.set_xlim(0, 24)
    ax.set_xlabel("Delivery hour (product time)")
    ax.set_ylabel("SoC (MWh)")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=8)
    if title:
        ax.set_title(title)

    return fig, ax


def rl_soc_plot_from_test_days(
    model_number: str | int,
    delivery_days: pd.Series,
    *,
    repo_root: Path | None = None,
    bs_folder: str = "bs15cr1rto0.86mc365mt10",
    q_low: float = 0.10,
    q_high: float = 0.90,
    **soc_kw,
):
    """
    Convenience: coordinated RL ``trades_dir`` from model id + ``df_test['delivery_day']``.
    Returns ``(fig, ax, mat_da, mat_all, summary_da, summary_all)``.
    """
    from evaluation.rl_model_registry import rl_rolling_intrinsic_trades_dir

    trades_dir = rl_rolling_intrinsic_trades_dir(
        model_number, repo_root=repo_root, bs_folder=bs_folder
    )
    mat_da = build_soc_profile_matrix(
        trades_dir, delivery_days, mode="da_only", **soc_kw
    )
    mat_all = build_soc_profile_matrix(
        trades_dir, delivery_days, mode="all", **soc_kw
    )
    s_da = summarize_soc_across_days(mat_da, q_low=q_low, q_high=q_high)
    s_all = summarize_soc_across_days(mat_all, q_low=q_low, q_high=q_high)
    fig, ax = plot_mean_quantile_soc(
        s_da,
        s_all,
        title=f"SoC profiles (coordinated RL, model {model_number})",
        q_low=q_low,
        q_high=q_high,
    )
    return fig, ax, mat_da, mat_all, s_da, s_all
