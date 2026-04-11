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
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.shared.config import RTE  # round-trip efficiency

# Mean lines: first two palette entries; bands: last two (user-specified hex).
SOC_PLOT_COLORS = {
    "mean_da": "#5EBEC4",
    "mean_actual": "#F92C85",
    "fill_da":"#9CD4D8",
    "fill_actual": "#F67EB7",
}


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


def single_market_trades_file(
    delivery_day: pd.Timestamp,
    *,
    repo_root: Path | None = None,
    bs_folder: str = "bs15cr1rto0.86mc365mt10",
) -> Path | None:
    """
    Resolve ``trades_YYYY-MM-DD.csv`` under ``qh/<year>/`` (or any ``qh/*`` fallback).
    Single-market RI stores outputs per calendar year.
    """
    root = repo_root or _REPO
    qh_root = root / "output/single_market/rolling_intrinsic/ri_basic/qh"
    dd = pd.Timestamp(delivery_day).tz_convert("Europe/Berlin").normalize()
    iso = dd.date().isoformat()
    year_first = qh_root / str(dd.year) / bs_folder / "trades" / f"trades_{iso}.csv"
    if year_first.is_file():
        return year_first
    if qh_root.is_dir():
        for sub in sorted(qh_root.iterdir()):
            if not sub.is_dir():
                continue
            cand = sub / bs_folder / "trades" / f"trades_{iso}.csv"
            if cand.is_file():
                return cand
    return None


def build_soc_profile_matrix_resolved(
    delivery_days: pd.Series | list | np.ndarray,
    trades_file_for_day: Callable[[pd.Timestamp], Path | None],
    *,
    mode: Literal["all", "da_only"] = "all",
    on_missing: str = "warn",
    **soc_kw,
) -> pd.DataFrame:
    """Like :func:`build_soc_profile_matrix` but paths come from a per-day resolver."""
    rows: list[np.ndarray] = []
    idx: list[pd.Timestamp] = []

    for d in delivery_days:
        dd = pd.Timestamp(d).tz_convert("Europe/Berlin").normalize()
        fp = trades_file_for_day(dd)
        if fp is None or not fp.is_file():
            msg = f"Missing single-market trades file for {dd.date()}"
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


def format_quantile_band_label(q_low: float, q_high: float) -> str:
    """Human-readable range for legends, e.g. ``q25–q75`` or ``q5–q95``."""

    def pct_part(q: float) -> str:
        p = q * 100.0
        if abs(p - round(p)) < 1e-6:
            return str(int(round(p)))
        return f"{p:g}"

    return f"q{pct_part(q_low)}–q{pct_part(q_high)}"


def summarize_soc_across_days(
    mat: pd.DataFrame,
    *,
    q_low: float = 0.25,
    q_high: float = 0.75,
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
    plot_da: bool = True,
    title: str | None = None,
    q_low: float = 0.25,
    q_high: float = 0.75,
    colors: dict[str, str] | None = None,
    band_alpha: float = 0.35,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (10, 4.5),
    swap_axes: bool = False,
    show_grid: bool = True,
    axis_labelsize: float | None = None,
    tick_labelsize: float | None = None,
    legend_fontsize: float | None = None,
    show_legend: bool = True,
):
    """
    One axes: optional DA-only vs actual (all trades) mean lines + quantile bands.

    If ``plot_da=False`` (e.g. single-market RI with no DA trades), only the
    **actual** trajectory band + mean are drawn (same colours as coordinated “Actual”).

    If ``swap_axes=True``, **x** = State of charge (MWh), **y** = delivery hour (with
    ``00:00``… tick labels on the vertical axis).
    """
    c = colors if colors is not None else SOC_PLOT_COLORS
    qlab = format_quantile_band_label(q_low, q_high)

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    h = summary_all["hour"].to_numpy()

    def _soc_xlim_pad(*series) -> tuple[float, float]:
        xs = np.concatenate(
            [np.asarray(s, dtype=float).ravel() for s in series if s is not None]
        )
        xs = xs[np.isfinite(xs)]
        if xs.size == 0:
            return 0.0, 1.0
        lo, hi = float(np.nanmin(xs)), float(np.nanmax(xs))
        pad = 0.05 * (hi - lo) if hi > lo else 0.05
        return lo - pad, hi + pad

    def _soc_xlim_zero_left(*series) -> tuple[float, float]:
        """SoC is non-negative: pin x=0 to the y-axis (no gap before zero)."""
        xs = np.concatenate(
            [np.asarray(s, dtype=float).ravel() for s in series if s is not None]
        )
        xs = xs[np.isfinite(xs)]
        if xs.size == 0:
            return 0.0, 1.0
        hi = float(np.nanmax(xs))
        pad = 0.05 * hi if hi > 0 else 0.05
        return 0.0, hi + pad

    if swap_axes:
        if plot_da:
            ax.fill_betweenx(
                h,
                summary_da["q_low"],
                summary_da["q_high"],
                alpha=band_alpha,
                color=c["fill_da"],
                label=f"DA SoC {qlab}",
            )
            ax.plot(
                summary_da["mean"],
                h,
                color=c["mean_da"],
                lw=2,
                label="DA SoC mean",
            )
            ax.fill_betweenx(
                h,
                summary_all["q_low"],
                summary_all["q_high"],
                alpha=band_alpha,
                color=c["fill_actual"],
                label=f"Actual SoC {qlab}",
            )
            ax.plot(
                summary_all["mean"],
                h,
                color=c["mean_actual"],
                lw=2,
                label="Actual SoC mean",
            )
            x0, x1 = _soc_xlim_zero_left(
                summary_da["q_low"],
                summary_da["q_high"],
                summary_all["q_low"],
                summary_all["q_high"],
                summary_da["mean"],
                summary_all["mean"],
            )
        else:
            ax.fill_betweenx(
                h,
                summary_all["q_low"],
                summary_all["q_high"],
                alpha=band_alpha,
                color=c["fill_actual"],
                label=f"SoC {qlab}",
            )
            ax.plot(
                summary_all["mean"],
                h,
                color=c["mean_actual"],
                lw=2,
                label="SoC mean",
            )
            x0, x1 = _soc_xlim_zero_left(
                summary_all["q_low"],
                summary_all["q_high"],
                summary_all["mean"],
            )
        ax.set_ylim(0, 24)
        yticks = np.arange(0, 25, 2, dtype=float)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{int(t):02d}:00" for t in yticks])
        ax.set_xlim(x0, x1)
        ax.margins(x=0)
        ax.set_xlabel("State of Charge (MWh)")
        ax.set_ylabel("Delivery hour (product time)")
    elif plot_da:
        ax.fill_between(
            h,
            summary_da["q_low"],
            summary_da["q_high"],
            alpha=band_alpha,
            color=c["fill_da"],
            label=f"DA SoC {qlab}",
        )
        ax.plot(
            h,
            summary_da["mean"],
            color=c["mean_da"],
            lw=2,
            label="DA SoC mean",
        )

        ax.fill_between(
            h,
            summary_all["q_low"],
            summary_all["q_high"],
            alpha=band_alpha,
            color=c["fill_actual"],
            label=f"Actual SoC {qlab}",
        )
        ax.plot(
            h,
            summary_all["mean"],
            color=c["mean_actual"],
            lw=2,
            label="Actual SoC mean",
        )

        ax.set_xlim(0, 24)
        xticks = np.arange(0, 25, 2, dtype=float)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{int(t):02d}:00" for t in xticks])
        ax.set_xlabel("Delivery hour (product time)")
        ax.set_ylabel("State of Charge (MWh)")
    else:
        ax.fill_between(
            h,
            summary_all["q_low"],
            summary_all["q_high"],
            alpha=band_alpha,
            color=c["fill_actual"],
            label=f"SoC {qlab}",
        )
        ax.plot(
            h,
            summary_all["mean"],
            color=c["mean_actual"],
            lw=2,
            label="SoC mean",
        )

        ax.set_xlim(0, 24)
        xticks = np.arange(0, 25, 2, dtype=float)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{int(t):02d}:00" for t in xticks])
        ax.set_xlabel("Delivery hour (product time)")
        ax.set_ylabel("State of Charge (MWh)")

    if show_grid:
        ax.grid(True, alpha=0.35)
    else:
        ax.grid(False)

    if show_legend:
        _lf = legend_fontsize if legend_fontsize is not None else 8
        leg = ax.legend(loc="best", fontsize=_lf)
        if leg.get_title().get_text():
            leg.get_title().set_fontsize(_lf)
    if axis_labelsize is not None:
        ax.xaxis.label.set_fontsize(axis_labelsize)
        ax.yaxis.label.set_fontsize(axis_labelsize)
    if tick_labelsize is not None:
        ax.tick_params(axis="both", which="major", labelsize=tick_labelsize)

    if title:
        ax.set_title(title)

    return fig, ax


def rl_soc_plot_from_test_days(
    model_number: str | int,
    delivery_days: pd.Series,
    *,
    repo_root: Path | None = None,
    bs_folder: str = "bs15cr1rto0.86mc365mt10",
    q_low: float = 0.25,
    q_high: float = 0.75,
    title: str | None = None,
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
        plot_da=True,
        title=title,
        q_low=q_low,
        q_high=q_high,
    )
    return fig, ax, mat_da, mat_all, s_da, s_all


def myopic_soc_plot_from_test_days(
    delivery_days: pd.Series,
    *,
    repo_root: Path | None = None,
    bs_folder: str = "bs15cr1rto0.86mc365mt10",
    q_low: float = 0.25,
    q_high: float = 0.75,
    title: str | None = None,
    **soc_kw,
):
    """
    Same SoC logic as coordinated RL: DA (ex. hour 13) + actual, myopic trade folder.
    Returns ``(fig, ax, mat_da, mat_all, summary_da, summary_all)``.
    """
    from evaluation.rl_model_registry import myopic_rolling_intrinsic_trades_dir

    trades_dir = myopic_rolling_intrinsic_trades_dir(
        repo_root=repo_root, bs_folder=bs_folder
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
        plot_da=True,
        title=title,
        q_low=q_low,
        q_high=q_high,
    )
    return fig, ax, mat_da, mat_all, s_da, s_all


def single_market_soc_plot_from_test_days(
    delivery_days: pd.Series,
    *,
    repo_root: Path | None = None,
    bs_folder: str = "bs15cr1rto0.86mc365mt10",
    q_low: float = 0.25,
    q_high: float = 0.75,
    title: str | None = None,
    **soc_kw,
):
    """
    Single-market RI: no DA layer — only **all-trades** SoC (intraday-only economically).

    Returns ``(fig, ax, mat_all, summary_all)`` (no ``mat_da`` / ``summary_da``).
    """
    mat_all = build_soc_profile_matrix_resolved(
        delivery_days,
        lambda dd: single_market_trades_file(dd, repo_root=repo_root, bs_folder=bs_folder),
        mode="all",
        **soc_kw,
    )
    s_all = summarize_soc_across_days(mat_all, q_low=q_low, q_high=q_high)
    s_da_empty = summarize_soc_across_days(
        pd.DataFrame(columns=[f"slot_{i:02d}" for i in range(96)]),
        q_low=q_low,
        q_high=q_high,
    )
    fig, ax = plot_mean_quantile_soc(
        s_da_empty,
        s_all,
        plot_da=False,
        title=title,
        q_low=q_low,
        q_high=q_high,
    )
    return fig, ax, mat_all, s_all
