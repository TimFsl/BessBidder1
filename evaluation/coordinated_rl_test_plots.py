"""
Thesis-style figures for **multi-market** test evaluation: cumulative DA vs intraday profit
(from ``trades_*.csv``) next to mean SoC bands (same colours as :mod:`evaluation.soc_profiles`).

Entry points:

- :func:`coordinated_rl_cumulative_profit_and_soc_figure` — coordinated RL trades under
  ``output/coordinated_multi_market/logging/<id>/.../trades/``.
- :func:`myopic_cumulative_profit_and_soc_figure` — myopic baseline trades under
  ``output/myopic_multi_market/.../trades/`` (same layout and styling as coordinated).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluation.soc_profiles import (
    SOC_PLOT_COLORS,
    build_soc_profile_matrix,
    is_da_execution,
    plot_mean_quantile_soc,
    prep_trades_for_soc,
    summarize_soc_across_days,
)


def daily_da_idc_profit_from_trades_file(path: Path | str) -> tuple[float, float]:
    """
    Sum ``profit`` for rows with DA execution (hour 13 Berlin) vs all other rows.

    Returns ``(profit_da_eur, profit_intraday_eur)``.
    """
    p = Path(path)
    df = pd.read_csv(p)
    if "profit" not in df.columns:
        raise ValueError(f"{p}: expected column 'profit'")
    if len(df) == 0:
        return 0.0, 0.0
    df = prep_trades_for_soc(df)
    pr = pd.to_numeric(df["profit"], errors="coerce").fillna(0.0)
    da_mask = is_da_execution(df["execution_time"])
    return float(pr[da_mask].sum()), float(pr[~da_mask].sum())


def build_daily_da_idc_profit_table(
    trades_dir: Path | str,
    delivery_days: pd.Series | list | np.ndarray,
    *,
    on_missing: str = "warn",
) -> pd.DataFrame:
    """
    One row per test ``delivery_day`` with daily DA and intraday profit (€).

    Uses the same day ordering and missing-file policy as SoC matrices.
    """
    import warnings

    td = Path(trades_dir)
    rows: list[dict] = []
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
        da_e, idc_e = daily_da_idc_profit_from_trades_file(fp)
        rows.append(
            {
                "delivery_day": dd,
                "profit_da_eur": da_e,
                "profit_idc_eur": idc_e,
                "profit_total_eur": da_e + idc_e,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["delivery_day", "profit_da_eur", "profit_idc_eur", "profit_total_eur"]
        )
    out = pd.DataFrame(rows).sort_values("delivery_day").reset_index(drop=True)
    return out


def plot_cumulative_da_idc_stacked(
    ax: plt.Axes,
    profit_table: pd.DataFrame,
    *,
    colors: dict[str, str] | None = None,
    day_index_offset: int = 0,
    legend_title: str = "Overall profit",
    sort_days_by_total_profit: bool = True,
    show_grid: bool = False,
    axis_labelsize: float | None = None,
    tick_labelsize: float | None = None,
    legend_fontsize: float | None = None,
    area_alpha: float = 0.5,
) -> None:
    """
    Stacked cumulative area: bottom = DA, top = intraday (IDC).

    Uses SoC **quantile fill** colours for areas. Only the **upper** boundaries of each
    stack (DA cumulative top, then total top) are drawn as lines — ``#5EBEC4`` and
    ``#F92C85``, matching coordinated mean SoC lines — not a full rectangle around the fills.

    If ``sort_days_by_total_profit``, days are ordered by increasing daily total profit
    so the cumulative total grows more smoothly (not chronological).
    """
    c = colors if colors is not None else SOC_PLOT_COLORS
    if profit_table.empty:
        ax.set_xlabel("Day number")
        ax.set_ylabel("Cumulative profit (€)")
        return

    df = profit_table.copy()
    if sort_days_by_total_profit and "profit_total_eur" in df.columns:
        df = df.sort_values("profit_total_eur", ascending=True).reset_index(drop=True)

    n = len(df)
    # Include origin (0, 0) like a classic cumulative curve; then one point per test day.
    x = np.concatenate([[0.0], np.arange(1.0, n + 1.0, dtype=float) + float(day_index_offset)])
    cum_da = np.concatenate([[0.0], df["profit_da_eur"].cumsum().to_numpy()])
    cum_idc = np.concatenate([[0.0], df["profit_idc_eur"].cumsum().to_numpy()])
    top = cum_da + cum_idc

    da_tot = float(cum_da[-1])
    idc_tot = float(cum_idc[-1])
    sum_tot = float(top[-1])

    edge_da = "#5EBEC4"
    edge_idc = "#F92C85"
    lw_edge = 2.5

    ax.fill_between(
        x,
        0.0,
        cum_da,
        facecolor=c["fill_da"],
        alpha=area_alpha,
        linewidth=0,
        edgecolor="none",
        label="_da_layer",
    )
    ax.fill_between(
        x,
        cum_da,
        top,
        facecolor=c["fill_actual"],
        alpha=area_alpha,
        linewidth=0,
        edgecolor="none",
        label="_idc_layer",
    )
    # Upper stack edges only (not a closed outline around each fill).
    ax.plot(x, cum_da, color=edge_da, linewidth=lw_edge, zorder=4, clip_on=True)
    ax.plot(x, top, color=edge_idc, linewidth=lw_edge, zorder=5, clip_on=True)

    _lf = legend_fontsize if legend_fontsize is not None else 8
    s_sum = int(round(sum_tot))
    s_idc = int(round(idc_tot))
    s_da = int(round(da_tot))

    # Slightly inset from axes; swatches wider than tall (small rectangles, not squares).
    x0 = 0.048
    y_title = 0.955
    line_h = 0.034 * (_lf / 8.0)
    sw_h = 0.018 * (_lf / 8.0)
    sw_w = 2.35 * sw_h
    gap = 0.006 * (_lf / 8.0)
    x_text = x0 + sw_w + gap
    gap_after_title = 0.026 * (_lf / 8.0)

    ax.text(
        x0,
        y_title,
        legend_title,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=_lf,
    )
    y_sum = y_title - line_h - gap_after_title
    ax.text(
        x_text,
        y_sum,
        f"SUM: {s_sum:,} €",
        transform=ax.transAxes,
        va="center",
        ha="left",
        fontsize=_lf,
    )
    y_idc = y_sum - line_h
    ax.add_patch(
        Rectangle(
            (x0, y_idc - sw_h / 2),
            sw_w,
            sw_h,
            transform=ax.transAxes,
            facecolor=c["fill_actual"],
            edgecolor=edge_idc,
            linewidth=1.2,
            clip_on=False,
            zorder=20,
        )
    )
    ax.text(
        x0 + sw_w + gap,
        y_idc,
        f"IDC: {s_idc:,} €",
        transform=ax.transAxes,
        va="center",
        ha="left",
        fontsize=_lf,
    )
    y_da = y_idc - line_h
    ax.add_patch(
        Rectangle(
            (x0, y_da - sw_h / 2),
            sw_w,
            sw_h,
            transform=ax.transAxes,
            facecolor=c["fill_da"],
            edgecolor=edge_da,
            linewidth=1.2,
            clip_on=False,
            zorder=20,
        )
    )
    ax.text(
        x0 + sw_w + gap,
        y_da,
        f"DA: {s_da:,} €",
        transform=ax.transAxes,
        va="center",
        ha="left",
        fontsize=_lf,
    )

    ax.set_xlabel("Day number")
    ax.set_ylabel("Cumulative profit (€)")
    if show_grid:
        ax.grid(True, alpha=0.35)
    else:
        ax.grid(False)

    y_max = float(np.nanmax(top))
    y_top_lim = y_max * 1.02 if y_max > 0 else 1.0
    ax.set_ylim(0.0, y_top_lim)
    ax.set_xlim(float(np.min(x)), float(np.max(x)) + 0.5)
    ax.margins(x=0, y=0)

    if axis_labelsize is not None:
        ax.xaxis.label.set_fontsize(axis_labelsize)
        ax.yaxis.label.set_fontsize(axis_labelsize)
    if tick_labelsize is not None:
        ax.tick_params(axis="both", which="major", labelsize=tick_labelsize)


def cumulative_profit_and_soc_figure_from_trades_dir(
    trades_dir: Path | str,
    delivery_days: pd.Series,
    *,
    q_low: float = 0.25,
    q_high: float = 0.75,
    figsize: tuple[float, float] = (12.5, 4.8),
    width_ratios: tuple[float, float] = (1.35, 1.0),
    on_missing_trades: str = "warn",
    spine_linewidth: float = 1.0,
    **soc_kw,
):
    """
    One row, two columns: cumulative DA + IDC profit (left), mean SoC DA vs actual (right).

    Reads ``trades_*.csv`` under ``trades_dir``. The returned **profit** table is ordered by
    increasing **daily total** profit (for the cumulative panel only); SoC summaries still
    aggregate over calendar test days.

    Returns
    -------
    fig, ax_cum, ax_soc, profit_table_plot, mat_da, mat_all, summary_da, summary_all
    """
    td = Path(trades_dir)
    profit_table = build_daily_da_idc_profit_table(
        td, delivery_days, on_missing=on_missing_trades
    )
    profit_table_plot = (
        profit_table.sort_values("profit_total_eur", ascending=True).reset_index(drop=True)
        if len(profit_table) > 0
        else profit_table
    )

    mat_da = build_soc_profile_matrix(
        td, delivery_days, mode="da_only", on_missing=on_missing_trades, **soc_kw
    )
    mat_all = build_soc_profile_matrix(
        td, delivery_days, mode="all", on_missing=on_missing_trades, **soc_kw
    )
    summary_da = summarize_soc_across_days(mat_da, q_low=q_low, q_high=q_high)
    summary_all = summarize_soc_across_days(mat_all, q_low=q_low, q_high=q_high)

    fig, (ax_cum, ax_soc) = plt.subplots(
        1,
        2,
        figsize=figsize,
        gridspec_kw={"width_ratios": list(width_ratios)},
        constrained_layout=False,
    )
    _lab = 18.0
    _tick = 16.0
    _leg = 18.0

    plot_cumulative_da_idc_stacked(
        ax_cum,
        profit_table_plot,
        sort_days_by_total_profit=False,
        show_grid=False,
        axis_labelsize=_lab,
        tick_labelsize=_tick,
        legend_fontsize=_leg,
    )

    plot_mean_quantile_soc(
        summary_da,
        summary_all,
        plot_da=True,
        ax=ax_soc,
        q_low=q_low,
        q_high=q_high,
        swap_axes=True,
        show_grid=False,
        show_legend=False,
        axis_labelsize=_lab,
        tick_labelsize=_tick,
        legend_fontsize=_leg,
    )

    for ax in (ax_cum, ax_soc):
        for spine in ax.spines.values():
            spine.set_linewidth(spine_linewidth)

    fig.tight_layout(w_pad=2.0)
    return fig, ax_cum, ax_soc, profit_table_plot, mat_da, mat_all, summary_da, summary_all


def coordinated_rl_cumulative_profit_and_soc_figure(
    model_number: str | int,
    delivery_days: pd.Series,
    *,
    repo_root: Path | None = None,
    bs_folder: str = "bs15cr1rto0.86mc365mt10",
    q_low: float = 0.25,
    q_high: float = 0.75,
    figsize: tuple[float, float] = (12.5, 4.8),
    width_ratios: tuple[float, float] = (1.35, 1.0),
    on_missing_trades: str = "warn",
    spine_linewidth: float = 1.0,
    **soc_kw,
):
    """
    Same as :func:`cumulative_profit_and_soc_figure_from_trades_dir` using coordinated RL
    ``trades/`` for ``model_number``.
    """
    from evaluation.rl_model_registry import rl_rolling_intrinsic_trades_dir

    trades_dir = rl_rolling_intrinsic_trades_dir(
        model_number, repo_root=repo_root, bs_folder=bs_folder
    )
    return cumulative_profit_and_soc_figure_from_trades_dir(
        trades_dir,
        delivery_days,
        q_low=q_low,
        q_high=q_high,
        figsize=figsize,
        width_ratios=width_ratios,
        on_missing_trades=on_missing_trades,
        spine_linewidth=spine_linewidth,
        **soc_kw,
    )


def myopic_cumulative_profit_and_soc_figure(
    delivery_days: pd.Series,
    *,
    repo_root: Path | None = None,
    bs_folder: str = "bs15cr1rto0.86mc365mt10",
    q_low: float = 0.25,
    q_high: float = 0.75,
    figsize: tuple[float, float] = (12.5, 4.8),
    width_ratios: tuple[float, float] = (1.35, 1.0),
    on_missing_trades: str = "warn",
    spine_linewidth: float = 1.0,
    **soc_kw,
):
    """
    Same layout and styling as :func:`coordinated_rl_cumulative_profit_and_soc_figure`, but
    using myopic multi-market ``output/myopic_multi_market/.../trades/`` (no RL model id).
    """
    from evaluation.rl_model_registry import myopic_rolling_intrinsic_trades_dir

    trades_dir = myopic_rolling_intrinsic_trades_dir(
        repo_root=repo_root, bs_folder=bs_folder
    )
    return cumulative_profit_and_soc_figure_from_trades_dir(
        trades_dir,
        delivery_days,
        q_low=q_low,
        q_high=q_high,
        figsize=figsize,
        width_ratios=width_ratios,
        on_missing_trades=on_missing_trades,
        spine_linewidth=spine_linewidth,
        **soc_kw,
    )
