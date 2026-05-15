"""
Build a long table from brute-force trade CSVs (``results_merged/trades``) and
optional monthly aggregates / combo charts (bars + line).

Each row is one (delivery_day, combo_id): ``n_trades``, ``daily_profit`` (sum of
per-trade ``profit`` in the combo file), ``profit_per_trade`` = daily_profit / n_trades.

Example::

    python brute_force_trades_daily_long_and_plots.py \\
        --trades-dir results_merged/trades \\
        --out-long results_merged/brute_force_daily_combo_trade_stats.parquet \\
        --monthly-csv results_merged/brute_force_monthly_trade_stats.csv \\
        --plot-png results_merged/brute_force_monthly_trades_and_profit_per_trade.png
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

N_COMBOS = 300


def combo_filename(combo_id: int) -> str:
    return f"trades_combo_{combo_id:03d}.csv"


def _process_one_day_impl(trades_root: Path, day_name: str, combo_ids: list[int]) -> list[dict[str, Any]]:
    day_dir = trades_root / day_name
    day_ts = pd.Timestamp(day_name, tz="Europe/Berlin").normalize()
    rows: list[dict[str, Any]] = []
    for cid in combo_ids:
        fp = day_dir / combo_filename(cid)
        if not fp.is_file():
            rows.append(
                {
                    "delivery_day": day_ts,
                    "combo_id": cid,
                    "n_trades": pd.NA,
                    "daily_profit": pd.NA,
                    "profit_per_trade": pd.NA,
                    "missing_file": True,
                }
            )
            continue
        df = pd.read_csv(fp, usecols=["profit"])
        df["profit"] = pd.to_numeric(df["profit"], errors="coerce").fillna(0.0)
        n = int(df.shape[0])
        prof = float(df["profit"].sum())
        ppt = (prof / n) if n > 0 else float("nan")
        rows.append(
            {
                "delivery_day": day_ts,
                "combo_id": cid,
                "n_trades": n,
                "daily_profit": prof,
                "profit_per_trade": ppt,
                "missing_file": False,
            }
        )
    return rows


def _worker_init(trades_root_str: str) -> None:
    global _TRADES_ROOT
    _TRADES_ROOT = Path(trades_root_str)


def _worker_job(day_name: str) -> list[dict[str, Any]]:
    combo_ids = list(range(N_COMBOS))
    return _process_one_day_impl(Path(_TRADES_ROOT), day_name, combo_ids)


def iter_delivery_day_names(trades_root: Path) -> list[str]:
    if not trades_root.is_dir():
        raise FileNotFoundError(f"Not a directory: {trades_root}")
    names = sorted(p.name for p in trades_root.iterdir() if p.is_dir())
    return names


def build_long_table(
    trades_root: Path,
    *,
    max_workers: int = 1,
) -> pd.DataFrame:
    days = iter_delivery_day_names(trades_root)
    if not days:
        raise FileNotFoundError(f"No day folders under {trades_root}")

    all_rows: list[dict[str, Any]] = []
    if max_workers <= 1:
        for d in days:
            all_rows.extend(_process_one_day_impl(trades_root, d, list(range(N_COMBOS))))
    else:
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_worker_init,
            initargs=(str(trades_root.resolve()),),
        ) as ex:
            futs = {ex.submit(_worker_job, d): d for d in days}
            for fut in as_completed(futs):
                all_rows.extend(fut.result())

    long_df = pd.DataFrame(all_rows)
    long_df = long_df.sort_values(["delivery_day", "combo_id"], kind="mergesort").reset_index(drop=True)
    return long_df


def monthly_aggregate(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per calendar month.

    - ``mean_n_trades``: mean of ``n_trades`` over all (day, combo) rows in that month
      (pandas mean skips NaN by default).
    - ``mean_profit_per_trade``: arithmetic mean of ``profit_per_trade`` only where
      ``n_trades`` > 0.
    - ``pooled_profit_per_trade``: sum(daily_profit) / sum(n_trades) over rows with
      ``n_trades`` > 0 (one pooled €/trade ratio for the month).
    """
    work = long_df.copy()
    work["month"] = work["delivery_day"].dt.to_period("M").dt.to_timestamp(how="start")

    def agg_month(g: pd.DataFrame) -> pd.Series:
        has = g["n_trades"].notna() & (g["n_trades"] > 0)
        valid = g.loc[has]
        tn = int(valid["n_trades"].sum())
        tp = float(valid["daily_profit"].sum()) if tn else float("nan")
        pooled = (tp / tn) if tn > 0 else float("nan")
        return pd.Series(
            {
                "mean_n_trades": float(g["n_trades"].mean()),
                "mean_profit_per_trade": float(valid["profit_per_trade"].mean())
                if len(valid)
                else float("nan"),
                "pooled_profit_per_trade": pooled,
                "n_rows": len(g),
                "n_rows_with_trades": int(len(valid)),
            }
        )

    out = work.groupby("month", sort=True).apply(agg_month).reset_index()
    return out


def monthly_aggregate_daily_first(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per calendar month, but with a "daily first, then monthly" approach.

    1) For each delivery day, average across available combos:
       - day_mean_n_trades = mean(n_trades)
       - day_mean_profit_per_trade = mean(profit_per_trade over rows with n_trades > 0)
       - day_pooled_profit_per_trade = sum(daily_profit) / sum(n_trades) over rows with n_trades > 0
    2) Then take monthly means of these daily values.
    """
    work = long_df.copy()
    work["delivery_day"] = pd.to_datetime(work["delivery_day"], errors="coerce")
    work = work.dropna(subset=["delivery_day"])
    work["month"] = work["delivery_day"].dt.to_period("M").dt.to_timestamp(how="start")

    def agg_day(g: pd.DataFrame) -> pd.Series:
        has = g["n_trades"].notna() & (g["n_trades"] > 0)
        valid = g.loc[has]
        tn = float(valid["n_trades"].sum())
        tp = float(valid["daily_profit"].sum()) if tn > 0 else float("nan")
        pooled = (tp / tn) if tn > 0 else float("nan")
        return pd.Series(
            {
                "day_mean_n_trades": float(g["n_trades"].mean()),
                "day_mean_profit_per_trade": float(valid["profit_per_trade"].mean())
                if len(valid)
                else float("nan"),
                "day_pooled_profit_per_trade": pooled,
                "day_rows": int(len(g)),
                "day_rows_with_trades": int(len(valid)),
            }
        )

    by_day = work.groupby("delivery_day", sort=True).apply(agg_day).reset_index()
    by_day["month"] = by_day["delivery_day"].dt.to_period("M").dt.to_timestamp(how="start")

    out = (
        by_day.groupby("month", sort=True)
        .agg(
            mean_n_trades=("day_mean_n_trades", "mean"),
            mean_profit_per_trade=("day_mean_profit_per_trade", "mean"),
            pooled_profit_per_trade=("day_pooled_profit_per_trade", "mean"),
            n_days=("delivery_day", "nunique"),
            mean_day_rows=("day_rows", "mean"),
            mean_day_rows_with_trades=("day_rows_with_trades", "mean"),
        )
        .reset_index()
    )
    return out


def plot_monthly_bars_and_line(
    monthly: pd.DataFrame,
    out_png: Path,
    *,
    use_pooled_line: bool = False,
    dpi: int = 150,
    figsize: tuple[float, float] = (12, 5),
    close_fig: bool = True,
):
    """
    Combo chart: monthly mean trade count (bars) and profit per trade (line).

    Returns the matplotlib figure. If ``close_fig`` is True (CLI default), the figure
    is closed after saving (return value is still valid until closed).
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    _axis_label_font = 16
    _tick_label_font = 12

    fig, ax1 = plt.subplots(figsize=figsize)
    x = monthly["month"].dt.to_pydatetime()
    w = 20  # bar width in days (approx)
    ax1.bar(
        x,
        monthly["mean_n_trades"],
        width=w,
        color="steelblue",
        alpha=0.75,
        label="Mean number of trades",
    )
    ax1.set_ylabel("Mean number of trades", fontsize=_axis_label_font)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax1.tick_params(axis="x", which="major", labelsize=_tick_label_font)
    ax1.tick_params(axis="y", which="major", labelsize=_tick_label_font)
    fig.autofmt_xdate(rotation=45)

    line_col = "pooled_profit_per_trade" if use_pooled_line else "mean_profit_per_trade"
    line_label = (
        "Profit per trade (pooled)"
        if use_pooled_line
        else "Profit per trade (mean of ratios)"
    )
    ax2 = ax1.twinx()
    ax2.plot(x, monthly[line_col], color="darkorange", marker="o", lw=1.5, ms=3, label=line_label)
    ax2.set_ylabel("Profit per trade (€)", fontsize=_axis_label_font)
    ax2.tick_params(axis="y", which="major", labelsize=_tick_label_font)

    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        lab1 + lab2,
        loc="upper left",
        fontsize=_axis_label_font,
        frameon=True,
    )



    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    if close_fig:
        plt.close(fig)
    return fig


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    p.add_argument(
        "--trades-dir",
        type=Path,
        default=root / "results_merged" / "trades",
        help="Root with YYYY-MM-DD folders containing trades_combo_*.csv",
    )
    p.add_argument(
        "--out-long",
        type=Path,
        default=root / "results_merged" / "brute_force_daily_combo_trade_stats.parquet",
        help="Output long table (.parquet or .csv by suffix)",
    )
    p.add_argument(
        "--monthly-csv",
        type=Path,
        default=None,
        help="Optional path to write monthly aggregate CSV",
    )
    p.add_argument(
        "--plot-png",
        type=Path,
        default=None,
        help="Optional path to save matplotlib combo chart",
    )
    p.add_argument(
        "--plot-line-pooled",
        action="store_true",
        help="Use pooled €/trade line instead of arithmetic mean of per-row ratios",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel day workers (1 = sequential). Uses processes.",
    )
    args = p.parse_args()

    long_df = build_long_table(args.trades_dir, max_workers=max(1, args.workers))
    out_long = Path(args.out_long)
    out_long.parent.mkdir(parents=True, exist_ok=True)
    if out_long.suffix.lower() == ".parquet":
        try:
            long_df.to_parquet(out_long, index=False)
        except ImportError as e:
            raise SystemExit(
                "Writing Parquet requires pyarrow or fastparquet. "
                "Install pyarrow or use --out-long ...csv"
            ) from e
    else:
        long_df.to_csv(out_long, index=False)

    monthly = monthly_aggregate(long_df)
    if args.monthly_csv:
        args.monthly_csv.parent.mkdir(parents=True, exist_ok=True)
        monthly.to_csv(args.monthly_csv, index=False)
    if args.plot_png:
        plot_monthly_bars_and_line(
            monthly,
            args.plot_png,
            use_pooled_line=args.plot_line_pooled,
            close_fig=True,
        )

    print(f"Wrote {len(long_df):,} rows -> {out_long}")
    if args.monthly_csv:
        print(f"Monthly rows -> {args.monthly_csv}")
    if args.plot_png:
        print(f"Plot -> {args.plot_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
