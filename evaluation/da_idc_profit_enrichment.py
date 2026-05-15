"""
Day-ahead vs intraday (IDC) profit splits from ``trades_YYYY-MM-DD.csv``.

Rolling-intrinsic ``profit.csv`` rows are **total** €; DA vs IDC comes from summing
``profit`` in trades where execution is the DA auction time (**13:00** Europe/Berlin)
on **delivery day - 1** vs not (same rule as
:func:`evaluation.coordinated_rl_test_plots.daily_da_idc_profit_from_trades_file`).

Use :func:`profit_comparison_summary` for a compact table (seven strategy columns × four
stat rows: mean total / mean DA / mean IDC / std of total daily €).
"""

from __future__ import annotations

import sys
import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluation.coordinated_rl_test_plots import (
    build_daily_da_idc_profit_table,
    daily_da_idc_profit_from_trades_file,
)


def _delivery_day_berlin(day_series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(day_series, utc=True, errors="coerce")
    return ts.dt.tz_convert("Europe/Berlin").dt.normalize()


def enrich_rolling_intrinsic_profit_csv(
    profit_csv: Path | str,
    trades_dir: Path | str,
    *,
    output_path: Path | str | None = None,
    on_missing_trades: str = "warn",
) -> pd.DataFrame:
    """
    Read a rolling-intrinsic ``profit.csv`` (columns ``day``, ``profit``, …), append
    ``da_profit``, ``idc_profit``, ``da_profit_share``, ``idc_profit_share`` from
    ``trades_dir/trades_YYYY-MM-DD.csv``.

    Shares use the CSV ``profit`` column as denominator (like ``da_profit_share`` in
    brute-force summary enrichment).

    Parameters
    ----------
    profit_csv
        Path to ``profit.csv``.
    trades_dir
        Directory containing ``trades_*.csv`` (same layout as next to ``profit.csv``).
    output_path
        If set, write the enriched table to this CSV path.
    on_missing_trades
        ``"warn"``, ``"skip"`` (silent NaN rows), or ``"raise"``.
    """
    p = Path(profit_csv)
    td = Path(trades_dir)
    df = pd.read_csv(p)
    if "day" not in df.columns or "profit" not in df.columns:
        raise ValueError(f"{p}: expected columns 'day' and 'profit'")

    da_vals: list[float] = []
    idc_vals: list[float] = []
    for _, row in df.iterrows():
        dd = _delivery_day_berlin(pd.Series([row["day"]])).iloc[0]
        if pd.isna(dd):
            da_vals.append(np.nan)
            idc_vals.append(np.nan)
            continue
        fp = td / f"trades_{dd.date().isoformat()}.csv"
        if not fp.is_file():
            msg = f"Missing trades file: {fp}"
            if on_missing_trades == "raise":
                raise FileNotFoundError(msg)
            if on_missing_trades == "warn":
                warnings.warn(msg, stacklevel=2)
            da_vals.append(np.nan)
            idc_vals.append(np.nan)
            continue
        da_e, idc_e = daily_da_idc_profit_from_trades_file(fp)
        da_vals.append(da_e)
        idc_vals.append(idc_e)

    out = df.copy()
    out["da_profit"] = da_vals
    out["idc_profit"] = idc_vals
    profit_num = pd.to_numeric(out["profit"], errors="coerce")
    da_num = pd.to_numeric(out["da_profit"], errors="coerce")
    idc_num = pd.to_numeric(out["idc_profit"], errors="coerce")
    out["da_profit_share"] = np.where(
        profit_num.abs() > 1e-9,
        da_num / profit_num,
        np.nan,
    )
    out["idc_profit_share"] = np.where(
        profit_num.abs() > 1e-9,
        idc_num / profit_num,
        np.nan,
    )
    out["da_profit_share"] = pd.to_numeric(out["da_profit_share"], errors="coerce").round(2)
    out["idc_profit_share"] = pd.to_numeric(out["idc_profit_share"], errors="coerce").round(2)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_path, index=False)

    return out


def attach_da_idc_to_comparison_df(
    df: pd.DataFrame,
    *,
    delivery_col: str = "delivery_day",
    rl_trades_dir: Path | str | None = None,
    myopic_trades_dir: Path | str | None = None,
    single_market_trades_dir: Path | str | None = None,
    on_missing: str = "warn",
) -> pd.DataFrame:
    """
    Left-merge DA / IDC profit columns from trades onto a test comparison frame.

    For each configured source, expects a total-profit column in ``df`` and adds
    ``<col>_da``, ``<col>_idc``, ``<col>_da_share``, ``<col>_idc_share`` (shares vs
    that total column).

    Typical columns: ``profit_rl_coordinated``, ``profit_myopic``,
    ``profit_single_market_ri``. Omit any ``*_trades_dir`` to skip that source.
    """
    if delivery_col not in df.columns:
        raise ValueError(f"df must contain {delivery_col!r}")

    out = df.copy()
    specs: list[tuple[str, Path | None]] = [
        ("profit_rl_coordinated", Path(rl_trades_dir) if rl_trades_dir is not None else None),
        ("profit_myopic", Path(myopic_trades_dir) if myopic_trades_dir is not None else None),
        (
            "profit_single_market_ri",
            Path(single_market_trades_dir) if single_market_trades_dir is not None else None,
        ),
    ]

    for profit_col, td in specs:
        if td is None:
            continue
        if profit_col not in out.columns:
            warnings.warn(
                f"Skipping DA/IDC merge for {profit_col!r}: column missing from df.",
                stacklevel=2,
            )
            continue

        tbl = build_daily_da_idc_profit_table(
            td, out[delivery_col], on_missing=on_missing
        )
        if tbl.empty:
            warnings.warn(f"No DA/IDC rows built from trades under {td}", stacklevel=2)
            continue

        t2 = tbl.rename(
            columns={
                "profit_da_eur": f"{profit_col}_da",
                "profit_idc_eur": f"{profit_col}_idc",
            }
        ).drop(columns=["profit_total_eur"], errors="ignore")

        out = out.merge(t2, on=delivery_col, how="left")
        tot = pd.to_numeric(out[profit_col], errors="coerce")
        da = pd.to_numeric(out[f"{profit_col}_da"], errors="coerce")
        idc = pd.to_numeric(out[f"{profit_col}_idc"], errors="coerce")
        out[f"{profit_col}_da_share"] = np.where(
            tot.abs() > 1e-9,
            da / tot,
            np.nan,
        )
        out[f"{profit_col}_idc_share"] = np.where(
            tot.abs() > 1e-9,
            idc / tot,
            np.nan,
        )
        out[f"{profit_col}_da_share"] = pd.to_numeric(
            out[f"{profit_col}_da_share"], errors="coerce"
        ).round(2)
        out[f"{profit_col}_idc_share"] = pd.to_numeric(
            out[f"{profit_col}_idc_share"], errors="coerce"
        ).round(2)

    return out


def day_ahead_profit_view(df: pd.DataFrame) -> pd.DataFrame:
    """
    Copy where ``profit_rl_coordinated``, ``profit_myopic``, ``profit_single_market_ri``
    are replaced by their ``*_da`` columns when present; other columns unchanged
    (e.g. brute-force stats stay full-day aggregates).
    """
    out = df.copy()
    for tot, da_col in (
        ("profit_rl_coordinated", "profit_rl_coordinated_da"),
        ("profit_myopic", "profit_myopic_da"),
        ("profit_single_market_ri", "profit_single_market_ri_da"),
    ):
        if da_col in out.columns and tot in out.columns:
            out[tot] = out[da_col]
    return out


def intraday_profit_view(df: pd.DataFrame) -> pd.DataFrame:
    """Same as :func:`day_ahead_profit_view` but using ``*_idc`` columns."""
    out = df.copy()
    for tot, idc_col in (
        ("profit_rl_coordinated", "profit_rl_coordinated_idc"),
        ("profit_myopic", "profit_myopic_idc"),
        ("profit_single_market_ri", "profit_single_market_ri_idc"),
    ):
        if idc_col in out.columns and tot in out.columns:
            out[tot] = out[idc_col]
    return out


# Columns for :func:`profit_comparison_summary` (must match ``df_test`` / ``df_test_bf`` names).
MODEL_TOTAL_COLUMNS: tuple[str, ...] = (
    "profit_rl_coordinated",
    "profit_myopic",
    "profit_single_market_ri",
)
BRUTE_TOTAL_STATS: tuple[str, ...] = (
    "theoretical_max_profit",
    "brute_median_profit",
    "brute_q075_profit",
    "brute_q090_profit",
)
BRUTE_DA_STATS: tuple[str, ...] = (
    "theoretical_max_da_profit",
    "brute_median_da_profit",
    "brute_q075_da_profit",
    "brute_q090_da_profit",
)
BRUTE_IDC_STATS: tuple[str, ...] = (
    "theoretical_max_idc_profit",
    "brute_median_idc_profit",
    "brute_q075_idc_profit",
    "brute_q090_idc_profit",
)
BRUTE_STAT_COLUMNS: tuple[str, ...] = (
    BRUTE_TOTAL_STATS + BRUTE_DA_STATS + BRUTE_IDC_STATS
)
DEFAULT_SUMMARY_COLUMN_LABELS: dict[str, str] = {
    "profit_rl_coordinated": "Coordinated multi-market",
    "profit_myopic": "Myopic multi-market",
    "profit_single_market_ri": "Single market (rolling intrinsic)",
    # Brute display names (also accepted as keys in ``column_labels``):
    "theoretical_max_profit": "Daily max (brute)",
    "brute_median_profit": "Daily median (brute)",
    "brute_q075_profit": "Daily q75 (brute)",
    "brute_q090_profit": "Daily q90 (brute)",
}

# One display column per brute level: (total, da, idc source cols, header).
BRUTE_SUMMARY_GROUPS: tuple[tuple[str, str, str, str], ...] = (
    (
        "theoretical_max_profit",
        "theoretical_max_da_profit",
        "theoretical_max_idc_profit",
        "Daily max (brute)",
    ),
    (
        "brute_median_profit",
        "brute_median_da_profit",
        "brute_median_idc_profit",
        "Daily median (brute)",
    ),
    (
        "brute_q075_profit",
        "brute_q075_da_profit",
        "brute_q075_idc_profit",
        "Daily q75 (brute)",
    ),
    (
        "brute_q090_profit",
        "brute_q090_da_profit",
        "brute_q090_idc_profit",
        "Daily q90 (brute)",
    ),
)


def profit_comparison_summary(
    df: pd.DataFrame,
    *,
    model_columns: Sequence[str] | None = None,
    column_labels: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Compact comparison table with **at most seven columns** (in order):

    1. Coordinated multi-market, 2. Myopic multi-market,
    3. Single market (rolling intrinsic) — if the corresponding ``profit_*`` column exists;
    4. Daily max / median / q75 / q90 (brute) — if the underlying triple of columns exists.

    **Rows** (same for every column present):

    - ``mean_total_profit`` — mean daily total €.
    - ``mean_da_profit`` — mean daily day-ahead € (``*_da`` for models; brute DA stat column).
    - ``mean_idc_profit`` — mean daily intraday € (``*_idc`` or brute IDC stat).
    - ``total_profit_std`` — sample std (``ddof=1``) of the **total** daily € series only.

    Brute DA/IDC columns come from
    :func:`evaluation.brute_force_summary_stats.profit_stats_from_summary_df`.
    """
    mc = tuple(model_columns) if model_columns is not None else MODEL_TOTAL_COLUMNS
    labels = {**DEFAULT_SUMMARY_COLUMN_LABELS, **(column_labels or {})}

    row_index = [
        "mean_total_profit",
        "mean_da_profit",
        "mean_idc_profit",
        "total_profit_std",
    ]
    mat: dict[str, list[float]] = {}

    for c in mc:
        if c not in df.columns:
            continue
        header = labels.get(c, c)
        ser = pd.to_numeric(df[c], errors="coerce")
        n_valid = int(ser.notna().sum())
        std_s = float(ser.std(ddof=1)) if n_valid > 1 else float("nan")

        da_c = f"{c}_da"
        idc_c = f"{c}_idc"
        if da_c in df.columns:
            mean_da = float(pd.to_numeric(df[da_c], errors="coerce").mean())
        else:
            mean_da = float("nan")
        if idc_c in df.columns:
            mean_idc = float(pd.to_numeric(df[idc_c], errors="coerce").mean())
        else:
            mean_idc = float("nan")
        mean_tot = float(ser.mean())
        mat[header] = [mean_tot, mean_da, mean_idc, std_s]

    for tot_c, da_c, idc_c, header in BRUTE_SUMMARY_GROUPS:
        if tot_c not in df.columns:
            continue
        disp = labels.get(tot_c, labels.get(header, header))
        ser = pd.to_numeric(df[tot_c], errors="coerce")
        n_valid = int(ser.notna().sum())
        std_s = float(ser.std(ddof=1)) if n_valid > 1 else float("nan")
        mean_tot = float(ser.mean())
        if da_c in df.columns:
            mean_da = float(pd.to_numeric(df[da_c], errors="coerce").mean())
        else:
            mean_da = float("nan")
        if idc_c in df.columns:
            mean_idc = float(pd.to_numeric(df[idc_c], errors="coerce").mean())
        else:
            mean_idc = float("nan")
        mat[disp] = [mean_tot, mean_da, mean_idc, std_s]

    if not mat:
        raise ValueError(
            "No summary columns found in df (expected model and/or brute profit columns)."
        )

    return pd.DataFrame(mat, index=row_index)
