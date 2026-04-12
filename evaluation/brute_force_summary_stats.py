"""
Statistics from per-day brute-force summaries (``summary_YYYY-MM-DD.csv``).

Each file lists many DA buy/sell-hour combinations with a ``profit`` column.
We aggregate per calendar day: max, median, 75% and 90% quantiles of ``profit``.

If ``da_profit`` is present (enriched summaries), the same aggregates are computed for
``da_profit`` and for ``idc_profit = profit - da_profit`` per row.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def default_results_merged_dir(repo_root: Path | None = None) -> Path:
    return (repo_root or _REPO) / "coordinated_market_upper_bound_analysis" / "results_merged"


def summary_csv_path(delivery_day: pd.Timestamp, results_dir: Path | None = None) -> Path:
    """``summary_YYYY-MM-DD.csv`` for a Berlin-normalized delivery day."""
    rd = results_dir or default_results_merged_dir()
    d = pd.Timestamp(delivery_day).tz_convert("Europe/Berlin").normalize().date()
    return rd / f"summary_{d.isoformat()}.csv"


def profit_stats_from_summary_df(df: pd.DataFrame) -> dict[str, float]:
    if "profit" not in df.columns:
        raise ValueError("Expected column 'profit' in summary CSV")
    p = pd.to_numeric(df["profit"], errors="coerce")
    out: dict[str, float] = {
        "theoretical_max_profit": float(p.max()),
        "brute_median_profit": float(p.median()),
        "brute_q075_profit": float(p.quantile(0.75)),
        "brute_q090_profit": float(p.quantile(0.90)),
    }
    if "da_profit" in df.columns:
        da = pd.to_numeric(df["da_profit"], errors="coerce")
        idc = p - da
        out.update(
            {
                "theoretical_max_da_profit": float(da.max()),
                "brute_median_da_profit": float(da.median()),
                "brute_q075_da_profit": float(da.quantile(0.75)),
                "brute_q090_da_profit": float(da.quantile(0.90)),
                "theoretical_max_idc_profit": float(idc.max()),
                "brute_median_idc_profit": float(idc.median()),
                "brute_q075_idc_profit": float(idc.quantile(0.75)),
                "brute_q090_idc_profit": float(idc.quantile(0.90)),
            }
        )
    return out


def profit_stats_from_summary_file(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    return profit_stats_from_summary_df(df)


def _rows_eligible_for_best_da_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove the **no day-ahead** scenario from pattern ranking: ``combo_id == 0`` and/or
    ``kind == no_da`` (as in ``summary_*.csv`` from the brute-force analysis).
    """
    if df.empty:
        return df
    drop = pd.Series(False, index=df.index)
    if "combo_id" in df.columns:
        cid = pd.to_numeric(df["combo_id"], errors="coerce")
        drop |= cid == 0
    if "kind" in df.columns:
        drop |= df["kind"].astype(str).str.strip().str.lower() == "no_da"
    if not drop.any():
        return df
    return df.loc[~drop].copy()


def best_mean_profit_da_pattern_over_days(
    delivery_days: Iterable[pd.Timestamp],
    *,
    results_dir: Path | None = None,
    on_missing: str = "warn",
) -> tuple[float | None, dict[str, Any]]:
    """
    Fixed DA-pattern baseline: same scenario rows in each ``summary_*.csv``. For each
    pattern, take the mean ``profit`` across the given delivery days, then return the
    **largest** of those means (and which pattern won).

    Rows with **no DA auction** are excluded from the competition: ``combo_id == 0``
    and/or ``kind == no_da``.

    If a ``kind`` column exists (e.g. ``buy_only`` / ``buy_sell``), it is included in the
    pattern key so rows are not merged incorrectly.
    """
    rd = results_dir or default_results_merged_dir()
    frames: list[pd.DataFrame] = []
    for d in delivery_days:
        dd = pd.Timestamp(d).tz_convert("Europe/Berlin").normalize()
        pth = summary_csv_path(dd, rd)
        if not pth.is_file():
            msg = f"Missing brute-force summary: {pth}"
            if on_missing == "raise":
                raise FileNotFoundError(msg)
            if on_missing == "warn":
                warnings.warn(msg, stacklevel=2)
            continue
        df = pd.read_csv(pth)
        if "profit" not in df.columns or "buy_hour" not in df.columns:
            raise ValueError(
                f"{pth}: expected columns 'profit' and 'buy_hour' "
                f"(and usually 'sell_hour')"
            )
        if "sell_hour" not in df.columns:
            df = df.copy()
            df["sell_hour"] = pd.NA
        frames.append(df)

    if not frames:
        return None, {}

    long = pd.concat(frames, ignore_index=True)
    long["profit"] = pd.to_numeric(long["profit"], errors="coerce")
    long = long.loc[long["profit"].notna()].copy()
    long = _rows_eligible_for_best_da_pattern(long)
    if long.empty:
        return None, {"n_days_loaded": 0, "n_rows_used": 0}

    group_cols: list[str] = []
    if "kind" in long.columns:
        long["kind"] = long["kind"].astype(str)
        group_cols.append("kind")
    long["buy_hour"] = pd.to_numeric(long["buy_hour"], errors="coerce")
    long["sell_hour"] = pd.to_numeric(long["sell_hour"], errors="coerce")
    group_cols.extend(["buy_hour", "sell_hour"])

    means = long.groupby(group_cols, dropna=False)["profit"].mean()
    if means.empty:
        return None, {"n_days_loaded": len(frames), "n_rows_used": len(long)}

    best_mean = float(means.max())
    best_key = means.idxmax()
    if isinstance(best_key, tuple):
        key_iter = iter(best_key)
    else:
        key_iter = iter((best_key,))
    info: dict[str, Any] = {
        "n_days_loaded": len(frames),
        "n_rows_used": int(len(long)),
        "n_unique_patterns": int(means.shape[0]),
    }
    for col in group_cols:
        info[col] = next(key_iter)
    return best_mean, info


def mean_profit_for_pattern_over_days(
    pattern_info: dict[str, Any],
    delivery_days: Iterable[pd.Timestamp],
    *,
    results_dir: Path | None = None,
    on_missing: str = "warn",
    profit_column: str = "profit",
) -> float | None:
    """
    Mean ``profit`` for one fixed DA pattern (``kind`` / ``buy_hour`` / ``sell_hour`` from
    :func:`best_mean_profit_da_pattern_over_days`) evaluated on each listed delivery day.

    Use this to score the **train-winning** pattern on the **validation** calendar days only.
    """
    if "buy_hour" not in pattern_info:
        return None

    rd = results_dir or default_results_merged_dir()
    profits: list[float] = []

    for d in delivery_days:
        dd = pd.Timestamp(d).tz_convert("Europe/Berlin").normalize()
        pth = summary_csv_path(dd, rd)
        if not pth.is_file():
            msg = f"Missing brute-force summary: {pth}"
            if on_missing == "raise":
                raise FileNotFoundError(msg)
            if on_missing == "warn":
                warnings.warn(msg, stacklevel=2)
            continue
        df = pd.read_csv(pth)
        if profit_column not in df.columns or "buy_hour" not in df.columns:
            raise ValueError(f"{pth}: need {profit_column!r} and 'buy_hour'")
        if "sell_hour" not in df.columns:
            df = df.copy()
            df["sell_hour"] = pd.NA

        sub = _summary_rows_matching_pattern(df, pattern_info)
        if sub.empty:
            warnings.warn(
                f"No summary row matches train pattern in {pth.name} "
                f"(kind={pattern_info.get('kind')!r} buy={pattern_info.get('buy_hour')!r} "
                f"sell={pattern_info.get('sell_hour')!r})",
                stacklevel=2,
            )
            continue
        if len(sub) > 1:
            warnings.warn(
                f"{pth.name}: {len(sub)} rows match pattern; using the first.",
                stacklevel=2,
            )
        pv = pd.to_numeric(sub[profit_column].iloc[0], errors="coerce")
        if pd.notna(pv):
            profits.append(float(pv))

    if not profits:
        return None
    return float(sum(profits) / len(profits))


def _summary_rows_matching_pattern(
    df: pd.DataFrame, pattern_info: dict[str, Any]
) -> pd.DataFrame:
    """Single-day summary slice matching train-fit winning keys."""
    m = pd.Series(True, index=df.index)
    if (
        "kind" in pattern_info
        and pd.notna(pattern_info.get("kind"))
        and str(pattern_info["kind"])
        and "kind" in df.columns
    ):
        m &= df["kind"].astype(str) == str(pattern_info["kind"])

    bh = float(pattern_info["buy_hour"])
    bh_s = pd.to_numeric(df["buy_hour"], errors="coerce")
    m &= (bh_s - bh).abs() < 1e-6

    sh_pat = pattern_info.get("sell_hour")
    sh_col = pd.to_numeric(df["sell_hour"], errors="coerce")
    if pd.isna(sh_pat):
        m &= df["sell_hour"].isna() | sh_col.isna()
    else:
        m &= (sh_col - float(sh_pat)).abs() < 1e-6

    return df.loc[m]


def brute_force_stats_for_delivery_days(
    delivery_days: Iterable[pd.Timestamp],
    *,
    results_dir: Path | None = None,
    on_missing: str = "warn",
) -> pd.DataFrame:
    """
    One row per delivery day with aggregated profit statistics (total, and DA/IDC if
    ``da_profit`` exists in each summary file).

    Parameters
    ----------
    delivery_days
        Iterable of timestamps (any tz); normalized to Berlin date for the filename.
    results_dir
        Folder containing ``summary_*.csv``.
    on_missing
        ``"warn"`` (log and skip row), ``"skip"`` (silent), or ``"raise"``.
    """
    rd = results_dir or default_results_merged_dir()
    rows: list[dict] = []
    for d in delivery_days:
        dd = pd.Timestamp(d).tz_convert("Europe/Berlin").normalize()
        pth = summary_csv_path(dd, rd)
        if not pth.is_file():
            msg = f"Missing brute-force summary: {pth}"
            if on_missing == "raise":
                raise FileNotFoundError(msg)
            if on_missing == "warn":
                warnings.warn(msg, stacklevel=2)
            continue
        stats = profit_stats_from_summary_file(pth)
        stats["delivery_day"] = dd
        rows.append(stats)

    if not rows:
        return pd.DataFrame(
            columns=[
                "delivery_day",
                "theoretical_max_profit",
                "brute_median_profit",
                "brute_q075_profit",
                "brute_q090_profit",
                "theoretical_max_da_profit",
                "brute_median_da_profit",
                "brute_q075_da_profit",
                "brute_q090_da_profit",
                "theoretical_max_idc_profit",
                "brute_median_idc_profit",
                "brute_q075_idc_profit",
                "brute_q090_idc_profit",
            ]
        )
    out = pd.DataFrame(rows)
    return out.sort_values("delivery_day").reset_index(drop=True)


def attach_brute_force_stats(
    df_test: pd.DataFrame,
    *,
    results_dir: Path | None = None,
    on_missing: str = "warn",
) -> pd.DataFrame:
    """
    Left-merge brute-force daily stats onto a test comparison frame that has ``delivery_day``.
    """
    if "delivery_day" not in df_test.columns:
        raise ValueError("df_test must contain column 'delivery_day'")
    bf = brute_force_stats_for_delivery_days(
        df_test["delivery_day"].unique(),
        results_dir=results_dir,
        on_missing=on_missing,
    )
    return df_test.merge(bf, on="delivery_day", how="left")


def build_brute_force_long_cache(
    results_dir: Path | None = None,
    output_path: Path | None = None,
    *,
    glob_pattern: str = "summary_*.csv",
) -> Path:
    """
    Concatenate all per-day summaries into one table (adds ``summary_file_date``).

    Use this once if you repeatedly need full scenario-level data; for test-only
    enrichment, :func:`brute_force_stats_for_delivery_days` is enough.

    Returns path to the written file (Parquet if suffix ``.parquet``, else CSV).
    """
    rd = results_dir or default_results_merged_dir()
    if output_path is None:
        output_path = rd.parent / "results_merged_all_scenarios.parquet"

    paths = sorted(rd.glob(glob_pattern))
    if not paths:
        raise FileNotFoundError(f"No files matching {glob_pattern!r} under {rd}")

    parts: list[pd.DataFrame] = []
    for p in paths:
        # summary_2019-01-01.csv -> 2019-01-01
        stem = p.stem  # summary_2019-01-01
        date_part = stem.replace("summary_", "", 1)
        day = pd.Timestamp(date_part, tz="Europe/Berlin").normalize()
        chunk = pd.read_csv(p)
        chunk["summary_file_date"] = day
        parts.append(chunk)

    long_df = pd.concat(parts, ignore_index=True)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".parquet":
        long_df.to_parquet(output_path, index=False)
    else:
        long_df.to_csv(output_path, index=False)
    return output_path
