"""
Statistics from per-day brute-force summaries (``summary_YYYY-MM-DD.csv``).

Each file lists many DA buy/sell-hour combinations with a ``profit`` column.
We aggregate per calendar day: max, median, 75% and 90% quantiles of ``profit``.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Iterable

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
    p = df["profit"].astype(float)
    return {
        "theoretical_max_profit": float(p.max()),
        "brute_median_profit": float(p.median()),
        "brute_q075_profit": float(p.quantile(0.75)),
        "brute_q090_profit": float(p.quantile(0.90)),
    }


def profit_stats_from_summary_file(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    return profit_stats_from_summary_df(df)


def brute_force_stats_for_delivery_days(
    delivery_days: Iterable[pd.Timestamp],
    *,
    results_dir: Path | None = None,
    on_missing: str = "warn",
) -> pd.DataFrame:
    """
    One row per delivery day with aggregated profit statistics.

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
