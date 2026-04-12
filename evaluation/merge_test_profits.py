"""
Join daily rolling-intrinsic profits for coordinated RL, myopic, and single-market RI.

The RL ``profit.csv`` defines the calendar days (test set as produced by your pipeline).
Myopic and single-market baselines are aligned on ``delivery_day`` (Europe/Berlin midnight),
so +01:00 / +02:00 offsets in the CSV do not break the merge.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


# Allow `python evaluation/merge_test_profits.py` and notebook imports of `evaluation.*`
_rr = _repo_root()
if str(_rr) not in sys.path:
    sys.path.insert(0, str(_rr))

from evaluation.rl_model_registry import paths_for_coordinated_model


def _delivery_day_berlin(day_series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(day_series, utc=True, errors="coerce")
    return ts.dt.tz_convert("Europe/Berlin").dt.normalize()


def load_profit_csv(path: str | Path) -> pd.DataFrame:
    """Read profit.csv; keep day + profit only."""
    p = Path(path)
    df = pd.read_csv(p, parse_dates=["day"])
    if "profit" not in df.columns:
        raise ValueError(f"{p}: expected column 'profit'")
    out = df[["day", "profit"]].copy()
    out["delivery_day"] = _delivery_day_berlin(out["day"])
    if out["delivery_day"].isna().any():
        raise ValueError(f"{p}: unparseable 'day' values")
    return out


def load_single_market_profits_from_glob(pattern: str) -> pd.DataFrame:
    """
    Load and concatenate multiple profit.csv files (e.g. one per calendar year).

    Example pattern (repo root), legacy multi-year layout::
        output/single_market/rolling_intrinsic/ri_basic/qh/*/bs15cr1rto0.86mc365mt10/profit.csv

    Prefer a single consolidated file via ``single_market_profit_csv`` (e.g.
    ``output/single_market/profit.csv``) when available.
    """
    root = _repo_root()
    paths = sorted(root.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No files matched glob relative to repo root: {pattern!r} "
            f"(resolved under {root})"
        )
    parts = [load_profit_csv(p) for p in paths]
    all_df = pd.concat(parts, ignore_index=True)
    # If the same calendar day appears twice, keep the last file order (sorted paths)
    all_df = all_df.drop_duplicates(subset=["delivery_day"], keep="last")
    return all_df


def build_test_comparison_df(
    rl_profit_csv: str | Path,
    myopic_profit_csv: str | Path,
    single_market_profit_csv: str | Path | None = None,
    *,
    single_market_glob: str | None = None,
    rl_col: str = "profit_rl_coordinated",
    myopic_col: str = "profit_myopic",
    single_col: str = "profit_single_market_ri",
) -> pd.DataFrame:
    """
    One row per delivery day in the RL profit file, with baseline profits joined.

    Parameters
    ----------
    rl_profit_csv
        Coordinated RL rolling-intrinsic output (defines which days are in scope).
    myopic_profit_csv
        Myopic multi-market RI profit.csv (full history is fine; joined on day).
    single_market_profit_csv
        Optional explicit path to one single-market profit.csv (e.g. .../qh/2019/.../profit.csv).
    single_market_glob
        Optional glob **relative to repo root** to merge several yearly single-market files.
        If given, ``single_market_profit_csv`` is ignored.

    Returns
    -------
    DataFrame with columns: delivery_day, day_rl (original RL timestamp string column as in CSV),
    and the three profit columns (NaN where a baseline has no row for that day).
    """
    rl = load_profit_csv(rl_profit_csv)
    myopic = load_profit_csv(myopic_profit_csv)

    if single_market_glob:
        single = load_single_market_profits_from_glob(single_market_glob)
    elif single_market_profit_csv is not None:
        single = load_profit_csv(single_market_profit_csv)
    else:
        raise ValueError(
            "Provide either single_market_profit_csv or single_market_glob "
            "(single-market results are often split by year folder)."
        )

    base = rl.rename(columns={"profit": rl_col, "day": "day_rl"})[
        ["delivery_day", "day_rl", rl_col]
    ].copy()

    m = myopic.rename(columns={"profit": myopic_col})[
        ["delivery_day", myopic_col]
    ]
    s = single.rename(columns={"profit": single_col})[
        ["delivery_day", single_col]
    ]

    out = base.merge(m, on="delivery_day", how="left")
    out = out.merge(s, on="delivery_day", how="left")
    out = out.sort_values("delivery_day").reset_index(drop=True)
    return out


def default_paths_bs15(
    model_number: str | int = "0",
    repo_root: Path | None = None,
    *,
    bs_folder: str = "bs15cr1rto0.86mc365mt10",
) -> dict[str, str]:
    """
    Paths for the bs15 bundle; RL path uses ``logging/<model_number>/``.

    Model numbers and training ranges: see ``evaluation.rl_model_registry``.
    """
    p = paths_for_coordinated_model(model_number, repo_root=repo_root, bs_folder=bs_folder)
    spec = p["spec"]
    return {
        "model_number": spec.model_number,
        "train_start": spec.train_start.isoformat(),
        "train_end_inclusive": spec.train_end_inclusive.isoformat(),
        "rl": str(p["rl"]),
        "myopic": str(p["myopic"]),
        "single_glob": str(p["single_glob"]),
        "single_file": str(p["single_file"]),
    }


if __name__ == "__main__":
    import os

    mid = os.environ.get("RL_MODEL_NUMBER", "0")
    paths = default_paths_bs15(mid)
    print("Model", paths["model_number"], "train", paths["train_start"], "→", paths["train_end_inclusive"])
    single_csv = Path(paths["single_file"])
    if single_csv.is_file():
        df = build_test_comparison_df(
            paths["rl"],
            paths["myopic"],
            single_market_profit_csv=paths["single_file"],
        )
    else:
        try:
            df = build_test_comparison_df(
                paths["rl"],
                paths["myopic"],
                single_market_glob=paths["single_glob"],
            )
        except FileNotFoundError:
            df = build_test_comparison_df(
                paths["rl"],
                paths["myopic"],
                single_market_profit_csv=paths["single_file"],
            )
    print(df.head())
    print("rows:", len(df), "non-null single_market:", df["profit_single_market_ri"].notna().sum())
