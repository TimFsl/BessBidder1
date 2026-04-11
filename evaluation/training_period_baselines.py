"""
Training-calendar baselines (no coordinated RL column): myopic, single-market RI, brute-force stats.

Rows are delivery days inside :attr:`CoordinatedRLModelSpec.train_start` …
``train_end_inclusive`` (Europe/Berlin), matched to myopic ``profit.csv``.
``is_validation_day`` flags days listed in the validation rollout log (random holdout from train).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluation.brute_force_summary_stats import attach_brute_force_stats
from evaluation.merge_test_profits import (
    load_profit_csv,
    load_single_market_profits_from_glob,
)
from evaluation.rl_model_registry import (
    baseline_paths_bs15,
    get_rl_model_spec,
    rl_validation_test_log_path,
)


def validation_delivery_days_from_test_log_csv(
    path: str | Path,
    *,
    time_column: str = "time",
) -> pd.DatetimeIndex:
    """Unique Berlin-normalized midnight timestamps from the validation env log."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Validation test log not found: {p}")
    df = pd.read_csv(p, parse_dates=[time_column])
    if time_column not in df.columns:
        raise ValueError(f"{p}: expected column {time_column!r}")
    ts = pd.to_datetime(df[time_column], utc=True, errors="coerce")
    if ts.isna().any():
        raise ValueError(f"{p}: unparseable values in {time_column!r}")
    days = ts.dt.tz_convert("Europe/Berlin").dt.normalize().unique()
    return pd.DatetimeIndex(sorted(days))


def build_training_period_baselines_df(
    model_number: str | int,
    *,
    repo_root: Path | None = None,
    bs_folder: str = "bs15cr1rto0.86mc365mt10",
    validation_test_log_path: str | Path | None = None,
    validation_checkpoint_subdir: str = "ppo_stacked_checkpoint_100000_steps",
    validation_log_filename: str = "basic_battery_dam_test_log_v3.csv",
    brute_results_dir: Path | None = None,
    on_missing_bf: str = "warn",
) -> pd.DataFrame:
    """
    One row per delivery day in the training window that appears in myopic ``profit.csv``,
    with single-market RI and brute-force columns merged. No RL profit column.

    ``is_validation_day`` is True when ``delivery_day`` appears in the validation log
    (unique calendar dates from ``time``).
    """
    root = repo_root or _REPO
    spec = get_rl_model_spec(model_number)
    t0, t1 = spec.train_start, spec.train_end_inclusive

    base_paths = baseline_paths_bs15(repo_root=root, bs_folder=bs_folder)
    myopic = load_profit_csv(base_paths["myopic"])
    mask = (myopic["delivery_day"] >= t0) & (myopic["delivery_day"] <= t1)
    myopic_win = myopic.loc[mask, ["delivery_day", "profit"]].rename(
        columns={"profit": "profit_myopic"}
    )

    if myopic_win.empty:
        warnings.warn(
            f"No myopic rows in training window {t0.date()}–{t1.date()} for model {model_number!r}.",
            stacklevel=2,
        )

    try:
        single = load_single_market_profits_from_glob(base_paths["single_glob"])
    except FileNotFoundError:
        single = load_profit_csv(base_paths["single_file"])
    single = single.rename(columns={"profit": "profit_single_market_ri"})[
        ["delivery_day", "profit_single_market_ri"]
    ]

    out = myopic_win.merge(single, on="delivery_day", how="left")
    out = attach_brute_force_stats(out, results_dir=brute_results_dir, on_missing=on_missing_bf)

    vpath = (
        Path(validation_test_log_path)
        if validation_test_log_path is not None
        else rl_validation_test_log_path(
            model_number,
            repo_root=root,
            checkpoint_subdir=validation_checkpoint_subdir,
            log_filename=validation_log_filename,
        )
    )
    val_days = validation_delivery_days_from_test_log_csv(vpath)
    out["is_validation_day"] = out["delivery_day"].isin(val_days)

    in_window = set(out["delivery_day"])
    missing_holdout = [d for d in val_days if d not in in_window]
    if missing_holdout:
        warnings.warn(
            f"{len(missing_holdout)} validation log day(s) have no myopic row in the training "
            f"window merge (e.g. {missing_holdout[0].date()}); they are omitted from this frame.",
            stacklevel=2,
        )

    return out.sort_values("delivery_day").reset_index(drop=True)


def split_training_baselines_train_vs_validation(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(train_fit, validation_holdout)`` using :attr:`is_validation_day`."""
    if "is_validation_day" not in df.columns:
        raise ValueError("df must contain is_validation_day (from build_training_period_baselines_df)")
    tr = df.loc[~df["is_validation_day"]].copy()
    va = df.loc[df["is_validation_day"]].copy()
    return tr, va


def mean_reference_values_train_fit(df: pd.DataFrame) -> dict[str, float]:
    """
    Column means on the **train-fit** subset only (``is_validation_day`` is False).

    Used as horizontal guides on TensorBoard-style plots (y only; x = timesteps).
    Omits keys whose column is missing or whose mean is NaN.
    """
    if "is_validation_day" not in df.columns:
        raise ValueError("df must contain is_validation_day")
    sub = df.loc[~df["is_validation_day"]]
    keys_cols = [
        ("mean_myopic_eur", "profit_myopic"),
        ("mean_single_market_ri_eur", "profit_single_market_ri"),
        ("mean_theoretical_max_eur", "theoretical_max_profit"),
        ("mean_brute_q075_eur", "brute_q075_profit"),
        ("mean_brute_q090_eur", "brute_q090_profit"),
    ]
    out: dict[str, float] = {}
    for name, col in keys_cols:
        if col not in sub.columns:
            continue
        m = sub[col].mean()
        if pd.notna(m):
            out[name] = float(m)
    return out


def mean_reference_values_validation_holdout(df: pd.DataFrame) -> dict[str, float]:
    """
    Column means on **validation holdout** rows only (``is_validation_day`` is True).

    Same keys as :func:`mean_reference_values_train_fit` for comparable horizontal guides.
    """
    if "is_validation_day" not in df.columns:
        raise ValueError("df must contain is_validation_day")
    sub = df.loc[df["is_validation_day"]]
    keys_cols = [
        ("mean_myopic_eur", "profit_myopic"),
        ("mean_single_market_ri_eur", "profit_single_market_ri"),
        ("mean_theoretical_max_eur", "theoretical_max_profit"),
        ("mean_brute_q075_eur", "brute_q075_profit"),
        ("mean_brute_q090_eur", "brute_q090_profit"),
    ]
    out: dict[str, float] = {}
    for name, col in keys_cols:
        if col not in sub.columns:
            continue
        m = sub[col].mean()
        if pd.notna(m):
            out[name] = float(m)
    return out
