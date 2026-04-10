"""
Coordinated multi-market RL runs: model folder id under ``logging/<id>/`` and training calendar range.

Training ranges are **inclusive** calendar days in Europe/Berlin (as you specified).
For half-open intervals (e.g. matching ``config.py``), use ``train_end_exclusive``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

TZ = "Europe/Berlin"


def _d(iso_date: str) -> pd.Timestamp:
    return pd.Timestamp(iso_date, tz=TZ).normalize()


@dataclass(frozen=True)
class CoordinatedRLModelSpec:
    """One row in :data:`RL_MODEL_REGISTRY`."""

    model_number: str
    train_start: pd.Timestamp
    train_end_inclusive: pd.Timestamp

    @property
    def train_end_exclusive(self) -> pd.Timestamp:
        """First calendar instant *after* the last training day (Berlin)."""
        return self.train_end_inclusive + pd.Timedelta(days=1)

    def describe_training(self) -> str:
        s = self.train_start.strftime("%d.%m.%Y")
        e = self.train_end_inclusive.strftime("%d.%m.%Y")
        return f"{s}–{e} ({TZ})"


# Model id = subfolder name under output/coordinated_multi_market/logging/<id>/
RL_MODEL_REGISTRY: dict[str, CoordinatedRLModelSpec] = {
    "0": CoordinatedRLModelSpec("0", _d("2019-01-01"), _d("2021-03-31")),
    "1": CoordinatedRLModelSpec("1", _d("2019-01-01"), _d("2021-03-31")),
    "2": CoordinatedRLModelSpec("2", _d("2023-01-01"), _d("2025-03-01")),
    "3": CoordinatedRLModelSpec("3", _d("2023-01-01"), _d("2025-03-01")),
    "4": CoordinatedRLModelSpec("4", _d("2019-01-01"), _d("2024-09-30")),
    "5": CoordinatedRLModelSpec("5", _d("2019-01-01"), _d("2024-09-30")),
}


def get_rl_model_spec(model_number: str | int) -> CoordinatedRLModelSpec:
    key = str(model_number).strip()
    if key not in RL_MODEL_REGISTRY:
        raise KeyError(
            f"Unknown coordinated RL model_number {key!r}. "
            f"Known: {sorted(RL_MODEL_REGISTRY)}"
        )
    return RL_MODEL_REGISTRY[key]


def registry_as_dataframe() -> pd.DataFrame:
    rows = []
    for k in sorted(RL_MODEL_REGISTRY, key=lambda x: int(x)):
        s = RL_MODEL_REGISTRY[k]
        rows.append(
            {
                "model_number": s.model_number,
                "train_start": s.train_start,
                "train_end_inclusive": s.train_end_inclusive,
                "train_end_exclusive": s.train_end_exclusive,
            }
        )
    return pd.DataFrame(rows)


def rl_rolling_intrinsic_profit_path(
    model_number: str | int,
    *,
    repo_root: Path | None = None,
    ri_subdir: str = "rolling_intrinsic_intelligently_stacked_on_day_ahead_qh",
    bs_folder: str = "bs15cr1rto0.86mc365mt10",
) -> Path:
    """Path to coordinated RL ``profit.csv`` for one model run."""
    root = repo_root or Path(__file__).resolve().parents[1]
    spec = get_rl_model_spec(model_number)
    return (
        root
        / "output"
        / "coordinated_multi_market"
        / "logging"
        / spec.model_number
        / ri_subdir
        / bs_folder
        / "profit.csv"
    )


def rl_rolling_intrinsic_trades_dir(
    model_number: str | int,
    *,
    repo_root: Path | None = None,
    ri_subdir: str = "rolling_intrinsic_intelligently_stacked_on_day_ahead_qh",
    bs_folder: str = "bs15cr1rto0.86mc365mt10",
) -> Path:
    """Directory with ``trades_YYYY-MM-DD.csv`` next to ``profit.csv``."""
    return rl_rolling_intrinsic_profit_path(
        model_number,
        repo_root=repo_root,
        ri_subdir=ri_subdir,
        bs_folder=bs_folder,
    ).parent / "trades"


def baseline_paths_bs15(
    repo_root: Path | None = None,
    bs_folder: str = "bs15cr1rto0.86mc365mt10",
) -> dict[str, str]:
    """Myopic + single-market paths (not tied to RL model id)."""
    root = repo_root or Path(__file__).resolve().parents[1]
    return {
        "myopic": str(
            root
            / "output/myopic_multi_market/rolling_intrinsic_stacked_on_day_ahead_qh"
            / bs_folder
            / "profit.csv"
        ),
        "single_glob": f"output/single_market/rolling_intrinsic/ri_basic/qh/*/{bs_folder}/profit.csv",
        "single_file": str(
            root
            / "output/single_market/rolling_intrinsic/ri_basic/qh/2019"
            / bs_folder
            / "profit.csv"
        ),
    }


def paths_for_coordinated_model(
    model_number: str | int,
    *,
    repo_root: Path | None = None,
    bs_folder: str = "bs15cr1rto0.86mc365mt10",
) -> dict[str, Path | str | CoordinatedRLModelSpec]:
    """
    All paths needed for test-profit comparison + the selected :class:`CoordinatedRLModelSpec`.

    Keys: ``spec``, ``rl``, ``trades_dir``, ``myopic``, ``single_glob``, ``single_file``.
    """
    root = repo_root or Path(__file__).resolve().parents[1]
    spec = get_rl_model_spec(model_number)
    base = baseline_paths_bs15(repo_root=root, bs_folder=bs_folder)
    rl_profit = rl_rolling_intrinsic_profit_path(
        model_number, repo_root=root, bs_folder=bs_folder
    )
    return {
        "spec": spec,
        "rl": rl_profit,
        "trades_dir": rl_profit.parent / "trades",
        "myopic": base["myopic"],
        "single_glob": base["single_glob"],
        "single_file": base["single_file"],
    }
