"""
Load scalar curves from coordinated RL TensorBoard runs (Stable-Baselines3).

Training logs ``episode_profit/combined_eur`` from :meth:`CustomPPO.collect_rollouts`
(step = environment timestep at log time).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tensorboard.backend.event_processing import event_accumulator

EPISODE_PROFIT_COMBINED_TAG = "episode_profit/combined_eur"


def _accumulator_for_run(logdir: Path) -> event_accumulator.EventAccumulator:
    logdir = Path(logdir)
    if not logdir.is_dir():
        raise FileNotFoundError(
            f"TensorBoard log directory does not exist: {logdir}. "
            "Expected a folder containing events.out.tfevents.* files."
        )
    ea = event_accumulator.EventAccumulator(
        str(logdir),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    ea.Reload()
    return ea


def list_tensorboard_scalar_tags(logdir: Path | str) -> list[str]:
    """All scalar tags in a run (sorted)."""
    ea = _accumulator_for_run(Path(logdir))
    return sorted(ea.Tags().get("scalars", []))


def load_tensorboard_scalar(logdir: Path | str, tag: str) -> pd.DataFrame:
    """
    One scalar series as a DataFrame with columns ``step``, ``wall_time``, ``value``.

    ``step`` is the TensorBoard step (for SB3 training metrics, typically the env timestep).
    """
    ea = _accumulator_for_run(Path(logdir))
    scalars = ea.Tags().get("scalars", [])
    if tag not in scalars:
        preview = ", ".join(scalars[:30])
        more = f" … (+{len(scalars) - 30} more)" if len(scalars) > 30 else ""
        raise KeyError(
            f"Scalar tag {tag!r} not found under {logdir!s}. "
            f"Example tags: {preview}{more}"
        )
    events = ea.Scalars(tag)
    return pd.DataFrame(
        [(e.step, e.wall_time, e.value) for e in events],
        columns=["step", "wall_time", "value"],
    )
