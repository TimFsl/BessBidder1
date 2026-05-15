"""
Build results_merged/trades by combining whole CSV files (not row-wise).

Per date folder YYYY-MM-DD:
  - If results_open_positions_V3/trades/<date>/ contains trades_combo_000.csv …
    trades_combo_299.csv (all 300 files), copy that entire day from V3 only.
  - Otherwise:
      trades_combo_000.csv … trades_combo_023.csv  → from V3 (fallback: results if missing in V3)
      trades_combo_024.csv … trades_combo_299.csv → from results
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

N_COMBOS = 300
V3_EXCLUSIVE_UPPER = 24  # combos 0..23 from V3; 24..299 from results when day not V3-complete


def combo_filename(combo_id: int) -> str:
    return f"trades_combo_{combo_id:03d}.csv"


def v3_day_is_complete(v3_day: Path) -> bool:
    if not v3_day.is_dir():
        return False
    for i in range(N_COMBOS):
        if not (v3_day / combo_filename(i)).is_file():
            return False
    return True


def iter_date_dirs(a: Path, b: Path) -> list[str]:
    names: set[str] = set()
    if a.is_dir():
        names.update(p.name for p in a.iterdir() if p.is_dir())
    if b.is_dir():
        names.update(p.name for p in b.iterdir() if p.is_dir())
    return sorted(names)


def copy_file(src: Path, dst: Path, *, dry_run: bool) -> bool:
    if not src.is_file():
        return False
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return True


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Per date: if V3 has all trades_combo_000..299, copy the whole day from V3; "
            "else copy combos 000-023 from V3 and 024-299 from results/trades (same date)."
        )
    )
    root = Path(__file__).resolve().parent
    p.add_argument(
        "--results-trades",
        type=Path,
        default=root / "results" / "trades",
        help="Base trades dir",
    )
    p.add_argument(
        "--v3-trades",
        type=Path,
        default=root / "results_open_positions_V3" / "trades",
        help="Open positions V3 trades dir",
    )
    p.add_argument(
        "--out-trades",
        type=Path,
        default=root / "results_merged" / "trades",
        help="Output trades root",
    )
    p.add_argument("--dry-run", action="store_true", help="Do not write files")
    p.add_argument("-q", "--quiet", action="store_true")
    args = p.parse_args()

    results_root: Path = args.results_trades
    v3_root: Path = args.v3_trades
    out_root: Path = args.out_trades

    if not v3_root.is_dir() and not results_root.is_dir():
        print(f"Neither {v3_root} nor {results_root} exists.", file=sys.stderr)
        return 1

    stats: dict[str, int] = {
        "days_v3_complete": 0,
        "days_split": 0,
        "files_copied_v3": 0,
        "files_copied_results": 0,
        "files_skipped_missing": 0,
        "files_fallback_results_early_combo": 0,
    }
    dates = iter_date_dirs(v3_root, results_root)

    for day in dates:
        v3_day = v3_root / day
        res_day = results_root / day
        out_day = out_root / day

        if v3_day_is_complete(v3_day):
            stats["days_v3_complete"] += 1
            for i in range(N_COMBOS):
                name = combo_filename(i)
                src = v3_day / name
                dst = out_day / name
                if copy_file(src, dst, dry_run=args.dry_run):
                    stats["files_copied_v3"] += 1
                else:
                    stats["files_skipped_missing"] += 1
                    if not args.quiet:
                        print(f"Missing (unexpected): {src}", file=sys.stderr)
            continue

        stats["days_split"] += 1
        for i in range(N_COMBOS):
            name = combo_filename(i)
            dst = out_day / name
            if i < V3_EXCLUSIVE_UPPER:
                src_v3 = v3_day / name if v3_day.is_dir() else None
                src_res = res_day / name if res_day.is_dir() else None
                if src_v3 and src_v3.is_file():
                    if copy_file(src_v3, dst, dry_run=args.dry_run):
                        stats["files_copied_v3"] += 1
                    else:
                        stats["files_skipped_missing"] += 1
                elif src_res and src_res.is_file():
                    if copy_file(src_res, dst, dry_run=args.dry_run):
                        stats["files_copied_results"] += 1
                        stats["files_fallback_results_early_combo"] += 1
                        if not args.quiet:
                            print(
                                f"Fallback {day}/{name}: V3 missing, used results",
                                file=sys.stderr,
                            )
                else:
                    stats["files_skipped_missing"] += 1
                    if not args.quiet:
                        print(f"Missing both sources: {day}/{name}", file=sys.stderr)
            else:
                src = res_day / name if res_day.is_dir() else None
                if src and src.is_file():
                    if copy_file(src, dst, dry_run=args.dry_run):
                        stats["files_copied_results"] += 1
                else:
                    stats["files_skipped_missing"] += 1
                    if not args.quiet:
                        print(f"Missing results file: {day}/{name}", file=sys.stderr)

    if not args.quiet:
        action = "Would write" if args.dry_run else "Wrote"
        print(f"{action} under {out_root}")
        for k, v in stats.items():
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
