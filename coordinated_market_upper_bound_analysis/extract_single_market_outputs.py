#!/usr/bin/env python3
"""
One-shot extraction for results_open_positions_V3:

1. Copy each trades/<YYYY-MM-DD>/trades_combo_000.csv to
   output/single_market/trades/trades_YYYY-MM-DD.csv

2. Concatenate rows with combo_id == 0 from each summary_YYYY-MM-DD.csv into
   output/single_market/profit.csv

Run: python3 coordinated_market_upper_bound_analysis/extract_single_market_outputs.py
"""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
REPO_ROOT = BASE.parent
TRADES_ROOT = BASE / "results_open_positions_V3" / "trades"
SUMMARY_ROOT = BASE / "results_open_positions_V3"
OUT = REPO_ROOT / "output" / "single_market"
OUT_TRADES = OUT / "trades"
PROFIT_CSV = OUT / "profit.csv"


def _combo_id_is_zero(value: str) -> bool:
    s = str(value).strip()
    try:
        return int(float(s)) == 0
    except ValueError:
        return False


def main() -> int:
    if not TRADES_ROOT.is_dir():
        print(f"Missing trades folder: {TRADES_ROOT}", file=sys.stderr)
        return 1
    if not SUMMARY_ROOT.is_dir():
        print(f"Missing summary folder: {SUMMARY_ROOT}", file=sys.stderr)
        return 1

    OUT_TRADES.mkdir(parents=True, exist_ok=True)

    date_dirs = sorted(p for p in TRADES_ROOT.iterdir() if p.is_dir())
    copied = 0
    missing_trades = 0
    for day_dir in date_dirs:
        src = day_dir / "trades_combo_000.csv"
        if not src.is_file():
            print(f"Skip (no trades_combo_000.csv): {day_dir.name}", file=sys.stderr)
            missing_trades += 1
            continue
        dst = OUT_TRADES / f"trades_{day_dir.name}.csv"
        shutil.copy2(src, dst)
        copied += 1

    summary_files = sorted(SUMMARY_ROOT.glob("summary_*.csv"))
    profit_rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    missing_summary_combo0 = 0

    for path in summary_files:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "combo_id" not in reader.fieldnames:
                print(f"Skip (no combo_id column): {path.name}", file=sys.stderr)
                continue
            if fieldnames is None:
                fieldnames = list(reader.fieldnames)
            found = False
            for row in reader:
                if row.get("combo_id") is not None and _combo_id_is_zero(row["combo_id"]):
                    profit_rows.append({k: row.get(k, "") for k in fieldnames})
                    found = True
                    break
            if not found:
                missing_summary_combo0 += 1
                print(f"Warning: no combo_id==0 row in {path.name}", file=sys.stderr)

    if not profit_rows or fieldnames is None:
        print("No summary rows with combo_id==0; profit.csv not written.", file=sys.stderr)
        return 1

    with PROFIT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(profit_rows)

    print(
        f"Done. Trades copied: {copied}"
        + (f" (missing trades_combo_000.csv in {missing_trades} date folders)" if missing_trades else "")
    )
    print(f"Profit rows: {len(profit_rows)} -> {PROFIT_CSV}")
    print(f"Trades output dir: {OUT_TRADES}")
    if missing_summary_combo0:
        print(f"Warnings: {missing_summary_combo0} summary files had no combo_id==0 row", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
