#!/usr/bin/env python3
"""
Merge summary CSV files by replacing profit values from a second folder.

Default behavior:
  - read files from results/
  - read matching files from results_open_positions_V3/
  - write merged files to results_merged/
  - replace profit only where kind == "buy_only"
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_base_dir = script_dir / "results"
    default_override_dir = script_dir / "results_open_positions_V3"
    default_output_dir = script_dir / "results_merged"

    parser = argparse.ArgumentParser(
        description="Merge summary CSVs by replacing profit values."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=default_base_dir,
        help="Folder with base summary_*.csv files (default: results).",
    )
    parser.add_argument(
        "--override-dir",
        type=Path,
        default=default_override_dir,
        help="Folder with override summary_*.csv files (default: results_open_positions_V3).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Folder to write merged summary_*.csv files (default: results_merged).",
    )
    parser.add_argument(
        "--replace-kind",
        default="buy_only",
        help='Only replace rows with this kind value (default: "buy_only").',
    )
    parser.add_argument(
        "--replace-first-n",
        type=int,
        default=None,
        help="Optional alternative mode: replace first N data rows, regardless of kind.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row.")
        rows = list(reader)
        return reader.fieldnames, rows


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def merge_file(
    base_path: Path,
    override_path: Path,
    output_path: Path,
    replace_kind: str,
    replace_first_n: int | None,
) -> int:
    _, override_rows = read_csv_rows(override_path)
    base_fieldnames, base_rows = read_csv_rows(base_path)

    if "profit" not in base_fieldnames:
        raise ValueError(f'{base_path} has no "profit" column.')
    if not override_rows:
        raise ValueError(f"{override_path} has no data rows.")
    if "profit" not in override_rows[0]:
        raise ValueError(f'{override_path} has no "profit" column.')

    replacements = 0

    override_by_combo: dict[str, str] = {}
    has_combo_id = "combo_id" in base_fieldnames and "combo_id" in override_rows[0]
    if has_combo_id:
        for row in override_rows:
            override_by_combo[row["combo_id"]] = row["profit"]

    for idx, base_row in enumerate(base_rows):
        should_replace = False
        if replace_first_n is not None:
            should_replace = idx < replace_first_n
        else:
            should_replace = base_row.get("kind") == replace_kind

        if not should_replace:
            continue

        new_profit: str | None = None
        if has_combo_id:
            combo_id = base_row.get("combo_id")
            if combo_id in override_by_combo:
                new_profit = override_by_combo[combo_id]

        if new_profit is None and idx < len(override_rows):
            new_profit = override_rows[idx]["profit"]

        if new_profit is None:
            continue

        base_row["profit"] = new_profit
        replacements += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv_rows(output_path, base_fieldnames, base_rows)
    return replacements


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir
    override_dir = args.override_dir
    output_dir = args.output_dir
    replace_kind = args.replace_kind
    replace_first_n = args.replace_first_n

    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")
    if not override_dir.exists():
        raise FileNotFoundError(f"Override directory not found: {override_dir}")

    base_files = sorted(base_dir.glob("summary_*.csv"))
    if not base_files:
        raise FileNotFoundError(f'No files matching "summary_*.csv" in {base_dir}')

    total_replacements = 0
    processed = 0
    skipped = 0

    for base_path in base_files:
        override_path = override_dir / base_path.name
        if not override_path.exists():
            skipped += 1
            print(f"SKIP (missing override): {base_path.name}")
            continue

        output_path = output_dir / base_path.name
        replacements = merge_file(
            base_path=base_path,
            override_path=override_path,
            output_path=output_path,
            replace_kind=replace_kind,
            replace_first_n=replace_first_n,
        )
        processed += 1
        total_replacements += replacements
        print(f"OK {base_path.name}: replaced {replacements} rows")

    print(
        f"\nDone. Processed {processed} files, skipped {skipped}, "
        f"total replaced rows: {total_replacements}"
    )


if __name__ == "__main__":
    main()
