#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove rows where total points for both seasons are 0 from a comparison CSV."
        )
    )
    parser.add_argument("csv_path", help="path to season comparison CSV")
    parser.add_argument(
        "--out",
        help="output CSV path (default: <input>_nonzero.csv)",
    )
    return parser.parse_args()


def find_points_columns(fieldnames: List[str]) -> List[str]:
    return [name for name in fieldnames if name.startswith("total_points_")]


def to_number(value: str) -> float:
    raw = (value or "").strip()
    if not raw:
        return 0.0
    try:
        return float(raw)
    except ValueError:
        return 0.0


def main() -> int:
    args = parse_args()
    in_path = args.csv_path
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"missing file: {in_path}")

    out_path = args.out
    if not out_path:
        base, ext = os.path.splitext(in_path)
        out_path = f"{base}_nonzero{ext or '.csv'}"

    with open(in_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        points_cols = find_points_columns(fieldnames)
        if len(points_cols) != 2:
            raise ValueError(
                f"expected 2 total_points columns, found {len(points_cols)}"
            )

        rows = list(reader)

    kept = []
    removed = 0
    for row in rows:
        points = [to_number(row.get(col, "")) for col in points_cols]
        if all(p == 0 for p in points):
            removed += 1
            continue
        kept.append(row)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    print(
        f"wrote {len(kept)} rows to {out_path} (removed {removed})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
