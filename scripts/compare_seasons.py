#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from typing import Dict, Tuple

NameKey = Tuple[str, str]


def load_total_points(season_dir: str) -> Dict[NameKey, int]:
    path = os.path.join(season_dir, "cleaned_players.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing file: {path}")

    totals: Dict[NameKey, int] = {}
    duplicate_count = 0

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"first_name", "second_name", "total_points"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing columns: {', '.join(sorted(missing))}")

        for row in reader:
            first = (row.get("first_name") or "").strip()
            second = (row.get("second_name") or "").strip()
            key = (first, second)

            if key in totals:
                duplicate_count += 1
                continue

            raw_points = (row.get("total_points") or "").strip()
            try:
                points = int(float(raw_points)) if raw_points else 0
            except ValueError:
                points = 0
            totals[key] = points

    if duplicate_count:
        print(
            f"warning: {duplicate_count} duplicate name rows ignored in {path}",
            file=sys.stderr,
        )

    return totals


def parse_args() -> argparse.Namespace:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    parser = argparse.ArgumentParser(
        description=(
            "Compare total fantasy points for players present in two seasons."
        )
    )
    parser.add_argument(
        "season_end_year",
        type=int,
        help="end year of the newer season, e.g. 2025 for 2024-25 vs 2023-24",
    )
    parser.add_argument(
        "--base-dir",
        default=os.path.join(repo_root, "Base Data"),
        help="base directory containing season folders",
    )
    parser.add_argument(
        "--out",
        help="output CSV path (default: season_compare_<older>_vs_<newer>.csv)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    end_year = args.season_end_year
    newer = f"{end_year - 1}-{str(end_year)[-2:]}"
    older = f"{end_year - 2}-{str(end_year - 1)[-2:]}"
    older_dir = os.path.join(args.base_dir, older)
    newer_dir = os.path.join(args.base_dir, newer)

    older_totals = load_total_points(older_dir)
    newer_totals = load_total_points(newer_dir)

    shared_keys = sorted(
        set(older_totals) & set(newer_totals), key=lambda k: (k[1], k[0])
    )

    out_path = args.out or f"season_compare_{older}_vs_{newer}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "first_name",
                "second_name",
                f"total_points_{older}",
                f"total_points_{newer}",
                "delta",
            ]
        )
        for first, second in shared_keys:
            older_points = older_totals[(first, second)]
            newer_points = newer_totals[(first, second)]
            writer.writerow(
                [
                    first,
                    second,
                    older_points,
                    newer_points,
                    newer_points - older_points,
                ]
            )

    print(
        f"wrote {len(shared_keys)} rows to {out_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
