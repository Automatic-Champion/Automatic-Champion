#!/usr/bin/env python3
import argparse
import csv
import os
import sys
import math
from typing import List, Optional, Tuple


def read_deltas(path: str) -> List[float]:
    deltas: List[float] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "delta" not in reader.fieldnames:
            raise ValueError(f"missing 'delta' column in {path}")
        for row in reader:
            raw = (row.get("delta") or "").strip()
            if not raw:
                continue
            try:
                deltas.append(float(raw))
            except ValueError:
                continue
    return deltas


def infer_seasons_from_header(path: str) -> Optional[Tuple[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, [])
    if not header:
        return None
    older = None
    newer = None
    for col in header:
        if col.startswith("total_points_"):
            season = col[len("total_points_") :]
            if older is None:
                older = season
            elif newer is None:
                newer = season
                break
    if older and newer:
        return older, newer
    return None


def infer_seasons_from_filename(path: str) -> Optional[Tuple[str, str]]:
    base = os.path.basename(path)
    prefix = "season_compare_"
    marker = "_vs_"
    if not base.startswith(prefix) or marker not in base:
        return None
    core = base[len(prefix) :]
    core = os.path.splitext(core)[0]
    parts = core.split(marker)
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a histogram of player delta points from a comparison CSV."
    )
    parser.add_argument("csv_path", help="path to season comparison CSV")
    parser.add_argument(
        "--bins",
        type=int,
        default=30,
        help="number of histogram bins (default: 30)",
    )
    parser.add_argument(
        "--bin-size",
        type=float,
        help="fixed bin size for the histogram (e.g. 5). Overrides --bins.",
    )
    parser.add_argument(
        "--out",
        help="output image path (e.g. deltas.png). If omitted, shows plot window.",
    )
    parser.add_argument(
        "--abs",
        action="store_true",
        help="plot absolute delta values",
    )
    parser.add_argument(
        "--title",
        help="plot title (default: inferred from the CSV when possible)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_path = args.csv_path
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"missing file: {csv_path}")

    deltas = read_deltas(csv_path)
    if not deltas:
        raise ValueError("no delta values found to plot")

    # Import here so the script can fail fast on missing dependency.
    import matplotlib.pyplot as plt

    title = args.title
    if not title:
        seasons = infer_seasons_from_header(csv_path) or infer_seasons_from_filename(
            csv_path
        )
        if seasons:
            prefix = "Absolute Delta Points" if args.abs else "Delta Points"
            title = f"{prefix}: {seasons[0]} vs {seasons[1]}"
        else:
            title = "Absolute Delta Points Histogram" if args.abs else "Delta Points Histogram"

    if args.abs:
        deltas = [abs(value) for value in deltas]

    plt.figure(figsize=(10, 6))
    if args.bin_size:
        min_val = min(deltas)
        max_val = max(deltas)
        start = math.floor(min_val / args.bin_size) * args.bin_size
        end = math.ceil(max_val / args.bin_size) * args.bin_size
        if start == end:
            end = start + args.bin_size
        bin_edges = []
        current = start
        # Build edges to include the max value in the last bin.
        while current <= end + 1e-9:
            bin_edges.append(current)
            current += args.bin_size
        plt.hist(deltas, bins=bin_edges, edgecolor="black")
    else:
        plt.hist(deltas, bins=args.bins, edgecolor="black")
    if args.abs:
        plt.xlabel("Absolute delta (points)")
    else:
        plt.xlabel("Delta (newer season points - older season points)")
    plt.ylabel("Number of players")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)

    if args.out:
        plt.tight_layout()
        plt.savefig(args.out, dpi=150)
        print(f"wrote histogram to {args.out}", file=sys.stderr)
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
