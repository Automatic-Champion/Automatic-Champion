from __future__ import annotations

from typing import Any

from src.team_builder import build_full_squad


class OptimizationError(ValueError):
    """Raised when the optimizer cannot build a valid squad."""


def generate_optimal_squad(
    budget: float,
    formation: str,
    data_path: str = "data/players_merged_2024-25.csv",
    models_dir: str = "models",
    locked_ids: set[str] | None = None,
    banned_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Thin wrapper around src.team_builder.build_full_squad()."""
    try:
        return build_full_squad(
            budget=budget,
            formation=formation,
            data_path=data_path,
            models_dir=models_dir,
            locked_ids=locked_ids,
            banned_ids=banned_ids,
        )
    except (ValueError, FileNotFoundError, ImportError, RuntimeError) as exc:
        raise OptimizationError(str(exc)) from exc
