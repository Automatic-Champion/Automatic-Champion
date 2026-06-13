from __future__ import annotations

"""NFR-01 performance test.

Engineering-report requirement NFR-01 / N1 / NP1: run 30 recommendation
requests covering BOTH initial squad generation (UC1) and weekly lineup
recommendation (UC2), measure runtime per request, and pass if the average is
<= 20 seconds.

These requests hit the REAL models end-to-end through the FastAPI app — the
seasonal RandomForest models for UC1 and the V8 CatBoost models for UC2 — with
only Firebase auth mocked (exactly as test_lineup_endpoint.py does). Nothing is
mocked on the prediction/optimization path, so the measured numbers reflect true
server-side runtime.
"""

import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)

# How the 30 requests are split between the two use cases.
N_SQUAD = 15  # UC1 — POST /squad/generate
N_LINEUP = 15  # UC2 — POST /lineup/recommend
NFR01_THRESHOLD_S = 20.0

# A fixed gameweek so the lineup endpoint does not hit the live FPL API to
# auto-detect the current GW; keeps the measurement self-contained and stable.
FIXED_GAMEWEEK = 20


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer faketoken"}


@pytest.fixture
def mock_verify_token():
    with patch("backend.app.auth.firebase_auth.verify_id_token") as mock:
        mock.return_value = {"uid": "test-user", "email": "test@example.com"}
        yield mock


def _assert_ok(resp, what: str) -> None:
    """Fail loudly with status + body on any non-2xx response."""
    if not (200 <= resp.status_code < 300):
        pytest.fail(f"{what} returned {resp.status_code}: {resp.text}")


def _squad_to_lineup_input(squad_resp: dict) -> list[dict]:
    """Map a /squad/generate response (15 real players with real FPL ids) to the
    LineupRecommendRequest squad shape so UC2 runs against real V8 feature rows
    instead of the FPL-API placeholder chain."""
    players = squad_resp["players"] + squad_resp["bench"]
    return [
        {
            "id": p["id"],
            "name": p["name"],
            "position": p["position"],
            "team": p["team"],
            "cost": p["cost"],
            "pred": p["predicted_points"],
        }
        for p in players
    ]


def _summary(label: str, times: list[float]) -> str:
    return (
        f"{label}: n={len(times)} "
        f"avg={sum(times) / len(times):.2f}s "
        f"min={min(times):.2f}s max={max(times):.2f}s"
    )


def test_nfr01_average_request_under_20s(mock_verify_token, auth_headers):
    squad_times: list[float] = []
    lineup_times: list[float] = []

    wall_start = time.perf_counter()

    # ── UC1: initial squad generation ──────────────────────────────────────
    last_squad_resp: dict | None = None
    for _ in range(N_SQUAD):
        t0 = time.perf_counter()
        resp = client.post(
            "/squad/generate",
            json={"budget": 100.0, "formation": "4-3-3"},
            headers=auth_headers,
        )
        elapsed = time.perf_counter() - t0
        _assert_ok(resp, "POST /squad/generate")
        squad_times.append(elapsed)
        last_squad_resp = resp.json()

    assert last_squad_resp is not None
    assert len(last_squad_resp["players"]) + len(last_squad_resp["bench"]) == 15

    # Reuse a real generated squad (real FPL ids/names) for the UC2 requests so
    # the V8 predictor scores from test.csv rather than the FPL-API fallback.
    lineup_squad = _squad_to_lineup_input(last_squad_resp)
    assert len(lineup_squad) == 15

    # ── UC2: weekly lineup recommendation ──────────────────────────────────
    for _ in range(N_LINEUP):
        t0 = time.perf_counter()
        resp = client.post(
            "/lineup/recommend",
            json={"squad": lineup_squad, "gameweek": FIXED_GAMEWEEK},
            headers=auth_headers,
        )
        elapsed = time.perf_counter() - t0
        _assert_ok(resp, "POST /lineup/recommend")
        lineup_times.append(elapsed)

    wall_elapsed = time.perf_counter() - wall_start

    all_times = squad_times + lineup_times
    overall_avg = sum(all_times) / len(all_times)
    squad_avg = sum(squad_times) / len(squad_times)
    lineup_avg = sum(lineup_times) / len(lineup_times)

    print("\n──────────── NFR-01 performance ────────────")
    print(f"requests: {len(all_times)} ({N_SQUAD} UC1 + {N_LINEUP} UC2)")
    print(_summary("UC1 (/squad/generate)", squad_times))
    print(_summary("UC2 (/lineup/recommend)", lineup_times))
    print(f"overall avg: {overall_avg:.2f}s  (threshold {NFR01_THRESHOLD_S:.0f}s)")
    print(f"total wall-clock for {len(all_times)} requests: {wall_elapsed:.2f}s")
    print("────────────────────────────────────────────")

    assert overall_avg <= NFR01_THRESHOLD_S, (
        f"NFR-01 FAILED: overall avg {overall_avg:.2f}s exceeds "
        f"{NFR01_THRESHOLD_S:.0f}s (UC1 avg {squad_avg:.2f}s, UC2 avg {lineup_avg:.2f}s)"
    )
