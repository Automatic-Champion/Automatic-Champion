# Chapter 10 — Completed Test Result Tables

These are the completed result tables for Chapter 10 of the engineering report. Every functional test case (TC01–TC06) and the automatable non-functional requirements (NFR01, NFR03) are now covered by automated tests. The full automated suite is **116 tests**, run with:

```bash
python -m pytest tests/ -v
```

Each result below cites the covering test(s) as `file::function`. NFR02 (usability) is the one item that cannot be automated; it is marked Pending and requires an informal study with five users.

---

## Table 10.1 — Functional Test Cases

| Test Case ID | Function or Scenario | Script Ref (S1–S6) | Valid and Fix Ref (P1–P6) | Test Result | Retest Notes |
|---|---|---|---|---|---|
| TC01 | Build initial squad with default preferences | S1 | P1 | **Pass** | `tests/test_full_squad.py::test_build_full_squad_returns_15_players`, `tests/test_full_squad.py::test_build_full_squad_different_formation` |
| TC02 | Build initial squad with valid user constraints | S2 | P2 | **Pass** | `tests/test_squad_constraints.py::test_locked_player_appears_in_squad`, `tests/test_squad_constraints.py::test_banned_player_excluded_from_squad` |
| TC03 | Handle infeasible initial squad constraints | S3 | P3 | **Pass** | `tests/test_squad_constraints.py::test_locked_and_banned_overlap_raises`, `::test_locked_id_not_found_raises`, `::test_locked_over_budget_raises`, `::test_locked_exceeds_position_count_raises`, `::test_locked_exceeds_team_cap_raises`, `::test_infeasible_budget_raises` |
| TC04 | Recommend weekly lineup with a valid squad | S4 | P4 | **Pass** | `tests/test_lineup_optimizer.py` (`TestBasicLineup`, `TestAutoFormation`, `TestBenchOrder`), `tests/test_lineup_endpoint.py::test_recommend_lineup_valid`, `::test_recommend_lineup_with_formation` |
| TC05 | Handle invalid squad input for lineup recommendation | S5 | P5 | **Pass** | `tests/test_squad_constraints.py::test_generate_squad_value_error_returns_400`, `::test_generate_squad_optimization_error_returns_400`, `tests/test_lineup_endpoint.py::test_recommend_lineup_invalid_squad_size`, `::test_recommend_lineup_invalid_formation` |
| TC06 | Handle missing or stale gameweek data | S6 | P6 | **Pass** | `tests/test_lineup_endpoint.py::test_recommend_lineup_warns_when_model_unavailable`, `::test_recommend_lineup_warns_on_partial_fallback`, `::test_recommend_lineup_no_warning_when_healthy`, `tests/test_gameweek_predictor.py::test_with_meta_reports_model_unavailable` |

### Per-case justification

**TC01 — Build initial squad with default preferences (P1).** P1 requires that the output contain exactly 15 players, respect the budget, the position structure, and the maximum-3-players-per-club rule, with expected points and at least one short explanation. `test_build_full_squad_returns_15_players` builds a squad with default settings and asserts the squad contains exactly 15 players within the position split, under the budget cap, and within the per-club limit, with expected points and explanations present. `test_build_full_squad_different_formation` repeats the same validity assertions for an alternate valid formation, confirming the rules hold beyond a single default layout. All P1 conditions are met, so no fix is required.

**TC02 — Build initial squad with valid user constraints (P2).** P2 requires all P1 validity conditions plus respecting feasible user constraints. `test_locked_player_appears_in_squad` supplies a feasible "must include" (locked) player and asserts that the player is present in the returned 15-player squad while the squad remains valid. `test_banned_player_excluded_from_squad` supplies a feasible "exclude" (banned) player and asserts that the player is absent while the squad remains valid. Together they confirm feasible constraints are honoured without violating any P1 rule, so no fix is required.

**TC03 — Handle infeasible initial squad constraints (P3).** P3 requires that the system does not return an invalid squad and instead clearly explains why the constraints conflict and asks the user to relax them. The six infeasibility tests in `test_squad_constraints.py` each force a distinct conflict — a player both locked and banned (`test_locked_and_banned_overlap_raises`), an unknown locked id (`test_locked_id_not_found_raises`), locked players exceeding the budget (`test_locked_over_budget_raises`), locked players exceeding a position count (`test_locked_exceeds_position_count_raises`), locked players breaching the per-club cap (`test_locked_exceeds_team_cap_raises`), and an outright infeasible budget (`test_infeasible_budget_raises`) — and assert that the builder raises a descriptive error rather than returning a rule-breaking squad. This matches P3's "explain why and do not return an invalid squad," so no fix is required.

**TC04 — Recommend weekly lineup with a valid squad (P4).** P4 requires exactly 11 starters with a valid formation and valid bench order, with expected points at lineup level and short explanations for key choices. The class-based optimizer tests in `tests/test_lineup_optimizer.py` cover the core selection logic: `TestBasicLineup` checks fixed-formation selection, captain (highest predicted) and vice-captain choice, and the captain-bonus total; `TestAutoFormation` checks that auto mode picks a valid best formation; and `TestBenchOrder` checks that the bench GK is always last and outfield bench players are sorted by predicted points. At the API level, `test_recommend_lineup_valid` asserts a valid 11 + bench, formation, captain/VC, expected points, and explanations from the endpoint, and `test_recommend_lineup_with_formation` confirms a user-supplied formation override produces an equally valid lineup. All P4 conditions are met, so no fix is required.

**TC05 — Handle invalid squad input for lineup recommendation (P5).** P5 requires that the system highlight missing or incorrect details, request correction, and not produce output from invalid input. On the squad-builder path, `test_generate_squad_value_error_returns_400` and `test_generate_squad_optimization_error_returns_400` assert that invalid or unsatisfiable requests return an HTTP 400 with an explanatory message rather than a malformed squad. On the lineup path, `test_recommend_lineup_invalid_squad_size` (wrong number of players) and `test_recommend_lineup_invalid_formation` (formation that cannot be satisfied) assert the endpoint blocks generation and returns an error instead of an invalid lineup. This matches P5's "block until the input becomes valid," so no fix is required.

**TC06 — Handle missing or stale gameweek data (P6).** P6 is satisfied either by warning and offering refresh, or by allowing continuation with the latest valid snapshot while still producing a valid lineup. The system takes the second branch: when the V8 weekly model is unavailable or only partially available it falls back to the latest valid snapshot (`ep_next` / points-per-game) and surfaces a `warnings` list on the `/lineup/recommend` response while still returning a valid lineup. `test_recommend_lineup_warns_when_model_unavailable` asserts a warning is present and a valid lineup is still produced when the model is unavailable; `test_recommend_lineup_warns_on_partial_fallback` asserts the same for a partial fallback; and `test_recommend_lineup_no_warning_when_healthy` asserts that no spurious warning appears when data is fresh. `test_gameweek_predictor.py::test_with_meta_reports_model_unavailable` confirms at the predictor layer that the model-unavailable condition is reported up the stack so the endpoint can warn. This matches P6 (warn and continue on the latest valid snapshot without producing an invalid lineup), so no fix is required.

---

## Table 10.2 — Non-Functional Test Cases

| Test Case ID | Requirement | Script Ref (N1–N3) | Pass Criteria Ref (NP1–NP3) | Test Result | Retest Notes |
|---|---|---|---|---|---|
| NFR01 | Performance | N1 | NP1 | **Pass** | Overall average **0.87 s** across 30 requests (15 UC1 `/squad/generate` at 0.67 s avg + 15 UC2 `/lineup/recommend` at 1.07 s avg) — well under the 20 s threshold. Real models end-to-end, no predictor/optimizer mocking. `tests/test_performance.py::test_nfr01_average_request_under_20s` |
| NFR02 | Usability | N2 | NP2 | **Pending — manual study** | Not automatable. Requires an informal study with 5 new users completing the main recommendation flow without help (≥80% success per NP2). To be run with classmates before the defense. |
| NFR03 | Supportability | N3 | NP3 | **Pass** | `tests/test_model_swap.py::test_all_dummy_models_still_build_valid_squad`, `tests/test_model_swap.py::test_dummy_model_with_constant_zero_still_valid` |

### Per-requirement justification

**NFR01 — Performance (N1 / NP1).** NP1 passes if the average runtime across the 30 requests is at most 20 seconds. `test_nfr01_average_request_under_20s` runs exactly the N1 protocol: 30 requests split as 15 UC1 initial-squad generations (`POST /squad/generate`) and 15 UC2 weekly-lineup recommendations (`POST /lineup/recommend`), measuring each request's wall-clock time. The requests hit the real seasonal RandomForest models (UC1) and the real V8 CatBoost models (UC2) end-to-end through the FastAPI app, with only Firebase auth mocked — nothing on the prediction or optimization path is mocked. The measured overall average is **0.87 s** (UC1 0.67 s avg, UC2 1.07 s avg), comfortably below the 20 s ceiling, so NP1 passes.

**NFR02 — Usability (N2 / NP2).** NP2 passes if at least 80 percent of five new users complete the main recommendation flow without external help. This is inherently a human study and cannot be expressed as an automated test, so it is marked **Pending — manual study** rather than Pass. The plan is an informal session with five classmates, each asked to obtain a recommendation unaided, recording success or failure; the result is recorded here once the study is run.

**NFR03 — Supportability (N3 / NP3).** NP3 passes if recommendations are still produced after the prediction component is replaced with a dummy model that keeps the same interface, without requiring frontend changes. `test_all_dummy_models_still_build_valid_squad` swaps in dummy models (preserving the predictor interface) and asserts the full-squad builder still produces a valid 15-player squad. `test_dummy_model_with_constant_zero_still_valid` pushes the boundary by using a dummy that returns a constant zero prediction and confirms the system still produces a valid squad rather than breaking on the degenerate scores. Because the swap occurs behind the stable predictor interface and no frontend change is involved, NP3 passes.
