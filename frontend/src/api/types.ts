// Player list (for search/select)

export interface PlayerListItem {
  id: number;
  name: string;
  position: string;
  team: string;
  cost: number;
}

// Shared types

export interface Explanation {
  feature: string;
  importance: number;
  explanation: string;
  value?: number | string | null;
}

// ── UC1: Squad Generation ──────────────────────────────────────────

export interface SquadGenerateRequest {
  budget: number;
  formation: string;
  locked_ids: number[];
  banned_ids: number[];
}

export interface SquadPlayerResponse {
  id: string;
  name: string;
  team: string;
  position: string;
  cost: number;
  predicted_points: number;
  is_starter: boolean;
  bench_order: number | null;
  explanations: Explanation[];
}

export interface SquadGenerateResponse {
  formation: string;
  budget: number;
  total_cost: number;
  total_predicted_points: number;
  players: SquadPlayerResponse[];
  bench: SquadPlayerResponse[];
}

// ── UC2: Lineup Recommendation ─────────────────────────────────────

export interface LineupPlayerInput {
  id: string;
  name: string;
  position: string;
  team: string;
  cost: number;
  pred: number;
}

export interface LineupRecommendRequest {
  squad: LineupPlayerInput[];
  gameweek: number | null;
  formation: string | null;
}

export interface LineupStarterResponse {
  id: string;
  name: string;
  position: string;
  team: string;
  gw_points: number;
  is_captain: boolean;
  is_vice_captain: boolean;
  explanations: Explanation[];
}

export interface LineupBenchResponse {
  id: string;
  name: string;
  position: string;
  team: string;
  gw_points: number;
  bench_order: number;
  explanations: Explanation[];
}

export interface LineupRecommendResponse {
  formation: string;
  gameweek: number | null;
  captain_id: string;
  vice_captain_id: string;
  total_gw_points: number;
  starters: LineupStarterResponse[];
  bench: LineupBenchResponse[];
}
