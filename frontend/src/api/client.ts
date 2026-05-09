import type {
  SquadGenerateRequest,
  SquadGenerateResponse,
  LineupRecommendRequest,
  LineupRecommendResponse,
  PlayerListItem,
} from "./types";

const API_BASE = "http://localhost:8000";

export async function fetchPlayers(): Promise<PlayerListItem[]> {
  const res = await fetch(`${API_BASE}/players`);

  if (!res.ok) {
    throw new Error(`Failed to fetch players (${res.status})`);
  }

  return res.json();
}

export async function generateSquad(
  req: SquadGenerateRequest,
): Promise<SquadGenerateResponse> {
  const res = await fetch(`${API_BASE}/squad/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });

  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail ?? `Squad generation failed (${res.status})`);
  }

  return res.json();
}

export async function recommendLineup(
  req: LineupRecommendRequest,
): Promise<LineupRecommendResponse> {
  const res = await fetch(`${API_BASE}/lineup/recommend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });

  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(
      body?.detail ?? `Lineup recommendation failed (${res.status})`,
    );
  }

  return res.json();
}
