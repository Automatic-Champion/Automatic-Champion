import { auth } from "../lib/firebase";
import type {
  SquadGenerateRequest,
  SquadGenerateResponse,
  LineupRecommendRequest,
  LineupRecommendResponse,
  PlayerListItem,
} from "./types";

const API_BASE = "http://localhost:8000";

async function buildHeaders(extra?: HeadersInit): Promise<Headers> {
  const headers = new Headers(extra);
  const user = auth.currentUser;
  if (user) {
    try {
      const token = await user.getIdToken();
      headers.set("Authorization", `Bearer ${token}`);
    } catch {
      // If we can't get a token, fall through; backend will return 401.
    }
  }
  return headers;
}

function handleUnauthorized(status: number) {
  if (status === 401 && typeof window !== "undefined") {
    if (window.location.pathname !== "/login") {
      window.location.assign("/login");
    }
  }
}

async function apiFetch(input: string, init: RequestInit = {}): Promise<Response> {
  const headers = await buildHeaders(init.headers);
  const res = await fetch(input, { ...init, headers });
  if (res.status === 401) {
    handleUnauthorized(res.status);
  }
  return res;
}

export async function fetchPlayers(): Promise<PlayerListItem[]> {
  const res = await apiFetch(`${API_BASE}/players`);

  if (!res.ok) {
    throw new Error(`Failed to fetch players (${res.status})`);
  }

  return res.json();
}

export async function generateSquad(
  req: SquadGenerateRequest,
): Promise<SquadGenerateResponse> {
  const res = await apiFetch(`${API_BASE}/squad/generate`, {
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
  const res = await apiFetch(`${API_BASE}/lineup/recommend`, {
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
