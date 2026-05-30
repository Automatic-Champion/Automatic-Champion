import { auth } from "../lib/firebase";
import type {
  SquadGenerateRequest,
  SquadGenerateResponse,
  LineupRecommendRequest,
  LineupRecommendResponse,
  PlayerListItem,
  SavedSquadSummary,
  SavedSquadResponse,
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

export async function listSavedSquads(): Promise<SavedSquadSummary[]> {
  const res = await apiFetch(`${API_BASE}/squads`);
  if (!res.ok) {
    throw new Error(`Failed to load saved squads (${res.status})`);
  }
  return res.json();
}

export async function saveSquad(
  name: string,
  squad: SquadGenerateResponse,
): Promise<SavedSquadResponse> {
  const res = await apiFetch(`${API_BASE}/squads`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, squad }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail ?? `Failed to save squad (${res.status})`);
  }
  return res.json();
}

export async function getSavedSquad(id: number): Promise<SavedSquadResponse> {
  const res = await apiFetch(`${API_BASE}/squads/${id}`);
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail ?? `Failed to load saved squad (${res.status})`);
  }
  return res.json();
}

export async function deleteSavedSquad(id: number): Promise<void> {
  const res = await apiFetch(`${API_BASE}/squads/${id}`, { method: "DELETE" });
  if (!res.ok && res.status !== 204) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail ?? `Failed to delete saved squad (${res.status})`);
  }
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
