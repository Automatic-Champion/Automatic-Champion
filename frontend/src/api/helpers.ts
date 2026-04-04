import type { SquadPlayerResponse, LineupPlayerInput } from "./types";

export function squadToLineupInput(
  players: SquadPlayerResponse[],
  bench: SquadPlayerResponse[],
): LineupPlayerInput[] {
  return [...players, ...bench].map((p) => ({
    id: p.id,
    name: p.name,
    position: p.position,
    team: p.team,
    cost: p.cost,
    pred: p.predicted_points,
  }));
}
