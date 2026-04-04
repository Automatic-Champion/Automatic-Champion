import PitchSVG from "./PitchSVG";
import PlayerNode from "./PlayerNode";
import { getFormationPositions } from "./formations";

export interface PitchViewPlayer {
  id: string;
  name: string;
  position: string;
  team: string;
  points: number;
  isCaptain?: boolean;
  isViceCaptain?: boolean;
}

export interface PitchViewProps {
  formation: string;
  players: PitchViewPlayer[];
  selectedPlayerId?: string | null;
  onPlayerClick?: (playerId: string) => void;
}

/**
 * Sort players into position groups and map them to formation slots.
 */
function mapPlayersToSlots(
  players: PitchViewPlayer[],
  formation: string
) {
  const slots = getFormationPositions(formation);

  // Group players by position
  const groups: Record<string, PitchViewPlayer[]> = {
    GK: [],
    DEF: [],
    MID: [],
    FWD: [],
  };
  for (const p of players) {
    groups[p.position]?.push(p);
  }

  // Map each slot to a player
  const counters: Record<string, number> = { GK: 0, DEF: 0, MID: 0, FWD: 0 };
  return slots.map((slot) => {
    const idx = counters[slot.position] ?? 0;
    counters[slot.position] = idx + 1;
    const player = groups[slot.position]?.[idx] ?? null;
    return { ...slot, player };
  });
}

export default function PitchView({
  formation,
  players,
  selectedPlayerId,
  onPlayerClick,
}: PitchViewProps) {
  const mapped = mapPlayersToSlots(players, formation);

  return (
    <div className="relative w-full max-w-[500px] mx-auto">
      {/* SVG pitch background */}
      <PitchSVG />

      {/* Player overlay */}
      <div className="absolute inset-0">
        {mapped.map((slot, i) =>
          slot.player ? (
            <div
              key={slot.player.id}
              className="absolute -translate-x-1/2 -translate-y-1/2"
              style={{
                left: `${slot.x}%`,
                top: `${slot.y}%`,
              }}
            >
              <PlayerNode
                name={slot.player.name}
                position={slot.player.position}
                team={slot.player.team}
                points={slot.player.points}
                isCaptain={slot.player.isCaptain}
                isViceCaptain={slot.player.isViceCaptain}
                isSelected={slot.player.id === selectedPlayerId}
                onClick={() => onPlayerClick?.(slot.player!.id)}
              />
            </div>
          ) : (
            <div
              key={`empty-${i}`}
              className="absolute -translate-x-1/2 -translate-y-1/2"
              style={{
                left: `${slot.x}%`,
                top: `${slot.y}%`,
              }}
            >
              <div className="w-10 h-10 rounded-full border-2 border-dashed border-white/40" />
            </div>
          )
        )}
      </div>
    </div>
  );
}
