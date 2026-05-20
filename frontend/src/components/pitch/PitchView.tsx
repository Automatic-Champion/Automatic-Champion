import { motion, useReducedMotion } from "framer-motion";
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

const POSITION_ORDER: Record<string, number> = { GK: 0, DEF: 1, MID: 2, FWD: 3 };

function mapPlayersToSlots(
  players: PitchViewPlayer[],
  formation: string
) {
  const slots = getFormationPositions(formation);

  const groups: Record<string, PitchViewPlayer[]> = {
    GK: [],
    DEF: [],
    MID: [],
    FWD: [],
  };
  for (const p of players) {
    groups[p.position]?.push(p);
  }

  const counters: Record<string, number> = { GK: 0, DEF: 0, MID: 0, FWD: 0 };
  return slots.map((slot) => {
    const idx = counters[slot.position] ?? 0;
    counters[slot.position] = idx + 1;
    const player = groups[slot.position]?.[idx] ?? null;
    return { ...slot, player };
  });
}

// Ambient particle dots for stadium lights effect
function AmbientParticles() {
  const particles = [
    { x: 5, y: 20, delay: 0 },
    { x: 95, y: 30, delay: 0.5 },
    { x: 8, y: 70, delay: 1.2 },
    { x: 92, y: 60, delay: 1.8 },
    { x: 3, y: 45, delay: 2.5 },
    { x: 97, y: 80, delay: 3.0 },
  ];

  return (
    <>
      {particles.map((p, i) => (
        <motion.div
          key={i}
          className="absolute w-1.5 h-1.5 rounded-full bg-amber-300/50 pointer-events-none"
          style={{ left: `${p.x}%`, top: `${p.y}%` }}
          animate={{
            opacity: [0.3, 0.7, 0.3],
            y: [0, -4, 0],
          }}
          transition={{
            duration: 3,
            delay: p.delay,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />
      ))}
    </>
  );
}

export default function PitchView({
  formation,
  players,
  selectedPlayerId,
  onPlayerClick,
}: PitchViewProps) {
  const mapped = mapPlayersToSlots(players, formation);
  const prefersReduced = useReducedMotion();
  const _dur = (d: number) => (prefersReduced ? 0 : d);
  void _dur;

  return (
    <div className="relative w-full max-w-[600px] mx-auto rounded-xl overflow-hidden shadow-lg ring-1 ring-amber-500/20">
      {/* SVG pitch background */}
      <PitchSVG />

      {/* Ambient glow — more visible with slow pulse */}
      <div
        className="absolute -inset-4 rounded-3xl bg-emerald-500/[0.18] blur-3xl dark:block hidden pointer-events-none"
        style={{ animation: prefersReduced ? "none" : "pulse 4s ease-in-out infinite" }}
      />

      {/* Gradient overlay for depth */}
      <div className="absolute inset-x-0 bottom-0 h-24 bg-gradient-to-t from-black/20 to-transparent pointer-events-none" />

      {/* Ambient particles — stadium lights */}
      {!prefersReduced && <AmbientParticles />}

      {/* Player overlay */}
      <div className="absolute inset-0">
        {mapped.map((slot, i) =>
          slot.player ? (
            <motion.div
              key={slot.player.id}
              className="absolute -translate-x-1/2 -translate-y-1/2"
              style={{
                left: `${slot.x}%`,
                top: `${slot.y}%`,
              }}
              initial={{ opacity: 0, scale: 0.5 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={
                prefersReduced
                  ? { duration: 0 }
                  : {
                      type: "spring",
                      stiffness: 300,
                      damping: 20,
                      delay: (POSITION_ORDER[slot.position] ?? 0) * 0.1 + 0.1,
                    }
              }
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
            </motion.div>
          ) : (
            <div
              key={`empty-${i}`}
              className="absolute -translate-x-1/2 -translate-y-1/2"
              style={{
                left: `${slot.x}%`,
                top: `${slot.y}%`,
              }}
            >
              <div className="w-10 h-[52px] rounded-lg border-2 border-dashed border-white/30" />
            </div>
          )
        )}
      </div>
    </div>
  );
}
