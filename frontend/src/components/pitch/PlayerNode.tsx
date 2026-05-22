import { motion, useReducedMotion } from "framer-motion";
import { cn } from "@/lib/utils";
import { getKitUrl } from "@/lib/teamKits";

export interface PlayerNodeProps {
  name: string;
  position: string; // "GK" | "DEF" | "MID" | "FWD"
  team: string;
  points: number;
  isCaptain?: boolean;
  isViceCaptain?: boolean;
  isSelected?: boolean;
  onClick?: () => void;
}

const POSITION_COLORS: Record<string, { gradient: string; jersey: string; glow: string }> = {
  GK: { gradient: "from-yellow-400 to-yellow-600", jersey: "#facc15", glow: "shadow-yellow-400/40" },
  DEF: { gradient: "from-blue-400 to-blue-600", jersey: "#3b82f6", glow: "shadow-blue-500/40" },
  MID: { gradient: "from-green-400 to-green-600", jersey: "#22c55e", glow: "shadow-green-500/40" },
  FWD: { gradient: "from-red-400 to-red-600", jersey: "#ef4444", glow: "shadow-red-500/40" },
};

function JerseyIcon({ color }: { color: string }) {
  return (
    <svg width="40" height="32" viewBox="0 0 20 16" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path
        d="M6 0L0 3V7L2 7.5V16H18V7.5L20 7V3L14 0H12.5C12.5 1.38 11.38 2.5 10 2.5C8.62 2.5 7.5 1.38 7.5 0H6Z"
        fill={color}
      />
      <path
        d="M6 0L0 3V7L2 7.5V16H18V7.5L20 7V3L14 0"
        stroke="rgba(255,255,255,0.3)"
        strokeWidth="0.5"
        fill="none"
      />
    </svg>
  );
}

export default function PlayerNode({
  name,
  position,
  team,
  points,
  isCaptain,
  isViceCaptain,
  isSelected,
  onClick,
}: PlayerNodeProps) {
  const prefersReduced = useReducedMotion();
  const colors = POSITION_COLORS[position] ?? POSITION_COLORS.MID;
  const kitUrl = getKitUrl(team);

  return (
    <motion.button
      type="button"
      onClick={onClick}
      className={cn(
        "group flex flex-col items-center gap-0 cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:rounded-lg",
        isSelected && "scale-110"
      )}
      whileHover={prefersReduced ? undefined : { rotateY: 5, scale: 1.08 }}
      transition={prefersReduced ? undefined : { type: "spring", stiffness: 300, damping: 20 }}
      style={{ perspective: "600px" }}
    >
      {/* Mini card shape */}
      <div
        className={cn(
          "relative flex flex-col items-center w-[88px] h-[116px] rounded-lg p-0.5 transition-shadow duration-200",
          `bg-gradient-to-b ${colors.gradient}`,
          `group-hover:shadow-lg group-hover:${colors.glow}`
        )}
        style={{
          boxShadow: isSelected ? `0 0 12px 2px ${colors.jersey}44` : undefined,
        }}
      >
        {/* Inner card */}
        <div className="flex flex-col items-center justify-between w-full h-full rounded-md bg-gray-900/80 backdrop-blur-sm px-1.5 py-2">
          {/* Kit image (fallback to jersey SVG if no kit available) */}
          {kitUrl ? (
            <img
              src={kitUrl}
              alt={team}
              loading="lazy"
              className="w-9 h-9 object-contain"
            />
          ) : (
            <JerseyIcon color={colors.jersey} />
          )}

          {/* Player name */}
          <span
            className="text-[11px] font-bold leading-tight text-white text-center w-full line-clamp-2 px-0.5"
            style={{ textShadow: "0 1px 2px rgba(0,0,0,0.8)" }}
          >
            {name}
          </span>

          {/* Points pill */}
          <motion.span
            className="px-1.5 py-0.5 rounded-full bg-amber-500/90 text-[11px] font-bold text-white leading-tight"
            whileHover={prefersReduced ? undefined : { y: -1 }}
          >
            {points.toFixed(1)}
          </motion.span>
        </div>

        {/* Captain badge — gold diamond */}
        {(isCaptain || isViceCaptain) && (
          <span
            className="absolute -top-2 -right-2 w-6 h-6 flex items-center justify-center text-[10px] font-black text-gray-900"
            style={{
              background: "linear-gradient(135deg, #fbbf24, #f59e0b)",
              clipPath: "polygon(50% 0%, 100% 50%, 50% 100%, 0% 50%)",
            }}
          >
            {isCaptain ? "C" : "V"}
          </span>
        )}
      </div>

      {/* Team abbreviation below card */}
      <span
        className="text-[10px] text-gray-300 leading-none mt-1"
        style={{ textShadow: "0 1px 2px rgba(0,0,0,0.5)" }}
      >
        {team}
      </span>
    </motion.button>
  );
}
