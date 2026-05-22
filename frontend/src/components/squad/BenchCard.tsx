import { motion, useReducedMotion } from "framer-motion";
import { Star } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import type { DisplayPlayer } from "@/api/types";
import { getKitUrl } from "@/lib/teamKits";

interface BenchCardProps {
  player: DisplayPlayer;
  benchOrder: number;
  isSelected?: boolean;
  onClick?: () => void;
}

const POSITION_COLORS: Record<string, { border: string; glow: string; jersey: string; stripe: string }> = {
  GK: { border: "border-l-yellow-400", glow: "hover:shadow-yellow-400/20", jersey: "#facc15", stripe: "from-yellow-400" },
  DEF: { border: "border-l-blue-500", glow: "hover:shadow-blue-500/20", jersey: "#3b82f6", stripe: "from-blue-500" },
  MID: { border: "border-l-green-500", glow: "hover:shadow-green-500/20", jersey: "#22c55e", stripe: "from-green-500" },
  FWD: { border: "border-l-red-500", glow: "hover:shadow-red-500/20", jersey: "#ef4444", stripe: "from-red-500" },
};

function JerseyIconSmall({ color }: { color: string }) {
  return (
    <svg width="16" height="13" viewBox="0 0 20 16" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path
        d="M6 0L0 3V7L2 7.5V16H18V7.5L20 7V3L14 0H12.5C12.5 1.38 11.38 2.5 10 2.5C8.62 2.5 7.5 1.38 7.5 0H6Z"
        fill={color}
      />
    </svg>
  );
}

export default function BenchCard({
  player,
  benchOrder: _benchOrder,
  isSelected,
  onClick,
}: BenchCardProps) {
  void _benchOrder; // used for rendering order
  const prefersReduced = useReducedMotion();
  const colors = POSITION_COLORS[player.position] ?? POSITION_COLORS.MID;
  const kitUrl = getKitUrl(player.team);

  return (
    <motion.div
      whileHover={prefersReduced ? undefined : { y: -2 }}
      transition={prefersReduced ? undefined : { type: "spring", stiffness: 400, damping: 25 }}
      className={`relative cursor-pointer rounded-lg border-l-4 px-4 py-3 transition-all duration-150
        border border-gray-200 bg-white/80 backdrop-blur-sm overflow-hidden
        dark:border-white/[0.08] dark:bg-white/[0.03] dark:backdrop-blur-xl dark:shadow-lg
        ${colors.border} ${colors.glow}
        hover:shadow-md dark:hover:bg-white/[0.06]
        ${isSelected ? "ring-2 ring-primary" : ""}`}
      onClick={onClick}
    >
      {/* Gradient top stripe */}
      <div className={`absolute top-0 left-0 right-0 h-[3px] bg-gradient-to-r ${colors.stripe} via-${colors.stripe}/50 to-transparent`} />

      {/* Diagonal stripe pattern overlay */}
      <div
        className="absolute inset-0 pointer-events-none opacity-[0.08] dark:opacity-[0.10]"
        style={{
          backgroundImage: `repeating-linear-gradient(
            45deg,
            transparent,
            transparent 10px,
            currentColor 10px,
            currentColor 11px
          )`,
        }}
      />

      <div className="relative flex items-center gap-3">
        {/* Kit image (fallback to jersey SVG if no kit available) */}
        <span className="flex size-7 shrink-0 items-center justify-center">
          {kitUrl ? (
            <img
              src={kitUrl}
              alt={player.team}
              loading="lazy"
              className="w-7 h-7 object-contain"
            />
          ) : (
            <JerseyIconSmall color={colors.jersey} />
          )}
        </span>
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-medium">{player.name}</p>
          <p className="text-xs text-muted-foreground">{player.team}</p>
        </div>
        <Badge variant="outline" className="shrink-0 text-xs">
          {player.position}
        </Badge>
        {player.cost != null && (
          <span className="shrink-0 text-xs text-muted-foreground">
            {"\u00A3"}{player.cost.toFixed(1)}m
          </span>
        )}
        <span className="shrink-0 flex items-center gap-0.5 text-sm font-bold text-amber-600 dark:text-amber-400">
          {player.points.toFixed(1)}
          <Star className="size-3 fill-amber-500 text-amber-500" />
        </span>
      </div>
    </motion.div>
  );
}
