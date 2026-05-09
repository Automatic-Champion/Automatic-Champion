import { cn } from "@/lib/utils";

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

const POSITION_COLORS: Record<string, string> = {
  GK: "bg-yellow-400",
  DEF: "bg-blue-500",
  MID: "bg-green-500",
  FWD: "bg-red-500",
};

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
  const displayName =
    name.length > 12 ? name.slice(0, 11) + "\u2026" : name;

  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "flex flex-col items-center gap-0.5 cursor-pointer transition-transform hover:scale-110 focus:outline-none",
        isSelected && "scale-110"
      )}
    >
      {/* Jersey circle */}
      <div className="relative">
        <div
          className={cn(
            "w-10 h-10 rounded-full flex items-center justify-center text-white text-xs font-bold shadow-md",
            POSITION_COLORS[position] ?? "bg-gray-500"
          )}
        >
          {position}
        </div>

        {/* Captain / Vice-captain badge */}
        {(isCaptain || isViceCaptain) && (
          <span className="absolute -top-1 -right-1 w-5 h-5 rounded-full bg-white text-gray-900 text-[10px] font-bold flex items-center justify-center shadow">
            {isCaptain ? "C" : "VC"}
          </span>
        )}
      </div>

      {/* Player name */}
      <span
        className="text-[11px] font-semibold leading-tight text-white text-center max-w-[80px] truncate"
        style={{ textShadow: "0 1px 3px rgba(0,0,0,0.7)" }}
      >
        {displayName}
      </span>

      {/* Team abbreviation */}
      <span
        className="text-[9px] text-gray-200 leading-none"
        style={{ textShadow: "0 1px 2px rgba(0,0,0,0.5)" }}
      >
        {team}
      </span>

      {/* Points badge */}
      <span className="mt-0.5 px-1.5 py-0.5 rounded-full bg-black/60 text-white text-[10px] font-medium leading-none">
        {points.toFixed(1)}
      </span>
    </button>
  );
}
