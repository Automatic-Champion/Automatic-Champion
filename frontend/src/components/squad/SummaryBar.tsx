import { LayoutGrid, Coins, Wallet, Trophy, Calendar } from "lucide-react";

interface SummaryBarProps {
  formation: string;
  totalCost: number;
  remainingBudget: number;
  totalPredictedPoints: number;
  gameweek?: number | null;
}

export default function SummaryBar({
  formation,
  totalCost,
  remainingBudget,
  totalPredictedPoints,
  gameweek,
}: SummaryBarProps) {
  const isLineup = gameweek != null;

  return (
    <div className="summary-bar-shine relative overflow-hidden flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-gray-200 bg-white/80 px-5 py-4 text-sm shadow-sm backdrop-blur-sm dark:border-white/[0.08] dark:bg-white/[0.03] dark:shadow-xl dark:backdrop-blur-xl">
      {/* Floodlight top gradient */}
      <div className="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-amber-400/60 to-transparent pointer-events-none" />

      {/* Top border gradient */}
      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-amber-500/30 via-transparent to-amber-500/30 pointer-events-none" />

      <Stat label="Formation" value={formation} icon={<LayoutGrid className="size-3.5" />} />
      {isLineup ? (
        <>
          <Divider />
          <Stat label="Gameweek" value={String(gameweek)} icon={<Calendar className="size-3.5" />} />
          <Divider />
          <Stat
            label="Total GW Pts"
            value={totalPredictedPoints.toFixed(1)}
            icon={<Trophy className="size-3.5" />}
            highlight
          />
        </>
      ) : (
        <>
          <Divider />
          <Stat label="Total Cost" value={`\u00A3${totalCost.toFixed(1)}m`} icon={<Coins className="size-3.5" />} />
          <Divider />
          <Stat
            label="Remaining"
            value={`\u00A3${remainingBudget.toFixed(1)}m`}
            icon={<Wallet className="size-3.5" />}
            muted={remainingBudget < 0}
          />
          <Divider />
          <Stat
            label="Predicted Pts"
            value={totalPredictedPoints.toFixed(1)}
            icon={<Trophy className="size-3.5" />}
            highlight
          />
        </>
      )}
    </div>
  );
}

function Stat({
  label,
  value,
  icon,
  highlight,
  muted,
}: {
  label: string;
  value: string;
  icon?: React.ReactNode;
  highlight?: boolean;
  muted?: boolean;
}) {
  return (
    <div className="text-center">
      <p className="flex items-center justify-center gap-1 text-[11px] text-muted-foreground uppercase tracking-wide">
        {icon}
        {label}
      </p>
      {highlight ? (
        <p className="mt-0.5 inline-flex items-center gap-1.5 rounded-md bg-amber-500/15 px-2.5 py-0.5 text-2xl font-black text-amber-600 dark:text-amber-400 scoreboard-glow">
          <svg className="size-4 shrink-0" viewBox="0 0 16 16" fill="none">
            <circle cx="8" cy="8" r="7" stroke="currentColor" strokeWidth="1.5" fill="none" />
            <path d="M8 1.5L9.5 5.5L13 6L10.5 8.5L11.5 12L8 10L4.5 12L5.5 8.5L3 6L6.5 5.5Z" fill="currentColor" opacity="0.3" />
          </svg>
          {value}
        </p>
      ) : (
        <p
          className={`text-xl font-bold mt-0.5 ${
            muted ? "text-destructive" : ""
          }`}
        >
          {value}
        </p>
      )}
    </div>
  );
}

function Divider() {
  return <div className="hidden h-10 w-px bg-border sm:block" />;
}
