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
    <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border bg-card px-4 py-3 text-sm ring-1 ring-foreground/10">
      <Stat label="Formation" value={formation} />
      {isLineup ? (
        <>
          <Divider />
          <Stat label="Gameweek" value={String(gameweek)} />
          <Divider />
          <Stat
            label="Total GW Pts"
            value={totalPredictedPoints.toFixed(1)}
            highlight
          />
        </>
      ) : (
        <>
          <Divider />
          <Stat label="Total Cost" value={`£${totalCost.toFixed(1)}m`} />
          <Divider />
          <Stat
            label="Remaining"
            value={`£${remainingBudget.toFixed(1)}m`}
            muted={remainingBudget < 0}
          />
          <Divider />
          <Stat
            label="Predicted Pts"
            value={totalPredictedPoints.toFixed(1)}
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
  highlight,
  muted,
}: {
  label: string;
  value: string;
  highlight?: boolean;
  muted?: boolean;
}) {
  return (
    <div className="text-center">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p
        className={`font-semibold ${
          highlight
            ? "text-primary"
            : muted
              ? "text-destructive"
              : ""
        }`}
      >
        {value}
      </p>
    </div>
  );
}

function Divider() {
  return <div className="hidden h-8 w-px bg-border sm:block" />;
}
