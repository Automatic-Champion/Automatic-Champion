import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import type { DisplayPlayer } from "@/api/types";

interface BenchCardProps {
  player: DisplayPlayer;
  benchOrder: number;
  isSelected?: boolean;
  onClick?: () => void;
}

export default function BenchCard({
  player,
  benchOrder,
  isSelected,
  onClick,
}: BenchCardProps) {
  return (
    <Card
      size="sm"
      className={`cursor-pointer transition-colors hover:bg-accent/50 ${
        isSelected ? "ring-2 ring-primary" : ""
      }`}
      onClick={onClick}
    >
      <CardContent className="flex items-center gap-3 py-0">
        <span className="flex size-6 shrink-0 items-center justify-center rounded-full bg-muted text-xs font-bold text-muted-foreground">
          {benchOrder}
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
            £{player.cost.toFixed(1)}m
          </span>
        )}
        <span className="shrink-0 text-sm font-semibold">
          {player.points.toFixed(1)}
        </span>
      </CardContent>
    </Card>
  );
}
