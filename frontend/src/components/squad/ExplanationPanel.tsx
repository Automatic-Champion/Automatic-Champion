import { X } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import type { SquadPlayerResponse } from "@/api/types";

interface ExplanationPanelProps {
  player: SquadPlayerResponse | null;
  onClose: () => void;
}

export default function ExplanationPanel({
  player,
  onClose,
}: ExplanationPanelProps) {
  return (
    <Sheet open={player !== null} onOpenChange={(open) => !open && onClose()}>
      <SheetContent className="w-full sm:max-w-md overflow-y-auto">
        {player && (
          <>
            <SheetHeader>
              <div className="flex items-start justify-between">
                <div>
                  <SheetTitle className="text-lg">{player.name}</SheetTitle>
                  <div className="mt-1 flex items-center gap-2">
                    <Badge variant="outline">{player.position}</Badge>
                    <span className="text-sm text-muted-foreground">
                      {player.team}
                    </span>
                  </div>
                </div>
                <Button variant="ghost" size="icon" onClick={onClose}>
                  <X className="size-4" />
                </Button>
              </div>
              <div className="mt-3 flex gap-4 text-sm">
                <div>
                  <span className="text-muted-foreground">Cost</span>
                  <p className="font-semibold">£{player.cost.toFixed(1)}m</p>
                </div>
                <div>
                  <span className="text-muted-foreground">Predicted Pts</span>
                  <p className="font-semibold">
                    {player.predicted_points.toFixed(1)}
                  </p>
                </div>
              </div>
            </SheetHeader>

            <div className="mt-6 space-y-4">
              <h3 className="text-sm font-semibold">Why this player?</h3>
              {player.explanations.length === 0 ? (
                <p className="text-sm text-muted-foreground">
                  No explanations available for this player.
                </p>
              ) : (
                <ul className="space-y-3">
                  {player.explanations.map((exp, i) => (
                    <li
                      key={i}
                      className="rounded-lg border bg-muted/30 p-3 text-sm"
                    >
                      <p>{exp.explanation}</p>
                      <div className="mt-2 flex items-center gap-2">
                        <span className="text-xs text-muted-foreground">
                          {exp.feature}
                        </span>
                        <div className="flex-1">
                          <div className="h-1.5 w-full rounded-full bg-muted">
                            <div
                              className="h-1.5 rounded-full bg-primary"
                              style={{
                                width: `${Math.min(Math.abs(exp.importance) * 100, 100)}%`,
                              }}
                            />
                          </div>
                        </div>
                        <span className="text-xs font-medium">
                          {(Math.abs(exp.importance) * 100).toFixed(0)}%
                        </span>
                      </div>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </>
        )}
      </SheetContent>
    </Sheet>
  );
}
