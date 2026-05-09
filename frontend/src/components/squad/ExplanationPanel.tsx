import { X } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import type { DisplayPlayer } from "@/api/types";

interface ExplanationPanelProps {
  player: DisplayPlayer | null;
  onClose: () => void;
}

const categoryColors: Record<string, string> = {
  performance: "bg-blue-100 text-blue-800",
  attacking: "bg-red-100 text-red-800",
  defensive: "bg-green-100 text-green-800",
  reliability: "bg-amber-100 text-amber-800",
  value: "bg-purple-100 text-purple-800",
  trending: "bg-cyan-100 text-cyan-800",
};

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
                {player.cost != null && (
                  <div>
                    <span className="text-muted-foreground">Cost</span>
                    <p className="font-semibold">£{player.cost.toFixed(1)}m</p>
                  </div>
                )}
                <div>
                  <span className="text-muted-foreground">
                    {player.pointsLabel}
                  </span>
                  <p className="font-semibold">
                    {player.points.toFixed(1)}
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
                      <p>{exp.text}</p>
                      <div className="mt-2">
                        <span
                          className={`inline-block rounded-full px-2 py-0.5 text-xs font-medium ${categoryColors[exp.category] ?? "bg-gray-100 text-gray-800"}`}
                        >
                          {exp.category}
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
