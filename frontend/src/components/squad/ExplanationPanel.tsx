import { X } from "lucide-react";
import { motion, useReducedMotion } from "framer-motion";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import type { DisplayPlayer } from "@/api/types";
import {
  categoryConfig,
  defaultCategoryConfig,
} from "@/lib/explanationCategories";

interface ExplanationPanelProps {
  player: DisplayPlayer | null;
  onClose: () => void;
}

const POSITION_BG: Record<string, string> = {
  GK: "from-yellow-500/15",
  DEF: "from-blue-500/15",
  MID: "from-green-500/15",
  FWD: "from-red-500/15",
};

const POSITION_TEXT: Record<string, string> = {
  GK: "text-yellow-500 border-yellow-500/30",
  DEF: "text-blue-500 border-blue-500/30",
  MID: "text-green-500 border-green-500/30",
  FWD: "text-red-500 border-red-500/30",
};

function ExplanationList({ player }: { player: DisplayPlayer }) {
  const prefersReduced = useReducedMotion();
  const dur = (d: number) => (prefersReduced ? 0 : d);

  return (
    <div className="mt-6 space-y-4">
      <h3 className="text-sm font-semibold">Why this player?</h3>
      {player.explanations.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          No explanations available for this player.
        </p>
      ) : (
        <ul className="space-y-3">
          {player.explanations.map((exp, i) => {
            const config = categoryConfig[exp.category] ?? defaultCategoryConfig;
            return (
              <motion.li
                key={i}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: dur(0.35), delay: dur(i * 0.08) }}
                className={`rounded-lg border border-l-4 ${config.borderColor} bg-muted/30 p-4 text-sm`}
              >
                <p>{exp.text}</p>
                <div className="mt-2">
                  <span
                    className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium ${config.color} ${config.darkColor}`}
                  >
                    {config.icon}
                    {exp.category}
                  </span>
                </div>
              </motion.li>
            );
          })}
        </ul>
      )}
    </div>
  );
}

export default function ExplanationPanel({
  player,
  onClose,
}: ExplanationPanelProps) {
  return (
    <Sheet open={player !== null} onOpenChange={(open) => !open && onClose()}>
      <SheetContent className="w-full sm:max-w-md overflow-y-auto overflow-x-visible">
        {player && (
          <>
            <SheetHeader>
              {/* Gradient header background */}
              <div className={`relative px-0 pt-16 pb-4 bg-gradient-to-b ${POSITION_BG[player.position] ?? "from-gray-500/15"} to-transparent`}>
                <div className="flex items-start justify-between">
                  <div className="flex items-start gap-3">
                    {/* Position badge circle */}
                    <div className={`flex size-10 shrink-0 items-center justify-center rounded-full border-2 ${POSITION_TEXT[player.position] ?? "text-gray-500 border-gray-500/30"} text-xs font-bold`}>
                      {player.position}
                    </div>
                    <div>
                      <SheetTitle className="text-xl font-bold">{player.name}</SheetTitle>
                      <div className="mt-1 flex items-center gap-2">
                        <Badge variant="outline">{player.position}</Badge>
                        <span className="text-sm text-muted-foreground">
                          {player.team}
                        </span>
                      </div>
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
                      <p className="font-semibold">{"\u00A3"}{player.cost.toFixed(1)}m</p>
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
              </div>
            </SheetHeader>

            <ExplanationList player={player} />
          </>
        )}
      </SheetContent>
    </Sheet>
  );
}
