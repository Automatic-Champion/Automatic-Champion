import { useState } from "react";
import { Loader2, Sparkles } from "lucide-react";
import { motion, useReducedMotion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import MultiPlayerSelect from "./MultiPlayerSelect";
import type { SquadGenerateRequest } from "@/api/types";
import type { PlayerListItem } from "@/api/types";

const FORMATIONS = ["3-4-3", "3-5-2", "4-3-3", "4-4-2", "4-5-1", "5-3-2", "5-4-1"];

interface ConstraintFormProps {
  onSubmit: (request: SquadGenerateRequest) => void;
  isLoading: boolean;
  players: PlayerListItem[];
}

export default function ConstraintForm({
  onSubmit,
  isLoading,
  players,
}: ConstraintFormProps) {
  const [budget, setBudget] = useState(100.0);
  const [formation, setFormation] = useState("4-4-2");
  const [lockedIds, setLockedIds] = useState<number[]>([]);
  const [bannedIds, setBannedIds] = useState<number[]>([]);
  const prefersReduced = useReducedMotion();

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    onSubmit({
      budget,
      formation,
      locked_ids: lockedIds,
      banned_ids: bannedIds,
    });
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* Budget Slider */}
      <div className="space-y-3 rounded-lg bg-muted/30 p-4">
        <div className="flex items-center justify-between">
          <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Budget
          </label>
          <span className="text-sm font-semibold text-gray-900 dark:text-white">
            {"\u00A3"}{budget.toFixed(1)}m
          </span>
        </div>
        <Slider
          min={50}
          max={150}
          step={0.5}
          value={[budget]}
          onValueChange={(val) => {
            if (Array.isArray(val)) setBudget(val[0]);
          }}
        />
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>{"\u00A3"}50.0m</span>
          <span>{"\u00A3"}150.0m</span>
        </div>
      </div>

      {/* Formation Select */}
      <div className="space-y-2 rounded-lg bg-muted/30 p-4">
        <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Formation
        </label>
        <Select value={formation} onValueChange={(val) => { if (val) setFormation(val); }}>
          <SelectTrigger className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {FORMATIONS.map((f) => (
              <SelectItem key={f} value={f}>
                {f}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Player selection */}
      <div className="space-y-4 rounded-lg bg-muted/30 p-4">
        {/* Must Include */}
        <MultiPlayerSelect
          label="Must Include"
          players={players}
          selectedIds={lockedIds}
          onSelectionChange={setLockedIds}
          excludeIds={bannedIds}
          placeholder="Search players to include..."
        />

        {/* Exclude */}
        <MultiPlayerSelect
          label="Exclude"
          players={players}
          selectedIds={bannedIds}
          onSelectionChange={setBannedIds}
          excludeIds={lockedIds}
          placeholder="Search players to exclude..."
        />
      </div>

      {/* Generate Button — premium animated gradient in dark mode */}
      <motion.div
        whileTap={prefersReduced ? undefined : { scale: 0.98 }}
        whileHover={prefersReduced ? undefined : { scale: 1.02 }}
        transition={{ type: "spring", stiffness: 400, damping: 25 }}
      >
        <Button
          type="submit"
          className="w-full shadow-md btn-gradient-shift dark:text-gray-900 dark:shadow-amber-500/20 dark:shadow-lg"
          style={{
            backgroundImage: "linear-gradient(90deg, #d97706, #f59e0b, #d97706)",
            backgroundSize: "200% 200%",
          }}
          disabled={isLoading}
        >
          {isLoading ? (
            <>
              <Loader2 className="mr-2 size-4 animate-spin" />
              Generating...
            </>
          ) : (
            <>
              <Sparkles className="mr-2 size-4" />
              Generate Squad
            </>
          )}
        </Button>
      </motion.div>
    </form>
  );
}
