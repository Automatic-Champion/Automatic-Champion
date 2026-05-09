import { useState } from "react";
import { Loader2 } from "lucide-react";
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
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Budget
          </label>
          <span className="text-sm font-semibold text-gray-900 dark:text-white">
            £{budget.toFixed(1)}m
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
          <span>£50.0m</span>
          <span>£150.0m</span>
        </div>
      </div>

      {/* Formation Select */}
      <div className="space-y-2">
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

      {/* Generate Button */}
      <Button type="submit" className="w-full" disabled={isLoading}>
        {isLoading ? (
          <>
            <Loader2 className="mr-2 size-4 animate-spin" />
            Generating...
          </>
        ) : (
          "Generate Squad"
        )}
      </Button>
    </form>
  );
}
