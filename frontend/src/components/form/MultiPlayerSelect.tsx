import { useState, useMemo } from "react";
import { X, ChevronsUpDown } from "lucide-react";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import {
  Command,
  CommandInput,
  CommandList,
  CommandEmpty,
  CommandGroup,
  CommandItem,
} from "@/components/ui/command";
import { Badge } from "@/components/ui/badge";
import type { PlayerListItem } from "@/api/types";

const POSITION_COLORS: Record<string, string> = {
  GK: "bg-yellow-500/20 text-yellow-700 dark:text-yellow-400",
  DEF: "bg-blue-500/20 text-blue-700 dark:text-blue-400",
  MID: "bg-green-500/20 text-green-700 dark:text-green-400",
  FWD: "bg-red-500/20 text-red-700 dark:text-red-400",
};

interface MultiPlayerSelectProps {
  label: string;
  players: PlayerListItem[];
  selectedIds: number[];
  onSelectionChange: (ids: number[]) => void;
  excludeIds?: number[];
  placeholder?: string;
}

export default function MultiPlayerSelect({
  label,
  players,
  selectedIds,
  onSelectionChange,
  excludeIds = [],
  placeholder = "Search players...",
}: MultiPlayerSelectProps) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");

  const excludeSet = useMemo(() => new Set(excludeIds), [excludeIds]);
  const selectedSet = useMemo(() => new Set(selectedIds), [selectedIds]);

  const availablePlayers = useMemo(
    () => players.filter((p) => !excludeSet.has(p.id)),
    [players, excludeSet],
  );

  const filteredPlayers = useMemo(() => {
    if (search.length < 2) return [];
    const query = search.toLowerCase();
    return availablePlayers
      .filter((p) => p.name.toLowerCase().includes(query))
      .slice(0, 50);
  }, [availablePlayers, search]);

  const selectedPlayers = useMemo(
    () => players.filter((p) => selectedSet.has(p.id)),
    [players, selectedSet],
  );

  function togglePlayer(id: number) {
    if (selectedSet.has(id)) {
      onSelectionChange(selectedIds.filter((sid) => sid !== id));
    } else {
      onSelectionChange([...selectedIds, id]);
    }
  }

  function removePlayer(id: number) {
    onSelectionChange(selectedIds.filter((sid) => sid !== id));
  }

  return (
    <div className="space-y-2">
      <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
        {label}
      </label>

      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger
          className="flex w-full items-center justify-between rounded-lg border border-input bg-transparent px-3 py-2 text-sm text-muted-foreground hover:bg-accent/50 dark:bg-input/30"
        >
          <span>{placeholder}</span>
          <ChevronsUpDown className="size-4 shrink-0 opacity-50" />
        </PopoverTrigger>
        <PopoverContent className="w-80 p-0" align="start">
          <Command shouldFilter={false}>
            <CommandInput placeholder="Type a player name..." value={search} onValueChange={setSearch} />
            <CommandList>
              <CommandEmpty>{search.length < 2 ? "Type at least 2 characters..." : "No players found."}</CommandEmpty>
              <CommandGroup>
                {filteredPlayers.map((player) => (
                  <CommandItem
                    key={player.id}
                    value={player.name}
                    onSelect={() => togglePlayer(player.id)}
                    data-checked={selectedSet.has(player.id)}
                  >
                    <span className="flex-1 truncate">{player.name}</span>
                    <span
                      className={`rounded px-1.5 py-0.5 text-xs font-medium ${POSITION_COLORS[player.position] ?? ""}`}
                    >
                      {player.position}
                    </span>
                    <span className="text-xs text-muted-foreground">
                      {player.team}
                    </span>
                    <span className="text-xs text-muted-foreground">
                      £{player.cost.toFixed(1)}m
                    </span>
                  </CommandItem>
                ))}
              </CommandGroup>
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>

      {selectedPlayers.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {selectedPlayers.map((player) => (
            <Badge key={player.id} variant="secondary" className="gap-1 pr-1">
              {player.name}
              <button
                type="button"
                onClick={() => removePlayer(player.id)}
                className="ml-0.5 rounded-full p-0.5 hover:bg-muted-foreground/20"
              >
                <X className="size-3" />
              </button>
            </Badge>
          ))}
        </div>
      )}
    </div>
  );
}
