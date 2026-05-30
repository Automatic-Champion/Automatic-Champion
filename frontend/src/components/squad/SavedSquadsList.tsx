import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { Loader2, Trash2, Upload } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  deleteSavedSquad,
  getSavedSquad,
  listSavedSquads,
} from "../../api/client";
import type { SavedSquadSummary } from "../../api/types";
import { useSquad } from "../../context/SquadContext";

function relativeTime(iso: string): string {
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "";
  const diffSec = Math.max(0, (Date.now() - then) / 1000);
  if (diffSec < 60) return "just now";
  const diffMin = Math.floor(diffSec / 60);
  if (diffMin < 60) return `${diffMin} minute${diffMin === 1 ? "" : "s"} ago`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr} hour${diffHr === 1 ? "" : "s"} ago`;
  const diffDay = Math.floor(diffHr / 24);
  if (diffDay < 30) return `${diffDay} day${diffDay === 1 ? "" : "s"} ago`;
  const diffMon = Math.floor(diffDay / 30);
  if (diffMon < 12) return `${diffMon} month${diffMon === 1 ? "" : "s"} ago`;
  const diffYr = Math.floor(diffMon / 12);
  return `${diffYr} year${diffYr === 1 ? "" : "s"} ago`;
}

interface Props {
  hasCurrentSquad: boolean;
}

export default function SavedSquadsList({ hasCurrentSquad }: Props) {
  const { setSquadFromSaved, currentSquadName } = useSquad();
  const [items, setItems] = useState<SavedSquadSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<number | null>(null);
  const [toast, setToast] = useState<string | null>(null);

  const prefersReduced = useReducedMotion();

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const list = await listSavedSquads();
      setItems(list);
    } catch (err) {
      if (err instanceof Error) setError(err.message);
      else setError("Failed to load saved squads.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  async function handleLoad(item: SavedSquadSummary) {
    setBusyId(item.id);
    setError(null);
    try {
      const full = await getSavedSquad(item.id);
      setSquadFromSaved(full);
      setToast(`Loaded "${full.name}"`);
      window.setTimeout(() => setToast(null), 3000);
    } catch (err) {
      if (err instanceof Error) setError(err.message);
      else setError("Failed to load squad.");
    } finally {
      setBusyId(null);
    }
  }

  async function handleDelete(item: SavedSquadSummary) {
    if (!window.confirm(`Delete saved squad "${item.name}"?`)) return;
    setBusyId(item.id);
    setError(null);
    try {
      await deleteSavedSquad(item.id);
      setItems((prev) => prev.filter((s) => s.id !== item.id));
    } catch (err) {
      if (err instanceof Error) setError(err.message);
      else setError("Failed to delete squad.");
    } finally {
      setBusyId(null);
    }
  }

  if (loading) {
    return (
      <div className="rounded-xl border border-gray-200 bg-white/80 p-5 shadow-sm backdrop-blur-sm dark:border-white/[0.08] dark:bg-white/[0.03] dark:shadow-xl dark:backdrop-blur-xl">
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="size-4 animate-spin" />
          Loading saved squads...
        </div>
      </div>
    );
  }

  if (!hasCurrentSquad && items.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center rounded-xl border border-gray-200 bg-white/80 py-12 backdrop-blur-sm dark:border-white/[0.08] dark:bg-white/[0.03] dark:backdrop-blur-xl">
        <p className="text-sm text-muted-foreground">
          No squads yet. Go to Squad Builder to create your first one.
        </p>
        <Link to="/">
          <Button variant="outline" className="mt-4">
            Go to Squad Builder
          </Button>
        </Link>
      </div>
    );
  }

  if (items.length === 0) {
    return null;
  }

  return (
    <div className="rounded-xl border border-gray-200 bg-white/80 p-5 shadow-sm backdrop-blur-sm dark:border-white/[0.08] dark:bg-white/[0.03] dark:shadow-xl dark:backdrop-blur-xl">
      <div className="mb-3 flex items-center justify-between gap-2">
        <h2 className="text-sm font-semibold text-gray-900 dark:text-white">
          Saved squads ({items.length}/10)
        </h2>
        {toast && (
          <span className="text-xs text-emerald-500" role="status">
            {toast}
          </span>
        )}
      </div>

      {error && (
        <div className="mb-3 rounded-lg bg-red-50 p-2 text-xs text-red-600 dark:bg-red-900/20 dark:text-red-400">
          {error}
        </div>
      )}

      <ul className="space-y-2">
        <AnimatePresence initial={false}>
          {items.map((item) => {
            const isActive = item.name === currentSquadName;
            return (
              <motion.li
                key={item.id}
                layout={prefersReduced ? false : true}
                initial={{ opacity: 0, y: -6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -6 }}
                transition={{ duration: prefersReduced ? 0 : 0.2 }}
                className={`flex flex-wrap items-center gap-3 rounded-lg border p-3 ${
                  isActive
                    ? "border-amber-500/40 bg-amber-500/5"
                    : "border-gray-200 dark:border-white/[0.08]"
                }`}
              >
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="truncate text-sm font-semibold">
                      {item.name}
                    </span>
                    <span className="shrink-0 rounded-md bg-muted px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
                      {item.formation}
                    </span>
                  </div>
                  <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-xs text-muted-foreground">
                    <span>
                      {item.total_predicted_points.toFixed(1)} pred. pts
                    </span>
                    <span>{relativeTime(item.created_at)}</span>
                  </div>
                </div>
                <div className="flex shrink-0 items-center gap-2">
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => handleLoad(item)}
                    disabled={busyId === item.id}
                  >
                    {busyId === item.id ? (
                      <Loader2 className="size-3.5 animate-spin" />
                    ) : (
                      <Upload className="size-3.5" />
                    )}
                    Load
                  </Button>
                  <Button
                    size="sm"
                    variant="destructive"
                    onClick={() => handleDelete(item)}
                    disabled={busyId === item.id}
                    aria-label={`Delete ${item.name}`}
                  >
                    <Trash2 className="size-3.5" />
                  </Button>
                </div>
              </motion.li>
            );
          })}
        </AnimatePresence>
      </ul>
    </div>
  );
}
