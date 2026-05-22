import { useState, useEffect } from "react";
import { motion, useReducedMotion } from "framer-motion";
import PageMeta from "../components/common/PageMeta";
import { PitchView } from "../components/pitch";
import type { PitchViewPlayer } from "../components/pitch";
import ConstraintForm from "../components/form/ConstraintForm";
import BenchCard from "../components/squad/BenchCard";
import ExplanationPanel from "../components/squad/ExplanationPanel";
import SummaryBar from "../components/squad/SummaryBar";
import PitchSVG from "../components/pitch/PitchSVG";
import { fetchPlayers, generateSquad } from "../api/client";
import { useSquad } from "../context/SquadContext";
import type {
  PlayerListItem,
  SquadGenerateRequest,
  SquadGenerateResponse,
  SquadPlayerResponse,
  DisplayPlayer,
} from "../api/types";

const EASE_OUT_EXPO = [0.16, 1, 0.3, 1] as const;

function squadPlayerToDisplay(p: SquadPlayerResponse): DisplayPlayer {
  return {
    id: p.id,
    name: p.name,
    position: p.position,
    team: p.team,
    points: p.predicted_points,
    pointsLabel: "Predicted Pts",
    cost: p.cost,
    benchOrder: p.bench_order,
    explanations: p.explanations,
  };
}

function startersToPitchPlayers(
  players: SquadPlayerResponse[]
): PitchViewPlayer[] {
  return players.map((p) => ({
    id: p.id,
    name: p.name,
    position: p.position,
    team: p.team,
    points: p.predicted_points,
  }));
}

function SkeletonPitch() {
  const circlePositions = [
    { x: 50, y: 88 },
    { x: 20, y: 68 }, { x: 40, y: 68 }, { x: 60, y: 68 }, { x: 80, y: 68 },
    { x: 20, y: 45 }, { x: 40, y: 45 }, { x: 60, y: 45 }, { x: 80, y: 45 },
    { x: 35, y: 22 }, { x: 65, y: 22 },
  ];

  return (
    <div className="relative w-full max-w-[600px] mx-auto rounded-xl overflow-hidden shadow-lg opacity-60">
      <PitchSVG />
      <div className="absolute inset-0">
        {circlePositions.map((pos, i) => (
          <div
            key={i}
            className="absolute -translate-x-1/2 -translate-y-1/2"
            style={{ left: `${pos.x}%`, top: `${pos.y}%` }}
          >
            <div
              className="w-8 h-8 flex items-center justify-center text-lg bounce-ball"
              style={{ animationDelay: `${i * 0.1}s` }}
            >
              <svg width="20" height="20" viewBox="0 0 20 20" className="opacity-70">
                <circle cx="10" cy="10" r="9" fill="white" stroke="#333" strokeWidth="1"/>
                <path d="M10 1L12 4L16 4L13 7L14 11L10 9L6 11L7 7L4 4L8 4Z" fill="#333" opacity="0.2"/>
              </svg>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function SkeletonSummaryBar() {
  return (
    <div className="skeleton-shimmer flex flex-wrap items-center justify-between gap-3 rounded-2xl border bg-card px-5 py-4 ring-1 ring-foreground/10">
      {[1, 2, 3, 4].map((i) => (
        <div key={i} className="text-center space-y-1">
          <div className="h-3 w-16 rounded bg-muted animate-pulse mx-auto" />
          <div className="h-6 w-14 rounded bg-muted animate-pulse mx-auto" />
        </div>
      ))}
    </div>
  );
}

function SkeletonBench() {
  return (
    <div className="space-y-2">
      <div className="h-4 w-12 rounded bg-muted animate-pulse" />
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
        {[1, 2, 3, 4].map((i) => (
          <div
            key={i}
            className="h-14 rounded-lg border bg-card animate-pulse"
          />
        ))}
      </div>
    </div>
  );
}

export default function SquadBuilder() {
  const [players, setPlayers] = useState<PlayerListItem[]>([]);
  const [playersLoading, setPlayersLoading] = useState(true);
  const [playersError, setPlayersError] = useState<string | null>(null);

  const [isGenerating, setIsGenerating] = useState(false);
  const { squad: squadResult, setSquad: setSquadResult } = useSquad();
  const [error, setError] = useState<string | null>(null);
  const [selectedPlayerId, setSelectedPlayerId] = useState<string | null>(null);

  const prefersReduced = useReducedMotion();
  const dur = (d: number) => (prefersReduced ? 0 : d);

  useEffect(() => {
    fetchPlayers()
      .then(setPlayers)
      .catch((err) => setPlayersError(err.message))
      .finally(() => setPlayersLoading(false));
  }, []);

  async function handleSubmit(request: SquadGenerateRequest) {
    setIsGenerating(true);
    setError(null);
    setSquadResult(null);
    setSelectedPlayerId(null);

    try {
      const result = await generateSquad(request);
      setSquadResult(result);
    } catch (err) {
      if (err instanceof TypeError && err.message === "Failed to fetch") {
        setError(
          "Cannot connect to the backend. Make sure the server is running on port 8000."
        );
      } else if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("An unexpected error occurred.");
      }
    } finally {
      setIsGenerating(false);
    }
  }

  function handlePlayerClick(playerId: string) {
    setSelectedPlayerId((prev) => (prev === playerId ? null : playerId));
  }

  const selectedRaw = selectedPlayerId
    ? (squadResult?.players.find((p) => p.id === selectedPlayerId) ??
      squadResult?.bench.find((p) => p.id === selectedPlayerId) ??
      null)
    : null;
  const selectedPlayer: DisplayPlayer | null = selectedRaw
    ? squadPlayerToDisplay(selectedRaw)
    : null;

  return (
    <>
      <PageMeta
        title="Squad Builder | Automatic Champion"
        description="Build your optimal FPL squad"
      />
      <motion.div
        className="space-y-6"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: dur(0.3) }}
      >
        <motion.h1
          className="text-4xl font-bold text-gray-900 dark:bg-gradient-to-r dark:from-white dark:via-amber-200 dark:to-amber-400 dark:bg-clip-text dark:text-transparent"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={prefersReduced ? { duration: 0 } : { type: "spring", stiffness: 200, damping: 20 }}
        >
          Squad Builder
        </motion.h1>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          {/* Left column: Constraint Form — glass card */}
          <div className="rounded-xl border border-gray-200 bg-white/80 p-5 shadow-sm backdrop-blur-sm dark:border-white/[0.08] dark:bg-white/[0.03] dark:shadow-xl dark:backdrop-blur-xl">
            {playersLoading ? (
              <div className="flex items-center justify-center py-12 text-sm text-muted-foreground">
                Loading players...
              </div>
            ) : playersError ? (
              <div className="rounded-lg bg-red-50 p-4 text-sm text-red-600 dark:bg-red-900/20 dark:text-red-400">
                Failed to load players: {playersError}
                <p className="mt-1 text-xs">
                  Make sure the backend is running on port 8000.
                </p>
              </div>
            ) : (
              <ConstraintForm
                onSubmit={handleSubmit}
                isLoading={isGenerating}
                players={players}
              />
            )}
          </div>

          {/* Right column: Results */}
          <div className="space-y-4 lg:col-span-2">
            {/* Error banner */}
            {error && (
              <div className="rounded-lg bg-red-50 p-4 text-sm text-red-600 dark:bg-red-900/20 dark:text-red-400">
                {error}
              </div>
            )}

            {/* Loading state */}
            {isGenerating && (
              <div className="space-y-4">
                <SkeletonSummaryBar />
                <SkeletonPitch />
                <SkeletonBench />
              </div>
            )}

            {/* Results */}
            {!isGenerating && squadResult && (
              <>
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: dur(0.5), ease: [...EASE_OUT_EXPO] }}
                >
                  <SummaryBar
                    formation={squadResult.formation}
                    totalCost={squadResult.total_cost}
                    remainingBudget={squadResult.budget - squadResult.total_cost}
                    totalPredictedPoints={squadResult.total_predicted_points}
                  />
                </motion.div>

                <motion.div
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: dur(0.6), delay: dur(0.15), ease: [...EASE_OUT_EXPO] }}
                  className="relative"
                >
                  {/* Ambient glow behind pitch — removed, now inside PitchView */}
                  <PitchView
                    formation={squadResult.formation}
                    players={startersToPitchPlayers(squadResult.players)}
                    selectedPlayerId={selectedPlayerId}
                    onPlayerClick={handlePlayerClick}
                  />
                </motion.div>

                {/* Bench */}
                <div className="space-y-2">
                  <h2 className="text-sm font-semibold text-muted-foreground">
                    Bench
                  </h2>
                  <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                    {[...squadResult.bench]
                      .sort(
                        (a, b) => (a.bench_order ?? 99) - (b.bench_order ?? 99)
                      )
                      .map((p, idx) => (
                        <motion.div
                          key={p.id}
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ duration: dur(0.4), delay: dur(0.3 + idx * 0.08), ease: [...EASE_OUT_EXPO] }}
                        >
                          <BenchCard
                            player={squadPlayerToDisplay(p)}
                            benchOrder={p.bench_order ?? 0}
                            isSelected={p.id === selectedPlayerId}
                            onClick={() => handlePlayerClick(p.id)}
                          />
                        </motion.div>
                      ))}
                  </div>
                </div>
              </>
            )}

            {/* Empty state */}
            {!isGenerating && !squadResult && !error && (
              <div className="flex items-center justify-center py-20 text-sm text-muted-foreground">
                Configure your constraints and click "Generate Squad" to begin.
              </div>
            )}
          </div>
        </div>
      </motion.div>

      {/* Explanation slide-out */}
      <ExplanationPanel
        player={selectedPlayer}
        onClose={() => setSelectedPlayerId(null)}
      />
    </>
  );
}
