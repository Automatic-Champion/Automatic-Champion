import { useState, useEffect } from "react";
import { Link } from "react-router";
import { Loader2 } from "lucide-react";
import { motion, useReducedMotion } from "framer-motion";
import PageMeta from "../components/common/PageMeta";
import { PitchView } from "../components/pitch";
import type { PitchViewPlayer } from "../components/pitch";
import BenchCard from "../components/squad/BenchCard";
import ExplanationPanel from "../components/squad/ExplanationPanel";
import SummaryBar from "../components/squad/SummaryBar";
import PitchSVG from "../components/pitch/PitchSVG";
import { recommendLineup } from "../api/client";
import { squadToLineupInput } from "../api/helpers";
import type {
  SquadGenerateResponse,
  LineupRecommendResponse,
  DisplayPlayer,
} from "../api/types";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

const FORMATIONS = [
  "Auto (best)",
  "3-4-3",
  "3-5-2",
  "4-3-3",
  "4-4-2",
  "4-5-1",
  "5-3-2",
  "5-4-1",
];

const GAMEWEEK_OPTIONS = [
  "Auto-detect",
  ...Array.from({ length: 38 }, (_, i) => String(i + 1)),
];

const EASE_OUT_EXPO = [0.16, 1, 0.3, 1] as const;

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
      {[1, 2, 3].map((i) => (
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

export default function LineupAdvisor() {
  const [squad, setSquad] = useState<SquadGenerateResponse | null>(null);
  const [formation, setFormation] = useState("Auto (best)");
  const [gameweek, setGameweek] = useState("Auto-detect");
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<LineupRecommendResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedPlayerId, setSelectedPlayerId] = useState<string | null>(null);

  const prefersReduced = useReducedMotion();
  const dur = (d: number) => (prefersReduced ? 0 : d);

  useEffect(() => {
    const stored = localStorage.getItem("lastSquad");
    if (stored) {
      try {
        setSquad(JSON.parse(stored) as SquadGenerateResponse);
      } catch {
        setSquad(null);
      }
    }
  }, []);

  async function handleRecommend() {
    if (!squad) return;
    setIsLoading(true);
    setError(null);
    setResult(null);
    setSelectedPlayerId(null);

    try {
      const lineupInput = squadToLineupInput(squad.players, squad.bench);
      const res = await recommendLineup({
        squad: lineupInput,
        formation: formation === "Auto (best)" ? null : formation,
        gameweek: gameweek === "Auto-detect" ? null : Number(gameweek),
      });
      setResult(res);
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
      setIsLoading(false);
    }
  }

  function handlePlayerClick(playerId: string) {
    setSelectedPlayerId((prev) => (prev === playerId ? null : playerId));
  }

  const allDisplayPlayers: DisplayPlayer[] = result
    ? [
        ...result.starters.map((s) => ({
          id: s.id,
          name: s.name,
          position: s.position,
          team: s.team,
          points: s.gw_points,
          pointsLabel: "GW Points",
          isCaptain: s.id === result.captain_id,
          isViceCaptain: s.id === result.vice_captain_id,
          explanations: s.explanations,
        })),
        ...result.bench.map((b) => ({
          id: b.id,
          name: b.name,
          position: b.position,
          team: b.team,
          points: b.gw_points,
          pointsLabel: "GW Points",
          benchOrder: b.bench_order,
          explanations: b.explanations,
        })),
      ]
    : [];

  const selectedPlayer: DisplayPlayer | null = selectedPlayerId
    ? (allDisplayPlayers.find((p) => p.id === selectedPlayerId) ?? null)
    : null;

  const pitchPlayers: PitchViewPlayer[] = result
    ? result.starters.map((s) => ({
        id: s.id,
        name: s.name,
        position: s.position,
        team: s.team,
        points: s.gw_points,
        isCaptain: s.id === result.captain_id,
        isViceCaptain: s.id === result.vice_captain_id,
      }))
    : [];

  const benchDisplayPlayers = allDisplayPlayers
    .filter((p) => p.benchOrder != null)
    .sort((a, b) => (a.benchOrder ?? 99) - (b.benchOrder ?? 99));

  return (
    <>
      <PageMeta
        title="Lineup Advisor | Automatic Champion"
        description="Get optimal weekly lineup recommendations"
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
          Lineup Advisor
        </motion.h1>

        {!squad ? (
          <div className="flex flex-col items-center justify-center rounded-xl border border-gray-200 bg-white/80 py-20 backdrop-blur-sm dark:border-white/[0.08] dark:bg-white/[0.03] dark:backdrop-blur-xl">
            <p className="text-sm text-muted-foreground">
              No squad found. Go to Squad Builder to generate one first.
            </p>
            <Link to="/">
              <Button variant="outline" className="mt-4">
                Go to Squad Builder
              </Button>
            </Link>
          </div>
        ) : (
          <>
            {/* Squad summary — glass card */}
            <div className="rounded-xl border border-gray-200 bg-white/80 p-5 shadow-sm backdrop-blur-sm dark:border-white/[0.08] dark:bg-white/[0.03] dark:shadow-xl dark:backdrop-blur-xl">
              <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
                <h2 className="text-sm font-semibold text-gray-900 dark:text-white">
                  Your Squad (15 players)
                </h2>
                <div className="flex gap-3 text-xs text-muted-foreground">
                  <span>Formation: {squad.formation}</span>
                  <span>Cost: {"\u00A3"}{squad.total_cost.toFixed(1)}m</span>
                </div>
              </div>
              <div className="grid grid-cols-1 gap-1 sm:grid-cols-3">
                {[...squad.players, ...squad.bench].map((p) => (
                  <div
                    key={p.id}
                    className="flex items-center gap-2 rounded px-2 py-1 text-sm"
                  >
                    <span className="w-8 shrink-0 text-xs font-medium text-muted-foreground">
                      {p.position}
                    </span>
                    <span className="min-w-0 flex-1 truncate">{p.name}</span>
                    <span className="shrink-0 text-xs text-muted-foreground">
                      {p.team}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Controls — glass card */}
            <div className="rounded-xl border border-gray-200 bg-white/80 p-5 shadow-sm backdrop-blur-sm dark:border-white/[0.08] dark:bg-white/[0.03] dark:shadow-xl dark:backdrop-blur-xl">
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <div className="space-y-1.5">
                  <label className="text-sm font-medium">Formation</label>
                  <Select value={formation} onValueChange={(v) => v && setFormation(v)}>
                    <SelectTrigger>
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
                <div className="space-y-1.5">
                  <label className="text-sm font-medium">Gameweek</label>
                  <Select value={gameweek} onValueChange={(v) => v && setGameweek(v)}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {GAMEWEEK_OPTIONS.map((gw) => (
                        <SelectItem key={gw} value={gw}>
                          {gw === "Auto-detect" ? "Auto-detect" : `GW ${gw}`}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <motion.div
                whileTap={prefersReduced ? undefined : { scale: 0.98 }}
                whileHover={prefersReduced ? undefined : { scale: 1.02 }}
                transition={{ type: "spring", stiffness: 400, damping: 25 }}
              >
                <Button
                  className="mt-4 w-full shadow-md btn-gradient-shift dark:text-gray-900 dark:shadow-amber-500/20 dark:shadow-lg"
                  style={{
                    backgroundImage: "linear-gradient(90deg, #d97706, #f59e0b, #d97706)",
                    backgroundSize: "200% 200%",
                  }}
                  onClick={handleRecommend}
                  disabled={isLoading}
                >
                  {isLoading && <Loader2 className="mr-2 size-4 animate-spin" />}
                  Get Lineup Recommendation
                </Button>
              </motion.div>
            </div>

            {/* Error banner */}
            {error && (
              <div className="rounded-lg bg-red-50 p-4 text-sm text-red-600 dark:bg-red-900/20 dark:text-red-400">
                {error}
              </div>
            )}

            {/* Loading state */}
            {isLoading && (
              <div className="space-y-4">
                <SkeletonSummaryBar />
                <SkeletonPitch />
                <SkeletonBench />
              </div>
            )}

            {/* Results */}
            {!isLoading && result && (
              <>
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: dur(0.5), ease: [...EASE_OUT_EXPO] }}
                >
                  <SummaryBar
                    formation={result.formation}
                    totalCost={0}
                    remainingBudget={0}
                    totalPredictedPoints={result.total_gw_points}
                    gameweek={result.gameweek}
                  />
                </motion.div>

                <motion.div
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: dur(0.6), delay: dur(0.15), ease: [...EASE_OUT_EXPO] }}
                  className="relative"
                >
                  {/* Ambient glow — removed, now inside PitchView */}
                  <PitchView
                    formation={result.formation}
                    players={pitchPlayers}
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
                    {benchDisplayPlayers.map((p, idx) => (
                      <motion.div
                        key={p.id}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ duration: dur(0.4), delay: dur(0.3 + idx * 0.08), ease: [...EASE_OUT_EXPO] }}
                      >
                        <BenchCard
                          player={p}
                          benchOrder={p.benchOrder ?? 0}
                          isSelected={p.id === selectedPlayerId}
                          onClick={() => handlePlayerClick(p.id)}
                        />
                      </motion.div>
                    ))}
                  </div>
                </div>
              </>
            )}
          </>
        )}
      </motion.div>

      {/* Explanation slide-out */}
      <ExplanationPanel
        player={selectedPlayer}
        onClose={() => setSelectedPlayerId(null)}
      />
    </>
  );
}
