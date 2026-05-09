import { useState, useEffect } from "react";
import { Link } from "react-router";
import { Loader2 } from "lucide-react";
import PageMeta from "../components/common/PageMeta";
import { PitchView } from "../components/pitch";
import type { PitchViewPlayer } from "../components/pitch";
import BenchCard from "../components/squad/BenchCard";
import ExplanationPanel from "../components/squad/ExplanationPanel";
import SummaryBar from "../components/squad/SummaryBar";
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

export default function LineupAdvisor() {
  const [squad, setSquad] = useState<SquadGenerateResponse | null>(null);
  const [formation, setFormation] = useState("Auto (best)");
  const [gameweek, setGameweek] = useState("Auto-detect");
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<LineupRecommendResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedPlayerId, setSelectedPlayerId] = useState<string | null>(null);

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

  // Build display players from result
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
      <div className="space-y-6">
        <h1 className="text-2xl font-semibold text-gray-900 dark:text-white">
          Lineup Advisor
        </h1>

        {!squad ? (
          <div className="flex flex-col items-center justify-center rounded-xl border border-gray-200 bg-white py-20 dark:border-gray-700 dark:bg-gray-800">
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
            {/* Squad summary */}
            <div className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm dark:border-gray-700 dark:bg-gray-800">
              <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
                <h2 className="text-sm font-semibold text-gray-900 dark:text-white">
                  Your Squad (15 players)
                </h2>
                <div className="flex gap-3 text-xs text-muted-foreground">
                  <span>Formation: {squad.formation}</span>
                  <span>Cost: £{squad.total_cost.toFixed(1)}m</span>
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

            {/* Controls */}
            <div className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm dark:border-gray-700 dark:bg-gray-800">
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
              <Button
                className="mt-4 w-full"
                onClick={handleRecommend}
                disabled={isLoading}
              >
                {isLoading && <Loader2 className="mr-2 size-4 animate-spin" />}
                Get Lineup Recommendation
              </Button>
            </div>

            {/* Error banner */}
            {error && (
              <div className="rounded-lg bg-red-50 p-4 text-sm text-red-600 dark:bg-red-900/20 dark:text-red-400">
                {error}
              </div>
            )}

            {/* Loading state */}
            {isLoading && (
              <div className="flex flex-col items-center justify-center py-20 text-muted-foreground">
                <Loader2 className="mb-3 size-8 animate-spin" />
                <p className="text-sm">Analyzing lineup...</p>
              </div>
            )}

            {/* Results */}
            {!isLoading && result && (
              <>
                <SummaryBar
                  formation={result.formation}
                  totalCost={0}
                  remainingBudget={0}
                  totalPredictedPoints={result.total_gw_points}
                  gameweek={result.gameweek}
                />

                <PitchView
                  formation={result.formation}
                  players={pitchPlayers}
                  selectedPlayerId={selectedPlayerId}
                  onPlayerClick={handlePlayerClick}
                />

                {/* Bench */}
                <div className="space-y-2">
                  <h2 className="text-sm font-semibold text-muted-foreground">
                    Bench
                  </h2>
                  <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                    {benchDisplayPlayers.map((p) => (
                      <BenchCard
                        key={p.id}
                        player={p}
                        benchOrder={p.benchOrder ?? 0}
                        isSelected={p.id === selectedPlayerId}
                        onClick={() => handlePlayerClick(p.id)}
                      />
                    ))}
                  </div>
                </div>
              </>
            )}
          </>
        )}
      </div>

      {/* Explanation slide-out */}
      <ExplanationPanel
        player={selectedPlayer}
        onClose={() => setSelectedPlayerId(null)}
      />
    </>
  );
}
