import { useState, useEffect } from "react";
import { Loader2 } from "lucide-react";
import PageMeta from "../components/common/PageMeta";
import { PitchView } from "../components/pitch";
import type { PitchViewPlayer } from "../components/pitch";
import ConstraintForm from "../components/form/ConstraintForm";
import BenchCard from "../components/squad/BenchCard";
import ExplanationPanel from "../components/squad/ExplanationPanel";
import SummaryBar from "../components/squad/SummaryBar";
import { fetchPlayers, generateSquad } from "../api/client";
import type {
  PlayerListItem,
  SquadGenerateRequest,
  SquadGenerateResponse,
  SquadPlayerResponse,
  DisplayPlayer,
} from "../api/types";

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

export default function SquadBuilder() {
  const [players, setPlayers] = useState<PlayerListItem[]>([]);
  const [playersLoading, setPlayersLoading] = useState(true);
  const [playersError, setPlayersError] = useState<string | null>(null);

  const [isGenerating, setIsGenerating] = useState(false);
  const [squadResult, setSquadResult] = useState<SquadGenerateResponse | null>(
    null
  );
  const [error, setError] = useState<string | null>(null);
  const [selectedPlayerId, setSelectedPlayerId] = useState<string | null>(null);

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
      localStorage.setItem("lastSquad", JSON.stringify(result));
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
      <div className="space-y-6">
        <h1 className="text-2xl font-semibold text-gray-900 dark:text-white">
          Squad Builder
        </h1>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          {/* Left column: Constraint Form */}
          <div className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm dark:border-gray-700 dark:bg-gray-800">
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
              <div className="flex flex-col items-center justify-center py-20 text-muted-foreground">
                <Loader2 className="mb-3 size-8 animate-spin" />
                <p className="text-sm">Optimizing squad...</p>
                <p className="mt-1 text-xs">This may take up to 30 seconds.</p>
              </div>
            )}

            {/* Results */}
            {!isGenerating && squadResult && (
              <>
                <SummaryBar
                  formation={squadResult.formation}
                  totalCost={squadResult.total_cost}
                  remainingBudget={squadResult.budget - squadResult.total_cost}
                  totalPredictedPoints={squadResult.total_predicted_points}
                />

                <PitchView
                  formation={squadResult.formation}
                  players={startersToPitchPlayers(squadResult.players)}
                  selectedPlayerId={selectedPlayerId}
                  onPlayerClick={handlePlayerClick}
                />

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
                      .map((p) => (
                        <BenchCard
                          key={p.id}
                          player={squadPlayerToDisplay(p)}
                          benchOrder={p.bench_order ?? 0}
                          isSelected={p.id === selectedPlayerId}
                          onClick={() => handlePlayerClick(p.id)}
                        />
                      ))}
                  </div>
                </div>
              </>
            )}

            {/* Empty state — no results yet, not loading */}
            {!isGenerating && !squadResult && !error && (
              <div className="flex items-center justify-center py-20 text-sm text-muted-foreground">
                Configure your constraints and click "Generate Squad" to begin.
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Explanation slide-out */}
      <ExplanationPanel
        player={selectedPlayer}
        onClose={() => setSelectedPlayerId(null)}
      />
    </>
  );
}
