import { useMemo, useState } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { Search, X } from "lucide-react";
import PageMeta from "../components/common/PageMeta";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  categoryConfig,
  defaultCategoryConfig,
} from "@/lib/explanationCategories";

type Where = "Squad Builder" | "Lineup Advisor" | "Both";

interface Entry {
  term: string;
  where: Where;
  description: string;
  /** Extra words the search should match (abbreviations, column names). */
  keywords?: string[];
}

interface Section {
  id: string;
  title: string;
  blurb: string;
  entries: Entry[];
}

// ── Content ───────────────────────────────────────────────────────────────
// Everything below describes what the app actually shows. If the optimiser,
// the explainers or the summary bar change, update these entries too.

const SCREEN_ENTRIES: Entry[] = [
  {
    term: "Predicted Pts",
    where: "Squad Builder",
    description:
      "How many FPL points the season model expects a player to score across a whole season. It is the number the optimiser maximises when it picks your 15.",
    keywords: ["prediction", "projected", "season points"],
  },
  {
    term: "Predicted Pts (squad total)",
    where: "Squad Builder",
    description:
      "The total in the summary bar counts your starting 11 only, so the four bench players never inflate it.",
    keywords: ["total predicted points", "summary"],
  },
  {
    term: "Total Cost",
    where: "Squad Builder",
    description:
      "What all 15 players cost together, in millions. It always includes the bench, because in FPL you pay for your bench too.",
    keywords: ["price", "budget", "money"],
  },
  {
    term: "Remaining",
    where: "Squad Builder",
    description:
      "Your budget minus the total cost — the money left in the bank. The optimiser is allowed to leave some behind if spending it would not buy better points.",
    keywords: ["budget", "bank", "left over"],
  },
  {
    term: "GW Points",
    where: "Lineup Advisor",
    description:
      "The weekly model's projection for one single gameweek, not the season. A player with a great season projection can still have a low GW number against a hard fixture.",
    keywords: ["gameweek points", "weekly prediction"],
  },
  {
    term: "Total GW Pts",
    where: "Lineup Advisor",
    description:
      "The sum of your starting 11 for that gameweek, with the captain counted twice — the same way FPL scores it.",
    keywords: ["total", "captain double"],
  },
  {
    term: "Formation",
    where: "Both",
    description:
      "The shape of the 10 outfield starters, e.g. 4-4-2 means 4 defenders, 4 midfielders and 2 forwards. A goalkeeper is always the 11th starter, so he is never written in the formation.",
    keywords: ["shape", "442", "343", "352"],
  },
  {
    term: "Gameweek",
    where: "Lineup Advisor",
    description:
      "Which round of Premier League matches the lineup is for. Change it and both the predictions and the reasons behind them change, because a different gameweek means a different fixture.",
    keywords: ["gw", "week", "round"],
  },
  {
    term: "Captain (C) and Vice-captain (VC)",
    where: "Lineup Advisor",
    description:
      "Your captain scores double points. The advisor gives the armband to the starter with the highest projected points and the vice-captaincy to the second highest; the vice-captain takes over automatically if the captain does not play.",
    keywords: ["armband", "double points", "c", "vc"],
  },
  {
    term: "Bench order (1, 2, 3)",
    where: "Both",
    description:
      "The order your substitutes come on if a starter does not play, best first. The reserve goalkeeper is kept separate from the outfield subs because he can only ever replace your goalkeeper.",
    keywords: ["subs", "substitutes", "sub order"],
  },
  {
    term: "Price (£m)",
    where: "Both",
    description:
      "What the player costs from your budget. FPL prices move slightly during the season as managers buy and sell.",
    keywords: ["cost", "value", "million"],
  },
  {
    term: "Positions (GK, DEF, MID, FWD)",
    where: "Both",
    description:
      "Goalkeeper, defender, midfielder and forward — colour-coded yellow, blue, green and red on the pitch. Every squad is 2 GK, 5 DEF, 5 MID and 3 FWD, with at most 3 players from any one club.",
    keywords: ["goalkeeper", "defender", "midfielder", "forward", "colours"],
  },
];

const CATEGORY_ENTRIES: (Entry & { category: string })[] = [
  {
    category: "form",
    term: "Form",
    where: "Lineup Advisor",
    description:
      "Points the player has actually been returning lately. Recent scoring counts for more than a good name.",
    keywords: ["recent", "hot", "streak"],
  },
  {
    category: "fixture",
    term: "Fixture",
    where: "Lineup Advisor",
    description:
      "Who the player faces and whether it is at home or away. Home games and weaker opponents mean easier points.",
    keywords: ["opponent", "home", "away", "difficulty"],
  },
  {
    category: "performance",
    term: "Performance",
    where: "Both",
    description:
      "All-round involvement in matches — ICT index, bonus and BPS, overall points. A player who is always near the action tends to keep scoring.",
    keywords: ["ict", "bonus", "bps", "overall"],
  },
  {
    category: "attacking",
    term: "Attacking",
    where: "Both",
    description:
      "Goal and assist output, plus the underlying numbers behind it: xG, xA, threat and creativity.",
    keywords: ["goals", "assists", "xg", "xa", "threat", "creativity"],
  },
  {
    category: "defensive",
    term: "Defensive",
    where: "Both",
    description:
      "Clean sheets, goals conceded and saves. This is where most of a goalkeeper's or defender's points come from.",
    keywords: ["clean sheet", "conceded", "saves"],
  },
  {
    category: "reliability",
    term: "Reliability",
    where: "Both",
    description:
      "Minutes and starts — how certain it is that the player will actually be on the pitch. A brilliant player who is rotated scores nothing from the bench.",
    keywords: ["minutes", "starts", "nailed", "rotation"],
  },
  {
    category: "value",
    term: "Value",
    where: "Both",
    description:
      "Points relative to price. A cheap player with a decent return frees up money for a premium elsewhere.",
    keywords: ["price", "points per million", "cheap", "budget"],
  },
  {
    category: "trending",
    term: "Trending",
    where: "Lineup Advisor",
    description:
      "How other FPL managers are behaving — ownership and transfers in or out. It reflects popularity rather than a footballing reason, so it only appears when there are not enough stronger reasons to show.",
    keywords: ["ownership", "transfers", "popular"],
  },
];

const STAT_ENTRIES: Entry[] = [
  {
    term: "Total points",
    where: "Both",
    description:
      "The FPL points a player scored — over a gameweek in the Lineup Advisor, and over the whole of last season in the Squad Builder's reasons.",
    keywords: ["pts", "score"],
  },
  {
    term: "xP (expected points)",
    where: "Lineup Advisor",
    description:
      "FPL's own projection of a player's points for a gameweek, based on his role and fixture. The weekly model treats it as one input among many rather than the answer.",
    keywords: ["expected points", "ep"],
  },
  {
    term: "Minutes",
    where: "Both",
    description:
      "How long the player was on the pitch. 60 minutes or more earns 2 points instead of 1, and consistently high minutes is the clearest sign of a guaranteed starter.",
    keywords: ["playing time", "nailed"],
  },
  {
    term: "Starts",
    where: "Lineup Advisor",
    description:
      "How often the player was in the starting 11 rather than coming off the bench.",
    keywords: ["lineups", "xi"],
  },
  {
    term: "Goals scored",
    where: "Both",
    description:
      "Goals are worth 6 points for a goalkeeper or defender, 5 for a midfielder and 4 for a forward.",
    keywords: ["goal"],
  },
  {
    term: "Assists",
    where: "Both",
    description: "The final pass before a goal — 3 points each, in any position.",
    keywords: ["assist", "chances"],
  },
  {
    term: "xG (expected goals)",
    where: "Lineup Advisor",
    description:
      "The quality of the chances a player got, where 1.0 is a chance you would expect to be scored every time. 0.6 xG in a match means good chances even if nothing went in — often a sign that goals are coming.",
    keywords: ["expected goals", "chance quality"],
  },
  {
    term: "xA (expected assists)",
    where: "Lineup Advisor",
    description:
      "The same idea for creators: how likely the passes a player played were to become assists, whether or not a teammate finished them.",
    keywords: ["expected assists"],
  },
  {
    term: "xGI (expected goal involvements)",
    where: "Lineup Advisor",
    description:
      "xG and xA added together — the player's total expected attacking output.",
    keywords: ["expected goal involvements", "involvement"],
  },
  {
    term: "Threat",
    where: "Both",
    description:
      "FPL's rating of how dangerous a player is in front of goal, built from the volume and quality of his shots.",
    keywords: ["shots", "danger"],
  },
  {
    term: "Creativity",
    where: "Both",
    description:
      "FPL's rating of chance creation — passing into dangerous areas and setting up teammates.",
    keywords: ["chances created", "key passes"],
  },
  {
    term: "Influence",
    where: "Both",
    description:
      "FPL's rating of how much a player affected the result of a match — decisive actions like goals, assists, saves and tackles.",
    keywords: ["impact"],
  },
  {
    term: "ICT index",
    where: "Both",
    description:
      "Influence, Creativity and Threat combined into a single number by FPL. A quick all-round measure of involvement that works across positions.",
    keywords: ["ict"],
  },
  {
    term: "BPS (Bonus Points System)",
    where: "Both",
    description:
      "A behind-the-scenes score FPL gives for everything a player does in a match — passes, tackles, saves, goals, cards. It is not points itself; it only decides who gets the bonus.",
    keywords: ["bps", "bonus points system"],
  },
  {
    term: "Bonus",
    where: "Both",
    description:
      "The extra 3, 2 and 1 points handed to the three best performers in each match, ranked by BPS. A high BPS player is repeatedly in line for them.",
    keywords: ["bonus points"],
  },
  {
    term: "Clean sheets",
    where: "Both",
    description:
      "Matches where the player's team conceded nothing while he was on the pitch for at least 60 minutes — 4 points for a goalkeeper or defender, 1 for a midfielder.",
    keywords: ["shutout", "cs"],
  },
  {
    term: "Goals conceded",
    where: "Both",
    description:
      "Goals let in by the player's team. Goalkeepers and defenders lose a point for every two conceded, so a low number here is a good sign.",
    keywords: ["conceded", "against"],
  },
  {
    term: "xGC (expected goals conceded)",
    where: "Lineup Advisor",
    description:
      "The quality of the chances the player's team allowed. A low xGC suggests the defence is genuinely solid rather than lucky, which makes future clean sheets more likely.",
    keywords: ["expected goals conceded", "defence"],
  },
  {
    term: "Saves",
    where: "Both",
    description:
      "Shots stopped by a goalkeeper — every 3 saves is worth 1 point, which is why a keeper behind a busy defence can still score well.",
    keywords: ["keeper", "stops"],
  },
  {
    term: "Selected by",
    where: "Lineup Advisor",
    description:
      "How many FPL managers own the player. It measures popularity, not quality, so the model treats it as a weak signal.",
    keywords: ["ownership", "owned", "popular"],
  },
  {
    term: "Transfers balance",
    where: "Lineup Advisor",
    description:
      "Transfers in minus transfers out for the gameweek. A big positive number means managers are piling in — often after a good run or an injury elsewhere.",
    keywords: ["transfers in", "transfers out", "momentum"],
  },
  {
    term: "Team goals scored / conceded",
    where: "Lineup Advisor",
    description:
      "How the player's whole team has been performing. A team scoring freely lifts its attackers; a team keeping things tight lifts its defenders and keeper.",
    keywords: ["team form", "club"],
  },
  {
    term: "Last gameweek / last 3, 5, 7 gameweeks",
    where: "Lineup Advisor",
    description:
      "Reasons often say a stat is measured over a window of recent matches. Short windows react quickly to a hot streak; longer ones are steadier and less easily fooled by one good afternoon.",
    keywords: ["rolling", "average", "window", "mean3", "mean5", "mean7"],
  },
];

const SECTIONS: Section[] = [
  {
    id: "screen",
    title: "Numbers on your screen",
    blurb:
      "What every figure on the pitch, the summary bar and the player cards actually means.",
    entries: SCREEN_ENTRIES,
  },
  {
    id: "stats",
    title: "Football stats A to Z",
    blurb: "The underlying stats that show up inside the reasons for each pick.",
    entries: STAT_ENTRIES,
  },
];

// ── Search ────────────────────────────────────────────────────────────────

function matches(entry: Entry, query: string): boolean {
  if (!query) return true;
  const haystack = [entry.term, entry.description, ...(entry.keywords ?? [])]
    .join(" ")
    .toLowerCase();
  return query
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean)
    .every((word) => haystack.includes(word));
}

// ── Pieces ────────────────────────────────────────────────────────────────

const WHERE_STYLE: Record<Where, string> = {
  "Squad Builder": "border-emerald-500/30 text-emerald-700 dark:text-emerald-300",
  "Lineup Advisor": "border-sky-500/30 text-sky-700 dark:text-sky-300",
  Both: "border-gray-400/30 text-gray-600 dark:text-gray-300",
};

function GlassCard({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`rounded-xl border border-gray-200 bg-white/80 p-5 shadow-sm backdrop-blur-sm dark:border-white/[0.08] dark:bg-white/[0.03] dark:shadow-xl dark:backdrop-blur-xl ${className}`}
    >
      {children}
    </div>
  );
}

function WhereBadge({ where }: { where: Where }) {
  return (
    <Badge variant="outline" className={`text-[10px] ${WHERE_STYLE[where]}`}>
      {where}
    </Badge>
  );
}

function EntryCard({ entry }: { entry: Entry }) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white/60 p-4 dark:border-white/[0.06] dark:bg-white/[0.02]">
      <div className="flex flex-wrap items-center gap-2">
        <h3 className="text-sm font-semibold">{entry.term}</h3>
        <WhereBadge where={entry.where} />
      </div>
      <p className="mt-2 text-sm text-muted-foreground">{entry.description}</p>
    </div>
  );
}

function CategoryCard({ entry }: { entry: Entry & { category: string } }) {
  const config = categoryConfig[entry.category] ?? defaultCategoryConfig;
  return (
    <div
      className={`rounded-lg border border-gray-200 border-l-4 ${config.borderColor} bg-white/60 p-4 dark:border-white/[0.06] dark:bg-white/[0.02]`}
    >
      <div className="flex flex-wrap items-center gap-2">
        <span
          className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium ${config.color} ${config.darkColor}`}
        >
          {config.icon}
          {entry.category}
        </span>
        <WhereBadge where={entry.where} />
      </div>
      <p className="mt-2 text-sm text-muted-foreground">{entry.description}</p>
    </div>
  );
}

function EntrySection({ section }: { section: Section }) {
  return (
    <GlassCard>
      <h2 className="text-lg font-semibold">{section.title}</h2>
      <p className="mt-1 text-sm text-muted-foreground">{section.blurb}</p>
      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        {section.entries.map((entry) => (
          <EntryCard key={entry.term} entry={entry} />
        ))}
      </div>
    </GlassCard>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────

export default function StatsGuide() {
  const [query, setQuery] = useState("");
  const prefersReduced = useReducedMotion();
  const dur = (d: number) => (prefersReduced ? 0 : d);

  const visibleSections = useMemo(
    () =>
      SECTIONS.map((section) => ({
        ...section,
        entries: section.entries.filter((e) => matches(e, query)),
      })).filter((section) => section.entries.length > 0),
    [query]
  );

  const visibleCategories = useMemo(
    () => CATEGORY_ENTRIES.filter((e) => matches(e, query)),
    [query]
  );

  const isSearching = query.trim().length > 0;
  const hasResults = visibleSections.length > 0 || visibleCategories.length > 0;
  const screenSection = visibleSections.find((s) => s.id === "screen");
  const statsSection = visibleSections.find((s) => s.id === "stats");

  return (
    <>
      <PageMeta
        title="Stats Guide | Automatic Champion"
        description="What every stat and number in Automatic Champion means"
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
          transition={
            prefersReduced
              ? { duration: 0 }
              : { type: "spring", stiffness: 200, damping: 20 }
          }
        >
          Stats Guide
        </motion.h1>

        <p className="max-w-3xl text-sm text-muted-foreground">
          Every number and badge on the Squad Builder and the Lineup Advisor,
          explained in plain English. No FPL experience assumed — if a stat in a
          player&apos;s reasons does not mean anything to you, look it up here.
        </p>

        <div className="relative max-w-md">
          <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search a stat, e.g. xG, bonus, clean sheet"
            aria-label="Search stats"
            className="pl-9 pr-9"
          />
          {isSearching && (
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setQuery("")}
              aria-label="Clear search"
              className="absolute right-1 top-1/2 size-7 -translate-y-1/2"
            >
              <X className="size-4" />
            </Button>
          )}
        </div>

        {!hasResults && (
          <GlassCard>
            <p className="text-sm text-muted-foreground">
              Nothing matches &ldquo;{query}&rdquo;. Try a shorter word, like{" "}
              <button
                type="button"
                className="underline underline-offset-2"
                onClick={() => setQuery("goals")}
              >
                goals
              </button>{" "}
              or{" "}
              <button
                type="button"
                className="underline underline-offset-2"
                onClick={() => setQuery("bonus")}
              >
                bonus
              </button>
              .
            </p>
          </GlassCard>
        )}

        {screenSection && <EntrySection section={screenSection} />}

        {visibleCategories.length > 0 && (
          <GlassCard>
            <h2 className="text-lg font-semibold">Reason badges</h2>
            <p className="mt-1 text-sm text-muted-foreground">
              Tap any player to see why they were picked. Every reason carries
              one of these coloured badges, telling you what kind of signal it
              is.
            </p>
            <div className="mt-4 grid gap-3 sm:grid-cols-2">
              {visibleCategories.map((entry) => (
                <CategoryCard key={entry.category} entry={entry} />
              ))}
            </div>
          </GlassCard>
        )}

        {statsSection && <EntrySection section={statsSection} />}

        {!isSearching && (
          <GlassCard>
            <h2 className="text-lg font-semibold">
              Where the numbers come from
            </h2>
            <div className="mt-4 grid gap-4 sm:grid-cols-2">
              <div>
                <h3 className="text-sm font-semibold">
                  Squad Builder — the season view
                </h3>
                <p className="mt-2 text-sm text-muted-foreground">
                  A separate model for each position projects a full season of
                  points from a player&apos;s record last season, and the
                  optimiser buys the best 15 it can afford under the FPL rules.
                  The reasons compare a player against others in his own
                  position, which is why they read like &ldquo;ranks 4th among
                  midfielders&rdquo;.
                </p>
              </div>
              <div>
                <h3 className="text-sm font-semibold">
                  Lineup Advisor — the gameweek view
                </h3>
                <p className="mt-2 text-sm text-muted-foreground">
                  The weekly (V8) model predicts one gameweek at a time, per
                  position, using recent form, the fixture and the underlying
                  stats above. Each reason comes from what actually moved that
                  particular prediction, so the explanation always matches the
                  number shown next to the player.
                </p>
              </div>
            </div>
          </GlassCard>
        )}
      </motion.div>
    </>
  );
}
