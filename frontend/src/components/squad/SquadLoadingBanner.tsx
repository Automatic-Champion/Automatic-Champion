import { useEffect, useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";

const DEFAULT_STAGES = [
  "Reading the player pool...",
  "Predicting season points...",
  "Balancing budget and club limits...",
  "Locking in your starting XI...",
];

type SquadLoadingBannerProps = {
  title?: string;
  stages?: string[];
  /** Time the progress bar takes to fill, in ms. Match the minimum loading hold. */
  durationMs?: number;
};

export default function SquadLoadingBanner({
  title = "Making your optimal squad",
  stages = DEFAULT_STAGES,
  durationMs = 3000,
}: SquadLoadingBannerProps) {
  const prefersReduced = useReducedMotion();
  const [stageIndex, setStageIndex] = useState(0);

  useEffect(() => {
    const step = Math.max(400, Math.round(durationMs / stages.length));
    const id = window.setInterval(() => {
      setStageIndex((prev) => Math.min(prev + 1, stages.length - 1));
    }, step);
    return () => window.clearInterval(id);
  }, [durationMs, stages.length]);

  return (
    <div className="relative overflow-hidden rounded-2xl border bg-card/70 px-5 py-4 ring-1 ring-foreground/10 backdrop-blur">
      <div className="flex items-center gap-3">
        <div className="flex h-9 w-9 items-center justify-center bounce-ball">
          <svg width="28" height="28" viewBox="0 0 20 20" aria-hidden="true">
            <circle cx="10" cy="10" r="9" fill="white" stroke="#333" strokeWidth="1" />
            <path
              d="M10 1L12 4L16 4L13 7L14 11L10 9L6 11L7 7L4 4L8 4Z"
              fill="#333"
              opacity="0.25"
            />
          </svg>
        </div>

        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-semibold tracking-tight">{title}</p>
          <div className="h-4">
            <AnimatePresence mode="wait">
              <motion.p
                key={stageIndex}
                initial={{ opacity: 0, y: prefersReduced ? 0 : 4 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: prefersReduced ? 0 : -4 }}
                transition={{ duration: prefersReduced ? 0 : 0.25 }}
                className="truncate text-xs text-muted-foreground"
                role="status"
                aria-live="polite"
              >
                {stages[stageIndex]}
              </motion.p>
            </AnimatePresence>
          </div>
        </div>
      </div>

      <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-muted">
        <motion.div
          className="h-full rounded-full bg-gradient-to-r from-emerald-400 to-sky-400"
          initial={{ width: "4%" }}
          animate={{ width: "100%" }}
          transition={{ duration: prefersReduced ? 0 : durationMs / 1000, ease: "easeInOut" }}
        />
      </div>
    </div>
  );
}
