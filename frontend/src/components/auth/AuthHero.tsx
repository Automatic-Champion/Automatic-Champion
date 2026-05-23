import { motion, useReducedMotion } from "framer-motion";
import PitchSVG from "../pitch/PitchSVG";

interface AuthHeroProps {
  mode: "login" | "register";
}

interface FloatingKit {
  src: string;
  top: string;
  left?: string;
  right?: string;
  rotate: number;
  duration: number;
  hideOnMobile?: boolean;
}

const KITS: FloatingKit[] = [
  { src: "/kits/arsenal1.png", top: "12%", left: "8%", rotate: -10, duration: 7 },
  { src: "/kits/liverpool1.png", top: "62%", left: "14%", rotate: 8, duration: 8.5 },
  { src: "/kits/mancity1.png", top: "30%", right: "22%", rotate: -6, duration: 6.5, hideOnMobile: true },
  { src: "/kits/Chelsea1.png", top: "72%", right: "10%", rotate: 11, duration: 9, hideOnMobile: true },
  { src: "/kits/manunited1.png", top: "8%", right: "8%", rotate: 4, duration: 7.5, hideOnMobile: true },
];

const PARTICLES = [
  { x: 6, y: 18, delay: 0 },
  { x: 22, y: 42, delay: 0.6 },
  { x: 38, y: 78, delay: 1.2 },
  { x: 55, y: 12, delay: 1.8 },
  { x: 68, y: 56, delay: 2.4 },
  { x: 82, y: 28, delay: 0.3 },
  { x: 90, y: 82, delay: 1.5 },
  { x: 14, y: 88, delay: 2.1 },
];

export default function AuthHero({ mode }: AuthHeroProps) {
  const prefersReduced = useReducedMotion();

  const headlineLines =
    mode === "login"
      ? ["Build the Squad.", "Win the Week."]
      : ["Join the Squad.", "Win the Week."];

  const easeInOut: [number, number, number, number] = [0.42, 0, 0.58, 1];

  return (
    <div className="stadium-bg relative overflow-hidden h-[35vh] min-h-[280px] lg:h-screen lg:flex-1 lg:basis-3/5">
      {/* Layer 1: stadium gradient is applied via stadium-bg utility above */}

      {/* Layer 1 entrance fade — fades the whole panel in */}
      <motion.div
        className="stadium-bg absolute inset-0"
        initial={prefersReduced ? false : { opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={prefersReduced ? { duration: 0 } : { duration: 0.3 }}
      />

      {/* Layer 2: pitch silhouette (desaturated, bleeding off bottom-right) */}
      <motion.div
        className="pointer-events-none absolute inset-0 overflow-hidden"
        initial={prefersReduced ? false : { opacity: 0 }}
        animate={{ opacity: 0.06 }}
        transition={
          prefersReduced
            ? { duration: 0 }
            : { delay: 0.2, duration: 0.3 }
        }
        aria-hidden="true"
      >
        <div
          className="absolute -bottom-[30%] -right-[20%] w-[120%]"
          style={{ filter: "saturate(0) brightness(0.4)" }}
        >
          <PitchSVG />
        </div>
      </motion.div>

      {/* Layer 3: floating kit cards */}
      <div className="pointer-events-none absolute inset-0 overflow-hidden" aria-hidden="true">
        {KITS.map((kit, i) => (
          <motion.img
            key={kit.src}
            src={kit.src}
            alt=""
            className={`absolute w-24 h-24 lg:w-32 lg:h-32 ${kit.hideOnMobile ? "hidden lg:block" : ""}`}
            style={{
              top: kit.top,
              left: kit.left,
              right: kit.right,
              filter: "blur(0.5px)",
            }}
            initial={prefersReduced ? { opacity: 0.18, rotate: kit.rotate } : { opacity: 0, rotate: kit.rotate, y: 0 }}
            animate={
              prefersReduced
                ? { opacity: 0.18, rotate: kit.rotate }
                : { opacity: 0.18, rotate: kit.rotate, y: [0, -10, 0] }
            }
            transition={
              prefersReduced
                ? { duration: 0 }
                : {
                    opacity: { delay: 0.4 + i * 0.08, duration: 0.5, ease: easeInOut },
                    y: { duration: kit.duration, repeat: Infinity, ease: "easeInOut" },
                  }
            }
          />
        ))}
      </div>

      {/* Layer 4: ambient particle dots */}
      {!prefersReduced && (
        <div className="pointer-events-none absolute inset-0" aria-hidden="true">
          {PARTICLES.map((p, i) => (
            <motion.div
              key={i}
              className="absolute h-1.5 w-1.5 rounded-full bg-amber-300/50"
              style={{ left: `${p.x}%`, top: `${p.y}%` }}
              animate={{
                opacity: [0.25, 0.7, 0.25],
                y: [0, -6, 0],
              }}
              transition={{
                duration: 3.5,
                delay: p.delay,
                repeat: Infinity,
                ease: "easeInOut",
              }}
            />
          ))}
        </div>
      )}

      {/* Layer 5: seam fade — bottom on mobile, right on desktop */}
      <div className="pointer-events-none absolute inset-x-0 bottom-0 h-24 bg-gradient-to-b from-transparent to-background lg:hidden" />
      <div className="pointer-events-none absolute inset-y-0 right-0 hidden w-32 bg-gradient-to-r from-transparent to-background lg:block" />

      {/* Hero copy */}
      <div className="relative z-10 flex h-full flex-col justify-center px-6 py-8 lg:px-16">
        <motion.h1
          className="bg-gradient-to-r from-white via-amber-200 to-amber-400 bg-clip-text font-black tracking-tight text-transparent text-3xl leading-tight lg:text-6xl lg:leading-[1.05]"
          style={{
            textShadow: "0 4px 24px rgba(0,0,0,0.5)",
          }}
          initial={prefersReduced ? false : { opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={
            prefersReduced
              ? { duration: 0 }
              : { delay: 0.5, type: "spring", stiffness: 220, damping: 22 }
          }
        >
          {headlineLines.map((line, i) => (
            <span key={i} className="block">
              {line}
            </span>
          ))}
        </motion.h1>

        <motion.p
          className="mt-4 text-base tracking-wide text-muted-foreground lg:text-lg"
          initial={prefersReduced ? false : { opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={
            prefersReduced ? { duration: 0 } : { delay: 0.7, duration: 0.5, ease: easeInOut }
          }
        >
          Predicts. Optimizes. Explains.
        </motion.p>

        <motion.p
          className="mt-3 text-xs text-muted-foreground/60"
          initial={prefersReduced ? false : { opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={
            prefersReduced ? { duration: 0 } : { delay: 0.78, duration: 0.5, ease: easeInOut }
          }
        >
          Premier League · 2024-25 season · 784 players analyzed
        </motion.p>
      </div>
    </div>
  );
}
