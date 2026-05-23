import { useState } from "react";
import type { FormEvent } from "react";
import { Link, useNavigate } from "react-router";
import { motion, useReducedMotion } from "framer-motion";
import { Eye, EyeOff, Loader2 } from "lucide-react";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { Label } from "../components/ui/label";
import AuthHero from "../components/auth/AuthHero";
import { useAuth } from "../context/AuthContext";

function mapFirebaseError(message: string): string {
  if (message.includes("invalid-credential") || message.includes("wrong-password")) {
    return "Invalid email or password";
  }
  if (message.includes("user-not-found")) {
    return "No account found with that email";
  }
  if (message.includes("invalid-email")) {
    return "Invalid email address";
  }
  if (message.includes("too-many-requests")) {
    return "Too many attempts. Please try again later.";
  }
  return "Could not sign in. Please try again.";
}

export default function Login() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const prefersReduced = useReducedMotion();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      await login(email, password);
      navigate("/", { replace: true });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(mapFirebaseError(msg));
    } finally {
      setSubmitting(false);
    }
  }

  const easeOutExpo: [number, number, number, number] = [0.16, 1, 0.3, 1];
  const fieldDelay = (i: number) =>
    prefersReduced ? 0 : 0.5 + 0.3 + i * 0.05;

  return (
    <div className="relative flex min-h-screen flex-col overflow-hidden bg-background lg:flex-row">
      <AuthHero mode="login" />

      <div className="relative flex flex-1 items-center justify-center px-4 py-10 lg:basis-2/5 lg:px-10">
        <motion.div
          className="relative w-full max-w-md overflow-hidden rounded-2xl border border-white/[0.08] bg-white/[0.03] p-8 shadow-xl backdrop-blur-xl"
          initial={prefersReduced ? false : { opacity: 0, x: 40 }}
          animate={{ opacity: 1, x: 0 }}
          transition={
            prefersReduced
              ? { duration: 0 }
              : { delay: 0.3, type: "spring", stiffness: 180, damping: 22 }
          }
        >
          {/* Gold accent line on top edge */}
          <div className="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-amber-400/40 to-transparent" />

          <motion.div
            initial={prefersReduced ? false : { opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={
              prefersReduced
                ? { duration: 0 }
                : { delay: 0.5, duration: 0.4, ease: easeOutExpo }
            }
            className="mb-6"
          >
            <p className="bg-gradient-to-r from-white via-amber-200 to-amber-400 bg-clip-text text-sm font-black uppercase tracking-[0.2em] text-transparent">
              Automatic Champion
            </p>
            <h2 className="mt-2 text-sm text-muted-foreground">Welcome back</h2>
          </motion.div>

          <form onSubmit={handleSubmit} className="space-y-4" noValidate>
            <motion.div
              className="space-y-2"
              initial={prefersReduced ? false : { opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={
                prefersReduced
                  ? { duration: 0 }
                  : { delay: fieldDelay(0), duration: 0.4, ease: easeOutExpo }
              }
            >
              <Label htmlFor="email">Email</Label>
              <Input
                id="email"
                type="email"
                autoComplete="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="h-11 focus-visible:ring-2 focus-visible:ring-amber-400/30"
              />
            </motion.div>

            <motion.div
              className="space-y-2"
              initial={prefersReduced ? false : { opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={
                prefersReduced
                  ? { duration: 0 }
                  : { delay: fieldDelay(1), duration: 0.4, ease: easeOutExpo }
              }
            >
              <Label htmlFor="password">Password</Label>
              <div className="relative">
                <Input
                  id="password"
                  type={showPassword ? "text" : "password"}
                  autoComplete="current-password"
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="h-11 pr-12 focus-visible:ring-2 focus-visible:ring-amber-400/30"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword((s) => !s)}
                  aria-label={showPassword ? "Hide password" : "Show password"}
                  aria-pressed={showPassword}
                  className="absolute right-0 top-1/2 flex h-11 w-11 -translate-y-1/2 items-center justify-center rounded-md text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-400/40"
                >
                  {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
              <div className="flex justify-end">
                <Link
                  to="/forgot-password"
                  className="text-xs text-amber-400 underline-offset-4 hover:underline"
                >
                  Forgot password?
                </Link>
              </div>
            </motion.div>

            {error && (
              <div
                role="alert"
                aria-live="polite"
                className="rounded-lg bg-red-50 p-3 text-sm text-red-600 dark:bg-red-900/20 dark:text-red-400"
              >
                {error}
              </div>
            )}

            <motion.div
              initial={prefersReduced ? false : { opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={
                prefersReduced
                  ? { duration: 0 }
                  : { delay: fieldDelay(2), duration: 0.4, ease: easeOutExpo }
              }
              whileTap={prefersReduced ? undefined : { scale: 0.97 }}
              whileHover={prefersReduced ? undefined : { scale: 1.02 }}
            >
              <Button
                type="submit"
                disabled={submitting}
                className="btn-gradient-shift h-11 w-full shadow-md dark:text-gray-900 dark:shadow-lg dark:shadow-amber-500/20"
                style={{
                  backgroundImage: "linear-gradient(90deg, #d97706, #f59e0b, #d97706)",
                  backgroundSize: "200% 200%",
                }}
              >
                {submitting ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Kicking off...
                  </>
                ) : (
                  "Kick off →"
                )}
              </Button>
            </motion.div>

            <motion.p
              className="text-center text-sm text-muted-foreground"
              initial={prefersReduced ? false : { opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={
                prefersReduced
                  ? { duration: 0 }
                  : { delay: fieldDelay(3), duration: 0.4, ease: easeOutExpo }
              }
            >
              Don&apos;t have an account?{" "}
              <Link
                to="/register"
                className="font-semibold text-amber-400 underline-offset-4 hover:underline"
              >
                Register
              </Link>
            </motion.p>
          </form>
        </motion.div>
      </div>
    </div>
  );
}
