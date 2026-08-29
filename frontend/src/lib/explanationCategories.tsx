import {
  TrendingUp,
  Swords,
  Shield,
  Clock,
  Coins,
  ArrowUpRight,
  Flame,
  MapPin,
} from "lucide-react";

export interface CategoryConfig {
  color: string;
  darkColor: string;
  borderColor: string;
  icon: React.ReactNode;
}

/**
 * Badge styling for the explanation categories the backend emits
 * (see src/explainer.py and src/weekly_explainer.py). Shared by the
 * ExplanationPanel and the Stats Guide so the two never drift apart.
 */
export const categoryConfig: Record<string, CategoryConfig> = {
  form: { color: "bg-orange-100 text-orange-800", darkColor: "dark:bg-orange-900/30 dark:text-orange-300", borderColor: "border-l-orange-400", icon: <Flame className="size-3.5" /> },
  fixture: { color: "bg-indigo-100 text-indigo-800", darkColor: "dark:bg-indigo-900/30 dark:text-indigo-300", borderColor: "border-l-indigo-400", icon: <MapPin className="size-3.5" /> },
  performance: { color: "bg-blue-100 text-blue-800", darkColor: "dark:bg-blue-900/30 dark:text-blue-300", borderColor: "border-l-blue-400", icon: <TrendingUp className="size-3.5" /> },
  attacking: { color: "bg-red-100 text-red-800", darkColor: "dark:bg-red-900/30 dark:text-red-300", borderColor: "border-l-red-400", icon: <Swords className="size-3.5" /> },
  defensive: { color: "bg-green-100 text-green-800", darkColor: "dark:bg-green-900/30 dark:text-green-300", borderColor: "border-l-green-400", icon: <Shield className="size-3.5" /> },
  reliability: { color: "bg-amber-100 text-amber-800", darkColor: "dark:bg-amber-900/30 dark:text-amber-300", borderColor: "border-l-amber-400", icon: <Clock className="size-3.5" /> },
  value: { color: "bg-purple-100 text-purple-800", darkColor: "dark:bg-purple-900/30 dark:text-purple-300", borderColor: "border-l-purple-400", icon: <Coins className="size-3.5" /> },
  trending: { color: "bg-cyan-100 text-cyan-800", darkColor: "dark:bg-cyan-900/30 dark:text-cyan-300", borderColor: "border-l-cyan-400", icon: <ArrowUpRight className="size-3.5" /> },
};

export const defaultCategoryConfig: CategoryConfig = {
  color: "bg-gray-100 text-gray-800",
  darkColor: "dark:bg-gray-800 dark:text-gray-300",
  borderColor: "border-l-gray-400",
  icon: null,
};
