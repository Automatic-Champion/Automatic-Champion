interface FormationSlot {
  x: number; // percentage 0-100
  y: number; // percentage 0-100
  position: string;
}

/**
 * Distributes `count` items evenly across the horizontal axis.
 * Returns an array of x-percentages.
 */
function spreadX(count: number): number[] {
  if (count === 1) return [50];
  const margin = count >= 5 ? 10 : 15;
  const step = (100 - 2 * margin) / (count - 1);
  return Array.from({ length: count }, (_, i) => margin + i * step);
}

// Row Y positions (top = FWD, bottom = GK)
const ROW_Y = {
  GK: 90,
  DEF: 70,
  MID: 45,
  FWD: 20,
};

function buildFormation(def: number, mid: number, fwd: number): FormationSlot[] {
  const slots: FormationSlot[] = [];

  // GK always 1
  slots.push({ x: 50, y: ROW_Y.GK, position: "GK" });

  for (const xVal of spreadX(def)) {
    slots.push({ x: xVal, y: ROW_Y.DEF, position: "DEF" });
  }
  for (const xVal of spreadX(mid)) {
    slots.push({ x: xVal, y: ROW_Y.MID, position: "MID" });
  }
  for (const xVal of spreadX(fwd)) {
    slots.push({ x: xVal, y: ROW_Y.FWD, position: "FWD" });
  }

  return slots;
}

const FORMATIONS: Record<string, FormationSlot[]> = {
  "3-4-3": buildFormation(3, 4, 3),
  "3-5-2": buildFormation(3, 5, 2),
  "4-3-3": buildFormation(4, 3, 3),
  "4-4-2": buildFormation(4, 4, 2),
  "4-5-1": buildFormation(4, 5, 1),
  "5-3-2": buildFormation(5, 3, 2),
  "5-4-1": buildFormation(5, 4, 1),
};

export function getFormationPositions(
  formation: string
): FormationSlot[] {
  return FORMATIONS[formation] ?? FORMATIONS["4-4-2"];
}
