const TEAM_KIT_MAP: Record<string, string> = {
  "Arsenal": "/kits/arsenal1.png",
  "Aston Villa": "/kits/avilla1.png",
  "Bournemouth": "/kits/bournemouth1.png",
  "Brentford": "/kits/Brentford1.png",
  "Brighton": "/kits/Brighton1.png",
  "Chelsea": "/kits/Chelsea1.png",
  "Crystal Palace": "/kits/cpalace1.png",
  "Everton": "/kits/everton1.png",
  "Fulham": "/kits/fulham1.png",
  "Liverpool": "/kits/liverpool1.png",
  "Man City": "/kits/mancity1.png",
  "Man Utd": "/kits/manunited1.png",
  "Newcastle": "/kits/newcastle1.png",
  "Nott'm Forest": "/kits/nforest1.png",
  "Spurs": "/kits/spurs1.png",
  "West Ham": "/kits/westham1.png",
  "Wolves": "/kits/wolves1.png",
};

export function getKitUrl(team: string): string | null {
  return TEAM_KIT_MAP[team] ?? null;
}
