import { createContext, useContext, useState } from "react";
import type { SquadGenerateResponse } from "../api/types";

type SquadContextType = {
  squad: SquadGenerateResponse | null;
  setSquad: (squad: SquadGenerateResponse | null) => void;
};

const SquadContext = createContext<SquadContextType | undefined>(undefined);

export const useSquad = () => {
  const context = useContext(SquadContext);
  if (!context) {
    throw new Error("useSquad must be used within a SquadProvider");
  }
  return context;
};

export const SquadProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [squad, setSquad] = useState<SquadGenerateResponse | null>(null);
  return (
    <SquadContext.Provider value={{ squad, setSquad }}>
      {children}
    </SquadContext.Provider>
  );
};
