import { createContext, useCallback, useContext, useState } from "react";
import type {
  SavedSquadResponse,
  SquadGenerateResponse,
} from "../api/types";

type SquadContextType = {
  squad: SquadGenerateResponse | null;
  currentSquadName: string | null;
  setSquad: (squad: SquadGenerateResponse | null) => void;
  setSquadFromSaved: (saved: SavedSquadResponse) => void;
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
  const [squad, setSquadState] = useState<SquadGenerateResponse | null>(null);
  const [currentSquadName, setCurrentSquadName] = useState<string | null>(null);

  const setSquad = useCallback((next: SquadGenerateResponse | null) => {
    setSquadState(next);
    setCurrentSquadName(null);
  }, []);

  const setSquadFromSaved = useCallback((saved: SavedSquadResponse) => {
    setSquadState(saved.payload);
    setCurrentSquadName(saved.name);
  }, []);

  return (
    <SquadContext.Provider
      value={{ squad, currentSquadName, setSquad, setSquadFromSaved }}
    >
      {children}
    </SquadContext.Provider>
  );
};
