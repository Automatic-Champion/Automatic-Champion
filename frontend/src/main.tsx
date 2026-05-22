import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App.tsx";
import { AppWrapper } from "./components/common/PageMeta.tsx";
import { ThemeProvider } from "./context/ThemeContext.tsx";
import { SquadProvider } from "./context/SquadContext.tsx";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ThemeProvider>
      <SquadProvider>
        <AppWrapper>
          <App />
        </AppWrapper>
      </SquadProvider>
    </ThemeProvider>
  </StrictMode>,
);
