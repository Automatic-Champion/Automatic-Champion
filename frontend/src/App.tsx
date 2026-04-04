import { BrowserRouter as Router, Routes, Route } from "react-router";
import AppLayout from "./layout/AppLayout";
import { ScrollToTop } from "./components/common/ScrollToTop";
import SquadBuilder from "./pages/SquadBuilder";
import LineupAdvisor from "./pages/LineupAdvisor";

export default function App() {
  return (
    <Router>
      <ScrollToTop />
      <Routes>
        <Route element={<AppLayout />}>
          <Route index path="/" element={<SquadBuilder />} />
          <Route path="/lineup" element={<LineupAdvisor />} />
        </Route>
      </Routes>
    </Router>
  );
}
