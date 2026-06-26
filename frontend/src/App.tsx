import { Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { AssistantPage } from "./pages/AssistantPage";
import { EvaluationPage } from "./pages/EvaluationPage";
import { SystemPage } from "./pages/SystemPage";

export function App() {
  return (
    <Routes>
      <Route element={<AppShell />} path="/">
        <Route element={<Navigate replace to="/assistant" />} index />
        <Route element={<AssistantPage />} path="assistant" />
        <Route element={<EvaluationPage />} path="evaluation" />
        <Route element={<SystemPage />} path="system" />
      </Route>
    </Routes>
  );
}
