import React from "react";
import ReactDOM from "react-dom/client";
import { App } from "./app/App";
import { initializeRustEngine } from "./game/rustEngine";
import "./app/styles.css";

await initializeRustEngine();

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
