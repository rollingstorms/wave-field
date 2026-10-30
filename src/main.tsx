import React from "react";
import ReactDOM from "react-dom/client";
import { SelfPlayViewer } from "./app/SelfPlayViewer";
import { App } from "./app/App";
import { initializeRustEngine } from "./game/rustEngine";
import "./app/styles.css";

await initializeRustEngine();

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    {new URLSearchParams(window.location.search).has("watch") ? <SelfPlayViewer /> : <App />}
  </React.StrictMode>,
);
