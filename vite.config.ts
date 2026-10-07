import path from "path";
import { fileURLToPath } from "url";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import { viteSingleFile } from "vite-plugin-singlefile";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// The Python inference backend (backend/app/main.py). The dev server proxies
// /api, /health and /ws to it so the browser only ever talks to its own origin
// — no CORS setup, and it keeps working behind a reverse proxy in production.
const backend = process.env.SIGNORA_BACKEND_URL ?? "http://127.0.0.1:8000";
const backendWs = backend.replace(/^http/, "ws");

const proxy = {
  "/api": { target: backend, changeOrigin: true },
  "/health": { target: backend, changeOrigin: true },
  "/ws": { target: backendWs, ws: true, changeOrigin: true },
};

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss(), viteSingleFile()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"),
    },
  },
  server: {
    host: "0.0.0.0",
    port: 5173,
    // Accept requests addressed to any hostname (tunnels, previews, LAN IPs).
    allowedHosts: true,
    proxy,
    // The Python virtualenv lives inside the repo and holds tens of thousands of
    // torch/TensorFlow files; watching them exhausts the OS inotify limit and
    // kills the dev server with ENOSPC.
    watch: {
      ignored: [
        "**/backend/.venv/**",
        "**/backend/data/**",
        "**/backend/**/__pycache__/**",
        "**/node_modules/**",
        "**/.git/**",
        "**/dist/**",
      ],
    },
  },
  preview: {
    host: "0.0.0.0",
    allowedHosts: true,
    proxy,
  },
});
