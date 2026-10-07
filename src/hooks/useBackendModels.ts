// ---------------------------------------------------------------------------
// MODULE: Backend connection state
//
// Polls /health and /api/models while the app is configured to use the backend,
// and keeps the result in the global store so every page (Live Translator,
// Recognition Engine, Settings) shows the same truth about what the server can
// actually serve. Polling stops when inference mode is "on-device" so a
// browser-only deployment never fires requests at a server that isn't there.
// ---------------------------------------------------------------------------
import { useCallback, useEffect, useRef } from "react";
import { BackendError, fetchHealth, fetchModels, loadModel, resolveBaseUrl, unloadModel } from "../lib/backend";
import { useAppStore } from "../store/useAppStore";
import type { BackendModelInfo } from "../types";

const POLL_INTERVAL_MS = 10_000;

export function useBackendModels(options: { autoRefresh?: boolean } = {}) {
  const { autoRefresh = true } = options;
  const settings = useAppStore((s) => s.settings);
  const setStatus = useAppStore((s) => s.setBackendStatus);
  const setModels = useAppStore((s) => s.setBackendModels);
  const setHealth = useAppStore((s) => s.setBackendHealth);
  const setError = useAppStore((s) => s.setBackendError);
  const updateSettings = useAppStore((s) => s.updateSettings);

  const baseUrl = resolveBaseUrl(settings.backendUrl);
  const enabled = settings.inferenceMode !== "on-device";
  const inFlight = useRef(false);

  const refresh = useCallback(async () => {
    if (inFlight.current) return;
    inFlight.current = true;
    try {
      const [health, models] = await Promise.all([fetchHealth(baseUrl), fetchModels(baseUrl)]);
      setHealth(health);
      setModels(models);
      setStatus("online");
      setError(health.problems.length ? health.problems[0] : null);

      // Adopt the server's first usable model when the user hasn't picked one,
      // or when the picked model no longer exists (renamed / deleted file).
      const current = useAppStore.getState().settings.activeModelId;
      const exists = current ? models.some((m) => m.id === current) : false;
      if (!exists) {
        const preferred = pickDefaultModel(models);
        if (preferred) updateSettings({ activeModelId: preferred });
      }
    } catch (error) {
      const message = error instanceof BackendError ? error.message : String(error);
      setModels([]);
      setHealth(null);
      setStatus("offline");
      setError(message);
    } finally {
      inFlight.current = false;
    }
  }, [baseUrl, setHealth, setModels, setStatus, setError, updateSettings]);

  useEffect(() => {
    if (!enabled) {
      setStatus("unknown");
      return;
    }
    void refresh();
    if (!autoRefresh) return;
    const id = window.setInterval(() => void refresh(), POLL_INTERVAL_MS);
    return () => window.clearInterval(id);
  }, [enabled, autoRefresh, refresh, setStatus]);

  const load = useCallback(
    async (modelId: string): Promise<BackendModelInfo | null> => {
      try {
        const info = await loadModel(baseUrl, modelId);
        setModels(useAppStore.getState().backendModels.map((m) => (m.id === info.id ? info : m)));
        await refresh();
        return info;
      } catch (error) {
        setError(error instanceof BackendError ? error.message : String(error));
        return null;
      }
    },
    [baseUrl, refresh, setModels, setError]
  );

  const unload = useCallback(
    async (modelId: string): Promise<BackendModelInfo | null> => {
      try {
        const info = await unloadModel(baseUrl, modelId);
        setModels(useAppStore.getState().backendModels.map((m) => (m.id === info.id ? info : m)));
        return info;
      } catch (error) {
        setError(error instanceof BackendError ? error.message : String(error));
        return null;
      }
    },
    [baseUrl, setModels, setError]
  );

  const models = useAppStore((s) => s.backendModels);
  const status = useAppStore((s) => s.backendStatus);
  const health = useAppStore((s) => s.backendHealth);
  const error = useAppStore((s) => s.backendError);

  return { baseUrl, enabled, status, models, health, error, refresh, load, unload };
}

/**
 * Choose the model the live translator should use: prefer one that is already
 * resident, then one that accepts landmark sequences, then anything loadable.
 */
export function pickDefaultModel(models: BackendModelInfo[]): string | null {
  if (!models.length) return null;
  const usable = models.filter((m) => !m.lfs_pointer && m.status !== "load_error" && m.status !== "needs_manifest");
  const pool = usable.length ? usable : models;
  const loaded = pool.find((m) => m.loaded);
  if (loaded) return loaded.id;
  const landmarks = pool.find((m) => m.modality === "landmarks");
  if (landmarks) return landmarks.id;
  const ready = pool.find((m) => m.status === "ready" || m.status === "idle");
  return ready?.id ?? pool[0]?.id ?? null;
}
