// ---------------------------------------------------------------------------
// MODULE: Lightweight local data storage
// Uses localStorage (JSON) for settings, emergency contacts and dataset
// samples. No video/audio is ever persisted or uploaded — only numeric
// landmark-derived feature vectors, in line with the project's privacy
// principles (see About > Privacy).
// ---------------------------------------------------------------------------
import type { AppSettings, DatasetSample, EmergencyContact } from "../types";

const KEYS = {
  settings: "isl.settings.v1",
  contacts: "isl.contacts.v1",
  dataset: "isl.dataset.v1",
  history: "isl.history.v1",
};

export const DEFAULT_SETTINGS: AppSettings = {
  confidenceThreshold: 0.72,
  smoothingWindow: 8,
  cooldownMs: 1400,
  outputLanguage: "hi",
  voiceURI: null,
  speechRate: 1,
  cameraDeviceId: null,
  mirrorPreview: true,
  showLandmarks: true,
  autoSpeak: false,

  // Backend inference. An empty URL means "same origin", which works through
  // the vite dev proxy (vite.config.ts) and behind any reverse proxy in prod.
  inferenceMode: "auto",
  // Default backend URL baked in at build time (VITE_BACKEND_URL, e.g. the
  // Vercel API project URL). An empty result means "same origin", which works
  // through the vite dev proxy (vite.config.ts) and behind any reverse proxy
  // in prod. Per-browser overrides in Settings always win.
  backendUrl: import.meta.env.VITE_BACKEND_URL ?? "",
  activeModelId: null,
  // Trained models spread probability over many classes, so their top-1 score
  // is usually lower than the geometric classifier's — hence a separate default.
  backendConfidenceThreshold: 0.5,
  streamStride: 1,
  streamSequenceLength: 30,
  backendHand: "first",
  fallbackToOnDevice: true,
  syncDatasetToBackend: true,
};

function readJSON<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return fallback;
    return { ...fallback, ...JSON.parse(raw) };
  } catch {
    return fallback;
  }
}

function writeJSON(key: string, value: unknown) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Storage may be unavailable (private mode / quota) — fail silently, non-critical.
  }
}

export function loadSettings(): AppSettings {
  return readJSON(KEYS.settings, DEFAULT_SETTINGS);
}
export function saveSettings(settings: AppSettings) {
  writeJSON(KEYS.settings, settings);
}

export function loadContacts(): EmergencyContact[] {
  try {
    const raw = localStorage.getItem(KEYS.contacts);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}
export function saveContacts(contacts: EmergencyContact[]) {
  writeJSON(KEYS.contacts, contacts);
}

export function loadDataset(): DatasetSample[] {
  try {
    const raw = localStorage.getItem(KEYS.dataset);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}
export function saveDataset(samples: DatasetSample[]) {
  writeJSON(KEYS.dataset, samples);
}

export function loadHistory(): string[] {
  try {
    const raw = localStorage.getItem(KEYS.history);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}
export function saveHistory(history: string[]) {
  writeJSON(KEYS.history, history);
}
