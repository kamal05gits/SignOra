// ---------------------------------------------------------------------------
// Global application state (zustand). Deliberately framework-agnostic data
// only — no DOM/camera/model objects live here, keeping the GUI decoupled
// from the vision/ML layer as required by the architecture.
// ---------------------------------------------------------------------------
import { create } from "zustand";
import type {
  AppSettings,
  BackendHealth,
  BackendModelInfo,
  BackendStatus,
  ClassificationResult,
  DatasetSample,
  EmergencyContact,
  EngineSource,
  EngineStatus,
  RecognizedToken,
  TranslationState,
} from "../types";
import {
  loadContacts,
  loadDataset,
  loadHistory,
  loadSettings,
  saveContacts,
  saveDataset,
  saveHistory,
  saveSettings,
} from "../lib/storage";
import { tokensToSentence } from "../lib/sentenceProcessor";

interface AppState {
  // Engine / pipeline status
  engineStatus: EngineStatus;
  statusMessage: string;
  setEngineStatus: (s: EngineStatus, message?: string) => void;

  fps: number;
  setFps: (fps: number) => void;

  handsDetected: number;
  setHandsDetected: (n: number) => void;

  currentClassification: ClassificationResult | null;
  setCurrentClassification: (c: ClassificationResult | null) => void;

  isPaused: boolean;
  togglePaused: () => void;
  setPaused: (v: boolean) => void;

  // Sentence building
  tokens: RecognizedToken[];
  addToken: (t: RecognizedToken) => void;
  removeLastToken: () => void;
  clearTokens: () => void;
  insertSpaceMarker: () => void;
  sentence: () => string;

  // Translation
  translation: TranslationState;
  setTranslation: (t: TranslationState) => void;

  // Settings
  settings: AppSettings;
  updateSettings: (patch: Partial<AppSettings>) => void;

  // Emergency contacts
  contacts: EmergencyContact[];
  addContact: (c: EmergencyContact) => void;
  removeContact: (id: string) => void;

  // Dataset
  dataset: DatasetSample[];
  addSample: (s: DatasetSample) => void;
  removeSample: (id: string) => void;
  clearDataset: () => void;

  // History (completed sentences that were spoken/cleared)
  history: string[];
  pushHistory: (sentence: string) => void;
  clearHistory: () => void;

  lastError: string | null;
  setError: (msg: string | null) => void;

  // Backend inference state
  backendStatus: BackendStatus;
  setBackendStatus: (s: BackendStatus) => void;
  backendModels: BackendModelInfo[];
  setBackendModels: (m: BackendModelInfo[]) => void;
  backendHealth: BackendHealth | null;
  setBackendHealth: (h: BackendHealth | null) => void;
  backendError: string | null;
  setBackendError: (msg: string | null) => void;
  engineSource: EngineSource;
  setEngineSource: (s: EngineSource) => void;
  backendLatencyMs: number;
  setBackendLatencyMs: (ms: number) => void;
}

export const useAppStore = create<AppState>((set, get) => ({
  engineStatus: "idle",
  statusMessage: "Camera is off. Press Start to begin.",
  setEngineStatus: (s, message) =>
    set({ engineStatus: s, statusMessage: message ?? get().statusMessage }),

  fps: 0,
  setFps: (fps) => set({ fps }),

  handsDetected: 0,
  setHandsDetected: (n) => set({ handsDetected: n }),

  currentClassification: null,
  setCurrentClassification: (c) => set({ currentClassification: c }),

  isPaused: false,
  togglePaused: () => set((s) => ({ isPaused: !s.isPaused })),
  setPaused: (v) => set({ isPaused: v }),

  tokens: [],
  addToken: (t) => set((s) => ({ tokens: [...s.tokens, t] })),
  removeLastToken: () => set((s) => ({ tokens: s.tokens.slice(0, -1) })),
  clearTokens: () => set({ tokens: [] }),
  insertSpaceMarker: () =>
    set((s) => ({
      tokens: [
        ...s.tokens,
        { id: `sp-${Date.now()}`, label: "SPACE", gloss: "—", confidence: 1, timestamp: Date.now() },
      ],
    })),
  sentence: () => tokensToSentence(get().tokens.filter((t) => t.gloss !== "—")),

  translation: { status: "idle", sourceText: "", translatedText: "", error: null },
  setTranslation: (t) => set({ translation: t }),

  settings: loadSettings(),
  updateSettings: (patch) => {
    const next = { ...get().settings, ...patch };
    saveSettings(next);
    set({ settings: next });
  },

  contacts: loadContacts(),
  addContact: (c) => {
    const next = [...get().contacts, c];
    saveContacts(next);
    set({ contacts: next });
  },
  removeContact: (id) => {
    const next = get().contacts.filter((c) => c.id !== id);
    saveContacts(next);
    set({ contacts: next });
  },

  dataset: loadDataset(),
  addSample: (sample) => {
    const next = [...get().dataset, sample];
    saveDataset(next);
    set({ dataset: next });
  },
  removeSample: (id) => {
    const next = get().dataset.filter((s) => s.id !== id);
    saveDataset(next);
    set({ dataset: next });
  },
  clearDataset: () => {
    saveDataset([]);
    set({ dataset: [] });
  },

  history: loadHistory(),
  pushHistory: (sentence) => {
    if (!sentence.trim()) return;
    const next = [sentence, ...get().history].slice(0, 50);
    saveHistory(next);
    set({ history: next });
  },
  clearHistory: () => {
    saveHistory([]);
    set({ history: [] });
  },

  lastError: null,
  setError: (msg) => set({ lastError: msg }),

  backendStatus: "unknown",
  setBackendStatus: (s) => set({ backendStatus: s }),

  backendModels: [],
  setBackendModels: (m) => set({ backendModels: m }),

  backendHealth: null,
  setBackendHealth: (h) => set({ backendHealth: h }),

  backendError: null,
  setBackendError: (msg) => set({ backendError: msg }),

  engineSource: "on-device",
  setEngineSource: (s) => set({ engineSource: s }),

  backendLatencyMs: 0,
  setBackendLatencyMs: (ms) => set({ backendLatencyMs: ms }),
}));
