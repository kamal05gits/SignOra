// ---------------------------------------------------------------------------
// Shared type definitions used across the vision, ML, language and UI layers
// ---------------------------------------------------------------------------

/** A single normalized hand landmark as returned by MediaPipe (0..1 range). */
export interface Landmark {
  x: number;
  y: number;
  z: number;
}

export type Handedness = "Left" | "Right";

export interface DetectedHand {
  landmarks: Landmark[];
  handedness: Handedness;
  score: number;
}

/** One camera frame: every hand MediaPipe detected in it (mirrors the backend's `Frame`). */
export interface Frame {
  hands: DetectedHand[];
}

/** Result of one frame of hand-landmark detection. */
export interface HandDetectionResult {
  hands: DetectedHand[];
  timestampMs: number;
  frameWidth: number;
  frameHeight: number;
}

/**
 * Output of one classification step, whichever engine produced it. Both the
 * on-device geometric classifier and the backend's trained models return this
 * shape, so smoothing, token building and speech stay engine-agnostic.
 */
export interface ClassificationResult {
  label: string | null;
  gloss: string | null;
  confidence: number;
  handsUsed: number;
  source?: EngineSource;
  /** Backend model id, when the prediction came from the API. */
  modelId?: string;
}

/** A token that has survived temporal smoothing / cooldown and is considered "recognized". */
export interface RecognizedToken {
  id: string;
  label: string;
  gloss: string;
  confidence: number;
  timestamp: number;
}

export interface SignDefinition {
  id: string;
  label: string; // short code e.g. "A"
  gloss: string; // word/phrase represented, e.g. "HELLO"
  description: string;
  category: "letter" | "number" | "word" | "phrase";
  match: (features: HandFeatureSet) => number; // returns 0..1 confidence
}

/** Geometric feature set extracted from a single detected hand, used by the classifier. */
export interface HandFeatureSet {
  handedness: Handedness;
  fingerCurl: {
    thumb: number;
    index: number;
    middle: number;
    ring: number;
    pinky: number;
  };
  fingerExtended: {
    thumb: boolean;
    index: boolean;
    middle: boolean;
    ring: boolean;
    pinky: boolean;
  };
  spread: number;
  thumbIndexDistance: number;
  palmOrientation: number;
  raw: Landmark[];
}

export interface DatasetSample {
  id: string;
  label: string;
  createdAt: number;
  features: HandFeatureSet[]; // one per detected hand at capture time
  /**
   * The raw landmark frame behind this sample. Optional because older samples
   * recorded before the backend existed only carry `features`; the backend uses
   * whichever of the two a given model can consume.
   */
  frames?: Frame[];
  thumbnail?: string; // small base64 preview
}

export interface AppSettings {
  confidenceThreshold: number; // 0..1
  smoothingWindow: number; // number of frames for majority vote
  cooldownMs: number; // minimum time between accepted tokens
  outputLanguage: string; // ISO code, e.g. "hi"
  voiceURI: string | null;
  speechRate: number;
  cameraDeviceId: string | null;
  mirrorPreview: boolean;
  showLandmarks: boolean;
  autoSpeak: boolean;

  // --- backend inference (see backend/README.md) ---
  inferenceMode: InferenceMode; // which engine produces signs
  backendUrl: string; // "" = same origin (the vite proxy); else http://host:port
  activeModelId: string | null; // model the backend should run
  backendConfidenceThreshold: number; // trained models are calibrated differently
  streamStride: number; // send every Nth frame to the backend
  streamSequenceLength: number; // temporal window the backend keeps
  backendHand: "first" | "left" | "right" | "dominant";
  fallbackToOnDevice: boolean; // use the geometric classifier when the API is down
  syncDatasetToBackend: boolean; // push captured samples to the API
}

export interface EmergencyContact {
  id: string;
  name: string;
  phone: string;
  relation: string;
}

export interface TranslationState {
  status: "idle" | "loading" | "success" | "error";
  sourceText: string;
  translatedText: string;
  error: string | null;
}

export type EngineStatus =
  | "idle"
  | "loading-model"
  | "starting-camera"
  | "running"
  | "paused"
  | "error";

// ---------------------------------------------------------------------------
// Backend inference (see backend/app/schemas.py for the server-side mirror)
// ---------------------------------------------------------------------------

/** Which engine turns frames into signs. */
export type InferenceMode = "on-device" | "backend" | "auto";

export type BackendFramework = "torch" | "keras";
export type BackendModality = "image" | "video" | "landmarks" | "features" | "unknown";
export type BackendModelStatus =
  | "ready"
  | "idle"
  | "unloaded"
  | "missing_weights"
  | "needs_manifest"
  | "load_error"
  | "runtime_missing";

export interface BackendModelInfo {
  id: string;
  filename: string;
  framework: BackendFramework;
  format: string;
  size_bytes: number;
  modality: BackendModality;
  status: BackendModelStatus;
  detail: string | null;
  architecture: string | null;
  input_shape: Array<number | null> | null;
  output_shape: Array<number | null> | null;
  num_classes: number | null;
  labels: string[] | null;
  labels_source: string | null;
  loaded: boolean;
  lfs_pointer: boolean;
  load_time_ms: number | null;
  last_inference_ms: number | null;
  inference_count: number;
  hint: string | null;
}

export interface BackendPrediction {
  index: number;
  label: string;
  gloss: string | null;
  score: number;
}

export interface BackendPredictionResponse {
  model_id: string;
  modality: BackendModality;
  predictions: BackendPrediction[];
  top: BackendPrediction | null;
  frames_used: number;
  latency_ms: number;
  warnings: string[];
}

export interface BackendHealth {
  status: "ok" | "degraded";
  version: string;
  runtimes: Record<string, string | null>;
  models_total: number;
  models_ready: number;
  problems: string[];
}

export type BackendStatus = "unknown" | "online" | "offline";

/** Which engine actually produced the sign currently on screen. */
export type EngineSource = "on-device" | "backend";

export interface StreamPredictionMessage {
  type: "prediction";
  frame: number;
  model_id: string;
  top: BackendPrediction | null;
  predictions: BackendPrediction[];
  latency_ms: number;
  frames_in_window: number;
  dropped: number;
  warnings: string[];
}

export interface StreamStatusMessage {
  type: "ready" | "pong" | "reset";
  model_id?: string;
  modality?: BackendModality;
  sequence_length?: number;
  stride?: number;
  hand?: string;
}

export interface StreamErrorMessage {
  type: "error";
  message: string;
  fatal: boolean;
}

export type StreamMessage = StreamPredictionMessage | StreamStatusMessage | StreamErrorMessage;

export interface ClassMetrics {
  label: string;
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface BackendEvaluationReport {
  model_id: string;
  total_samples: number;
  correct: number;
  accuracy: number;
  rejected: number;
  classes: string[];
  confusion_matrix: number[][];
  per_class: ClassMetrics[];
  latency_ms: number;
}
