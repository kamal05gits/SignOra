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

/** Result of one frame of hand-landmark detection. */
export interface HandDetectionResult {
  hands: DetectedHand[];
  timestampMs: number;
  frameWidth: number;
  frameHeight: number;
}

/** Output of the heuristic geometric sign classifier for a single frame. */
export interface ClassificationResult {
  label: string | null;
  gloss: string | null;
  confidence: number;
  handsUsed: number;
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
