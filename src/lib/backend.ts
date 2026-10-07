// ---------------------------------------------------------------------------
// MODULE: Backend API client
//
// Typed wrapper around the FastAPI service in backend/. Every call funnels
// through `request()` so that error handling is uniform:
//
//   * a network failure            -> BackendError with status 0 (unreachable)
//   * 404 unknown model            -> BackendError with status 404
//   * 409 wrong input modality     -> BackendError with status 409
//   * 503 model not loadable       -> BackendError with status 503
//
// Callers decide what to do with those; nothing in here swallows an error or
// invents a result. `SignStream` is the WebSocket session used by the live
// translator: the browser keeps MediaPipe locally and ships only the 21x3
// landmark vectors, the backend runs the trained model over its window.
// ---------------------------------------------------------------------------
import type {
  BackendEvaluationReport,
  BackendHealth,
  BackendModelInfo,
  BackendPredictionResponse,
  DetectedHand,
  Frame,
  Landmark,
  StreamMessage,
} from "../types";

export class BackendError extends Error {
  readonly status: number;
  readonly payload: unknown;

  constructor(message: string, status = 0, payload?: unknown) {
    super(message);
    this.name = "BackendError";
    this.status = status;
    this.payload = payload;
  }

  /** True when the server could not be reached at all (DNS/refused/offline). */
  get unreachable(): boolean {
    return this.status === 0;
  }
}

/** Empty string means "same origin" — the vite dev proxy forwards /api and /ws. */
export function resolveBaseUrl(configured: string | null | undefined): string {
  const url = (configured ?? "").trim();
  if (!url) return "";
  return url.replace(/\/+$/, "");
}

export function websocketUrl(baseUrl: string, params: Record<string, string | number | null>): string {
  const query = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === null || value === undefined || value === "") continue;
    query.set(key, String(value));
  }
  const suffix = `/ws/stream${query.toString() ? `?${query}` : ""}`;
  if (baseUrl) return `${baseUrl.replace(/^http/, "ws")}${suffix}`;
  const protocol = typeof window !== "undefined" && window.location.protocol === "https:" ? "wss" : "ws";
  return `${protocol}://${window.location.host}${suffix}`;
}

async function request<T>(baseUrl: string, path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${baseUrl}${path}`, init);
  } catch (error) {
    throw new BackendError(
      `Cannot reach the inference backend at ${baseUrl || window.location.origin}${path}. ` +
        "Is it running? Start it with ./scripts/dev.sh (see backend/README.md).",
      0,
      error
    );
  }

  const text = await response.text();
  const parsed = text ? safeJson(text) : null;

  if (!response.ok) {
    const detail = extractDetail(parsed) ?? `Request failed with HTTP ${response.status}`;
    throw new BackendError(detail, response.status, parsed);
  }
  return parsed as T;
}

function safeJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

function extractDetail(payload: unknown): string | null {
  if (typeof payload === "string") return payload;
  if (payload && typeof payload === "object" && "detail" in payload) {
    const detail = (payload as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail
        .map((entry) => (entry && typeof entry === "object" ? String((entry as { msg?: string }).msg ?? "") : String(entry)))
        .filter(Boolean)
        .join("; ");
    }
    return JSON.stringify(detail);
  }
  return null;
}

// ---------------------------------------------------------------------------
// REST
// ---------------------------------------------------------------------------
export function fetchHealth(baseUrl: string): Promise<BackendHealth> {
  return request<BackendHealth>(baseUrl, "/health");
}

export function fetchModels(baseUrl: string): Promise<BackendModelInfo[]> {
  return request<BackendModelInfo[]>(baseUrl, "/api/models");
}

export function refreshModels(baseUrl: string): Promise<BackendModelInfo[]> {
  return request<BackendModelInfo[]>(baseUrl, "/api/models/refresh", { method: "POST" });
}

export function loadModel(baseUrl: string, modelId: string): Promise<BackendModelInfo> {
  return request<BackendModelInfo>(baseUrl, `/api/models/${encodeURIComponent(modelId)}/load`, { method: "POST" });
}

export function unloadModel(baseUrl: string, modelId: string): Promise<BackendModelInfo> {
  return request<BackendModelInfo>(baseUrl, `/api/models/${encodeURIComponent(modelId)}/unload`, { method: "POST" });
}

export function predictImage(
  baseUrl: string,
  payload: { model_id?: string | null; image_base64: string; top_k?: number }
): Promise<BackendPredictionResponse> {
  return request<BackendPredictionResponse>(baseUrl, "/api/predict/image", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function predictLandmarks(
  baseUrl: string,
  payload: { model_id?: string | null; frames: Frame[]; top_k?: number; hand?: string }
): Promise<BackendPredictionResponse> {
  return request<BackendPredictionResponse>(baseUrl, "/api/predict/landmarks", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function evaluateDataset(
  baseUrl: string,
  payload: { model_id?: string | null; samples: unknown[]; confidence_threshold?: number }
): Promise<BackendEvaluationReport> {
  return request<BackendEvaluationReport>(baseUrl, "/api/dataset/evaluate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function syncSamples(
  baseUrl: string,
  samples: unknown[],
  replace = false
): Promise<{ written: number; total: number; path: string }> {
  return request(baseUrl, "/api/dataset/samples", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ samples, replace }),
  });
}

export function fetchDatasetSummary(
  baseUrl: string
): Promise<{ total: number; classes: Record<string, number>; path: string }> {
  return request(baseUrl, "/api/dataset/summary");
}

export function datasetExportHref(baseUrl: string, format: "npz" | "csv" | "jsonl"): string {
  return `${baseUrl}/api/dataset/export?format=${format}`;
}

// ---------------------------------------------------------------------------
// Live WebSocket session
// ---------------------------------------------------------------------------
export type StreamState = "idle" | "connecting" | "open" | "closed";

export interface SignStreamConfig {
  modelId: string | null;
  sequenceLength: number;
  stride: number;
  hand: "first" | "left" | "right" | "dominant";
}

export interface SignStreamHandlers {
  onPrediction?: (message: Extract<StreamMessage, { type: "prediction" }>) => void;
  onReady?: (message: Extract<StreamMessage, { type: "ready" | "pong" | "reset" }>) => void;
  onError?: (message: Extract<StreamMessage, { type: "error" }>) => void;
  onStateChange?: (state: StreamState) => void;
}

const MAX_RECONNECT_ATTEMPTS = 5;

export class SignStream {
  private socket: WebSocket | null = null;
  private state: StreamState = "idle";
  private attempts = 0;
  private closedByCaller = false;
  private reconnectTimer: number | null = null;
  private lastLatencyMs = 0;

  constructor(
    private baseUrl: string,
    private config: SignStreamConfig,
    private handlers: SignStreamHandlers = {}
  ) {}

  get currentState(): StreamState {
    return this.state;
  }

  get latencyMs(): number {
    return this.lastLatencyMs;
  }

  connect(): void {
    this.closedByCaller = false;
    this.openSocket();
  }

  private openSocket(): void {
    this.setState("connecting");
    const url = websocketUrl(this.baseUrl, {
      model_id: this.config.modelId,
      sequence_length: this.config.sequenceLength,
      stride: this.config.stride,
      hand: this.config.hand,
    });

    let socket: WebSocket;
    try {
      socket = new WebSocket(url);
    } catch (error) {
      this.handlers.onError?.({
        type: "error",
        message: `Could not open a WebSocket to ${url}: ${error instanceof Error ? error.message : String(error)}`,
        fatal: true,
      });
      this.setState("closed");
      return;
    }
    this.socket = socket;

    socket.onopen = () => {
      this.attempts = 0;
      this.setState("open");
    };

    socket.onmessage = (event) => {
      let message: StreamMessage;
      try {
        message = JSON.parse(String(event.data)) as StreamMessage;
      } catch {
        return;
      }
      if (message.type === "prediction") {
        this.lastLatencyMs = message.latency_ms;
        this.handlers.onPrediction?.(message);
      } else if (message.type === "error") {
        this.handlers.onError?.(message);
        if (message.fatal) this.close();
      } else {
        this.handlers.onReady?.(message);
      }
    };

    socket.onclose = () => {
      this.socket = null;
      if (this.closedByCaller) {
        this.setState("closed");
        return;
      }
      this.setState("closed");
      this.scheduleReconnect();
    };

    socket.onerror = () => {
      // onclose follows immediately and carries the reconnect logic.
    };
  }

  private scheduleReconnect(): void {
    if (this.closedByCaller || this.attempts >= MAX_RECONNECT_ATTEMPTS) return;
    this.attempts += 1;
    const delay = Math.min(5000, 500 * 2 ** (this.attempts - 1));
    this.reconnectTimer = window.setTimeout(() => {
      this.reconnectTimer = null;
      if (!this.closedByCaller) this.openSocket();
    }, delay);
  }

  private setState(state: StreamState): void {
    if (this.state === state) return;
    this.state = state;
    this.handlers.onStateChange?.(state);
  }

  /** Push one camera frame's hands into the server-side temporal window. */
  sendFrame(hands: DetectedHand[]): void {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) return;
    const payload = {
      type: "frame",
      hands: hands.map((hand) => ({
        landmarks: hand.landmarks.map((point: Landmark) => ({ x: point.x, y: point.y, z: point.z })),
        handedness: hand.handedness,
        score: hand.score,
      })),
    };
    this.socket.send(JSON.stringify(payload));
  }

  /** Send a frame that had no hand, so the server window stays time-aligned. */
  sendEmptyFrame(): void {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) return;
    this.socket.send(JSON.stringify({ type: "frame", hands: [] }));
  }

  configure(config: Partial<SignStreamConfig>): void {
    this.config = { ...this.config, ...config };
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify({ type: "configure", ...config }));
    }
  }

  reset(): void {
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify({ type: "reset" }));
    }
  }

  close(): void {
    this.closedByCaller = true;
    if (this.reconnectTimer != null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.socket) {
      try {
        this.socket.close();
      } catch {
        // Already closed — nothing to clean up.
      }
      this.socket = null;
    }
    this.setState("closed");
  }
}

// ---------------------------------------------------------------------------
// Helpers shared by the UI
// ---------------------------------------------------------------------------

/** Grab the current camera frame as a base64 JPEG for /api/predict/image. */
export function captureFrameBase64(video: HTMLVideoElement | null, size = 256, quality = 0.85): string | null {
  if (!video || video.readyState < 2 || !video.videoWidth) return null;
  const canvas = document.createElement("canvas");
  const scale = size / Math.max(video.videoWidth, video.videoHeight);
  canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
  canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const dataUrl = canvas.toDataURL("image/jpeg", quality);
  return dataUrl.includes(",") ? dataUrl.split(",")[1] : null;
}

export function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error(`Could not read ${file.name}`));
    reader.onload = () => {
      const result = String(reader.result ?? "");
      resolve(result.includes(",") ? result.split(",")[1] : result);
    };
    reader.readAsDataURL(file);
  });
}

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const exponent = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
  return `${(bytes / 1024 ** exponent).toFixed(exponent === 0 ? 0 : 1)} ${units[exponent]}`;
}
