// ---------------------------------------------------------------------------
// Orchestration layer: wires together camera frames -> MediaPipe landmarks ->
// classification -> temporal smoothing -> global store updates. This is the
// only place that couples the independent modules together; the GUI components
// only ever read/write the zustand store, keeping presentation decoupled from
// the vision/ML pipeline.
//
// TWO CLASSIFICATION ENGINES, ONE PIPELINE
// ------------------------------------------------------------------------
// 1. on-device  — the transparent geometric classifier in src/lib/signClassifier.ts.
//    Runs entirely in the browser, needs no server.
// 2. backend    — the trained models in models/, served by backend/ (FastAPI).
//    The browser keeps MediaPipe locally (landmark detection is cheap and no
//    video leaves the machine) and streams the 21x3 landmark vectors over a
//    WebSocket; the server keeps the temporal window and runs the model.
//
// Both engines produce the same ClassificationResult, so everything after the
// classifier — smoothing, cooldown, token building, speech, translation — is
// shared and cannot drift between the two. Settings decide which engine runs:
//   on-device | backend | auto (backend when reachable, else on-device)
//
// NOTE: the recursive requestAnimationFrame loop always re-reads state via
// useAppStore.getState() rather than closing over React props/state, so
// pause/resume and live settings changes (confidence threshold, smoothing
// window, engine mode) take effect immediately without restarting the loop.
// ---------------------------------------------------------------------------
import { useEffect, useMemo, useRef } from "react";
import { useCamera } from "./useCamera";
import { useHandLandmarker } from "./useHandLandmarker";
import { useBackendModels } from "./useBackendModels";
import { extractFeaturesFromFrame } from "../lib/geometry";
import { classifyHandFeatures } from "../lib/signClassifier";
import { TemporalSmoother } from "../lib/sequenceBuffer";
import { HAND_CONNECTIONS } from "../lib/handConnections";
import { resolveBaseUrl, SignStream } from "../lib/backend";
import { useAppStore } from "../store/useAppStore";
import type { ClassificationResult, DetectedHand, HandFeatureSet, StreamMessage } from "../types";

export function useRecognitionEngine() {
  const camera = useCamera();
  const landmarker = useHandLandmarker();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const smootherRef = useRef(new TemporalSmoother());
  const rafRef = useRef<number | null>(null);
  const fpsCounter = useRef({ frames: 0, last: performance.now() });
  const latestFeaturesRef = useRef<HandFeatureSet[]>([]);
  const latestHandsRef = useRef<DetectedHand[]>([]);
  const streamRef = useRef<SignStream | null>(null);
  const backendResultRef = useRef<ClassificationResult | null>(null);
  const frameCounter = useRef(0);

  const settings = useAppStore((s) => s.settings);
  const setEngineStatus = useAppStore((s) => s.setEngineStatus);
  const setFps = useAppStore((s) => s.setFps);
  const backendStatus = useAppStore((s) => s.backendStatus);

  // Keeps /health polling and the model list fresh for the whole app.
  const backend = useBackendModels();

  const baseUrl = resolveBaseUrl(settings.backendUrl);
  const backendActive = useMemo(() => {
    if (settings.inferenceMode === "on-device") return false;
    if (settings.inferenceMode === "backend") return true;
    return backendStatus === "online"; // "auto"
  }, [settings.inferenceMode, backendStatus]);

  const streamEnabled = backendActive && camera.isActive;

  // -------------------------------------------------------------------------
  // Smoother configuration. The geometric classifier and a softmax over N
  // classes are calibrated differently, so each engine has its own threshold.
  // -------------------------------------------------------------------------
  useEffect(() => {
    smootherRef.current.updateConfig({
      windowSize: settings.smoothingWindow,
      cooldownMs: settings.cooldownMs,
      confidenceThreshold: backendActive ? settings.backendConfidenceThreshold : settings.confidenceThreshold,
    });
  }, [
    settings.smoothingWindow,
    settings.cooldownMs,
    settings.confidenceThreshold,
    settings.backendConfidenceThreshold,
    backendActive,
  ]);

  // -------------------------------------------------------------------------
  // WebSocket lifecycle: (re)created whenever the model, window or hand
  // selection changes, torn down when the camera stops or the mode switches.
  // -------------------------------------------------------------------------
  useEffect(() => {
    if (!streamEnabled || !settings.activeModelId) {
      streamRef.current?.close();
      streamRef.current = null;
      backendResultRef.current = null;
      return;
    }

    const stream = new SignStream(
      baseUrl,
      {
        modelId: settings.activeModelId,
        sequenceLength: settings.streamSequenceLength,
        stride: settings.streamStride,
        hand: settings.backendHand,
      },
      {
        onPrediction: (message: Extract<StreamMessage, { type: "prediction" }>) => {
          const top = message.top;
          backendResultRef.current = {
            label: top?.label ?? null,
            gloss: top?.gloss ?? top?.label ?? null,
            confidence: top?.score ?? 0,
            handsUsed: latestHandsRef.current.length,
            source: "backend",
            modelId: message.model_id,
          };
          useAppStore.getState().setBackendLatencyMs(message.latency_ms);
        },
        onError: (message) => {
          useAppStore.getState().setBackendError(message.message);
          if (message.fatal) {
            useAppStore.getState().setEngineStatus("error", `Backend model error: ${message.message}`);
          }
        },
        onReady: (message) => {
          if (message.type === "ready") {
            useAppStore.getState().setBackendError(null);
            useAppStore.getState().setEngineStatus("running", `Streaming to backend model '${message.model_id}'.`);
          }
        },
      }
    );

    stream.connect();
    streamRef.current = stream;
    frameCounter.current = 0;

    return () => {
      stream.close();
      if (streamRef.current === stream) streamRef.current = null;
    };
  }, [
    streamEnabled,
    baseUrl,
    settings.activeModelId,
    settings.streamSequenceLength,
    settings.streamStride,
    settings.backendHand,
  ]);

  // -------------------------------------------------------------------------
  // Status reporting
  // -------------------------------------------------------------------------
  useEffect(() => {
    if (landmarker.error) {
      setEngineStatus("error", landmarker.error);
    } else if (landmarker.isLoading) {
      setEngineStatus("loading-model", "Loading on-device hand-detection model…");
    } else if (landmarker.isReady && !camera.isActive) {
      setEngineStatus("idle", "Model ready. Press Start Camera to begin.");
    }
  }, [landmarker.isReady, landmarker.isLoading, landmarker.error, camera.isActive, setEngineStatus]);

  useEffect(() => {
    if (camera.error) setEngineStatus("error", camera.error);
  }, [camera.error, setEngineStatus]);

  useEffect(() => {
    const state = useAppStore.getState();
    if (!camera.isActive) return;
    if (backendActive && settings.inferenceMode === "backend" && backendStatus !== "online") {
      setEngineStatus(
        "error",
        "Backend inference is selected but the API is not reachable. Start it with ./scripts/dev.sh or switch to on-device in Settings."
      );
      return;
    }
    if (backendActive && settings.activeModelId) {
      state.setEngineSource("backend");
    } else {
      state.setEngineSource("on-device");
      if (settings.inferenceMode === "auto" && backendStatus === "offline") {
        setEngineStatus("running", "Backend unreachable — using the on-device classifier.");
      }
    }
  }, [backendActive, backendStatus, camera.isActive, settings.activeModelId, settings.inferenceMode, setEngineStatus]);

  // -------------------------------------------------------------------------
  function drawOverlay(hands: DetectedHand[], width: number, height: number, showLandmarks: boolean) {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);
    if (!showLandmarks) return;

    for (const hand of hands) {
      ctx.strokeStyle = hand.handedness === "Right" ? "#22d3ee" : "#a78bfa";
      ctx.lineWidth = 3;
      for (const [a, b] of HAND_CONNECTIONS) {
        const p1 = hand.landmarks[a];
        const p2 = hand.landmarks[b];
        if (!p1 || !p2) continue;
        ctx.beginPath();
        ctx.moveTo(p1.x * width, p1.y * height);
        ctx.lineTo(p2.x * width, p2.y * height);
        ctx.stroke();
      }
      ctx.fillStyle = "#f8fafc";
      for (const p of hand.landmarks) {
        ctx.beginPath();
        ctx.arc(p.x * width, p.y * height, 3.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  function loop() {
    rafRef.current = requestAnimationFrame(loop);
    const video = camera.videoRef.current;
    if (!video || !landmarker.isReady || video.readyState < 2) return;

    const now = performance.now();
    const detection = landmarker.detectVideoFrame(video, now);

    // FPS accounting
    fpsCounter.current.frames += 1;
    if (now - fpsCounter.current.last >= 1000) {
      setFps(fpsCounter.current.frames);
      fpsCounter.current.frames = 0;
      fpsCounter.current.last = now;
    }

    if (!detection) return;

    // Always read the freshest state/settings directly from the store so that
    // pausing or changing settings mid-stream takes effect on the very next frame.
    const state = useAppStore.getState();
    const { settings: live } = state;

    drawOverlay(
      detection.hands,
      detection.frameWidth || video.videoWidth,
      detection.frameHeight || video.videoHeight,
      live.showLandmarks
    );
    state.setHandsDetected(detection.hands.length);
    latestHandsRef.current = detection.hands;

    if (state.isPaused) return;

    frameCounter.current += 1;
    const stream = streamRef.current;
    const streaming = stream != null && stream.currentState === "open";
    state.setEngineSource(streaming ? "backend" : "on-device");

    if (streaming && stream) {
      // Send every Nth frame so a 30 fps camera does not outrun the model.
      if (frameCounter.current % Math.max(1, live.streamStride) === 0) {
        if (detection.hands.length) stream.sendFrame(detection.hands);
        else stream.sendEmptyFrame();
      }
    }

    const features = extractFeaturesFromFrame(detection.hands);
    latestFeaturesRef.current = features;

    // Backend predictions arrive asynchronously; use the newest one available.
    const classification =
      streaming && backendResultRef.current ? backendResultRef.current : classifyHandFeatures(features);

    const previous = state.currentClassification;
    if (
      !previous ||
      previous.label !== classification.label ||
      previous.confidence !== classification.confidence ||
      previous.source !== classification.source
    ) {
      state.setCurrentClassification(classification);
    }

    const threshold = streaming ? live.backendConfidenceThreshold : live.confidenceThreshold;
    const stable = smootherRef.current.push(classification, now);
    if (stable && stable.confidence >= threshold) {
      state.addToken({
        id: `${now}-${Math.random().toString(36).slice(2, 8)}`,
        label: stable.label,
        gloss: stable.gloss,
        confidence: stable.confidence,
        timestamp: Date.now(),
      });
    }
  }

  async function startCamera() {
    await camera.start(useAppStore.getState().settings.cameraDeviceId);
    if (!camera.error) {
      setEngineStatus("running", "Recognition running. Show a sign to the camera.");
    }
    smootherRef.current.reset();
    backendResultRef.current = null;
    frameCounter.current = 0;
    if (rafRef.current == null) {
      rafRef.current = requestAnimationFrame(loop);
    }
  }

  function stopCamera() {
    if (rafRef.current != null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    streamRef.current?.close();
    streamRef.current = null;
    backendResultRef.current = null;
    camera.stop();
    const state = useAppStore.getState();
    state.setCurrentClassification(null);
    state.setHandsDetected(0);
    setFps(0);
    setEngineStatus("idle", "Camera stopped.");
  }

  useEffect(() => {
    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
      streamRef.current?.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return {
    camera,
    landmarker,
    canvasRef,
    startCamera,
    stopCamera,
    getLatestFeatures: () => latestFeaturesRef.current,
    getLatestHands: () => latestHandsRef.current,
    backend,
    streamState: streamRef.current?.currentState ?? "idle",
  };
}
