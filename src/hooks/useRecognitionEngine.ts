// ---------------------------------------------------------------------------
// Orchestration layer: wires together camera frames -> MediaPipe landmarks ->
// feature extraction -> heuristic classification -> temporal smoothing ->
// global store updates. This is the only place that couples the independent
// modules together; the GUI components only ever read/write the zustand
// store, keeping presentation decoupled from the vision/ML pipeline.
//
// NOTE: the recursive requestAnimationFrame loop always re-reads state via
// useAppStore.getState() rather than closing over React props/state, so
// pause/resume and live settings changes (confidence threshold, smoothing
// window, etc.) take effect immediately without needing to restart the loop.
// ---------------------------------------------------------------------------
import { useEffect, useRef } from "react";
import { useCamera } from "./useCamera";
import { useHandLandmarker } from "./useHandLandmarker";
import { extractFeaturesFromFrame } from "../lib/geometry";
import { classifyHandFeatures } from "../lib/signClassifier";
import { TemporalSmoother } from "../lib/sequenceBuffer";
import { HAND_CONNECTIONS } from "../lib/handConnections";
import { useAppStore } from "../store/useAppStore";
import type { DetectedHand, HandFeatureSet } from "../types";

export function useRecognitionEngine() {
  const camera = useCamera();
  const landmarker = useHandLandmarker();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const smootherRef = useRef(new TemporalSmoother());
  const rafRef = useRef<number | null>(null);
  const fpsCounter = useRef({ frames: 0, last: performance.now() });
  const latestFeaturesRef = useRef<HandFeatureSet[]>([]);

  const settings = useAppStore((s) => s.settings);
  const setEngineStatus = useAppStore((s) => s.setEngineStatus);
  const setFps = useAppStore((s) => s.setFps);

  // Keep the smoother config in sync with user settings without resetting it on every render
  useEffect(() => {
    smootherRef.current.updateConfig({
      windowSize: settings.smoothingWindow,
      cooldownMs: settings.cooldownMs,
      confidenceThreshold: settings.confidenceThreshold,
    });
  }, [settings.smoothingWindow, settings.cooldownMs, settings.confidenceThreshold]);

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

    drawOverlay(
      detection.hands,
      detection.frameWidth || video.videoWidth,
      detection.frameHeight || video.videoHeight,
      state.settings.showLandmarks
    );
    state.setHandsDetected(detection.hands.length);

    if (state.isPaused) return;

    const features = extractFeaturesFromFrame(detection.hands);
    latestFeaturesRef.current = features;
    const classification = classifyHandFeatures(features);
    state.setCurrentClassification(classification);

    const stable = smootherRef.current.push(classification, now);
    if (stable && stable.confidence >= state.settings.confidenceThreshold) {
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
    if (rafRef.current == null) {
      rafRef.current = requestAnimationFrame(loop);
    }
  }

  function stopCamera() {
    if (rafRef.current != null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
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
  };
}
