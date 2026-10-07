// ---------------------------------------------------------------------------
// MODULE: MediaPipe hand-landmark detection
// Loads the official MediaPipe HandLandmarker (WASM, runs fully on-device /
// locally in the browser — no frames are ever uploaded anywhere) and exposes
// a simple detect() function plus loading/error state.
// ---------------------------------------------------------------------------
import { useEffect, useRef, useState } from "react";
import { FilesetResolver, HandLandmarker, type HandLandmarkerResult } from "@mediapipe/tasks-vision";
import type { DetectedHand, HandDetectionResult, Handedness } from "../types";

const WASM_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

export function useHandLandmarker() {
  const landmarkerRef = useRef<HandLandmarker | null>(null);
  const [isReady, setIsReady] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setIsLoading(true);
      setError(null);
      try {
        const vision = await FilesetResolver.forVisionTasks(WASM_BASE);
        const landmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
          runningMode: "VIDEO",
          numHands: 2,
          minHandDetectionConfidence: 0.55,
          minHandPresenceConfidence: 0.55,
          minTrackingConfidence: 0.55,
        });
        if (cancelled) {
          landmarker.close();
          return;
        }
        landmarkerRef.current = landmarker;
        setIsReady(true);
      } catch (err) {
        if (!cancelled) {
          setError(
            "Failed to load the hand-detection model. Check your internet connection (the model loads from a CDN on first use) and try again."
          );
          console.error(err);
        }
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    }
    load();
    return () => {
      cancelled = true;
      landmarkerRef.current?.close();
      landmarkerRef.current = null;
    };
  }, []);

  function detectVideoFrame(video: HTMLVideoElement, timestampMs: number): HandDetectionResult | null {
    const landmarker = landmarkerRef.current;
    if (!landmarker || video.readyState < 2) return null;

    let result: HandLandmarkerResult;
    try {
      result = landmarker.detectForVideo(video, timestampMs);
    } catch {
      return null;
    }

    const hands: DetectedHand[] = (result.landmarks ?? []).map((lm, i) => {
      const handednessInfo = result.handednesses?.[i]?.[0];
      return {
        landmarks: lm.map((p) => ({ x: p.x, y: p.y, z: p.z })),
        handedness: (handednessInfo?.categoryName as Handedness) ?? "Right",
        score: handednessInfo?.score ?? 0,
      };
    });

    return {
      hands,
      timestampMs,
      frameWidth: video.videoWidth,
      frameHeight: video.videoHeight,
    };
  }

  return { isReady, isLoading, error, detectVideoFrame };
}
