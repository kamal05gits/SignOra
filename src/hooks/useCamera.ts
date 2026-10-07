// ---------------------------------------------------------------------------
// MODULE: Live webcam capture
// Wraps getUserMedia with device enumeration, start/stop lifecycle and
// correct resource release. Designed to never block the UI thread.
// ---------------------------------------------------------------------------
import { useCallback, useEffect, useRef, useState } from "react";

export interface CameraDeviceInfo {
  deviceId: string;
  label: string;
}

export function useCamera() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [devices, setDevices] = useState<CameraDeviceInfo[]>([]);
  const [isActive, setIsActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isStarting, setIsStarting] = useState(false);

  const refreshDevices = useCallback(async () => {
    try {
      if (!navigator.mediaDevices?.enumerateDevices) return;
      const list = await navigator.mediaDevices.enumerateDevices();
      const cams = list
        .filter((d) => d.kind === "videoinput")
        .map((d, i) => ({ deviceId: d.deviceId, label: d.label || `Camera ${i + 1}` }));
      setDevices(cams);
    } catch {
      // Device enumeration failing is non-fatal; selection UI just stays empty.
    }
  }, []);

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setIsActive(false);
  }, []);

  const start = useCallback(
    async (deviceId?: string | null) => {
      setError(null);
      setIsStarting(true);
      try {
        if (!navigator.mediaDevices?.getUserMedia) {
          throw new Error("This browser does not support camera access (getUserMedia unavailable).");
        }
        // Release any previous stream first
        streamRef.current?.getTracks().forEach((t) => t.stop());

        const constraints: MediaStreamConstraints = {
          audio: false,
          video: deviceId
            ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
            : { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
        };

        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play().catch(() => undefined);
        }
        setIsActive(true);
        await refreshDevices();
      } catch (err) {
        let message = "Unable to access the webcam.";
        if (err instanceof DOMException) {
          if (err.name === "NotAllowedError") message = "Camera permission was denied. Please allow camera access in your browser.";
          else if (err.name === "NotFoundError") message = "No camera device was found on this system.";
          else if (err.name === "NotReadableError") message = "The camera is already in use by another application.";
          else message = `Camera error: ${err.name}`;
        } else if (err instanceof Error) {
          message = err.message;
        }
        setError(message);
        setIsActive(false);
      } finally {
        setIsStarting(false);
      }
    },
    [refreshDevices]
  );

  useEffect(() => {
    refreshDevices();
    navigator.mediaDevices?.addEventListener?.("devicechange", refreshDevices);
    return () => {
      navigator.mediaDevices?.removeEventListener?.("devicechange", refreshDevices);
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, [refreshDevices]);

  return { videoRef, devices, isActive, isStarting, error, start, stop, refreshDevices };
}
