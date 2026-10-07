import { useEffect, useRef, useState } from "react";
import { Camera, Download, Upload, Trash2, CircleDot, Database } from "lucide-react";
import { useCamera } from "../hooks/useCamera";
import { useHandLandmarker } from "../hooks/useHandLandmarker";
import { extractFeaturesFromFrame } from "../lib/geometry";
import { HAND_CONNECTIONS } from "../lib/handConnections";
import { useAppStore } from "../store/useAppStore";
import { CameraView } from "../components/CameraView";
import { Badge, Button, Card, CardHeader, EmptyState } from "../components/ui";
import type { DatasetSample } from "../types";

export default function DatasetPage() {
  const camera = useCamera();
  const landmarker = useHandLandmarker();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const rafRef = useRef<number | null>(null);
  const latestHandsRef = useRef<ReturnType<typeof extractFeaturesFromFrame>>([]);
  const [handCount, setHandCount] = useState(0);
  const [label, setLabel] = useState("");
  const [flash, setFlash] = useState(false);

  const { dataset, addSample, removeSample, clearDataset } = useAppStore();

  useEffect(() => {
    function loop() {
      rafRef.current = requestAnimationFrame(loop);
      const video = camera.videoRef.current;
      if (!video || !landmarker.isReady || video.readyState < 2) return;
      const detection = landmarker.detectVideoFrame(video, performance.now());
      if (!detection) return;
      const canvas = canvasRef.current;
      if (canvas) {
        canvas.width = detection.frameWidth || video.videoWidth;
        canvas.height = detection.frameHeight || video.videoHeight;
        const ctx = canvas.getContext("2d");
        if (ctx) {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          for (const hand of detection.hands) {
            ctx.strokeStyle = "#14b8a6";
            ctx.lineWidth = 3;
            for (const [a, b] of HAND_CONNECTIONS) {
              const p1 = hand.landmarks[a];
              const p2 = hand.landmarks[b];
              if (!p1 || !p2) continue;
              ctx.beginPath();
              ctx.moveTo(p1.x * canvas.width, p1.y * canvas.height);
              ctx.lineTo(p2.x * canvas.width, p2.y * canvas.height);
              ctx.stroke();
            }
          }
        }
      }
      setHandCount(detection.hands.length);
      latestHandsRef.current = extractFeaturesFromFrame(detection.hands);
    }
    if (camera.isActive) rafRef.current = requestAnimationFrame(loop);
    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
    };
  }, [camera.isActive, landmarker.isReady, landmarker, camera.videoRef]);

  function handleCapture() {
    if (!label.trim()) return;
    if (latestHandsRef.current.length === 0) return;
    const sample: DatasetSample = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
      label: label.trim().toUpperCase(),
      createdAt: Date.now(),
      features: latestHandsRef.current,
    };
    addSample(sample);
    setFlash(true);
    setTimeout(() => setFlash(false), 250);
  }

  function handleExport() {
    const blob = new Blob([JSON.stringify(dataset, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `isl-dataset-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function handleImport(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const parsed = JSON.parse(String(reader.result));
        if (!Array.isArray(parsed)) throw new Error("Invalid format: expected an array of samples.");
        parsed.forEach((s: DatasetSample) => {
          if (s.id && s.label && Array.isArray(s.features)) addSample(s);
        });
      } catch (err) {
        alert("Could not import dataset file: " + (err instanceof Error ? err.message : "unknown error"));
      }
    };
    reader.readAsText(file);
    e.target.value = "";
  }

  const counts = dataset.reduce<Record<string, number>>((acc, s) => {
    acc[s.label] = (acc[s.label] ?? 0) + 1;
    return acc;
  }, {});

  const total = dataset.length;
  const trainCount = Math.round(total * 0.7);
  const valCount = Math.round(total * 0.15);
  const testCount = total - trainCount - valCount;

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-6">
      <header>
        <h1 className="text-2xl font-bold text-slate-900">Dataset Collection</h1>
        <p className="mt-1 max-w-3xl text-sm text-slate-500">
          No pre-existing labelled ISL video dataset ships with this project (none is fabricated or claimed). Use this
          tool to record your own labelled hand-landmark samples via webcam — the same feature-extraction pipeline
          used here is used for live recognition, so a future trained model can consume this data directly.
        </p>
      </header>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="flex flex-col gap-4 lg:col-span-2">
          <Card className="overflow-hidden">
            <CameraView
              videoRef={camera.videoRef}
              canvasRef={canvasRef}
              isActive={camera.isActive}
              isStarting={camera.isStarting}
              mirror
              overlayTop={
                <Badge tone="slate" className="bg-slate-900/60 text-white backdrop-blur">
                  {handCount} hand{handCount === 1 ? "" : "s"} detected
                </Badge>
              }
            />
          </Card>
          <Card className="flex flex-wrap items-center gap-3 p-4">
            {!camera.isActive ? (
              <Button onClick={() => camera.start(null)} disabled={!landmarker.isReady}>
                <Camera className="h-4 w-4" /> {landmarker.isLoading ? "Loading model…" : "Start Camera"}
              </Button>
            ) : (
              <Button variant="danger" onClick={camera.stop}>
                Stop Camera
              </Button>
            )}
            <input
              value={label}
              onChange={(e) => setLabel(e.target.value)}
              placeholder="Sign label, e.g. HELLO"
              className="min-w-[180px] flex-1 rounded-lg border border-slate-300 px-3 py-2 text-sm outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500"
            />
            <Button
              onClick={handleCapture}
              disabled={!camera.isActive || !label.trim() || handCount === 0}
              className={flash ? "ring-2 ring-teal-300" : ""}
            >
              <CircleDot className="h-4 w-4" /> Capture Sample
            </Button>
          </Card>
          {camera.error && (
            <p className="text-xs font-medium text-rose-600">{camera.error}</p>
          )}
        </div>

        <div className="flex flex-col gap-4">
          <Card>
            <CardHeader title="Dataset Summary" icon={<Database className="h-4 w-4" />} />
            <div className="space-y-3 p-4">
              <div className="flex justify-between text-sm">
                <span className="text-slate-500">Total samples</span>
                <span className="font-semibold">{total}</span>
              </div>
              <div className="grid grid-cols-3 gap-2 text-center text-xs">
                <div className="rounded-lg bg-slate-50 p-2">
                  <p className="font-semibold text-slate-700">{trainCount}</p>
                  <p className="text-slate-400">train (70%)</p>
                </div>
                <div className="rounded-lg bg-slate-50 p-2">
                  <p className="font-semibold text-slate-700">{valCount}</p>
                  <p className="text-slate-400">val (15%)</p>
                </div>
                <div className="rounded-lg bg-slate-50 p-2">
                  <p className="font-semibold text-slate-700">{testCount}</p>
                  <p className="text-slate-400">test (15%)</p>
                </div>
              </div>
              <p className="text-[11px] text-slate-400">
                Split shown is a suggested, reproducible 70/15/15 partition by sample order to prevent leakage during
                a future offline training run — no training happens in-browser.
              </p>
              <div className="flex gap-2 pt-1">
                <Button size="sm" variant="outline" className="flex-1" onClick={handleExport} disabled={total === 0}>
                  <Download className="h-3.5 w-3.5" /> Export JSON
                </Button>
                <label className="flex flex-1 cursor-pointer items-center justify-center gap-1.5 rounded-xl border border-slate-300 bg-white px-4 py-2 text-xs font-medium text-slate-700 hover:bg-slate-50">
                  <Upload className="h-3.5 w-3.5" /> Import
                  <input type="file" accept="application/json" className="hidden" onChange={handleImport} />
                </label>
              </div>
              <Button size="sm" variant="ghost" className="w-full text-rose-600" onClick={clearDataset} disabled={total === 0}>
                <Trash2 className="h-3.5 w-3.5" /> Clear all samples
              </Button>
            </div>
          </Card>

          <Card>
            <CardHeader title="Labels Recorded" />
            <div className="max-h-80 space-y-2 overflow-y-auto p-4">
              {Object.keys(counts).length === 0 ? (
                <EmptyState title="No samples yet" description="Start the camera, type a label, and capture a few samples." />
              ) : (
                Object.entries(counts).map(([lab, count]) => (
                  <div key={lab} className="flex items-center justify-between rounded-lg bg-slate-50 px-3 py-2 text-sm">
                    <span className="font-medium text-slate-700">{lab}</span>
                    <div className="flex items-center gap-2">
                      <Badge tone="teal">{count}</Badge>
                      <button
                        className="text-slate-300 hover:text-rose-500"
                        onClick={() => dataset.filter((d) => d.label === lab).forEach((d) => removeSample(d.id))}
                        title="Remove all samples for this label"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
