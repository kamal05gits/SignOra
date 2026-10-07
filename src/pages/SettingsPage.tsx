import { useEffect, useState } from "react";
import { Settings as SettingsIcon, Mic, Camera, ShieldCheck, History, Trash2, ServerCog, PlugZap } from "lucide-react";
import { useAppStore } from "../store/useAppStore";
import { useCamera } from "../hooks/useCamera";
import { useBackendModels } from "../hooks/useBackendModels";
import { BackendError, fetchHealth, resolveBaseUrl } from "../lib/backend";
import { Button, Card, CardHeader, EmptyState, Select, Slider, Toggle } from "../components/ui";
import { getAvailableVoices, isSpeechSupported } from "../lib/speech";
import type { AppSettings } from "../types";

export default function SettingsPage() {
  const { settings, updateSettings, history, clearHistory } = useAppStore();
  const camera = useCamera();
  const { models } = useBackendModels();
  const [voices, setVoices] = useState(getAvailableVoices());
  const [testing, setTesting] = useState(false);
  const [connectionResult, setConnectionResult] = useState<{ ok: boolean; message: string } | null>(null);

  async function testConnection() {
    setTesting(true);
    setConnectionResult(null);
    const base = resolveBaseUrl(settings.backendUrl);
    try {
      const health = await fetchHealth(base);
      setConnectionResult({
        ok: true,
        message: `Connected to v${health.version} · ${health.models_total} model(s) found · runtimes: ${Object.entries(
          health.runtimes
        )
          .map(([name, version]) => `${name} ${version ?? "missing"}`)
          .join(", ")}`,
      });
    } catch (error) {
      setConnectionResult({
        ok: false,
        message: error instanceof BackendError ? error.message : String(error),
      });
    } finally {
      setTesting(false);
    }
  }

  useEffect(() => {
    function refresh() {
      setVoices(getAvailableVoices());
    }
    refresh();
    if (isSpeechSupported()) {
      window.speechSynthesis.onvoiceschanged = refresh;
    }
  }, []);

  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-6">
      <header>
        <h1 className="text-2xl font-bold text-slate-900">Settings</h1>
        <p className="mt-1 text-sm text-slate-500">Configure recognition behaviour, camera, speech and data.</p>
      </header>

      <Card>
        <CardHeader
          title="Inference Engine"
          subtitle="Where signs are recognised: in this browser, or by the trained models in the Python backend"
          icon={<ServerCog className="h-4 w-4" />}
          action={
            <Button size="sm" variant="outline" onClick={testConnection} disabled={testing}>
              <PlugZap className="h-3.5 w-3.5" /> {testing ? "Testing…" : "Test connection"}
            </Button>
          }
        />
        <div className="space-y-4 p-4">
          <div>
            <label className="mb-1.5 block text-xs font-medium text-slate-500">Recognition engine</label>
            <Select
              value={settings.inferenceMode}
              onChange={(e) => updateSettings({ inferenceMode: e.target.value as AppSettings["inferenceMode"] })}
            >
              <option value="auto">Auto — backend when reachable, else on-device</option>
              <option value="backend">Backend only — trained models via the API</option>
              <option value="on-device">On-device only — geometric classifier, no server</option>
            </Select>
          </div>

          <div>
            <label className="mb-1.5 block text-xs font-medium text-slate-500">Backend URL</label>
            <input
              value={settings.backendUrl}
              onChange={(e) => updateSettings({ backendUrl: e.target.value })}
              placeholder="(empty = same origin, via the dev proxy)"
              className="w-full rounded-lg border border-slate-300 px-3 py-2 font-mono text-sm outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500"
            />
            <p className="mt-1 text-[11px] text-slate-400">
              Leave empty in development: vite proxies <code>/api</code> and <code>/ws</code> to{" "}
              <code>http://127.0.0.1:8000</code>. Set it only when the API lives on another host.
            </p>
          </div>

          <div>
            <label className="mb-1.5 block text-xs font-medium text-slate-500">Model used for live recognition</label>
            <Select
              value={settings.activeModelId ?? ""}
              onChange={(e) => updateSettings({ activeModelId: e.target.value || null })}
              disabled={models.length === 0}
            >
              <option value="">{models.length ? "Server default" : "Backend offline — no models listed"}</option>
              {models.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.id} — {model.framework}/{model.modality}
                  {model.lfs_pointer ? " (weights missing)" : ""}
                </option>
              ))}
            </Select>
          </div>

          <Slider
            label="Backend confidence threshold"
            min={0.05}
            max={0.95}
            step={0.01}
            value={settings.backendConfidenceThreshold}
            valueLabel={`${Math.round(settings.backendConfidenceThreshold * 100)}%`}
            onChange={(v) => updateSettings({ backendConfidenceThreshold: v })}
          />

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div>
              <label className="mb-1.5 block text-xs font-medium text-slate-500">Send every Nth frame</label>
              <Select
                value={String(settings.streamStride)}
                onChange={(e) => updateSettings({ streamStride: Number(e.target.value) })}
              >
                {[1, 2, 3, 4, 5].map((n) => (
                  <option key={n} value={n}>
                    {n === 1 ? "Every frame" : `Every ${n} frames`}
                  </option>
                ))}
              </Select>
            </div>
            <div>
              <label className="mb-1.5 block text-xs font-medium text-slate-500">Which hand the model sees</label>
              <Select
                value={settings.backendHand}
                onChange={(e) => updateSettings({ backendHand: e.target.value as AppSettings["backendHand"] })}
              >
                <option value="first">First detected</option>
                <option value="dominant">Highest confidence</option>
                <option value="left">Left hand</option>
                <option value="right">Right hand</option>
              </Select>
            </div>
          </div>

          <Slider
            label="Temporal window sent to the model (frames)"
            min={5}
            max={120}
            step={1}
            value={settings.streamSequenceLength}
            onChange={(v) => updateSettings({ streamSequenceLength: v })}
          />

          <Toggle
            label="Fall back to the on-device classifier when the backend is down"
            checked={settings.fallbackToOnDevice}
            onChange={(v) => updateSettings({ fallbackToOnDevice: v })}
          />
          <Toggle
            label="Sync captured dataset samples to the backend"
            checked={settings.syncDatasetToBackend}
            onChange={(v) => updateSettings({ syncDatasetToBackend: v })}
          />

          {connectionResult && (
            <div
              className={`rounded-xl px-4 py-3 text-sm ${
                connectionResult.ok ? "bg-emerald-50 text-emerald-800" : "bg-rose-50 text-rose-700"
              }`}
            >
              {connectionResult.message}
            </div>
          )}
        </div>
      </Card>

      <Card>
        <CardHeader title="Recognition" icon={<SettingsIcon className="h-4 w-4" />} />
        <div className="space-y-4 p-4">
          <Slider
            label="Confidence threshold"
            min={0.3}
            max={0.95}
            step={0.01}
            value={settings.confidenceThreshold}
            valueLabel={`${Math.round(settings.confidenceThreshold * 100)}%`}
            onChange={(v) => updateSettings({ confidenceThreshold: v })}
          />
          <Slider
            label="Smoothing window (frames)"
            min={4}
            max={20}
            step={1}
            value={settings.smoothingWindow}
            onChange={(v) => updateSettings({ smoothingWindow: v })}
          />
          <Slider
            label="Cooldown between tokens (ms)"
            min={400}
            max={3000}
            step={100}
            value={settings.cooldownMs}
            onChange={(v) => updateSettings({ cooldownMs: v })}
          />
          <Toggle
            label="Show hand landmark overlay"
            checked={settings.showLandmarks}
            onChange={(v) => updateSettings({ showLandmarks: v })}
          />
          <Toggle
            label="Mirror camera preview"
            checked={settings.mirrorPreview}
            onChange={(v) => updateSettings({ mirrorPreview: v })}
          />
        </div>
      </Card>

      <Card>
        <CardHeader title="Camera" icon={<Camera className="h-4 w-4" />} />
        <div className="space-y-3 p-4">
          <label className="text-xs font-medium text-slate-500">Preferred camera device</label>
          <Select
            value={settings.cameraDeviceId ?? ""}
            onChange={(e) => updateSettings({ cameraDeviceId: e.target.value || null })}
          >
            <option value="">Default camera</option>
            {camera.devices.map((d) => (
              <option key={d.deviceId} value={d.deviceId}>
                {d.label}
              </option>
            ))}
          </Select>
          <p className="text-[11px] text-slate-400">
            Camera list populates fully once permission has been granted at least once on the Live Translator page.
          </p>
        </div>
      </Card>

      <Card>
        <CardHeader title="Speech & Output" icon={<Mic className="h-4 w-4" />} />
        <div className="space-y-4 p-4">
          <div>
            <label className="mb-1.5 block text-xs font-medium text-slate-500">Preferred voice</label>
            <Select
              value={settings.voiceURI ?? ""}
              onChange={(e) => updateSettings({ voiceURI: e.target.value || null })}
            >
              <option value="">System default</option>
              {voices.map((v) => (
                <option key={v.voiceURI} value={v.voiceURI}>
                  {v.name} ({v.lang})
                </option>
              ))}
            </Select>
            {!isSpeechSupported() && (
              <p className="mt-1 text-xs text-rose-600">Speech synthesis is not supported in this browser.</p>
            )}
          </div>
          <Slider
            label="Speech rate"
            min={0.5}
            max={1.8}
            step={0.1}
            value={settings.speechRate}
            onChange={(v) => updateSettings({ speechRate: v })}
          />
          <Toggle
            label="Auto-speak each recognized word"
            checked={settings.autoSpeak}
            onChange={(v) => updateSettings({ autoSpeak: v })}
          />
        </div>
      </Card>

      <Card>
        <CardHeader
          title="Sentence History"
          subtitle="Stored locally in your browser only"
          icon={<History className="h-4 w-4" />}
          action={
            <Button size="sm" variant="ghost" className="text-rose-600" onClick={clearHistory} disabled={history.length === 0}>
              <Trash2 className="h-3.5 w-3.5" /> Clear
            </Button>
          }
        />
        <div className="max-h-56 space-y-1 overflow-y-auto p-4">
          {history.length === 0 ? (
            <EmptyState title="No sentences saved yet" description="Cleared sentences from the Live Translator are kept here for reference." />
          ) : (
            history.map((h, i) => (
              <div key={i} className="rounded-lg bg-slate-50 px-3 py-2 text-sm text-slate-600">
                {h}
              </div>
            ))
          )}
        </div>
      </Card>

      <Card className="border-emerald-200 bg-emerald-50/60 p-4 text-sm text-emerald-800">
        <div className="flex gap-3">
          <ShieldCheck className="h-5 w-5 shrink-0 text-emerald-600" />
          <p>
            <strong>Privacy:</strong> Hand-landmark detection always runs locally in your browser with MediaPipe, and
            no video is ever recorded or uploaded. In <em>on-device</em> mode nothing at all leaves the machine. In{" "}
            <em>backend</em> mode the 21×3 landmark coordinates of each processed frame are sent over a WebSocket to
            the inference API you configured (your own server by default) — still no pixels, and dataset samples synced
            there contain landmark numbers plus your label. Settings, contacts and history stay in this browser's local
            storage. Translation requests send only the recognized text to the MyMemory API; the emergency lookup sends
            only a country code.
          </p>
        </div>
      </Card>
    </div>
  );
}
