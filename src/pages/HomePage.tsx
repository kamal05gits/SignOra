import { useEffect, useMemo, useState } from "react";
import {
  Play,
  Square,
  Pause,
  Volume2,
  VolumeX,
  RotateCcw,
  Eraser,
  Space,
  Delete,
  Languages,
  Siren,
  Activity,
  Hand,
  Cpu,
} from "lucide-react";
import { useRecognitionEngine } from "../hooks/useRecognitionEngine";
import { useAppStore } from "../store/useAppStore";
import { CameraView } from "../components/CameraView";
import { Badge, Button, Card, CardHeader, Select, StatusDot } from "../components/ui";
import { SUPPORTED_LANGUAGES, translateText } from "../lib/translator";
import { getAvailableVoices, isSpeaking, speak, stopSpeech } from "../lib/speech";
import { traceTokens } from "../lib/sentenceProcessor";
import { Link } from "react-router-dom";

export default function HomePage() {
  const engine = useRecognitionEngine();
  const {
    engineStatus,
    statusMessage,
    fps,
    handsDetected,
    currentClassification,
    isPaused,
    togglePaused,
    tokens,
    sentence,
    removeLastToken,
    clearTokens,
    insertSpaceMarker,
    settings,
    translation,
    setTranslation,
    pushHistory,
  } = useAppStore();

  const [speaking, setSpeaking] = useState(false);
  const sentenceText = sentence();

  useEffect(() => {
    const id = setInterval(() => setSpeaking(isSpeaking()), 400);
    return () => clearInterval(id);
  }, []);

  // Auto-speak newly recognized tokens when enabled in Settings
  const lastTokenId = tokens.length ? tokens[tokens.length - 1].id : null;
  useEffect(() => {
    if (!settings.autoSpeak || !lastTokenId) return;
    const last = tokens[tokens.length - 1];
    if (last && last.gloss !== "—") {
      speak(last.gloss, { voiceURI: settings.voiceURI, rate: settings.speechRate, onError: () => undefined });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lastTokenId]);

  const confidencePct = Math.round((currentClassification?.confidence ?? 0) * 100);
  // The backend's trained models and the on-device classifier are calibrated
  // differently, so each has its own acceptance threshold.
  const fromBackend = currentClassification?.source === "backend";
  const activeThreshold = fromBackend ? settings.backendConfidenceThreshold : settings.confidenceThreshold;
  const confidenceOk = (currentClassification?.confidence ?? 0) >= activeThreshold;

  const voices = useMemo(() => getAvailableVoices(), []);

  async function handleTranslate() {
    if (!sentenceText.trim()) {
      setTranslation({ status: "error", sourceText: "", translatedText: "", error: "Nothing recognized yet to translate." });
      return;
    }
    setTranslation({ status: "loading", sourceText: sentenceText, translatedText: "", error: null });
    const result = await translateText(sentenceText, settings.outputLanguage);
    if (result.success) {
      setTranslation({ status: "success", sourceText: sentenceText, translatedText: result.text ?? "", error: null });
    } else {
      setTranslation({ status: "error", sourceText: sentenceText, translatedText: "", error: result.error ?? "Translation failed." });
    }
  }

  function handleSpeak(text: string, lang: string) {
    speak(text, {
      voiceURI: settings.voiceURI,
      lang,
      rate: settings.speechRate,
      onError: (m) => console.warn(m),
    });
  }

  function handleClear() {
    if (sentenceText.trim()) pushHistory(sentenceText);
    clearTokens();
    setTranslation({ status: "idle", sourceText: "", translatedText: "", error: null });
  }

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-6">
      <header className="flex flex-col gap-1">
        <h1 className="text-2xl font-bold tracking-tight text-slate-900">Live Translator</h1>
        <p className="text-sm text-slate-500">
          Capture hand gestures from your webcam and convert them into text and speech in real time.
        </p>
      </header>

      {engineStatus === "error" && (
        <div className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
          <strong className="font-semibold">Something needs attention: </strong>
          {statusMessage}
        </div>
      )}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Camera + recognition column */}
        <div className="flex flex-col gap-4 lg:col-span-2">
          <Card className="overflow-hidden">
            <CameraView
              videoRef={engine.camera.videoRef}
              canvasRef={engine.canvasRef}
              isActive={engine.camera.isActive}
              isStarting={engine.camera.isStarting}
              mirror={settings.mirrorPreview}
              overlayTop={
                <div className="flex items-center justify-between">
                  <StatusDot status={isPaused && engine.camera.isActive ? "paused" : engineStatus} />
                  <div className="flex gap-2">
                    <Badge tone="slate" className="bg-slate-900/60 text-white backdrop-blur">
                      {fps} FPS
                    </Badge>
                    <Badge tone="slate" className="bg-slate-900/60 text-white backdrop-blur">
                      <Hand className="h-3 w-3" /> {handsDetected} hand{handsDetected === 1 ? "" : "s"}
                    </Badge>
                    <Badge
                      tone={currentClassification?.source === "backend" ? "teal" : "slate"}
                      className="bg-slate-900/60 text-white backdrop-blur"
                    >
                      <Cpu className="h-3 w-3" />{" "}
                      {currentClassification?.source === "backend"
                        ? currentClassification.modelId ?? "backend"
                        : "on-device"}
                    </Badge>
                  </div>
                </div>
              }
              overlayBottom={
                engine.camera.isActive && currentClassification?.gloss ? (
                  <div
                    className={`flex items-center justify-between rounded-xl px-3 py-2 backdrop-blur ${
                      confidenceOk ? "bg-emerald-600/80" : "bg-amber-500/80"
                    }`}
                  >
                    <span className="text-sm font-semibold text-white">
                      {confidenceOk ? currentClassification.gloss : "Uncertain sign…"}
                    </span>
                    <span className="text-xs font-medium text-white/90">{confidencePct}% confidence</span>
                  </div>
                ) : null
              }
            />
          </Card>

          <Card className="p-4">
            <div className="flex flex-wrap items-center gap-2">
              {!engine.camera.isActive ? (
                <Button onClick={engine.startCamera} disabled={!engine.landmarker.isReady}>
                  <Play className="h-4 w-4" />
                  {engine.landmarker.isLoading ? "Loading model…" : "Start Camera"}
                </Button>
              ) : (
                <Button variant="danger" onClick={engine.stopCamera}>
                  <Square className="h-4 w-4" /> Stop Camera
                </Button>
              )}
              <Button
                variant="outline"
                disabled={!engine.camera.isActive}
                onClick={togglePaused}
              >
                <Pause className="h-4 w-4" /> {isPaused ? "Resume Recognition" : "Pause Recognition"}
              </Button>

              <div className="ml-auto flex items-center gap-2">
                <label className="text-xs font-medium text-slate-500">Camera:</label>
                <Select
                  className="w-44"
                  value={settings.cameraDeviceId ?? ""}
                  onChange={(e) => useAppStore.getState().updateSettings({ cameraDeviceId: e.target.value || null })}
                >
                  <option value="">Default camera</option>
                  {engine.camera.devices.map((d) => (
                    <option key={d.deviceId} value={d.deviceId}>
                      {d.label}
                    </option>
                  ))}
                </Select>
              </div>
            </div>
            {engine.camera.error && (
              <p className="mt-2 text-xs font-medium text-rose-600">{engine.camera.error}</p>
            )}
            <p className="mt-3 text-xs text-slate-400">{statusMessage}</p>
          </Card>

          <Card className="p-4">
            <CardHeader
              title="Recognized Sentence"
              subtitle="Stable signs are appended automatically. Edit manually using the controls below."
              icon={<Activity className="h-4 w-4" />}
            />
            <div className="p-4">
              <div className="min-h-[72px] rounded-xl border border-slate-200 bg-slate-50 p-4 text-lg font-medium text-slate-800">
                {sentenceText || <span className="text-slate-400">Recognized words will appear here…</span>}
              </div>
              {tokens.length > 0 && (
                <p className="mt-2 truncate text-xs text-slate-400" title={traceTokens(tokens)}>
                  Token trace: {traceTokens(tokens)}
                </p>
              )}
              <div className="mt-3 flex flex-wrap gap-2">
                <Button size="sm" variant="outline" onClick={insertSpaceMarker}>
                  <Space className="h-3.5 w-3.5" /> Space
                </Button>
                <Button size="sm" variant="outline" onClick={removeLastToken} disabled={tokens.length === 0}>
                  <Delete className="h-3.5 w-3.5" /> Remove last
                </Button>
                <Button size="sm" variant="outline" onClick={handleClear} disabled={tokens.length === 0}>
                  <Eraser className="h-3.5 w-3.5" /> Clear
                </Button>
                <Link to="/emergency">
                  <Button size="sm" variant="danger">
                    <Siren className="h-3.5 w-3.5" /> Emergency SOS
                  </Button>
                </Link>
              </div>
            </div>
          </Card>
        </div>

        {/* Side panel: language + speech + status */}
        <div className="flex flex-col gap-4">
          <Card>
            <CardHeader title="Current Sign" subtitle="Confidence-aware prediction" icon={<Hand className="h-4 w-4" />} />
            <div className="space-y-3 p-4">
              <div className="flex items-center justify-between">
                <span className="text-sm text-slate-500">Gloss</span>
                <span className="text-base font-semibold text-slate-900">
                  {currentClassification?.gloss ?? "—"}
                </span>
              </div>
              <div>
                <div className="mb-1 flex items-center justify-between text-xs text-slate-500">
                  <span>Confidence</span>
                  <span>{confidencePct}%</span>
                </div>
                <div className="h-2 w-full overflow-hidden rounded-full bg-slate-100">
                  <div
                    className={`h-full rounded-full ${confidenceOk ? "bg-emerald-500" : "bg-amber-400"}`}
                    style={{ width: `${confidencePct}%` }}
                  />
                </div>
                <p className="mt-1 text-[11px] text-slate-400">
                  Threshold: {Math.round(activeThreshold * 100)}% (
                  {fromBackend ? "backend model" : "on-device classifier"}) — predictions below this are shown as
                  "uncertain" and never added to the sentence.
                </p>
              </div>
            </div>
          </Card>

          <Card>
            <CardHeader title="Language & Translation" icon={<Languages className="h-4 w-4" />} />
            <div className="space-y-3 p-4">
              <Select
                value={settings.outputLanguage}
                onChange={(e) => useAppStore.getState().updateSettings({ outputLanguage: e.target.value })}
              >
                {SUPPORTED_LANGUAGES.map((l) => (
                  <option key={l.code} value={l.code}>
                    {l.label}
                  </option>
                ))}
              </Select>
              <Button className="w-full" variant="outline" onClick={handleTranslate}>
                Translate Sentence
              </Button>
              <div className="rounded-xl border border-slate-200 bg-slate-50 p-3 text-sm min-h-[56px]">
                {translation.status === "loading" && <span className="text-slate-400">Translating…</span>}
                {translation.status === "error" && <span className="text-rose-600">{translation.error}</span>}
                {translation.status === "success" && <span className="text-slate-800">{translation.translatedText}</span>}
                {translation.status === "idle" && <span className="text-slate-400">Translated text will appear here.</span>}
              </div>
            </div>
          </Card>

          <Card>
            <CardHeader title="Text-to-Speech" icon={<Volume2 className="h-4 w-4" />} />
            <div className="space-y-2 p-4">
              <Button
                className="w-full"
                onClick={() => handleSpeak(sentenceText, "en-US")}
                disabled={!sentenceText.trim()}
              >
                <Volume2 className="h-4 w-4" /> Speak (English)
              </Button>
              <Button
                className="w-full"
                variant="secondary"
                onClick={() => handleSpeak(translation.translatedText || sentenceText, settings.outputLanguage)}
                disabled={!translation.translatedText}
              >
                <RotateCcw className="h-4 w-4" /> Speak Translation
              </Button>
              <Button className="w-full" variant="outline" onClick={stopSpeech} disabled={!speaking}>
                <VolumeX className="h-4 w-4" /> Stop Speaking
              </Button>
              {voices.length === 0 && (
                <p className="text-[11px] text-amber-600">
                  No system voices detected yet — they usually load a moment after the page opens.
                </p>
              )}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
