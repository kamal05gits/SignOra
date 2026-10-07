import { useCallback, useMemo, useRef, useState } from "react";
import {
  Cpu,
  PlayCircle,
  AlertTriangle,
  BookOpen,
  Server,
  RefreshCw,
  Power,
  PowerOff,
  Target,
  Image as ImageIcon,
  Upload,
  CheckCircle2,
  Boxes,
} from "lucide-react";
import { SIGN_VOCABULARY } from "../lib/signVocabulary";
import { evaluateDataset, type EvaluationReport } from "../lib/evaluate";
import { useAppStore } from "../store/useAppStore";
import { useBackendModels } from "../hooks/useBackendModels";
import { BackendError, datasetExportHref, evaluateDataset as backendEvaluate, fileToBase64, formatBytes, predictImage } from "../lib/backend";
import { Badge, Button, Card, CardHeader, EmptyState, Select, Slider } from "../components/ui";
import type { BackendEvaluationReport, BackendModelInfo, BackendModelStatus, BackendPrediction } from "../types";

const STATUS_TONE: Record<BackendModelStatus, "emerald" | "slate" | "amber" | "rose"> = {
  ready: "emerald",
  idle: "slate",
  unloaded: "slate",
  missing_weights: "amber",
  needs_manifest: "amber",
  load_error: "rose",
  runtime_missing: "rose",
};

const STATUS_LABEL: Record<BackendModelStatus, string> = {
  ready: "Loaded",
  idle: "Available",
  unloaded: "Unloaded",
  missing_weights: "Weights missing (Git LFS)",
  needs_manifest: "Needs manifest entry",
  load_error: "Load failed",
  runtime_missing: "Runtime not installed",
};

export default function ModelPage() {
  const { dataset, settings, updateSettings } = useAppStore();
  const [report, setReport] = useState<EvaluationReport | null>(null);
  const canEvaluate = dataset.length > 0;

  function runEvaluation() {
    setReport(evaluateDataset(dataset, settings.confidenceThreshold));
  }

  const vocabByCategory = useMemo(() => {
    const groups: Record<string, typeof SIGN_VOCABULARY> = {};
    for (const s of SIGN_VOCABULARY) {
      groups[s.category] = groups[s.category] ?? [];
      groups[s.category].push(s);
    }
    return groups;
  }, []);

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-6">
      <header>
        <h1 className="text-2xl font-bold text-slate-900">Recognition Engine</h1>
        <p className="mt-1 max-w-3xl text-sm text-slate-500">
          Two engines are wired into the live pipeline: the trained models served by the Python backend in{" "}
          <code className="rounded bg-slate-100 px-1">models/</code>, and the on-device geometric classifier that runs
          with no server at all. Everything below reports live state from the API — nothing is a placeholder number.
        </p>
      </header>

      <BackendPanel />

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader
            title="On-device Demo Vocabulary"
            subtitle={`${SIGN_VOCABULARY.length} static signs used when the backend is not in use`}
            icon={<BookOpen className="h-4 w-4" />}
          />
          <div className="space-y-4 p-4">
            {Object.entries(vocabByCategory).map(([cat, items]) => (
              <div key={cat}>
                <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">{cat}</p>
                <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                  {items.map((s) => (
                    <div key={s.id} className="rounded-lg border border-slate-200 p-3">
                      <div className="flex items-center justify-between">
                        <span className="font-semibold text-slate-800">{s.gloss}</span>
                        <Badge tone="teal">{s.label}</Badge>
                      </div>
                      <p className="mt-1 text-xs text-slate-500">{s.description}</p>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </Card>

        <Card>
          <CardHeader title="Recognition Parameters" icon={<Cpu className="h-4 w-4" />} />
          <div className="space-y-4 p-4">
            <Slider
              label="Confidence threshold (on-device)"
              min={0.3}
              max={0.95}
              step={0.01}
              value={settings.confidenceThreshold}
              valueLabel={`${Math.round(settings.confidenceThreshold * 100)}%`}
              onChange={(v) => updateSettings({ confidenceThreshold: v })}
            />
            <Slider
              label="Confidence threshold (backend)"
              min={0.05}
              max={0.95}
              step={0.01}
              value={settings.backendConfidenceThreshold}
              valueLabel={`${Math.round(settings.backendConfidenceThreshold * 100)}%`}
              onChange={(v) => updateSettings({ backendConfidenceThreshold: v })}
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
            <p className="text-[11px] text-slate-400">
              A softmax over many classes rarely exceeds 0.72, so the backend engine has its own threshold. Both are
              live on the Live Translator page and persisted locally.
            </p>
          </div>
        </Card>
      </div>

      <Card>
        <CardHeader
          title="On-device Evaluation"
          subtitle="Runs the geometric classifier against your recorded samples and reports measured metrics"
          icon={<PlayCircle className="h-4 w-4" />}
          action={
            <Button size="sm" onClick={runEvaluation} disabled={!canEvaluate}>
              Run evaluation
            </Button>
          }
        />
        <div className="p-4">
          {!canEvaluate && (
            <EmptyState
              title="No dataset samples recorded yet"
              description="Go to Dataset Collection and record a few labelled samples first, then come back and run the evaluation."
            />
          )}
          {canEvaluate && !report && (
            <p className="text-sm text-slate-400">
              Press "Run evaluation" to compute accuracy on your {dataset.length} recorded samples.
            </p>
          )}
          {report && <EvaluationTables report={report} />}
        </div>
      </Card>

      <BackendEvaluationCard />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Backend connection + model registry
// ---------------------------------------------------------------------------
function BackendPanel() {
  const { baseUrl, status, models, health, error, refresh, load, unload } = useBackendModels();
  const settings = useAppStore((s) => s.settings);
  const updateSettings = useAppStore((s) => s.updateSettings);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  async function handleRefresh() {
    setRefreshing(true);
    await refresh();
    setRefreshing(false);
  }

  async function toggleModel(model: BackendModelInfo) {
    setBusyId(model.id);
    if (model.loaded) await unload(model.id);
    else await load(model.id);
    setBusyId(null);
  }

  return (
    <div className="flex flex-col gap-4">
      <Card>
        <CardHeader
          title="Inference Backend"
          subtitle={
            status === "online"
              ? `${baseUrl || window.location.origin} · v${health?.version ?? "?"} · ${health?.models_ready ?? 0}/${
                  health?.models_total ?? 0
                } models servable`
              : status === "offline"
                ? `${baseUrl || window.location.origin} is not reachable`
                : "Not checked yet (inference mode is on-device)"
          }
          icon={<Server className="h-4 w-4" />}
          action={
            <div className="flex items-center gap-2">
              <Badge tone={status === "online" ? "emerald" : status === "offline" ? "rose" : "slate"}>
                {status === "online" ? "Online" : status === "offline" ? "Offline" : "Idle"}
              </Badge>
              <Button size="sm" variant="outline" onClick={handleRefresh} disabled={refreshing}>
                <RefreshCw className={`h-3.5 w-3.5 ${refreshing ? "animate-spin" : ""}`} /> Refresh
              </Button>
            </div>
          }
        />
        <div className="space-y-3 p-4">
          {health && (
            <div className="flex flex-wrap gap-2">
              {Object.entries(health.runtimes).map(([name, version]) => (
                <Badge key={name} tone={version ? "teal" : "rose"}>
                  {name}: {version ?? "not installed"}
                </Badge>
              ))}
            </div>
          )}
          {status === "offline" && (
            <div className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
              <strong className="font-semibold">Backend unreachable. </strong>
              {error ?? ""} Run <code className="rounded bg-rose-100 px-1">./scripts/dev.sh</code> from the repository
              root, or set the URL in Settings.
            </div>
          )}
          {health?.problems?.length ? (
            <ul className="space-y-1 text-xs text-amber-700">
              {health.problems.slice(0, 4).map((problem) => (
                <li key={problem} className="flex gap-2">
                  <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                  <span>{problem}</span>
                </li>
              ))}
            </ul>
          ) : null}

          {models.length === 0 ? (
            <EmptyState
              icon={<Boxes className="h-6 w-6" />}
              title="No models discovered"
              description="The backend scans models/ for .pth, .pt, .keras and .h5 files. Pull Git LFS and press Refresh."
            />
          ) : (
            <div className="flex flex-col gap-3">
              {models.map((model) => (
                <ModelRow
                  key={model.id}
                  model={model}
                  active={settings.activeModelId === model.id}
                  busy={busyId === model.id}
                  onToggle={() => toggleModel(model)}
                  onSelect={() => updateSettings({ activeModelId: model.id })}
                />
              ))}
            </div>
          )}
        </div>
      </Card>

      <ImageTestCard models={models} />
    </div>
  );
}

function ModelRow({
  model,
  active,
  busy,
  onToggle,
  onSelect,
}: {
  model: BackendModelInfo;
  active: boolean;
  busy: boolean;
  onToggle: () => void;
  onSelect: () => void;
}) {
  return (
    <div
      className={`rounded-xl border p-3 ${active ? "border-teal-300 bg-teal-50/50" : "border-slate-200 bg-white"}`}
    >
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-mono text-sm font-semibold text-slate-800">{model.id}</span>
        <Badge tone="indigo">{model.framework}</Badge>
        <Badge tone="slate">{model.modality}</Badge>
        {model.architecture && <Badge tone="slate">{model.architecture}</Badge>}
        <Badge tone={STATUS_TONE[model.status]}>{STATUS_LABEL[model.status]}</Badge>
        {active && <Badge tone="teal">live engine</Badge>}
        <span className="ml-auto text-[11px] text-slate-400">{formatBytes(model.size_bytes)}</span>
      </div>

      <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-[11px] text-slate-500 sm:grid-cols-4">
        <span>input: {model.input_shape ? JSON.stringify(model.input_shape) : "—"}</span>
        <span>output: {model.output_shape ? JSON.stringify(model.output_shape) : "—"}</span>
        <span>classes: {model.num_classes ?? "—"}</span>
        <span>labels: {model.labels_source ?? "none"}</span>
        {model.load_time_ms != null && <span>load: {model.load_time_ms.toFixed(0)} ms</span>}
        {model.last_inference_ms != null && <span>last inference: {model.last_inference_ms.toFixed(1)} ms</span>}
        <span>predictions: {model.inference_count}</span>
      </div>

      {model.detail && (
        <p className="mt-2 rounded-lg bg-slate-50 px-3 py-2 text-[11px] leading-relaxed text-slate-600">{model.detail}</p>
      )}
      {model.hint && (
        <pre className="mt-2 max-h-40 overflow-auto rounded-lg bg-slate-900 px-3 py-2 text-[10px] leading-relaxed text-slate-100">
          {model.hint}
        </pre>
      )}

      <div className="mt-3 flex flex-wrap gap-2">
        <Button size="sm" variant={model.loaded ? "outline" : "primary"} onClick={onToggle} disabled={busy}>
          {model.loaded ? <PowerOff className="h-3.5 w-3.5" /> : <Power className="h-3.5 w-3.5" />}
          {busy ? "Working…" : model.loaded ? "Unload" : "Load"}
        </Button>
        <Button size="sm" variant={active ? "secondary" : "outline"} onClick={onSelect} disabled={active}>
          <Target className="h-3.5 w-3.5" /> {active ? "In use for live recognition" : "Use for live recognition"}
        </Button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Single-image test against a backend model
// ---------------------------------------------------------------------------
function ImageTestCard({ models }: { models: BackendModelInfo[] }) {
  const { baseUrl } = useBackendModels({ autoRefresh: false });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<{ model: string; predictions: BackendPrediction[]; latency: number } | null>(null);
  const [modelId, setModelId] = useState<string>("");
  const fileInput = useRef<HTMLInputElement | null>(null);

  const imageModels = models.filter((m) => m.modality === "image" || m.modality === "unknown");

  async function run(payloadBase64: string, chosenModel: string) {
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const response = await predictImage(baseUrl, {
        model_id: chosenModel || null,
        image_base64: payloadBase64,
        top_k: 5,
      });
      setResult({ model: response.model_id, predictions: response.predictions, latency: response.latency_ms });
    } catch (err) {
      setError(err instanceof BackendError ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function handleFile(file: File | undefined) {
    if (!file) return;
    try {
      const base64 = await fileToBase64(file);
      await run(base64, modelId);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setBusy(false);
    }
  }

  return (
    <Card>
      <CardHeader
        title="Test a model with one image"
        subtitle="Sends the image to /api/predict/image and shows the model's real top-5 output"
        icon={<ImageIcon className="h-4 w-4" />}
      />
      <div className="space-y-3 p-4">
        <div className="flex flex-wrap items-center gap-2">
          <Select className="max-w-xs" value={modelId} onChange={(e) => setModelId(e.target.value)}>
            <option value="">Server default</option>
            {imageModels.map((m) => (
              <option key={m.id} value={m.id}>
                {m.id} ({m.modality})
              </option>
            ))}
          </Select>
          <label className="inline-flex cursor-pointer items-center gap-2 rounded-xl border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50">
            <Upload className="h-4 w-4" /> Choose image
            <input
              ref={fileInput}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => {
                void handleFile(e.target.files?.[0]);
                e.target.value = "";
              }}
            />
          </label>
          {busy && <span className="text-xs text-slate-400">Running inference…</span>}
        </div>

        {error && <p className="rounded-lg bg-rose-50 px-3 py-2 text-xs text-rose-700">{error}</p>}

        {result && (
          <div className="space-y-2">
            <p className="text-xs text-slate-500">
              Model <span className="font-mono font-semibold">{result.model}</span> · {result.latency.toFixed(1)} ms
            </p>
            <div className="space-y-1">
              {result.predictions.map((prediction) => (
                <div key={prediction.index} className="flex items-center gap-3">
                  <span className="w-32 truncate font-mono text-xs text-slate-700">{prediction.label}</span>
                  <div className="h-2 flex-1 overflow-hidden rounded-full bg-slate-100">
                    <div className="h-full rounded-full bg-teal-500" style={{ width: `${prediction.score * 100}%` }} />
                  </div>
                  <span className="w-14 text-right text-xs text-slate-500">{(prediction.score * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>
        )}
        {!result && !error && (
          <p className="text-xs text-slate-400">
            Nothing run yet. Pick an image model above and choose a photo of a sign.
          </p>
        )}
      </div>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Score the recorded dataset with a trained backend model
// ---------------------------------------------------------------------------
function BackendEvaluationCard() {
  const { baseUrl, models, status } = useBackendModels({ autoRefresh: false });
  const dataset = useAppStore((s) => s.dataset);
  const settings = useAppStore((s) => s.settings);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<BackendEvaluationReport | null>(null);
  const [modelId, setModelId] = useState<string>(settings.activeModelId ?? "");

  const scoredSamples = dataset.filter((s) => (s.frames?.length ?? 0) > 0 || (s.features?.length ?? 0) > 0);

  const run = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      const response = await backendEvaluate(baseUrl, {
        model_id: modelId || null,
        samples: scoredSamples,
        confidence_threshold: settings.backendConfidenceThreshold,
      });
      setReport(response);
    } catch (err) {
      setError(err instanceof BackendError ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }, [baseUrl, modelId, scoredSamples, settings.backendConfidenceThreshold]);

  return (
    <Card>
      <CardHeader
        title="Backend Evaluation"
        subtitle="Scores your recorded samples with a trained model served by the API — the same metrics as above"
        icon={<CheckCircle2 className="h-4 w-4" />}
        action={
          <div className="flex items-center gap-2">
            <Select className="w-52" value={modelId} onChange={(e) => setModelId(e.target.value)}>
              <option value="">Server default</option>
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.id}
                </option>
              ))}
            </Select>
            <Button size="sm" onClick={() => void run()} disabled={busy || scoredSamples.length === 0 || status !== "online"}>
              Evaluate
            </Button>
          </div>
        }
      />
      <div className="space-y-3 p-4">
        {status !== "online" && (
          <p className="text-xs text-amber-700">The backend is offline — start it to evaluate with a trained model.</p>
        )}
        {scoredSamples.length === 0 && (
          <EmptyState
            title="No samples to score"
            description="Record labelled samples in Dataset Collection first. Samples captured with the camera include the raw landmarks these models need."
          />
        )}
        {error && <p className="rounded-lg bg-rose-50 px-3 py-2 text-xs text-rose-700">{error}</p>}
        {report && (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-3 text-center sm:grid-cols-4">
              <div className="rounded-xl bg-slate-50 p-3">
                <p className="text-2xl font-bold text-slate-800">{report.total_samples}</p>
                <p className="text-xs text-slate-400">samples scored</p>
              </div>
              <div className="rounded-xl bg-slate-50 p-3">
                <p className="text-2xl font-bold text-slate-800">{report.correct}</p>
                <p className="text-xs text-slate-400">correct</p>
              </div>
              <div className="rounded-xl bg-emerald-50 p-3">
                <p className="text-2xl font-bold text-emerald-700">{(report.accuracy * 100).toFixed(1)}%</p>
                <p className="text-xs text-emerald-600">accuracy</p>
              </div>
              <div className="rounded-xl bg-slate-50 p-3">
                <p className="text-2xl font-bold text-slate-800">{report.latency_ms.toFixed(0)} ms</p>
                <p className="text-xs text-slate-400">total inference</p>
              </div>
            </div>
            <p className="text-[11px] text-slate-400">
              Model <span className="font-mono">{report.model_id}</span> · {report.rejected} prediction(s) below the{" "}
              {Math.round(settings.backendConfidenceThreshold * 100)}% threshold ·{" "}
              <a className="text-teal-600 underline" href={datasetExportHref(baseUrl, "npz")} download>
                download the training set (.npz)
              </a>
            </p>
            <EvaluationTables
              report={{
                totalSamples: report.total_samples,
                correct: report.correct,
                accuracy: report.accuracy,
                classes: report.classes,
                confusionMatrix: report.confusion_matrix,
                perClass: report.per_class.map((row) => ({
                  label: row.label,
                  precision: row.precision,
                  recall: row.recall,
                  f1: row.f1,
                  support: row.support,
                })),
              }}
            />
          </div>
        )}
      </div>
    </Card>
  );
}

// ---------------------------------------------------------------------------
function EvaluationTables({ report }: { report: EvaluationReport }) {
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 gap-3 text-center">
        <div className="rounded-xl bg-slate-50 p-3">
          <p className="text-2xl font-bold text-slate-800">{report.totalSamples}</p>
          <p className="text-xs text-slate-400">samples evaluated</p>
        </div>
        <div className="rounded-xl bg-slate-50 p-3">
          <p className="text-2xl font-bold text-slate-800">{report.correct}</p>
          <p className="text-xs text-slate-400">correct predictions</p>
        </div>
        <div className="rounded-xl bg-emerald-50 p-3">
          <p className="text-2xl font-bold text-emerald-700">{(report.accuracy * 100).toFixed(1)}%</p>
          <p className="text-xs text-emerald-600">measured accuracy</p>
        </div>
      </div>

      <div>
        <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">Per-class metrics</p>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[480px] text-left text-sm">
            <thead>
              <tr className="border-b border-slate-200 text-xs uppercase text-slate-400">
                <th className="py-2">Label</th>
                <th className="py-2">Precision</th>
                <th className="py-2">Recall</th>
                <th className="py-2">F1</th>
                <th className="py-2">Support</th>
              </tr>
            </thead>
            <tbody>
              {report.perClass.map((row) => (
                <tr key={row.label} className="border-b border-slate-100">
                  <td className="py-2 font-medium text-slate-700">{row.label}</td>
                  <td className="py-2">{(row.precision * 100).toFixed(0)}%</td>
                  <td className="py-2">{(row.recall * 100).toFixed(0)}%</td>
                  <td className="py-2">{(row.f1 * 100).toFixed(0)}%</td>
                  <td className="py-2">{row.support}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div>
        <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">Confusion matrix</p>
        <div className="overflow-x-auto">
          <table className="text-center text-xs">
            <thead>
              <tr>
                <th className="p-1"></th>
                {report.classes.map((c) => (
                  <th key={c} className="p-1 font-medium text-slate-500">
                    {c}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {report.classes.map((rowLabel, i) => (
                <tr key={rowLabel}>
                  <td className="p-1 pr-2 font-medium text-slate-500">{rowLabel}</td>
                  {report.confusionMatrix[i].map((v, j) => (
                    <td
                      key={j}
                      className={`p-1.5 ${
                        i === j && v > 0
                          ? "bg-emerald-100 font-semibold text-emerald-700"
                          : v > 0
                            ? "bg-rose-50 text-rose-600"
                            : "text-slate-300"
                      }`}
                    >
                      {v}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
