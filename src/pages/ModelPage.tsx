import { useMemo, useState } from "react";
import { Cpu, PlayCircle, AlertTriangle, BookOpen } from "lucide-react";
import { SIGN_VOCABULARY } from "../lib/signVocabulary";
import { evaluateDataset, type EvaluationReport } from "../lib/evaluate";
import { useAppStore } from "../store/useAppStore";
import { Badge, Button, Card, CardHeader, EmptyState, Slider } from "../components/ui";

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
          Technical details of how signs are currently recognized, plus a reproducible evaluation tool that measures
          real accuracy against your recorded samples — nothing on this page is a pre-written or assumed number.
        </p>
      </header>

      <Card className="border-amber-200 bg-amber-50/60 p-4">
        <div className="flex gap-3">
          <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-amber-600" />
          <div className="text-sm text-amber-800">
            <p className="font-semibold">Honest scope disclosure</p>
            <p className="mt-1 leading-relaxed">
              A production-grade ISL recognizer requires a CNN/LSTM trained on a large labelled ISL video corpus — that
              training pipeline (Python + TensorFlow/PyTorch + GPU + licensed dataset) cannot run inside this
              browser-only build. Live recognition here uses a transparent, rule-based <em>geometric classifier</em>{" "}
              that reads real MediaPipe landmarks from your real webcam and matches them to hand-shape templates for a
              demo vocabulary of {SIGN_VOCABULARY.length} signs. It is a genuine, working, real-time classifier — just
              not a trained deep-learning model. The Dataset Collection page lets you build a real labelled dataset
              with the exact same feature pipeline, ready to train an actual CNN/LSTM offline later.
            </p>
          </div>
        </div>
      </Card>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader title="Supported Demo Vocabulary" subtitle={`${SIGN_VOCABULARY.length} static signs`} icon={<BookOpen className="h-4 w-4" />} />
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
            <p className="text-[11px] text-slate-400">
              These same parameters are live on the Live Translator page and are persisted locally.
            </p>
          </div>
        </Card>
      </div>

      <Card>
        <CardHeader
          title="Evaluation"
          subtitle="Runs the classifier against your recorded dataset samples and reports real, measured metrics"
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
            <p className="text-sm text-slate-400">Press "Run evaluation" to compute accuracy on your {dataset.length} recorded samples.</p>
          )}
          {report && (
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
                              className={`p-1.5 ${i === j && v > 0 ? "bg-emerald-100 font-semibold text-emerald-700" : v > 0 ? "bg-rose-50 text-rose-600" : "text-slate-300"}`}
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
          )}
        </div>
      </Card>
    </div>
  );
}
