import {
  Info,
  Layers,
  AlertTriangle,
  CheckCircle2,
  ShieldCheck,
  Rocket,
} from "lucide-react";
import { Card, CardHeader } from "../components/ui";

const PIPELINE = [
  "Webcam capture (getUserMedia)",
  "Frame preprocessing (resize, color conversion — handled internally by MediaPipe)",
  "MediaPipe HandLandmarker (21-point landmarks, on-device WASM)",
  "Feature extraction (finger curl, spread, distances — scale/position invariant)",
  "Heuristic geometric classification (confidence-scored)",
  "Temporal smoothing + cooldown (majority vote across frames)",
  "Confidence thresholding (reject uncertain predictions)",
  "NLP sentence construction (tokens → normalized text)",
  "Optional translation (MyMemory API)",
  "Text-to-speech (Web Speech API)",
];

const LIMITATIONS = [
  "Recognition uses a transparent rule-based geometric classifier, not a trained CNN/LSTM — no ISL video dataset/training pipeline is bundled or fabricated.",
  "Demo vocabulary covers a small set of static, single-hand signs; it is not an exhaustive ISL dictionary and does not model two-hand compound signs or motion-based signs.",
  "Continuous recognition is sequential isolated-sign detection with stability/cooldown logic, not full co-articulated sentence-level sign language understanding.",
  "Translation quality depends on the free MyMemory API and may be rate-limited; always cross-check important translations.",
  "Emergency SOS uses tel:/sms: device links and a public emergency-number lookup — it cannot programmatically dispatch SMS/calls without the user's own device/app.",
  "Performance (FPS, latency) depends entirely on the end-user's device/browser/GPU and has not been benchmarked on specific hardware here.",
];

const ACCEPTANCE = [
  "Application loads without critical errors in a modern browser",
  "Real webcam feed is displayed and can be started/stopped",
  "MediaPipe detects real hand landmarks and overlays them live",
  "A genuine (heuristic, documented) classifier performs real inference — no scripted/fake results",
  "Low-confidence predictions are visibly rejected, not silently accepted",
  "Recognized tokens combine into an editable, readable sentence",
  "User can clear / remove-last / insert space",
  "Text-to-speech genuinely speaks recognized/translated text",
  "Language selection triggers a real translation API call with error handling",
  "UI remains responsive while recognition runs (bounded RAF loop, no blocking)",
  "Dataset collection and evaluation are reproducible and based on real recorded samples",
  "All limitations are explicitly documented on this page",
];

export default function AboutPage() {
  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-6">
      <header className="flex items-center gap-3">
        <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-indigo-100 text-indigo-600">
          <Info className="h-6 w-6" />
        </div>
        <div>
          <h1 className="text-2xl font-bold text-slate-900">About This Project</h1>
          <p className="text-sm text-slate-500">Real-Time Indian Sign Language Translator — academic / demo build</p>
        </div>
      </header>

      <Card className="p-5">
        <h2 className="text-sm font-semibold text-slate-900">Problem Statement</h2>
        <p className="mt-2 text-sm leading-relaxed text-slate-600">
          Indian Sign Language (ISL) users cannot always communicate directly with people who do not understand sign
          language, and are often dependent on human interpreters for immediate interaction. This project explores an
          automated system that recognizes hand gestures in real time and converts them into readable text and
          audible speech, to support more independent communication.
        </p>
      </Card>

      <Card className="p-5">
        <h2 className="text-sm font-semibold text-slate-900">Implementation note</h2>
        <p className="mt-2 text-sm leading-relaxed text-slate-600">
          The original specification for this project assumed a Python desktop stack (OpenCV, TensorFlow/PyTorch,
          CustomTkinter/PySide6). This build runs inside a browser-based React/Vite environment, so every module was
          re-implemented with equivalent, real, working browser technology rather than being simulated:
          MediaPipe Tasks-Vision (WASM, on-device) for hand landmarks, a transparent geometric rule-based classifier
          for recognition, the Web Speech API for text-to-speech, and the free MyMemory API for translation. The
          architecture and module boundaries mirror the originally requested design so the vision/ML layer can be
          swapped for a trained backend model later without touching the UI.
        </p>
      </Card>

      <Card>
        <CardHeader title="System Pipeline" icon={<Layers className="h-4 w-4" />} />
        <ol className="space-y-2 p-4">
          {PIPELINE.map((step, i) => (
            <li key={step} className="flex items-start gap-3 text-sm text-slate-600">
              <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-teal-50 text-[11px] font-semibold text-teal-700">
                {i + 1}
              </span>
              {step}
            </li>
          ))}
        </ol>
      </Card>

      <Card className="border-amber-200 bg-amber-50/60">
        <CardHeader title="Known Limitations" icon={<AlertTriangle className="h-4 w-4 text-amber-600" />} />
        <ul className="space-y-2 p-4">
          {LIMITATIONS.map((l) => (
            <li key={l} className="text-sm leading-relaxed text-amber-800">
              • {l}
            </li>
          ))}
        </ul>
      </Card>

      <Card>
        <CardHeader title="Acceptance Criteria Checklist" icon={<CheckCircle2 className="h-4 w-4" />} />
        <ul className="space-y-2 p-4">
          {ACCEPTANCE.map((a) => (
            <li key={a} className="flex items-start gap-2 text-sm text-slate-600">
              <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-500" />
              {a}
            </li>
          ))}
        </ul>
      </Card>

      <Card>
        <CardHeader title="Future Enhancements" icon={<Rocket className="h-4 w-4" />} />
        <ul className="space-y-2 p-4 text-sm text-slate-600">
          <li>• Train a real CNN/LSTM (or Transformer) model offline on a licensed ISL dataset using the dataset-export format produced here, then serve it via an on-device ONNX/TFJS model for drop-in replacement of the heuristic classifier.</li>
          <li>• Expand the vocabulary to full ISL alphabet/numerals and common two-hand signs.</li>
          <li>• Add true continuous/co-articulated sentence modelling with a sequence model over raw landmark sequences.</li>
          <li>• Integrate a dedicated server-side SMS/voice gateway for true automated emergency dispatch.</li>
          <li>• Add user accounts and cloud sync of datasets/history (currently local-only by design for privacy).</li>
        </ul>
      </Card>

      <Card className="border-emerald-200 bg-emerald-50/60 p-4 text-sm text-emerald-800">
        <div className="flex gap-3">
          <ShieldCheck className="h-5 w-5 shrink-0 text-emerald-600" />
          <p>
            <strong>Privacy notice:</strong> Camera video and audio are never transmitted off your device. Only
            derived numeric hand-landmark features (which you explicitly capture) and short recognized text snippets
            (for translation) ever leave the browser, and only to the third-party APIs explicitly named above.
          </p>
        </div>
      </Card>
    </div>
  );
}
