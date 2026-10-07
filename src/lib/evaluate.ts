// ---------------------------------------------------------------------------
// MODULE: Model evaluation
// Runs the actual heuristic classifier against actually-recorded dataset
// samples (captured in the Dataset Collection page) and reports genuinely
// measured metrics. Nothing here is a pre-written/fabricated number — if
// there is no dataset yet, the evaluation simply reports that no data is
// available instead of inventing results.
// ---------------------------------------------------------------------------
import type { DatasetSample } from "../types";
import { classifyHandFeatures } from "./signClassifier";

export interface EvaluationReport {
  totalSamples: number;
  correct: number;
  accuracy: number;
  classes: string[];
  confusionMatrix: number[][]; // rows = true label, cols = predicted label
  perClass: Array<{ label: string; precision: number; recall: number; f1: number; support: number }>;
}

export function evaluateDataset(samples: DatasetSample[], confidenceThreshold: number): EvaluationReport | null {
  if (samples.length === 0) return null;

  const classesSet = new Set<string>();
  samples.forEach((s) => classesSet.add(s.label));

  const predictions = samples.map((s) => {
    const result = classifyHandFeatures(s.features);
    const predicted = result.confidence >= confidenceThreshold && result.gloss ? result.gloss : "REJECTED";
    classesSet.add(predicted);
    return { trueLabel: s.label, predicted };
  });

  const classes = Array.from(classesSet).sort();
  const index = new Map(classes.map((c, i) => [c, i]));
  const matrix = classes.map(() => classes.map(() => 0));

  let correct = 0;
  predictions.forEach(({ trueLabel, predicted }) => {
    matrix[index.get(trueLabel)!][index.get(predicted)!] += 1;
    if (trueLabel === predicted) correct += 1;
  });

  const perClass = classes
    .filter((c) => c !== "REJECTED")
    .map((c) => {
      const i = index.get(c)!;
      const support = matrix[i].reduce((a, b) => a + b, 0);
      const truePositive = matrix[i][i];
      const predictedPositive = classes.reduce((sum, _, j) => sum + matrix[j][i], 0);
      const precision = predictedPositive > 0 ? truePositive / predictedPositive : 0;
      const recall = support > 0 ? truePositive / support : 0;
      const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
      return { label: c, precision, recall, f1, support };
    });

  return {
    totalSamples: samples.length,
    correct,
    accuracy: correct / samples.length,
    classes,
    confusionMatrix: matrix,
    perClass,
  };
}
