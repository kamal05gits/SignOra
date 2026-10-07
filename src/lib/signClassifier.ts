// ---------------------------------------------------------------------------
// MODULE: Confidence-aware prediction
// Picks the best-matching sign definition for the dominant hand in frame and
// returns a calibrated confidence score. Confidence thresholding / rejection
// of uncertain predictions happens one layer up, in the recognition engine,
// so this module stays a pure, testable function.
// ---------------------------------------------------------------------------
import type { ClassificationResult, HandFeatureSet } from "../types";
import { SIGN_VOCABULARY } from "./signVocabulary";

export function classifyHandFeatures(features: HandFeatureSet[]): ClassificationResult {
  if (!features.length) {
    return { label: null, gloss: null, confidence: 0, handsUsed: 0 };
  }

  // Use the first detected hand as the primary signing hand (documented limitation:
  // two-hand compound signs are not composed together, see About > Limitations).
  const primary = features[0];

  let bestScore = 0;
  let bestDef = null as (typeof SIGN_VOCABULARY)[number] | null;

  for (const def of SIGN_VOCABULARY) {
    const score = def.match(primary);
    if (score > bestScore) {
      bestScore = score;
      bestDef = def;
    }
  }

  if (!bestDef) {
    return { label: null, gloss: null, confidence: 0, handsUsed: features.length };
  }

  return {
    label: bestDef.label,
    gloss: bestDef.gloss,
    confidence: Math.min(1, Math.max(0, bestScore)),
    handsUsed: features.length,
  };
}
