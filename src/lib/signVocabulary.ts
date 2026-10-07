// ---------------------------------------------------------------------------
// MODULE: Recognition Engine — Sign Vocabulary & Heuristic Classifier
//
// IMPORTANT / HONEST DISCLOSURE
// ------------------------------------------------------------------------
// A production ISL recognizer needs a CNN/LSTM trained on a large labelled
// video dataset (e.g. INCLUDE, ISL-CSLRT). That training pipeline cannot run
// inside this browser-only project environment (no Python runtime, no GPU,
// no access to a licensed ISL corpus). Rather than FAKE a trained model or
// hardcode scripted "recognition" results, this module implements a
// transparent, fully-explainable GEOMETRIC RULE-BASED classifier that reads
// real MediaPipe hand-landmarks captured from a real webcam, in real time,
// and matches them against hand-shape templates for a small demo vocabulary.
//
// This is clearly a different technique from a trained CNN/LSTM and is
// labelled as such everywhere in the UI. The Dataset Collection module lets
// you record real labelled samples; the architecture (features -> fixed
// vector -> classifier) is kept identical to what a real trained model would
// consume, so the heuristic classifier here can be swapped for a trained
// model later (see docs in the About page) without touching the UI layer.
// ---------------------------------------------------------------------------
import type { HandFeatureSet, SignDefinition } from "../types";

function fe(f: HandFeatureSet) {
  return f.fingerExtended;
}

/** Scores how well a boolean extension pattern matches, tolerating partial matches. */
function patternScore(
  f: HandFeatureSet,
  pattern: Partial<Record<"thumb" | "index" | "middle" | "ring" | "pinky", boolean>>
) {
  const keys = Object.keys(pattern) as Array<keyof typeof pattern>;
  let hits = 0;
  for (const k of keys) {
    if (fe(f)[k] === pattern[k]) hits += 1;
  }
  return hits / keys.length;
}

export const SIGN_VOCABULARY: SignDefinition[] = [
  {
    id: "open_palm",
    label: "G1",
    gloss: "HELLO",
    category: "word",
    description: "Open palm, all five fingers extended",
    match: (f) =>
      patternScore(f, { thumb: true, index: true, middle: true, ring: true, pinky: true }),
  },
  {
    id: "fist",
    label: "G2",
    gloss: "YES",
    category: "word",
    description: "Closed fist, all fingers curled",
    match: (f) =>
      patternScore(f, { thumb: false, index: false, middle: false, ring: false, pinky: false }),
  },
  {
    id: "index_only",
    label: "G3",
    gloss: "ONE",
    category: "number",
    description: "Only the index finger extended",
    match: (f) =>
      patternScore(f, { thumb: false, index: true, middle: false, ring: false, pinky: false }),
  },
  {
    id: "index_middle",
    label: "G4",
    gloss: "TWO",
    category: "number",
    description: "Index and middle fingers extended (V shape)",
    match: (f) => {
      const base = patternScore(f, {
        thumb: false,
        index: true,
        middle: true,
        ring: false,
        pinky: false,
      });
      // Reward a visible gap between index & middle (true V, not crossed fingers)
      const spreadBonus = Math.min(1, f.spread / 2.4);
      return base * 0.8 + spreadBonus * 0.2;
    },
  },
  {
    id: "index_middle_ring",
    label: "G5",
    gloss: "THREE",
    category: "number",
    description: "Index, middle and ring fingers extended",
    match: (f) =>
      patternScore(f, { thumb: false, index: true, middle: true, ring: true, pinky: false }),
  },
  {
    id: "four_fingers",
    label: "G6",
    gloss: "FOUR",
    category: "number",
    description: "Four fingers extended, thumb folded across palm",
    match: (f) =>
      patternScore(f, { thumb: false, index: true, middle: true, ring: true, pinky: true }),
  },
  {
    id: "thumbs_up",
    label: "G7",
    gloss: "GOOD",
    category: "word",
    description: "Thumb extended upward, all other fingers curled",
    match: (f) =>
      patternScore(f, { thumb: true, index: false, middle: false, ring: false, pinky: false }),
  },
  {
    id: "pinky_only",
    label: "G8",
    gloss: "I / ME",
    category: "word",
    description: "Only the little finger extended",
    match: (f) =>
      patternScore(f, { thumb: false, index: false, middle: false, ring: false, pinky: true }),
  },
  {
    id: "thumb_pinky",
    label: "G9",
    gloss: "CALL ME",
    category: "phrase",
    description: "Thumb and little finger extended (shaka shape)",
    match: (f) =>
      patternScore(f, { thumb: true, index: false, middle: false, ring: false, pinky: true }),
  },
  {
    id: "ok_sign",
    label: "G10",
    gloss: "OK",
    category: "word",
    description: "Thumb and index tips touching, other fingers extended",
    match: (f) => {
      const touch = f.thumbIndexDistance < 0.45 ? 1 : Math.max(0, 1 - (f.thumbIndexDistance - 0.45));
      const others = patternScore(f, { middle: true, ring: true, pinky: true });
      return touch * 0.6 + others * 0.4;
    },
  },
  {
    id: "no_sign",
    label: "G11",
    gloss: "NO",
    category: "word",
    description: "Index and middle extended, pressed together, thumb out",
    match: (f) => {
      const base = patternScore(f, { thumb: true, index: true, middle: true, ring: false, pinky: false });
      const closeBonus = f.spread < 1.1 ? 1 : Math.max(0, 1 - (f.spread - 1.1));
      return base * 0.7 + closeBonus * 0.3;
    },
  },
  {
    id: "help_sign",
    label: "G12",
    gloss: "HELP",
    category: "phrase",
    description: "Both-hand friendly: fist with thumb tucked, raised",
    match: (f) => patternScore(f, { thumb: false, index: false, middle: false, ring: true, pinky: true }) * 0.9,
  },
];

export function describeVocabulary() {
  return SIGN_VOCABULARY.map((s) => ({
    label: s.label,
    gloss: s.gloss,
    category: s.category,
    description: s.description,
  }));
}
