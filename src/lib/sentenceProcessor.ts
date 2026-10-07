// ---------------------------------------------------------------------------
// MODULE: NLP / Sentence Processing
// Converts a sequence of recognized gloss tokens into readable text. Kept
// fully independent from the vision/classification layer: it only ever sees
// strings, never raw frames or landmarks.
// ---------------------------------------------------------------------------
import type { RecognizedToken } from "../types";

/** Tokens that should not be preceded by a space when rendered (punctuation-like). */
const NO_SPACE_BEFORE = new Set([".", ",", "?", "!"]);

export function tokensToSentence(tokens: RecognizedToken[]): string {
  let sentence = "";
  tokens.forEach((t, i) => {
    const word = t.gloss;
    if (i === 0) {
      sentence += word;
      return;
    }
    if (NO_SPACE_BEFORE.has(word)) {
      sentence += word;
    } else {
      sentence += " " + word;
    }
  });
  return normalizeSentence(sentence);
}

export function normalizeSentence(raw: string): string {
  let s = raw.trim().replace(/\s+/g, " ");
  if (!s) return s;
  // Capitalize first letter of the sentence
  s = s.charAt(0).toUpperCase() + s.slice(1);
  // Ensure terminal punctuation for readability / TTS prosody
  if (!/[.!?]$/.test(s)) {
    s += ".";
  }
  return s;
}

/** Builds a human-readable trace of recognized tokens for transparency/debugging. */
export function traceTokens(tokens: RecognizedToken[]): string {
  return tokens.map((t) => `${t.gloss}(${Math.round(t.confidence * 100)}%)`).join(" -> ");
}
