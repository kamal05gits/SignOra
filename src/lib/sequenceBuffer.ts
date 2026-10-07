// ---------------------------------------------------------------------------
// MODULE: Continuous recognition — temporal smoothing & cooldown
//
// Isolated per-frame predictions are noisy. This module maintains a rolling
// window of recent classifications, requires a majority vote above the
// confidence threshold to consider a sign "stable", and enforces a cooldown
// so the same sign is not appended to the sentence on every single frame.
//
// NOTE (scope honesty): this performs sequential ISOLATED-SIGN recognition
// with stability detection — it does NOT perform full continuous
// sentence-level co-articulation modelling (which would require a trained
// temporal model, e.g. an LSTM/Transformer over raw video). See About page.
// ---------------------------------------------------------------------------
import type { ClassificationResult } from "../types";

export interface StableDetection {
  gloss: string;
  label: string;
  confidence: number;
}

export class TemporalSmoother {
  private windowSize: number;
  private cooldownMs: number;
  private confidenceThreshold: number;
  private buffer: ClassificationResult[] = [];
  private lastAcceptedGloss: string | null = null;
  private lastAcceptedAt = 0;

  constructor(windowSize = 8, cooldownMs = 1400, confidenceThreshold = 0.72) {
    this.windowSize = windowSize;
    this.cooldownMs = cooldownMs;
    this.confidenceThreshold = confidenceThreshold;
  }

  updateConfig(opts: { windowSize?: number; cooldownMs?: number; confidenceThreshold?: number }) {
    if (opts.windowSize) this.windowSize = opts.windowSize;
    if (opts.cooldownMs !== undefined) this.cooldownMs = opts.cooldownMs;
    if (opts.confidenceThreshold !== undefined) this.confidenceThreshold = opts.confidenceThreshold;
  }

  reset() {
    this.buffer = [];
    this.lastAcceptedGloss = null;
    this.lastAcceptedAt = 0;
  }

  /** Feed one frame's classification result. Returns a StableDetection if a new token should be emitted. */
  push(result: ClassificationResult, now = performance.now()): StableDetection | null {
    this.buffer.push(result);
    if (this.buffer.length > this.windowSize) this.buffer.shift();

    if (this.buffer.length < Math.min(this.windowSize, 5)) return null;

    // Majority vote among frames that pass the confidence threshold
    const counts = new Map<string, { count: number; confSum: number; label: string }>();
    for (const r of this.buffer) {
      if (!r.gloss || r.confidence < this.confidenceThreshold) continue;
      const entry = counts.get(r.gloss) ?? { count: 0, confSum: 0, label: r.label ?? r.gloss };
      entry.count += 1;
      entry.confSum += r.confidence;
      counts.set(r.gloss, entry);
    }

    let winner: { gloss: string; count: number; confSum: number; label: string } | null = null;
    for (const [gloss, v] of counts.entries()) {
      if (!winner || v.count > winner.count) winner = { gloss, ...v };
    }

    if (!winner) return null;

    const requiredVotes = Math.ceil(this.buffer.length * 0.6);
    if (winner.count < requiredVotes) return null;

    const avgConfidence = winner.confSum / winner.count;

    const sinceLast = now - this.lastAcceptedAt;
    const isRepeat = winner.gloss === this.lastAcceptedGloss;
    if (isRepeat && sinceLast < this.cooldownMs) {
      return null; // duplicate suppression / cooldown
    }

    this.lastAcceptedGloss = winner.gloss;
    this.lastAcceptedAt = now;
    // Clear the buffer after accepting so the next sign needs fresh evidence
    this.buffer = [];

    return { gloss: winner.gloss, label: winner.label, confidence: avgConfidence };
  }
}
