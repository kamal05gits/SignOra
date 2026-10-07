// ---------------------------------------------------------------------------
// MODULE: Feature Extraction (spatial)
// Converts raw MediaPipe hand landmarks into a normalized, scale/position
// invariant feature set that is reused identically by both the live
// recognition engine and the dataset-collection tool, so training-time and
// inference-time preprocessing never diverge.
// ---------------------------------------------------------------------------
import type { DetectedHand, HandFeatureSet, Landmark } from "../types";

// MediaPipe Hands landmark indices (21 points per hand)
const WRIST = 0;
const THUMB_TIP = 4;
const INDEX_TIP = 8;
const INDEX_PIP = 6;
const INDEX_MCP = 5;
const MIDDLE_TIP = 12;
const MIDDLE_PIP = 10;
const MIDDLE_MCP = 9;
const RING_TIP = 16;
const RING_PIP = 14;
const RING_MCP = 13;
const PINKY_TIP = 20;
const PINKY_PIP = 18;
const PINKY_MCP = 17;

function dist(a: Landmark, b: Landmark): number {
  return Math.hypot(a.x - b.x, a.y - b.y, (a.z - b.z) * 0.5);
}

/** Hand scale reference: wrist to middle-finger MCP distance. Used to normalize all other distances. */
function handScale(lm: Landmark[]): number {
  const s = dist(lm[WRIST], lm[MIDDLE_MCP]);
  return s < 1e-5 ? 1e-5 : s;
}

/**
 * Curl ratio for a finger: 0 = fully extended (straight), 1 = fully curled (folded into palm).
 * Computed from the ratio of (tip-to-mcp distance) vs (pip-to-mcp + tip-to-pip), which drops
 * sharply when the finger bends.
 */
function fingerCurl(lm: Landmark[], tip: number, pip: number, mcp: number): number {
  const straightLine = dist(lm[tip], lm[mcp]);
  const curledPath = dist(lm[mcp], lm[pip]) + dist(lm[pip], lm[tip]);
  if (curledPath < 1e-6) return 0;
  const ratio = straightLine / curledPath; // ~1 when straight, drops when curled
  const curl = 1 - ratio;
  return Math.min(1, Math.max(0, curl * 1.8));
}

/** Thumb uses a distance-to-palm heuristic since it bends differently from other fingers. */
function thumbCurl(lm: Landmark[], scale: number): number {
  const d = dist(lm[THUMB_TIP], lm[INDEX_MCP]) / scale;
  // Larger distance => thumb extended away from palm => low curl.
  const curl = 1 - Math.min(1, d / 1.6);
  return Math.min(1, Math.max(0, curl));
}

export function extractHandFeatures(hand: DetectedHand): HandFeatureSet {
  const lm = hand.landmarks;
  const scale = handScale(lm);

  const curlIndex = fingerCurl(lm, INDEX_TIP, INDEX_PIP, INDEX_MCP);
  const curlMiddle = fingerCurl(lm, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP);
  const curlRing = fingerCurl(lm, RING_TIP, RING_PIP, RING_MCP);
  const curlPinky = fingerCurl(lm, PINKY_TIP, PINKY_PIP, PINKY_MCP);
  const curlThumb = thumbCurl(lm, scale);

  const EXT_THRESHOLD = 0.45;

  const spread =
    (dist(lm[INDEX_TIP], lm[PINKY_TIP]) / scale +
      dist(lm[INDEX_TIP], lm[MIDDLE_TIP]) / scale) /
    2;

  const thumbIndexDistance = dist(lm[THUMB_TIP], lm[INDEX_TIP]) / scale;

  // Palm orientation: angle (radians) of the vector from wrist to middle-MCP, used to
  // roughly estimate whether the palm faces the camera (used for diagnostics, not classification).
  const palmOrientation = Math.atan2(
    lm[MIDDLE_MCP].y - lm[WRIST].y,
    lm[MIDDLE_MCP].x - lm[WRIST].x
  );

  return {
    handedness: hand.handedness,
    fingerCurl: {
      thumb: curlThumb,
      index: curlIndex,
      middle: curlMiddle,
      ring: curlRing,
      pinky: curlPinky,
    },
    fingerExtended: {
      thumb: curlThumb < EXT_THRESHOLD,
      index: curlIndex < EXT_THRESHOLD,
      middle: curlMiddle < EXT_THRESHOLD,
      ring: curlRing < EXT_THRESHOLD,
      pinky: curlPinky < EXT_THRESHOLD,
    },
    spread,
    thumbIndexDistance,
    palmOrientation,
    raw: lm,
  };
}

export function extractFeaturesFromFrame(hands: DetectedHand[]): HandFeatureSet[] {
  return hands.map(extractHandFeatures);
}
