// ---------------------------------------------------------------------------
// MODULE: Text-to-Speech
// Thin, defensive wrapper around the browser's native Web Speech Synthesis
// API. Runs fully asynchronously and never blocks the camera/recognition
// loop. Degrades gracefully with a clear error when unsupported.
// ---------------------------------------------------------------------------

export function isSpeechSupported(): boolean {
  return typeof window !== "undefined" && "speechSynthesis" in window;
}

export function getAvailableVoices(): SpeechSynthesisVoice[] {
  if (!isSpeechSupported()) return [];
  return window.speechSynthesis.getVoices();
}

export interface SpeakOptions {
  voiceURI?: string | null;
  lang?: string;
  rate?: number;
  pitch?: number;
  onEnd?: () => void;
  onError?: (message: string) => void;
}

export function speak(text: string, options: SpeakOptions = {}) {
  if (!isSpeechSupported()) {
    options.onError?.("Text-to-speech is not supported in this browser.");
    return;
  }
  if (!text.trim()) {
    options.onError?.("There is no text to speak yet.");
    return;
  }

  window.speechSynthesis.cancel();

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = options.rate ?? 1;
  utterance.pitch = options.pitch ?? 1;
  if (options.lang) utterance.lang = options.lang;

  if (options.voiceURI) {
    const voice = getAvailableVoices().find((v) => v.voiceURI === options.voiceURI);
    if (voice) utterance.voice = voice;
  }

  utterance.onend = () => options.onEnd?.();
  utterance.onerror = (e) => options.onError?.(`Speech synthesis error: ${e.error}`);

  window.speechSynthesis.speak(utterance);
}

export function pauseSpeech() {
  if (isSpeechSupported()) window.speechSynthesis.pause();
}

export function resumeSpeech() {
  if (isSpeechSupported()) window.speechSynthesis.resume();
}

export function stopSpeech() {
  if (isSpeechSupported()) window.speechSynthesis.cancel();
}

export function isSpeaking(): boolean {
  return isSpeechSupported() && window.speechSynthesis.speaking;
}
