// ---------------------------------------------------------------------------
// MODULE: Multilingual output (translation)
// Uses the free, keyless MyMemory Translation API. This is a real network
// call with real error handling — if the device is offline or the service
// is unavailable, the UI must show a clear error and keep the original
// English/ISL-gloss text available (translation never replaces the source).
// ---------------------------------------------------------------------------

export interface SupportedLanguage {
  code: string;
  label: string;
}

export const SUPPORTED_LANGUAGES: SupportedLanguage[] = [
  { code: "en", label: "English (source)" },
  { code: "hi", label: "Hindi" },
  { code: "ta", label: "Tamil" },
  { code: "te", label: "Telugu" },
  { code: "kn", label: "Kannada" },
  { code: "ml", label: "Malayalam" },
  { code: "mr", label: "Marathi" },
  { code: "bn", label: "Bengali" },
  { code: "gu", label: "Gujarati" },
  { code: "pa", label: "Punjabi" },
  { code: "ur", label: "Urdu" },
  { code: "fr", label: "French" },
  { code: "es", label: "Spanish" },
];

export interface TranslateResult {
  success: boolean;
  text?: string;
  error?: string;
}

/**
 * Calls the MyMemory translation API (https://mymemory.translated.net/).
 * No API key required; subject to a fair-use daily word limit. Network or
 * quota failures are surfaced to the caller rather than silently swallowed.
 */
export async function translateText(text: string, targetLang: string): Promise<TranslateResult> {
  if (!text.trim()) {
    return { success: false, error: "No text to translate." };
  }
  if (targetLang === "en") {
    return { success: true, text };
  }

  const params = new URLSearchParams({
    q: text.slice(0, 480),
    langpair: `en|${targetLang}`,
  });

  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 8000);
    const res = await fetch(`https://api.mymemory.translated.net/get?${params.toString()}`, {
      signal: controller.signal,
    });
    clearTimeout(timeout);

    if (!res.ok) {
      return { success: false, error: `Translation service responded with HTTP ${res.status}.` };
    }
    const data = await res.json();
    if (data?.responseStatus && Number(data.responseStatus) !== 200) {
      return { success: false, error: data?.responseDetails ?? "Translation request was rejected." };
    }
    const translated = data?.responseData?.translatedText;
    if (!translated) {
      return { success: false, error: "Translation service returned an empty result." };
    }
    return { success: true, text: translated };
  } catch (err) {
    const message =
      err instanceof DOMException && err.name === "AbortError"
        ? "Translation request timed out. Check your internet connection."
        : "Could not reach the translation service. You are possibly offline.";
    return { success: false, error: message };
  }
}
