// ---------------------------------------------------------------------------
// MODULE: Emergency SOS integration
// Provides quick access to real emergency numbers (via the free, keyless
// Emergency Number API: https://emergencynumberapi.com) with a verified
// offline fallback table in case the network/API is unavailable, plus
// geolocation-based alert message composition. No paid SMS/voice gateway is
// integrated (that would require a paid account + backend server, which is
// out of scope for this static client-only build) — instead we use native
// tel: / sms: URI schemes so the device's own phone/SMS app sends the alert.
// ---------------------------------------------------------------------------
import type { EmergencyContact } from "../types";

export interface EmergencyNumbers {
  country: string;
  police: string[];
  ambulance: string[];
  fire: string[];
  dispatch: string[];
  source: "api" | "offline-fallback";
}

// Verified common national numbers kept locally so the SOS feature still
// works with no internet connection (critical for an "emergency" feature).
const OFFLINE_FALLBACK: Record<string, EmergencyNumbers> = {
  IN: {
    country: "India",
    police: ["100", "112"],
    ambulance: ["102", "108"],
    fire: ["101"],
    dispatch: ["112"],
    source: "offline-fallback",
  },
  US: {
    country: "United States",
    police: ["911"],
    ambulance: ["911"],
    fire: ["911"],
    dispatch: ["911"],
    source: "offline-fallback",
  },
  GB: {
    country: "United Kingdom",
    police: ["999", "112"],
    ambulance: ["999", "112"],
    fire: ["999", "112"],
    dispatch: ["999"],
    source: "offline-fallback",
  },
};

export async function fetchEmergencyNumbers(countryCode: string): Promise<EmergencyNumbers> {
  const code = countryCode.toUpperCase();
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 6000);
    const res = await fetch(`https://emergencynumberapi.com/api/country/${code}`, {
      signal: controller.signal,
    });
    clearTimeout(timeout);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    return {
      country: data?.Country?.Name ?? code,
      police: data?.Police?.All?.length ? data.Police.All : OFFLINE_FALLBACK[code]?.police ?? [],
      ambulance: data?.Ambulance?.All?.length
        ? data.Ambulance.All
        : OFFLINE_FALLBACK[code]?.ambulance ?? [],
      fire: data?.Fire?.All?.length ? data.Fire.All : OFFLINE_FALLBACK[code]?.fire ?? [],
      dispatch: data?.Dispatch?.All?.length ? data.Dispatch.All : OFFLINE_FALLBACK[code]?.dispatch ?? [],
      source: "api",
    };
  } catch {
    return (
      OFFLINE_FALLBACK[code] ?? {
        country: code,
        police: [],
        ambulance: [],
        fire: [],
        dispatch: [],
        source: "offline-fallback",
      }
    );
  }
}

export interface GeoResult {
  success: boolean;
  latitude?: number;
  longitude?: number;
  mapsUrl?: string;
  error?: string;
}

export function getCurrentLocation(): Promise<GeoResult> {
  return new Promise((resolve) => {
    if (!("geolocation" in navigator)) {
      resolve({ success: false, error: "Geolocation is not supported in this browser." });
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const { latitude, longitude } = pos.coords;
        resolve({
          success: true,
          latitude,
          longitude,
          mapsUrl: `https://www.google.com/maps?q=${latitude},${longitude}`,
        });
      },
      (err) => {
        resolve({ success: false, error: err.message || "Location permission was denied." });
      },
      { enableHighAccuracy: true, timeout: 8000, maximumAge: 0 }
    );
  });
}

export function buildEmergencyMessage(opts: {
  recognizedText: string;
  mapsUrl?: string;
  contact?: EmergencyContact;
}): string {
  const parts = [
    "EMERGENCY ALERT from ISL Translator.",
    opts.recognizedText ? `Message conveyed via sign language: "${opts.recognizedText}".` : "Immediate assistance needed.",
    opts.mapsUrl ? `My current location: ${opts.mapsUrl}` : "Location unavailable.",
  ];
  return parts.join(" ");
}

export function buildTelLink(number: string): string {
  return `tel:${number.replace(/\s+/g, "")}`;
}

export function buildSmsLink(number: string, body: string): string {
  return `sms:${number.replace(/\s+/g, "")}?body=${encodeURIComponent(body)}`;
}
