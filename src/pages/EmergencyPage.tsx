import { useEffect, useState } from "react";
import { Siren, Phone, MessageSquareText, MapPin, Plus, Trash2, Volume2, Globe2 } from "lucide-react";
import { useAppStore } from "../store/useAppStore";
import { Badge, Button, Card, CardHeader, EmptyState, Select } from "../components/ui";
import {
  buildEmergencyMessage,
  buildSmsLink,
  buildTelLink,
  fetchEmergencyNumbers,
  getCurrentLocation,
  type EmergencyNumbers,
  type GeoResult,
} from "../lib/emergency";
import { speak } from "../lib/speech";
import type { EmergencyContact } from "../types";

const COUNTRIES = [
  { code: "IN", label: "India" },
  { code: "US", label: "United States" },
  { code: "GB", label: "United Kingdom" },
];

export default function EmergencyPage() {
  const { contacts, addContact, removeContact, sentence, settings } = useAppStore();
  const [country, setCountry] = useState("IN");
  const [numbers, setNumbers] = useState<EmergencyNumbers | null>(null);
  const [loadingNumbers, setLoadingNumbers] = useState(false);
  const [geo, setGeo] = useState<GeoResult | null>(null);
  const [locating, setLocating] = useState(false);
  const [form, setForm] = useState({ name: "", phone: "", relation: "" });

  useEffect(() => {
    setLoadingNumbers(true);
    fetchEmergencyNumbers(country)
      .then(setNumbers)
      .finally(() => setLoadingNumbers(false));
  }, [country]);

  const sentenceText = sentence();
  const message = buildEmergencyMessage({ recognizedText: sentenceText, mapsUrl: geo?.mapsUrl });

  async function handleLocate() {
    setLocating(true);
    const result = await getCurrentLocation();
    setGeo(result);
    setLocating(false);
  }

  function handleAddContact(e: React.FormEvent) {
    e.preventDefault();
    if (!form.name.trim() || !form.phone.trim()) return;
    const contact: EmergencyContact = {
      id: `${Date.now()}`,
      name: form.name.trim(),
      phone: form.phone.trim(),
      relation: form.relation.trim() || "Contact",
    };
    addContact(contact);
    setForm({ name: "", phone: "", relation: "" });
  }

  function speakAlert() {
    speak(message, { rate: settings.speechRate, onError: console.warn });
  }

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-6">
      <header className="flex items-center gap-3">
        <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-rose-100 text-rose-600">
          <Siren className="h-6 w-6" />
        </div>
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Emergency SOS</h1>
          <p className="text-sm text-slate-500">
            Quick access to real emergency numbers, your location, and your personal emergency contacts.
          </p>
        </div>
      </header>

      <Card className="border-rose-200 bg-rose-50/60 p-4 text-sm text-rose-800">
        <p className="font-semibold">Scope disclosure</p>
        <p className="mt-1 leading-relaxed text-rose-700">
          This is a browser-only build with no paid backend SMS/voice gateway. "Send" actions open your device's own
          phone/SMS app with the number and message pre-filled (via <code>tel:</code> / <code>sms:</code> links) — they
          do not silently dispatch messages on their own. Emergency numbers are fetched from the free{" "}
          <a className="underline" href="https://emergencynumberapi.com" target="_blank" rel="noreferrer">
            Emergency Number API
          </a>{" "}
          with a verified offline fallback table if the network is unavailable.
        </p>
      </Card>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader
            title="National Emergency Numbers"
            icon={<Globe2 className="h-4 w-4" />}
            action={
              <Select className="w-40" value={country} onChange={(e) => setCountry(e.target.value)}>
                {COUNTRIES.map((c) => (
                  <option key={c.code} value={c.code}>
                    {c.label}
                  </option>
                ))}
              </Select>
            }
          />
          <div className="p-4">
            {loadingNumbers && <p className="text-sm text-slate-400">Fetching emergency numbers…</p>}
            {!loadingNumbers && numbers && (
              <>
                <div className="mb-2 flex items-center gap-2">
                  <Badge tone={numbers.source === "api" ? "emerald" : "amber"}>
                    {numbers.source === "api" ? "Live from Emergency Number API" : "Offline verified fallback"}
                  </Badge>
                </div>
                <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                  {[
                    { label: "Police", nums: numbers.police },
                    { label: "Ambulance", nums: numbers.ambulance },
                    { label: "Fire", nums: numbers.fire },
                  ].map((group) => (
                    <div key={group.label} className="rounded-xl border border-slate-200 p-3">
                      <p className="mb-2 text-xs font-semibold uppercase text-slate-400">{group.label}</p>
                      {group.nums.length === 0 && <p className="text-xs text-slate-400">Not available</p>}
                      <div className="flex flex-wrap gap-2">
                        {group.nums.map((n) => (
                          <a
                            key={n}
                            href={buildTelLink(n)}
                            className="inline-flex items-center gap-1 rounded-lg bg-rose-600 px-3 py-1.5 text-sm font-semibold text-white hover:bg-rose-700"
                          >
                            <Phone className="h-3.5 w-3.5" /> {n}
                          </a>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        </Card>

        <Card>
          <CardHeader title="Your Location" icon={<MapPin className="h-4 w-4" />} />
          <div className="space-y-3 p-4">
            <Button className="w-full" variant="outline" onClick={handleLocate} disabled={locating}>
              <MapPin className="h-4 w-4" /> {locating ? "Locating…" : "Share My Location"}
            </Button>
            {geo?.success && (
              <a href={geo.mapsUrl} target="_blank" rel="noreferrer" className="block text-xs text-teal-700 underline">
                View on Google Maps ({geo.latitude?.toFixed(4)}, {geo.longitude?.toFixed(4)})
              </a>
            )}
            {geo && !geo.success && <p className="text-xs text-rose-600">{geo.error}</p>}
          </div>
        </Card>
      </div>

      <Card>
        <CardHeader title="Alert Message Preview" subtitle="Includes your last recognized sentence and location, when available" icon={<MessageSquareText className="h-4 w-4" />} />
        <div className="space-y-3 p-4">
          <div className="rounded-xl border border-slate-200 bg-slate-50 p-3 text-sm text-slate-700">{message}</div>
          <Button variant="outline" onClick={speakAlert}>
            <Volume2 className="h-4 w-4" /> Speak alert message aloud
          </Button>
        </div>
      </Card>

      <Card>
        <CardHeader title="Emergency Contacts" icon={<Siren className="h-4 w-4" />} />
        <div className="space-y-4 p-4">
          <form onSubmit={handleAddContact} className="grid grid-cols-1 gap-2 sm:grid-cols-4">
            <input
              value={form.name}
              onChange={(e) => setForm({ ...form, name: e.target.value })}
              placeholder="Name"
              className="rounded-lg border border-slate-300 px-3 py-2 text-sm outline-none focus:border-teal-500"
            />
            <input
              value={form.phone}
              onChange={(e) => setForm({ ...form, phone: e.target.value })}
              placeholder="Phone number"
              className="rounded-lg border border-slate-300 px-3 py-2 text-sm outline-none focus:border-teal-500"
            />
            <input
              value={form.relation}
              onChange={(e) => setForm({ ...form, relation: e.target.value })}
              placeholder="Relation (e.g. Mother)"
              className="rounded-lg border border-slate-300 px-3 py-2 text-sm outline-none focus:border-teal-500"
            />
            <Button type="submit">
              <Plus className="h-4 w-4" /> Add contact
            </Button>
          </form>

          {contacts.length === 0 ? (
            <EmptyState title="No emergency contacts saved" description="Add a trusted contact above to enable one-tap call/SMS alerts." />
          ) : (
            <div className="space-y-2">
              {contacts.map((c) => (
                <div key={c.id} className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-slate-200 p-3">
                  <div>
                    <p className="text-sm font-semibold text-slate-800">
                      {c.name} <span className="font-normal text-slate-400">· {c.relation}</span>
                    </p>
                    <p className="text-xs text-slate-500">{c.phone}</p>
                  </div>
                  <div className="flex gap-2">
                    <a href={buildTelLink(c.phone)} className="rounded-lg bg-emerald-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-emerald-700">
                      <Phone className="mr-1 inline h-3 w-3" /> Call
                    </a>
                    <a
                      href={buildSmsLink(c.phone, message)}
                      className="rounded-lg bg-teal-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-teal-700"
                    >
                      <MessageSquareText className="mr-1 inline h-3 w-3" /> Send SMS alert
                    </a>
                    <button onClick={() => removeContact(c.id)} className="text-slate-300 hover:text-rose-500">
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </Card>
    </div>
  );
}
