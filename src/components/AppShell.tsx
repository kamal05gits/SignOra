import { NavLink, Outlet } from "react-router-dom";
import {
  Hand,
  Video,
  Database,
  Cpu,
  Siren,
  Settings as SettingsIcon,
  Info,
} from "lucide-react";
import { cn } from "../utils/cn";

const NAV_ITEMS = [
  { to: "/", label: "Live Translator", icon: Video, end: true },
  { to: "/dataset", label: "Dataset Collection", icon: Database },
  { to: "/model", label: "Recognition Engine", icon: Cpu },
  { to: "/emergency", label: "Emergency SOS", icon: Siren, danger: true },
  { to: "/settings", label: "Settings", icon: SettingsIcon },
  { to: "/about", label: "About Project", icon: Info },
];

export function AppShell() {
  return (
    <div className="flex min-h-screen bg-slate-50 text-slate-900">
      <aside className="hidden w-64 shrink-0 flex-col border-r border-slate-200 bg-white px-4 py-6 md:flex">
        <div className="mb-8 flex items-center gap-3 px-2">
          <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-gradient-to-br from-teal-500 to-emerald-600 text-white shadow-md shadow-teal-200">
            <Hand className="h-6 w-6" />
          </div>
          <div>
            <p className="text-sm font-bold leading-tight text-slate-900">ISL Translator</p>
            <p className="text-[11px] leading-tight text-slate-400">Real-Time Sign Language AI</p>
          </div>
        </div>
        <nav className="flex flex-1 flex-col gap-1">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) =>
                cn(
                  "flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition-colors",
                  isActive
                    ? item.danger
                      ? "bg-rose-50 text-rose-700"
                      : "bg-teal-50 text-teal-700"
                    : "text-slate-500 hover:bg-slate-100 hover:text-slate-800"
                )
              }
            >
              <item.icon className="h-4.5 w-4.5" />
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className="rounded-xl bg-slate-50 p-3 text-[11px] leading-relaxed text-slate-400">
          Final-year CSE academic project. All vision processing runs locally in
          your browser — no video is uploaded.
        </div>
      </aside>

      <div className="flex min-h-screen flex-1 flex-col">
        <MobileNav />
        <main className="flex-1 px-4 py-6 md:px-8 md:py-8">
          <Outlet />
        </main>
      </div>
    </div>
  );
}

function MobileNav() {
  return (
    <div className="flex items-center justify-between border-b border-slate-200 bg-white px-4 py-3 md:hidden">
      <div className="flex items-center gap-2">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-teal-500 to-emerald-600 text-white">
          <Hand className="h-4 w-4" />
        </div>
        <span className="text-sm font-bold">ISL Translator</span>
      </div>
      <nav className="flex gap-1 overflow-x-auto">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.end}
            className={({ isActive }) =>
              cn(
                "flex h-8 w-8 items-center justify-center rounded-lg",
                isActive ? "bg-teal-50 text-teal-700" : "text-slate-400"
              )
            }
            title={item.label}
          >
            <item.icon className="h-4 w-4" />
          </NavLink>
        ))}
      </nav>
    </div>
  );
}
