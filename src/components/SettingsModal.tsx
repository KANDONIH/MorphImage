import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { getVersion } from "@tauri-apps/api/app";
import { openUrl } from "@tauri-apps/plugin-opener";
import { Check, Copy, Cpu, FolderOpen, Info, Loader2, RefreshCw, RotateCcw, Settings2, X } from "lucide-react";
import { t, useLang, setLang, type Lang } from "../i18n";

export type Theme = "liquid-glass" | "sky-frost" | "system" | "light" | "dark";
export type ExecMode = "auto" | "gpu" | "cpu";

/** Settings persisted under `morphimage_settings`; theme and language have their own keys. */
export interface AppSettings {
  autoOpen: boolean;
  naming: "suffix" | "overwrite";
  execMode: ExecMode;
  /** null = auto (logical cores - 2). */
  workerThreads: number | null;
}

export const DEFAULT_SETTINGS: AppSettings = { autoOpen: false, naming: "suffix", execMode: "auto", workerThreads: null };
export const DEFAULT_THEME: Theme = "liquid-glass";
const THEME_KEY = "morphimage_theme";
const SETTINGS_KEY = "morphimage_settings";
const THEMES: Theme[] = ["liquid-glass", "sky-frost", "system", "light", "dark"];

// ponytail: repo is a constant; move to a settings field if forks need their own update channel.
export const RELEASES_API = "https://api.github.com/repos/kando/MorphImage/releases/latest";

export function loadTheme(): Theme {
  try {
    const v = localStorage.getItem(THEME_KEY) as Theme | null;
    return v && THEMES.includes(v) ? v : DEFAULT_THEME;
  } catch {
    return DEFAULT_THEME;
  }
}

export function saveTheme(v: Theme) {
  try {
    localStorage.setItem(THEME_KEY, v);
  } catch {
    /* in-memory only */
  }
}

export function loadSettings(): AppSettings {
  try {
    return { ...DEFAULT_SETTINGS, ...JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? "{}") };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

export function saveSettings(s: AppSettings) {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(s));
  } catch {
    /* in-memory only */
  }
}

/** True when `latest` (e.g. "v0.2.0") is newer than `current` ("0.1.0"). */
export function isNewer(latest: string, current: string): boolean {
  const p = (v: string) => v.replace(/^v/i, "").split(/[.-]/).map((n) => parseInt(n, 10) || 0);
  const [a, b] = [p(latest), p(current)];
  for (let i = 0; i < Math.max(a.length, b.length); i++) {
    if ((a[i] ?? 0) !== (b[i] ?? 0)) return (a[i] ?? 0) > (b[i] ?? 0);
  }
  return false;
}

interface GpuInfo {
  available: boolean;
  gpu_name: string;
  vram_mb: number;
  backend: string;
}

interface Props {
  settings: AppSettings;
  onSettings: (s: AppSettings) => void;
  theme: Theme;
  onTheme: (t: Theme) => void;
  /** Live backend label from get_system_stats. */
  aiBackend: string;
  onReset: () => void;
  onClose: () => void;
}

type Tab = "general" | "ai" | "about";

type UpdateState =
  | { kind: "idle" }
  | { kind: "checking" }
  | { kind: "latest" }
  | { kind: "available"; version: string; url: string }
  | { kind: "error"; msg: string };

const CARD = "space-y-3 rounded-xl border border-line bg-panel p-4";
const H3 = "text-xs font-semibold uppercase tracking-wider text-muted";
const BTN =
  "inline-flex h-8 items-center justify-center gap-1.5 rounded-md border border-line px-3 text-xs font-medium text-fg transition-colors hover:bg-hover disabled:opacity-40";
const chip = (on: boolean) =>
  `rounded-lg border px-3 py-2 text-left text-xs transition-colors ${
    on ? "border-accent bg-accent-soft font-medium text-accent-text" : "border-line text-fg hover:bg-hover"
  }`;

function formatBytes(b: number): string {
  if (b >= 1_073_741_824) return (b / 1_073_741_824).toFixed(2) + " GB";
  if (b >= 1_048_576) return (b / 1_048_576).toFixed(1) + " MB";
  return (b / 1024).toFixed(0) + " KB";
}

export default function SettingsModal({ settings, onSettings, theme, onTheme, aiBackend, onReset, onClose }: Props) {
  const lang = useLang();
  const [tab, setTab] = useState<Tab>("general");
  const closeRef = useRef<HTMLButtonElement>(null);
  const set = (patch: Partial<AppSettings>) => onSettings({ ...settings, ...patch });

  useEffect(() => {
    closeRef.current?.focus();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  // Hardware & storage: loaded once when the modal opens.
  const [cores, setCores] = useState<number | null>(null);
  const [gpu, setGpu] = useState<GpuInfo | null>(null);
  const [storage, setStorage] = useState<{ path: string; total_bytes: number; file_count: number } | null>(null);
  const [version, setVersion] = useState("");
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    invoke<number>("get_logical_cores").then(setCores).catch(() => {});
    invoke<GpuInfo>("get_gpu_info").then(setGpu).catch(() => {});
    invoke<typeof storage>("get_models_storage_info").then(setStorage).catch((e) => setErr(String(e)));
    getVersion().then(setVersion).catch(() => {});
  }, []);
  const autoThreads = Math.max(1, (cores ?? 3) - 2);

  const [update, setUpdate] = useState<UpdateState>({ kind: "idle" });
  async function checkUpdates() {
    setUpdate({ kind: "checking" });
    try {
      const r = await fetch(RELEASES_API, { headers: { Accept: "application/vnd.github+json" } });
      if (r.status === 404) return setUpdate({ kind: "error", msg: t("No releases published yet.") });
      if (!r.ok) return setUpdate({ kind: "error", msg: t("GitHub returned HTTP {code}.", { code: r.status }) });
      const j = (await r.json()) as { tag_name?: string; html_url?: string };
      const tag = j.tag_name ?? "";
      setUpdate(
        version && isNewer(tag, version)
          ? { kind: "available", version: tag, url: j.html_url ?? "" }
          : { kind: "latest" },
      );
    } catch {
      // fetch rejects only on network failure (offline, DNS, blocked).
      setUpdate({ kind: "error", msg: t("You appear to be offline. Check your connection and try again.") });
    }
  }

  const [copied, setCopied] = useState(false);
  function copyDiagnostics() {
    const text = [
      `MorphImage ${version || "?"}`,
      `GPU: ${gpu?.gpu_name ?? "?"}`,
      `VRAM: ${gpu?.vram_mb ?? 0} MB`,
      `CPU threads: ${cores ?? "?"}`,
      `Active engine: ${aiBackend}`,
      `Execution mode: ${settings.execMode}`,
      `Worker threads: ${settings.workerThreads ?? `auto (${autoThreads})`}`,
      `OS: ${navigator.userAgent}`,
    ].join("\n");
    void navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }

  const TABS: { key: Tab; label: string; icon: React.ReactNode }[] = [
    { key: "general", label: t("General"), icon: <Settings2 size={14} /> },
    { key: "ai", label: t("AI & Performance"), icon: <Cpu size={14} /> },
    { key: "about", label: t("About & Diagnostics"), icon: <Info size={14} /> },
  ];
  const THEME_LABEL: Record<Theme, [string, string]> = {
    "liquid-glass": [t("Liquid Glass"), t("Translucent glass over an ambient gradient")],
    "sky-frost": [t("Sky Frost"), t("Pastel sky-blue glass with azure accents")],
    system: [t("System"), t("Follows your OS setting")],
    light: [t("Light"), t("Clean and bright")],
    dark: [t("Dark"), t("High contrast")],
  };
  const SWATCH: Record<Theme, string> = {
    "liquid-glass": "bg-[radial-gradient(at_20%_20%,#60a5fa,transparent_60%),radial-gradient(at_80%_80%,#22d3ee,transparent_60%),#a5b4fc]",
    "sky-frost": "bg-[radial-gradient(at_10%_0%,#7dd3fc_0,transparent_55%),linear-gradient(135deg,#e0f2fe,#dbeafe_50%,#f0f9ff)]",
    system: "bg-[linear-gradient(135deg,#f8fafc_50%,#09090b_50%)]",
    light: "bg-[#f8fafc]",
    dark: "bg-[#09090b]",
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="settings-title"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      onClick={onClose}
    >
      <div
        className="modal-surface flex max-h-[90vh] w-full max-w-2xl flex-col rounded-2xl border border-line bg-bg shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-2 border-b border-line px-5 py-3">
          <Settings2 size={16} className="text-accent" />
          <h2 id="settings-title" className="text-sm font-semibold">
            {t("Settings")}
          </h2>
          <button
            ref={closeRef}
            onClick={onClose}
            aria-label={t("Close settings")}
            className="ml-auto inline-flex h-7 w-7 items-center justify-center rounded-md text-muted hover:bg-hover hover:text-fg"
          >
            <X size={15} />
          </button>
        </div>

        <div role="tablist" aria-label={t("Settings")} className="flex gap-1 border-b border-line px-3 py-2">
          {TABS.map((x) => (
            <button
              key={x.key}
              role="tab"
              id={`stab-${x.key}`}
              aria-selected={tab === x.key}
              aria-controls={`spanel-${x.key}`}
              onClick={() => setTab(x.key)}
              className={`inline-flex h-8 items-center gap-1.5 rounded-md px-3 text-xs transition-colors ${
                tab === x.key ? "bg-accent-soft font-medium text-accent-text" : "text-muted hover:bg-hover hover:text-fg"
              }`}
            >
              {x.icon}
              {x.label}
            </button>
          ))}
        </div>

        <div
          key={tab}
          role="tabpanel"
          id={`spanel-${tab}`}
          aria-labelledby={`stab-${tab}`}
          className="tab-fade-in space-y-4 overflow-y-auto p-5"
        >
          {err && (
            <p role="alert" className="rounded-md border border-bad/40 bg-bad/10 px-3 py-2 text-xs text-bad">
              {err}
            </p>
          )}

          {tab === "general" && (
            <>
              <section className={CARD}>
                <h3 className={H3}>{t("Language")}</h3>
                <div className="grid grid-cols-2 gap-2">
                  {(
                    [
                      ["en", "English (Default)"],
                      ["ms", "Bahasa Melayu"],
                    ] as [Lang, string][]
                  ).map(([k, label]) => (
                    <button key={k} onClick={() => setLang(k)} aria-pressed={lang === k} className={chip(lang === k)}>
                      {label}
                    </button>
                  ))}
                </div>
              </section>

              <section className={CARD}>
                <h3 className={H3}>{t("Appearance")}</h3>
                <div className="grid grid-cols-2 gap-2 sm:grid-cols-5">
                  {THEMES.map((k) => (
                    <button key={k} onClick={() => onTheme(k)} aria-pressed={theme === k} className={chip(theme === k)}>
                      <span aria-hidden className={`mb-2 block h-10 rounded-md border border-line ${SWATCH[k]}`} />
                      <span className="block font-medium">{THEME_LABEL[k][0]}</span>
                      <span className="block text-[11px] text-muted">{THEME_LABEL[k][1]}</span>
                    </button>
                  ))}
                </div>
              </section>

              <section className={CARD}>
                <h3 className={H3}>{t("Output Behavior")}</h3>
                <div className="flex items-center justify-between gap-3">
                  <span id="auto-open-label" className="text-sm">
                    {t("Auto-open output folder after processing")}
                  </span>
                  <button
                    role="switch"
                    aria-checked={settings.autoOpen}
                    aria-labelledby="auto-open-label"
                    onClick={() => set({ autoOpen: !settings.autoOpen })}
                    className={`relative h-5 w-9 shrink-0 rounded-full transition-colors ${settings.autoOpen ? "bg-accent" : "bg-line"}`}
                  >
                    <span
                      className={`absolute left-0.5 top-0.5 h-4 w-4 rounded-full bg-white shadow-sm transition-transform ${
                        settings.autoOpen ? "translate-x-4" : ""
                      }`}
                    />
                  </button>
                </div>
                <fieldset className="space-y-2">
                  <legend className="mb-1.5 text-sm">{t("File naming")}</legend>
                  <div className="grid grid-cols-2 gap-2">
                    <button onClick={() => set({ naming: "suffix" })} aria-pressed={settings.naming === "suffix"} className={chip(settings.naming === "suffix")}>
                      <span className="block font-medium">{t("Add _morph suffix")}</span>
                      <span className="block text-[11px] text-muted">photo.jpg → photo_morph.jpg</span>
                    </button>
                    <button onClick={() => set({ naming: "overwrite" })} aria-pressed={settings.naming === "overwrite"} className={chip(settings.naming === "overwrite")}>
                      <span className="block font-medium">{t("Overwrite original file")}</span>
                      <span className="block text-[11px] text-muted">{t("Replaces the source. Cannot be undone.")}</span>
                    </button>
                  </div>
                </fieldset>
              </section>
            </>
          )}

          {tab === "ai" && (
            <>
              <section className={CARD}>
                <h3 className={H3}>{t("AI Execution Mode Override")}</h3>
                <div className="grid gap-2 sm:grid-cols-3">
                  {(
                    [
                      ["auto", t("Auto (4GB VRAM Safe Mode)"), t("Heavy models run on CPU when VRAM ≤ 4 GB.")],
                      ["gpu", t("Force GPU (DirectML)"), t("Skips Safe Mode. May reset the GPU on low VRAM.")],
                      ["cpu", t("Force CPU"), t("Slowest, but never touches the GPU.")],
                    ] as [ExecMode, string, string][]
                  ).map(([k, label, hint]) => (
                    <button key={k} onClick={() => set({ execMode: k })} aria-pressed={settings.execMode === k} className={chip(settings.execMode === k)}>
                      <span className="block font-medium">{label}</span>
                      <span className="block text-[11px] text-muted">{hint}</span>
                    </button>
                  ))}
                </div>
              </section>

              <section className={CARD}>
                <h3 className={H3}>{t("CPU Worker Threads")}</h3>
                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={settings.workerThreads === null}
                    onChange={(e) => set({ workerThreads: e.target.checked ? null : autoThreads })}
                    className="h-4 w-4 accent-accent"
                  />
                  {t("Auto (Total Threads − 2) = {n}", { n: autoThreads })}
                </label>
                <label className={`block space-y-1.5 ${settings.workerThreads === null ? "opacity-40" : ""}`}>
                  <span className="flex justify-between text-xs">
                    <span>{t("Manual threads")}</span>
                    <span className="tabular-nums text-accent-text">
                      {settings.workerThreads ?? autoThreads} / {cores ?? "—"}
                    </span>
                  </span>
                  <input
                    type="range"
                    min={1}
                    max={cores ?? 1}
                    value={settings.workerThreads ?? autoThreads}
                    disabled={settings.workerThreads === null || !cores}
                    onChange={(e) => set({ workerThreads: Number(e.target.value) })}
                  />
                </label>
                {settings.workerThreads !== null && cores !== null && settings.workerThreads > cores - 2 && (
                  <p className="text-xs text-amber-600 dark:text-amber-400">
                    {t("Using all cores can make Windows unresponsive during large batches.")}
                  </p>
                )}
              </section>

              <section className={CARD}>
                <h3 className={H3}>{t("Model Storage")}</h3>
                <div className="flex items-center justify-between gap-3">
                  <div className="min-w-0">
                    <p className="text-lg font-semibold tabular-nums">
                      {storage ? formatBytes(storage.total_bytes) : "—"}
                    </p>
                    <p className="truncate text-xs text-muted" title={storage?.path}>
                      {storage ? t("{n} files · {path}", { n: storage.file_count, path: storage.path }) : t("Loading…")}
                    </p>
                  </div>
                  <button
                    onClick={() => void invoke("open_models_folder").catch((e) => setErr(String(e)))}
                    className={BTN + " shrink-0"}
                  >
                    <FolderOpen size={14} />
                    {t("Open Models Folder")}
                  </button>
                </div>
              </section>
            </>
          )}

          {tab === "about" && (
            <>
              <section className={CARD}>
                <h3 className={H3}>{t("Version")}</h3>
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <p className="text-sm">
                    MorphImage <b className="tabular-nums">v{version || "—"}</b>
                  </p>
                  <button onClick={() => void checkUpdates()} disabled={update.kind === "checking"} className={BTN}>
                    {update.kind === "checking" ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
                    {t("Check for Updates")}
                  </button>
                </div>
                <div aria-live="polite" className="text-xs">
                  {update.kind === "latest" && <p className="text-ok">{t("You're on the latest version.")}</p>}
                  {update.kind === "error" && <p className="text-bad">{update.msg}</p>}
                  {update.kind === "available" && (
                    <p className="text-accent-text">
                      {t("Version {v} is available.", { v: update.version })}{" "}
                      {update.url && (
                        <button onClick={() => void openUrl(update.url).catch(() => {})} className="font-medium underline">
                          {t("View release")}
                        </button>
                      )}
                    </p>
                  )}
                </div>
              </section>

              <section className={CARD}>
                <h3 className={H3}>{t("Hardware Diagnostics")}</h3>
                <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-sm">
                  <dt className="text-muted">{t("GPU")}</dt>
                  <dd className="truncate" title={gpu?.gpu_name}>{gpu?.gpu_name ?? "—"}</dd>
                  <dt className="text-muted">{t("VRAM detected")}</dt>
                  <dd className="tabular-nums">{gpu ? (gpu.vram_mb > 0 ? `${gpu.vram_mb} MB` : "—") : "—"}</dd>
                  <dt className="text-muted">{t("CPU threads")}</dt>
                  <dd className="tabular-nums">{cores ?? "—"}</dd>
                  <dt className="text-muted">{t("Active engine")}</dt>
                  <dd>{aiBackend === "Idle" ? t("Idle") : aiBackend}</dd>
                </dl>
                <div className="flex flex-wrap gap-2 pt-1">
                  <button onClick={copyDiagnostics} className={BTN}>
                    {copied ? <Check size={14} className="text-ok" /> : <Copy size={14} />}
                    {copied ? t("Copied") : t("Copy Diagnostics")}
                  </button>
                  <button onClick={onReset} className={BTN + " text-bad"}>
                    <RotateCcw size={14} />
                    {t("Reset All Settings to Default")}
                  </button>
                </div>
              </section>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
