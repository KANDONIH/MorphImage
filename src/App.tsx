import { useState, useEffect, useRef, useMemo, useCallback } from "react";
import { invoke, convertFileSrc } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { open, save, ask } from "@tauri-apps/plugin-dialog";
import AiGuideModal from "./components/AiGuideModal";
import SettingsModal, {
  DEFAULT_SETTINGS,
  DEFAULT_THEME,
  loadSettings,
  loadTheme,
  saveSettings,
  saveTheme,
  type AppSettings,
  type Theme,
} from "./components/SettingsModal";
import { t, useLang, setLang } from "./i18n";
import ModelSelect from "./components/ModelSelect";
import {
  BookOpen,
  Plus,
  ImageIcon,
  FolderOpen,
  Layers,
  Zap,
  Trash2,
  UploadCloud,
  Check,
  AlertTriangle,
  Loader2,
  Columns2,
  X,
  ClipboardPaste,
  Copy,
  RotateCcw,
  RotateCw,
  FlipHorizontal,
  FlipVertical,
  Crop,
  FileText,
  Settings as SettingsIcon,
  HardDrive,
  TrendingDown,
} from "lucide-react";

interface ImageInfo {
  path: string;
  name: string;
  size_bytes: number;
  width: number;
  height: number;
  format: string;
}

interface ProcessResult {
  source_path: string;
  output_path: string;
  original_size: number;
  new_size: number;
  saved_percent: number;
  success: boolean;
  error: string | null;
}

interface BatchProgressEvent {
  index: number;
  total: number;
  result: ProcessResult;
}

/** Normalized crop region (0.0–1.0 of the oriented source image). */
interface CropRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

interface SystemStats {
  cpu_usage: number;
  app_memory_mb: number;
  active_threads: number;
  total_threads: number;
  gpu_name: string;
  gpu_usage: number;
  gpu_vram_used_mb: number;
  gpu_active: boolean;
  ai_backend: string;
}

type OutputFormat =
  | "Original" | "Webp" | "Avif" | "Jpeg" | "Png" | "Ico" | "Tiff" | "Bmp" | "Gif";
type WidthChoice = "orig" | "1920" | "1280" | "800" | "custom";

type PanelTab = "format" | "size" | "ai";
const TABS: { key: PanelTab; label: string }[] = [
  { key: "format", label: "Format" },
  { key: "size", label: "Size" },
  { key: "ai", label: "AI Studio" },
];
const TAB_ORDER: PanelTab[] = TABS.map((t) => t.key);

const INPUT_EXTS = [
  "jpg", "jpeg", "png", "webp", "gif", "tiff", "tif", "bmp", "svg", "ico", "jxl", "psd",
  "cr2", "cr3", "nef", "arw", "dng", "raw",
];

/** Format yang WebView tak boleh render sebagai <img> — tunjuk ikon fail. */
const NO_THUMB = ["psd", "jxl", "cr2", "cr3", "nef", "arw", "dng", "raw"];

/** Sama urutan dengan enjin Rust: putar → cermin mendatar → flip menegak. */
function previewTransform(deg: number, fh: boolean, fv: boolean): string {
  return `scaleX(${fh ? -1 : 1}) scaleY(${fv ? -1 : 1}) rotate(${deg}deg)`;
}

const FORMAT_GROUPS: { label: string; items: { value: OutputFormat; label: string }[] }[] = [
  {
    label: "Web & Modern",
    items: [
      { value: "Original", label: "Original (Auto)" },
      { value: "Webp", label: "WebP" },
      { value: "Avif", label: "AVIF" },
      { value: "Jpeg", label: "JPEG" },
      { value: "Png", label: "PNG" },
    ],
  },
  {
    label: "Icons & Publishing",
    items: [
      { value: "Ico", label: "ICO (Favicon / Windows)" },
      { value: "Tiff", label: "TIFF" },
      { value: "Bmp", label: "BMP" },
      { value: "Gif", label: "GIF" },
    ],
  },
];

const FORMAT_HINTS: Partial<Record<OutputFormat, string>> = {
  Original: "Keeps the original format. RAW/PSD/SVG/JXL become PNG (transparent) or JPEG",
  Webp: "Small size, ideal for the web",
  Avif: "Best compression, slower encoding",
  Jpeg: "Works everywhere; no transparency",
  Ico: "Auto-resized to 256×256 for icons & favicons",
  Tiff: "Lossless, ideal for print; large files",
  Bmp: "Uncompressed; very large files",
  Gif: "Max 256 colors; no animation",
};

/** Formats where the quality slider has no effect. */
const NO_QUALITY: OutputFormat[] = ["Png", "Ico", "Tiff", "Bmp", "Gif"];

type PresetKey = "custom" | "web" | "ecom" | "form";

type UpscaleModelId = "x4v3" | "x4plus" | "ultrasharp" | "anime-6b";
const UPSCALE_MODELS: Record<UpscaleModelId, { label: string; hint: string; mb: number }> = {
  x4v3: { label: "Fast (x4v3)", hint: "Fast & versatile.", mb: 5 },
  x4plus: { label: "Photo Pro (x4+)", hint: "Real photos & complex textures.", mb: 67 },
  ultrasharp: { label: "UltraSharp", hint: "Maximum sharpness for blurry images & text.", mb: 67 },
  "anime-6b": { label: "Anime / Graphics (6B)", hint: "Illustrations, anime & logos.", mb: 18 },
};

type BgModelId = "rmbg-1.4" | "birefnet-lite";
const BG_MODELS: Record<BgModelId, { label: string; hint: string; mb: number }> = {
  "rmbg-1.4": { label: "Fast (RMBG-1.4)", hint: "Light & fast (~44 MB).", mb: 44 },
  "birefnet-lite": {
    label: "Ultra (BiRefNet Lite)",
    hint: "Sharpest hair & edge cutouts (~115 MB • MIT).",
    mb: 115,
  },
};

/** Model .onnx yang diimport pengguna (indeks custom_models.json di Rust). */
interface CustomModel {
  id: string;
  kind: "upscale" | "bg";
  name: string;
  file_name: string;
  scale?: number;
  input_size?: number;
  norm_type: "standard" | "imagenet";
  size_mb: number;
}

/** Model pulih wajah diimport (indeks face_models.json di Rust). */
interface FaceModel {
  id: string;
  name: string;
  file_name: string;
  size_mb: number;
  dtype: "float16" | "float32" | "float64";
  has_fidelity: boolean;
}

/** Sama dengan had Rust: model diimport > 60 MB pada GPU ≤ 4608 MB berjalan di CPU. */
const SAFE_VRAM_MB = 4608;
const PIP_CMD = "pip install torch spandrel onnx";

/** Dialog fail .onnx/.pth; .pth perlu pengesahan storan dahulu. Pulangkan null jika dibatalkan. */
async function pickModel(): Promise<string | null> {
  const path = await open({ multiple: false, filters: [{ name: t("AI Model (.onnx, .pth, .pt)"), extensions: ["onnx", "pth", "pt"] }] });
  if (typeof path !== "string") return null;
  if (!/\.(pth|pt)$/i.test(path)) return path;
  const ok = await ask(
    t("The .pth file will be converted to .onnx using Python on your PC. This needs the torch, spandrel and onnx libraries (~2.5 GB of storage if not yet installed). It may take about a minute. Continue?"),
    { title: t("Convert .pth Model"), kind: "warning", okLabel: t("Continue"), cancelLabel: t("Cancel") },
  );
  return ok ? path : null;
}
const HEAVY_MB = 60;

const PRESETS: Record<
  Exclude<PresetKey, "custom">,
  { format: OutputFormat; quality: number; width: WidthChoice; customWidth?: number; targetKb: string }
> = {
  web: { format: "Webp", quality: 80, width: "1920", targetKb: "" },
  ecom: { format: "Jpeg", quality: 85, width: "custom", customWidth: 1200, targetKb: "" },
  form: { format: "Jpeg", quality: 80, width: "1280", targetKb: "500" },
};

function formatSize(bytes: number): string {
  if (bytes >= 1_048_576) return (bytes / 1_048_576).toFixed(2) + " MB";
  return (bytes / 1024).toFixed(1) + " KB";
}

function dirname(p: string): string {
  return p.slice(0, Math.max(p.lastIndexOf("/"), p.lastIndexOf("\\")));
}

export default function App() {
  const [images, setImages] = useState<ImageInfo[]>([]);
  /** Path imej yang ditanda. Path basi diabaikan oleh `targets` di bawah. */
  const [selected, setSelected] = useState<Set<string>>(() => new Set());
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);

  useLang(); // re-render on language change

  // Theme (morphimage_theme) + app settings (morphimage_settings)
  const [theme, setTheme] = useState<Theme>(loadTheme);
  const [settings, setSettings] = useState<AppSettings>(loadSettings);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const closeSettings = useCallback(() => setSettingsOpen(false), []);
  const [logoOk, setLogoOk] = useState(true);
  useEffect(() => {
    saveTheme(theme);
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const apply = () => {
      const root = document.documentElement;
      // Liquid Glass and System both take light/dark from the OS; Sky Frost is always light.
      root.classList.toggle("dark", theme === "dark" || (theme !== "light" && theme !== "sky-frost" && mq.matches));
      root.classList.toggle("glass", theme === "liquid-glass");
      root.classList.toggle("frost", theme === "sky-frost");
    };
    apply();
    mq.addEventListener("change", apply);
    return () => mq.removeEventListener("change", apply);
  }, [theme]);
  useEffect(() => saveSettings(settings), [settings]);

  async function resetAllSettings() {
    if (!(await ask(t("Reset all settings to default?"), { title: t("Reset Settings"), kind: "warning" }))) return;
    setSettings(DEFAULT_SETTINGS);
    setTheme(DEFAULT_THEME);
    setLang("en");
  }

  // Tetapan
  const [outputFormat, setOutputFormat] = useState<OutputFormat>("Original");
  const [quality, setQuality] = useState(80);
  const [widthChoice, setWidthChoice] = useState<WidthChoice>("orig");
  const [customWidth, setCustomWidth] = useState(1600);
  const [stripMetadata, setStripMetadata] = useState(true);
  const [customOutDir, setCustomOutDir] = useState<string | null>(null);
  const [useCustomDir, setUseCustomDir] = useState(false);
  const [preset, setPreset] = useState<PresetKey>("custom");
  const [targetKb, setTargetKb] = useState(""); // kosong = tiada sasaran
  const [pngLossy, setPngLossy] = useState(false);
  const [suffixInput, setSuffixInput] = useState<string | null>(null); // null = auto
  const [useNumbering, setUseNumbering] = useState(false);
  const [compare, setCompare] = useState<ProcessResult | null>(null);
  const [activeTab, setActiveTab] = useState<PanelTab>("format");

  // Fasa 7B: papan keratan, putar/cermin, crop, PDF
  const [rotateDeg, setRotateDeg] = useState(0); // 0/90/180/270 (ikut arah jam)
  const [flipH, setFlipH] = useState(false);
  const [flipV, setFlipV] = useState(false);
  const [crops, setCrops] = useState<Record<string, CropRect>>({});
  const [cropTarget, setCropTarget] = useState<ImageInfo | null>(null);
  const [copiedPath, setCopiedPath] = useState<string | null>(null);
  const [pdfInfo, setPdfInfo] = useState<{ path: string; size: number } | null>(null);
  const [exporting, setExporting] = useState(false);

  // Upscaler
  const [upscaleFactor, setUpscaleFactor] = useState<2 | 4 | null>(null);
  const [upscaleMode, setUpscaleMode] = useState<"fast" | "ai">("fast");
  const [aiStrength, setAiStrength] = useState(85);
  const [gpu, setGpu] = useState<{ available: boolean; gpu_name: string; vram_mb: number; backend: string } | null>(null);
  const [modelReady, setModelReady] = useState<boolean | null>(null);
  const [modelPct, setModelPct] = useState<number | null>(null); // null = tidak memuat turun
  const [upscaleModel, setUpscaleModel] = useState<string>("x4v3");
  const [upscaleDownloading, setUpscaleDownloading] = useState<string | null>(null);
  const upscaleDownloadingRef = useRef<string | null>(null);
  const upscaleModelRef = useRef(upscaleModel);
  upscaleModelRef.current = upscaleModel;

  // Buang latar belakang
  const [removeBg, setRemoveBg] = useState(false);
  const [bgChoice, setBgChoice] = useState<"transparent" | "white" | "custom">("transparent");
  const [bgCustomColor, setBgCustomColor] = useState("#FFFFFF");
  const [bgModelReady, setBgModelReady] = useState<boolean | null>(null);
  const [bgModelPct, setBgModelPct] = useState<number | null>(null);
  const [bgModel, setBgModel] = useState<string>("rmbg-1.4");
  const [bgDownloading, setBgDownloading] = useState<string | null>(null);
  const bgDownloadingRef = useRef<string | null>(null);
  const bgModelRef = useRef(bgModel);
  bgModelRef.current = bgModel;

  // Model .onnx diimport + Buku Panduan
  const [customModels, setCustomModels] = useState<CustomModel[]>([]);
  const [importing, setImporting] = useState<"upscale" | "bg" | null>(null);
  const [bgImportNorm, setBgImportNorm] = useState<"standard" | "imagenet" | null>(null); // null = panel tertutup
  const [guideOpen, setGuideOpen] = useState(false);
  const closeGuide = useCallback(() => setGuideOpen(false), []);
  const customUp = customModels.filter((m) => m.kind === "upscale");
  const customBg = customModels.filter((m) => m.kind === "bg");
  const pickedCustomUp = customUp.find((m) => m.id === upscaleModel);
  const pickedCustomBg = customBg.find((m) => m.id === bgModel);
  const lowVram = !!gpu && gpu.vram_mb > 0 && gpu.vram_mb <= SAFE_VRAM_MB;

  // Pulih Wajah AI
  const [restoreFace, setRestoreFace] = useState(false);
  const [faceModels, setFaceModels] = useState<FaceModel[]>([]);
  const [faceModelId, setFaceModelId] = useState("");
  const [faceFidelity, setFaceFidelity] = useState(70); // 0–100
  const [faceImporting, setFaceImporting] = useState(false);
  const pickedFace = faceModels.find((m) => m.id === faceModelId);

  useEffect(() => {
    invoke<FaceModel[]>("list_face_models")
      .then((l) => {
        setFaceModels(l);
        setFaceModelId((id) => id || (l[0]?.id ?? ""));
      })
      .catch((e) => setError(String(e)));
  }, []);

  async function importFaceModel() {
    const path = await pickModel();
    if (!path) return;
    setFaceImporting(true);
    try {
      const m = await invoke<FaceModel>("import_face_model", { sourcePath: path });
      setFaceModels((l) => [...l, m]);
      setFaceModelId(m.id);
    } catch (e) {
      setError(String(e));
    } finally {
      setFaceImporting(false);
    }
  }

  async function deleteFaceModel(m: FaceModel) {
    if (!(await ask(t("Delete model \"{name}\"? Your original file is not affected.", { name: m.name }), { title: t("Delete Model"), kind: "warning" }))) return;
    try {
      await invoke("delete_face_model", { id: m.id });
      const rest = faceModels.filter((x) => x.id !== m.id);
      setFaceModels(rest);
      if (faceModelId === m.id) setFaceModelId(rest[0]?.id ?? "");
    } catch (e) {
      setError(String(e));
    }
  }

  useEffect(() => {
    invoke<CustomModel[]>("list_custom_models").then(setCustomModels).catch((e) => setError(String(e)));
  }, []);

  async function importModel(kind: "upscale" | "bg", norm?: "standard" | "imagenet") {
    const path = await pickModel();
    if (!path) return;
    setImporting(kind);
    try {
      const m =
        kind === "upscale"
          ? await invoke<CustomModel>("import_upscale_model", { sourcePath: path })
          : await invoke<CustomModel>("import_bg_model", { sourcePath: path, normPreset: norm });
      setCustomModels((l) => [...l, m]);
      if (kind === "upscale") setUpscaleModel(m.id);
      else {
        setBgModel(m.id);
        setBgImportNorm(null);
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setImporting(null);
    }
  }

  async function deleteModel(m: CustomModel) {
    if (!(await ask(t("Delete model \"{name}\"? Your original file is not affected.", { name: m.name }), { title: t("Delete Model"), kind: "warning" }))) return;
    try {
      await invoke("delete_custom_model", { id: m.id });
      setCustomModels((l) => l.filter((x) => x.id !== m.id));
      if (upscaleModelRef.current === m.id) setUpscaleModel("x4v3");
      if (bgModelRef.current === m.id) setBgModel("rmbg-1.4");
    } catch (e) {
      setError(String(e));
    }
  }

  // Hardware & model diperiksa apabila Mod AI dipilih
  useEffect(() => {
    if (upscaleFactor === null || upscaleMode !== "ai") return;
    if (!gpu) invoke<typeof gpu>("get_gpu_info").then(setGpu).catch((e) => setError(String(e)));
    let stale = false;
    // Model diimport sudah ada di cakera; hanya model bawaan perlu dimuat turun.
    if (!(upscaleModel in UPSCALE_MODELS)) {
      setModelReady(true);
      return;
    }
    setModelReady(null);
    invoke<boolean>("check_upscale_model", { model: upscaleModel })
      .then((ok) => !stale && setModelReady(ok))
      .catch((e) => setError(String(e)));
    return () => {
      stale = true;
    };
  }, [upscaleFactor, upscaleMode, gpu, upscaleModel]);

  useEffect(() => {
    const un = listen<{ model: string; percent: number }>("model-download-progress", (e) => {
      if (e.payload.model === upscaleDownloadingRef.current) setModelPct(e.payload.percent);
    });
    return () => void un.then((f) => f());
  }, []);

  // Model buang latar diperiksa apabila suis dihidupkan atau model ditukar
  useEffect(() => {
    if (!removeBg) return;
    if (!gpu) invoke<typeof gpu>("get_gpu_info").then(setGpu).catch((e) => setError(String(e)));
    let stale = false;
    if (!(bgModel in BG_MODELS)) {
      setBgModelReady(true);
      return;
    }
    setBgModelReady(null);
    invoke<boolean>("check_bg_model", { model: bgModel })
      .then((ok) => !stale && setBgModelReady(ok))
      .catch((e) => setError(String(e)));
    return () => {
      stale = true;
    };
  }, [removeBg, gpu, bgModel]);

  useEffect(() => {
    const un = listen<{ model: string; percent: number }>("bg-model-download-progress", (e) => {
      if (e.payload.model === bgDownloadingRef.current) setBgModelPct(e.payload.percent);
    });
    return () => void un.then((f) => f());
  }, []);

  async function downloadModel() {
    const model = upscaleModel;
    upscaleDownloadingRef.current = model;
    setUpscaleDownloading(model);
    setModelPct(0);
    try {
      await invoke("download_upscale_model", { model });
      if (upscaleModelRef.current === model) setModelReady(true);
    } catch (e) {
      setError(String(e));
    } finally {
      upscaleDownloadingRef.current = null;
      setUpscaleDownloading(null);
      setModelPct(null);
    }
  }

  async function downloadBgModel() {
    const model = bgModel;
    bgDownloadingRef.current = model;
    setBgDownloading(model);
    setBgModelPct(0);
    try {
      await invoke("download_bg_model", { model });
      if (bgModelRef.current === model) setBgModelReady(true);
    } catch (e) {
      setError(String(e));
    } finally {
      bgDownloadingRef.current = null;
      setBgDownloading(null);
      setBgModelPct(null);
    }
  }

  // Settings → "Add _morph suffix" default; user can still type their own.
  const suffix = suffixInput ?? "_morph";

  function applyPreset(key: PresetKey) {
    setPreset(key);
    if (key === "custom") return;
    const p = PRESETS[key];
    setOutputFormat(p.format);
    setQuality(p.quality);
    setWidthChoice(p.width);
    if (p.customWidth) setCustomWidth(p.customWidth);
    setTargetKb(p.targetKb);
  }

  // Kelompok
  const [results, setResults] = useState<Record<string, ProcessResult>>({});
  const [batchTotal, setBatchTotal] = useState(0);
  const [batchDone, setBatchDone] = useState(0);
  const [running, setRunning] = useState(false);
  const [sysStats, setSysStats] = useState<SystemStats | null>(null);

  const busy = loading || running;
  const busyRef = useRef(busy);
  busyRef.current = busy;

  // Statistik sistem masa nyata: 400ms semasa memproses, 2000ms semasa melahu
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>;
    let cancelled = false;
    const tick = async () => {
      try {
        const s = await invoke<SystemStats>("get_system_stats");
        if (!cancelled) setSysStats(s);
      } catch {
        /* abaikan: command tiada di luar Tauri */
      }
      if (!cancelled) timer = setTimeout(tick, busyRef.current ? 400 : 2000);
    };
    void tick();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, []);

  const loadPaths = useCallback(async (paths: string[]) => {
    if (paths.length === 0) return;
    setError(null);
    setLoading(true);
    try {
      const found = await invoke<ImageInfo[]>("inspect_images", { paths });
      setImages((prev) => {
        const seen = new Set(prev.map((i) => i.path));
        return [...prev, ...found.filter((i) => !seen.has(i.path))];
      });
      setResults({});
      setBatchDone(0);
      setBatchTotal(0);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  // Seret-dan-lepas natif Tauri
  useEffect(() => {
    let unlisten: (() => void) | undefined;
    let cancelled = false;
    getCurrentWebview()
      .onDragDropEvent((event) => {
        const p = event.payload;
        if (p.type === "enter" || p.type === "over") {
          if (!busyRef.current) setDragging(true);
        } else if (p.type === "drop") {
          setDragging(false);
          if (!busyRef.current) void loadPaths(p.paths);
        } else {
          setDragging(false);
        }
      })
      .then((fn) => (cancelled ? fn() : (unlisten = fn)));
    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, [loadPaths]);

  async function pickImages() {
    const selected = await open({
      multiple: true,
      filters: [{ name: t("Images"), extensions: INPUT_EXTS }],
    });
    if (selected) await loadPaths(Array.isArray(selected) ? selected : [selected]);
  }

  async function pickFolder() {
    const selected = await open({ directory: true, title: t("Select Image Folder") });
    if (selected && typeof selected === "string") await loadPaths([selected]);
  }

  /** Tampal imej daripada papan keratan (Ctrl+V / Cmd+V atau butang). */
  const pasteClipboard = useCallback(async () => {
    if (busyRef.current) return;
    setError(null);
    try {
      const img = await invoke<ImageInfo>("paste_image_from_clipboard");
      setImages((prev) =>
        prev.some((i) => i.path === img.path) ? prev : [...prev, img]
      );
    } catch (e) {
      setError(String(e));
    }
  }, []);

  // Pintasan papan keratan: abaikan apabila fokus dalam medan input
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!(e.ctrlKey || e.metaKey) || e.key.toLowerCase() !== "v") return;
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable)) return;
      e.preventDefault();
      void pasteClipboard();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [pasteClipboard]);

  /** Salin imej hasil ke papan keratan, dengan maklum balas 2 saat. */
  async function copyImage(path: string) {
    setError(null);
    try {
      await invoke("copy_image_to_clipboard", { path });
      setCopiedPath(path);
      setTimeout(() => setCopiedPath((p) => (p === path ? null : p)), 2000);
    } catch (e) {
      setError(String(e));
    }
  }

  /** Eksport semua imej sebagai satu PDF, kemudian tunjuk saiz fail. */
  async function exportPdf() {
    if (targets.length === 0) return;
    const target = await save({
      title: t("Save PDF"),
      defaultPath: "morphimage-gabungan.pdf",
      filters: [{ name: "PDF", extensions: ["pdf"] }],
    });
    if (!target) return;
    setError(null);
    setExporting(true);
    setPdfInfo(null);
    try {
      const size = await invoke<number>("export_images_to_pdf", {
        paths: targets.map((i) => i.path),
        options: processOptions(),
        outputPdfPath: target,
        crops,
      });
      setPdfInfo({ path: target, size });
    } catch (e) {
      setError(String(e));
    } finally {
      setExporting(false);
    }
  }

  async function pickOutDir() {
    const selected = await open({ directory: true, title: t("Select Output Folder") });
    if (selected && typeof selected === "string") {
      setCustomOutDir(selected);
      setUseCustomDir(true);
    }
  }

  function clearAll() {
    setImages([]);
    setSelected(new Set());
    setResults({});
    setBatchDone(0);
    setBatchTotal(0);
    setCrops({});
    setPdfInfo(null);
    setError(null);
  }

  const maxWidth: number | null =
    widthChoice === "orig" ? null : widthChoice === "custom" ? customWidth : Number(widthChoice);

  /** Imej yang masih ada dalam senarai DAN ditanda (path basi diabaikan). */
  const picked = useMemo(() => images.filter((i) => selected.has(i.path)), [images, selected]);
  /** Sasaran proses/eksport: pilihan jika sebahagian sahaja, jika tidak semua. */
  const targets = picked.length > 0 ? picked : images;
  const allChecked = images.length > 0 && picked.length === images.length;
  /** Sebahagian sahaja ditanda — butang utama jadi "Proses Pilihan". */
  const partial = picked.length > 0 && picked.length < images.length;

  // Kotak "pilih semua" tunjuk keadaan separa (indeterminate)
  const selectAllRef = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (selectAllRef.current) selectAllRef.current.indeterminate = partial;
  }, [partial]);

  function toggleOne(path: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
  }

  function toggleAll() {
    setSelected(allChecked ? new Set() : new Set(images.map((i) => i.path)));
  }

  /** Buang imej tertentu daripada senarai (tanda, hasil & crop ikut dibuang). */
  function removeImages(paths: string[]) {
    if (paths.length === 0) return;
    const gone = new Set(paths);
    setImages((prev) => prev.filter((i) => !gone.has(i.path)));
    setSelected((prev) => {
      const next = new Set(prev);
      gone.forEach((p) => next.delete(p));
      return next;
    });
    setResults((prev) => {
      const next = { ...prev };
      gone.forEach((p) => delete next[p]);
      return next;
    });
    setCrops((prev) => {
      const next = { ...prev };
      gone.forEach((p) => delete next[p]);
      return next;
    });
    setCompare((c) => (c && gone.has(c.source_path) ? null : c));
    setCropTarget((c) => (c && gone.has(c.path) ? null : c));
  }

  /** Bina ProcessOptions daripada tetapan semasa (dikongsi oleh proses & eksport PDF). */
  function processOptions(kb: number | null = targetKb.trim() === "" ? null : Number(targetKb)) {
    return {
      output_format: outputFormat,
      quality,
      max_width: maxWidth,
      strip_metadata: stripMetadata,
      output_dir: useCustomDir ? customOutDir : null,
      png_lossy: outputFormat === "Png" && pngLossy,
      target_size_kb:
        kb !== null && (outputFormat === "Jpeg" || outputFormat === "Webp" || outputFormat === "Avif")
          ? Math.floor(kb)
          : null,
      custom_suffix: suffix,
      use_numbering: useNumbering,
      rotate_deg: rotateDeg,
      flip_horizontal: flipH,
      flip_vertical: flipV,
      crop: null,
      upscale_factor: upscaleFactor,
      upscale_mode: upscaleMode,
      upscale_model: upscaleModel,
      ai_strength: aiStrength,
      remove_bg: removeBg,
      bg_model: bgModel,
      restore_face: restoreFace && faceModelId !== "",
      face_model_id: faceModelId,
      face_fidelity: faceFidelity / 100,
      execution_mode: settings.execMode,
      worker_threads: settings.workerThreads,
      overwrite_original: settings.naming === "overwrite",
      bg_replace_color: removeBg
        ? bgChoice === "transparent"
          ? null
          : bgChoice === "white"
            ? "#FFFFFF"
            : bgCustomColor
        : null,
    };
  }

  async function processAll() {
    if (targets.length === 0) return;
    if (widthChoice === "custom" && !(customWidth >= 16)) {
      setError(t("Custom width must be at least 16px."));
      return;
    }
    if (useCustomDir && !customOutDir) {
      setError(t("Select a custom output folder first."));
      return;
    }
    const kb = targetKb.trim() === "" ? null : Number(targetKb);
    if (kb !== null && !(kb >= 1)) {
      setError(t("Target size must be at least 1 KB."));
      return;
    }
    if (/[\\/:*?"<>|]/.test(suffix)) {
      setError(t("File suffix cannot contain the characters \\ / : * ? \" < > |"));
      return;
    }
    if (restoreFace && !faceModelId) {
      setError(t("Import a Face Restoration model (.onnx) first, or turn off the Face Restoration switch."));
      return;
    }
    setError(null);
    setResults({});
    setBatchDone(0);
    setBatchTotal(targets.length);
    setRunning(true);

    const unlisten = await listen<BatchProgressEvent>("batch-progress", (event) => {
      const { result } = event.payload;
      setResults((prev) => ({ ...prev, [result.source_path]: result }));
      setBatchDone((n) => n + 1);
    });

    if (
      settings.naming === "overwrite" &&
      !(await ask(t("Overwrite {n} original file(s)? This cannot be undone.", { n: targets.length }), {
        title: t("Overwrite original file"),
        kind: "warning",
      }))
    ) {
      unlisten();
      setRunning(false);
      return;
    }

    try {
      const out = await invoke<ProcessResult[]>("process_batch_images", {
        paths: targets.map((i) => i.path),
        options: processOptions(kb),
        crops,
      });
      const first = out.find((r) => r.success);
      if (settings.autoOpen && first) {
        void invoke("open_folder", { path: dirname(first.output_path) }).catch(() => {});
      }
    } catch (e) {
      setError(String(e));
    } finally {
      unlisten();
      setRunning(false);
    }
  }

  const stats = useMemo(() => {
    const total = images.reduce((a, i) => a + i.size_bytes, 0);
    const done = Object.values(results).filter((r) => r.success);
    const origDone = done.reduce((a, r) => a + r.original_size, 0);
    const after = done.reduce((a, r) => a + r.new_size, 0);
    const saved = origDone - after;
    return {
      total,
      after: done.length ? after : null,
      savedBytes: done.length ? saved : null,
      savedPct: origDone > 0 ? (saved / origDone) * 100 : 0,
    };
  }, [images, results]);

  const firstOut = Object.values(results).find((r) => r.success)?.output_path;
  const finished = !running && batchTotal > 0 && batchDone >= batchTotal;
  const progressPct = batchTotal > 0 ? Math.round((batchDone / batchTotal) * 100) : 0;
  const pngSelected = outputFormat === "Png";
  const noQuality = NO_QUALITY.includes(outputFormat);

  // Penunjuk tab: tetapan bukan lalai sedang aktif
  const cropCount = Object.keys(crops).length;
  const transformActive = rotateDeg !== 0 || flipH || flipV || cropCount > 0;
  const tabDots: Record<PanelTab, boolean> = {
    format: outputFormat !== "Original" || quality !== 80 || targetKb !== "" || useCustomDir || !stripMetadata,
    size: widthChoice !== "orig" || transformActive,
    ai: removeBg || upscaleFactor !== null,
  };

  function resetTransforms() {
    setRotateDeg(0);
    setFlipH(false);
    setFlipV(false);
    setCrops({});
  }

  async function openOutput() {
    const dir = useCustomDir && customOutDir ? customOutDir : firstOut ? dirname(firstOut) : null;
    if (!dir) return;
    try {
      await invoke("open_folder", { path: dir });
    } catch (e) {
      setError(String(e));
    }
  }

  return (
    <div className="flex h-screen flex-col bg-bg text-fg">
      {guideOpen && <AiGuideModal onClose={closeGuide} />}
      {settingsOpen && (
        <SettingsModal
          settings={settings}
          onSettings={setSettings}
          theme={theme}
          onTheme={setTheme}
          aiBackend={sysStats?.ai_backend ?? "Idle"}
          onReset={() => void resetAllSettings()}
          onClose={closeSettings}
        />
      )}
      <header className="flex h-14 shrink-0 items-center gap-3 border-b border-line bg-panel px-4">
        <div className="flex items-center gap-2.5">
          {logoOk ? (
            <img
              src="/logo-morphimage.png"
              alt="MorphImage"
              className="h-7 w-7 rounded-md object-contain"
              onError={() => setLogoOk(false)}
            />
          ) : (
            <ImageIcon className="text-accent" size={22} />
          )}
          <h1 className="text-sm font-semibold tracking-tight">MorphImage</h1>
        </div>
        <div className="ml-auto flex items-center gap-2">
          <button
            onClick={pickImages}
            disabled={busy}
            className="inline-flex h-8 items-center gap-1.5 rounded-md bg-gradient-to-r from-accent to-accent-2 px-3 text-xs font-medium text-white shadow-sm transition-opacity hover:opacity-90 disabled:opacity-40"
          >
            <FolderOpen size={14} />
            {t("Select Images")}
          </button>
          <button onClick={pickFolder} disabled={busy} className={BTN}>
            <Layers size={14} />
            {t("Select Folder")}
          </button>
          <button
            onClick={() => void pasteClipboard()}
            disabled={busy}
            title={t("Paste image from clipboard (Ctrl+V)")}
            className={BTN}
          >
            <ClipboardPaste size={14} />
            {t("Paste Ctrl+V")}
          </button>
          <span className="mx-1 h-5 w-px bg-line" />
          <button
            onClick={() => setGuideOpen(true)}
            title={t("AI Guidebook")}
            aria-label={t("AI Guidebook")}
            className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-line text-muted transition-colors hover:bg-hover hover:text-fg"
          >
            <BookOpen size={15} />
          </button>
          <button
            onClick={() => setSettingsOpen(true)}
            title={t("Settings")}
            aria-label={t("Settings")}
            className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-line text-muted transition-colors hover:bg-hover hover:text-fg"
          >
            <SettingsIcon size={15} />
          </button>
        </div>
      </header>

      <div className="grid min-h-0 flex-1 grid-cols-[1fr_340px]">
        {/* Kiri */}
        <main className="flex min-h-0 flex-col gap-4 overflow-y-auto p-5">
          <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
            <StatCard label={t("Total Files")} value={String(images.length)} icon={<ImageIcon size={13} />} />
            <StatCard label={t("Original Size")} value={formatSize(stats.total)} icon={<HardDrive size={13} />} />
            <StatCard
              label={t("New Size")}
              value={stats.after === null ? "—" : formatSize(stats.after)}
              icon={<Layers size={13} />}
            />
            <StatCard
              label={t("Savings")}
              value={
                stats.savedBytes === null
                  ? "—"
                  : `${stats.savedPct.toFixed(1)}% · ${(stats.savedBytes / 1_048_576).toFixed(2)} MB`
              }
              icon={<TrendingDown size={13} />}
              accent
            />
          </div>

          {error && (
            <p
              role="alert"
              className="rounded-md border border-bad/40 bg-bad/10 px-3 py-2 text-sm text-bad"
            >
              {error}
              {error.includes(PIP_CMD) && (
                <button
                  onClick={() => void navigator.clipboard.writeText(PIP_CMD).catch(() => {})}
                  className="ml-2 inline-flex items-center gap-1 rounded border border-bad/40 px-2 py-0.5 text-xs font-medium hover:bg-bad/10"
                >
                  <Copy size={12} /> {t("Copy Command")}
                </button>
              )}
            </p>
          )}

          {/* Dropzone + senarai */}
          <section
            className={`relative flex min-h-[280px] flex-1 flex-col overflow-hidden rounded-xl border transition-colors ${
              images.length === 0 ? "border-dashed" : "shadow-sm"
            } ${dragging ? "border-accent bg-accent-soft" : "border-line bg-panel"}`}
          >
            {images.length === 0 ? (
              <div className="flex flex-1 flex-col items-center justify-center gap-2 p-8 text-center">
                <UploadCloud size={32} strokeWidth={1.5} className={dragging ? "text-accent" : "text-muted"} />
                <p className="text-sm font-medium">
                  {dragging ? t("Drop to add files") : t("Drag and drop images or folders here")}
                </p>
                <p className="text-xs text-muted">JPG, PNG, WebP, GIF, BMP, TIFF, ICO, SVG, PSD, JXL, RAW</p>
              </div>
            ) : (
              <>
                <div className="flex items-center justify-between gap-2 border-b border-line px-4 py-2">
                  <span className="text-xs text-muted tabular-nums">
                    {picked.length > 0
                      ? t("{picked} / {images} selected", { picked: picked.length, images: images.length })
                      : t("{images} files", { images: images.length })}
                  </span>
                  <div className="flex items-center gap-1">
                    <button
                      onClick={() => setRotateDeg((d) => (d + 270) % 360)}
                      title={t("Rotate 90° left")}
                      aria-label={t("Rotate 90 degrees left")}
                      className={iconBar}
                    >
                      <RotateCcw size={14} />
                    </button>
                    <button
                      onClick={() => setRotateDeg((d) => (d + 90) % 360)}
                      title={t("Rotate 90° right")}
                      aria-label={t("Rotate 90 degrees right")}
                      className={iconBar}
                    >
                      <RotateCw size={14} />
                    </button>
                    <button
                      role="switch"
                      aria-checked={flipH}
                      onClick={() => setFlipH((v) => !v)}
                      title={t("Mirror horizontally")}
                      aria-label={t("Mirror horizontally")}
                      className={flipH ? iconBarOn : iconBar}
                    >
                      <FlipHorizontal size={14} />
                    </button>
                    <button
                      role="switch"
                      aria-checked={flipV}
                      onClick={() => setFlipV((v) => !v)}
                      title={t("Flip vertically")}
                      aria-label={t("Flip vertically")}
                      className={flipV ? iconBarOn : iconBar}
                    >
                      <FlipVertical size={14} />
                    </button>
                    <span className="mx-1 h-5 w-px bg-line" />
                    {picked.length > 0 && (
                      <button
                        onClick={() => removeImages([...selected])}
                        disabled={busy}
                        className="inline-flex h-7 items-center gap-1.5 rounded-md bg-bad/10 px-2 text-xs font-medium text-bad transition-colors hover:bg-bad/20 disabled:opacity-40"
                      >
                        <Trash2 size={13} />
                        {t("Delete Selected ({n})", { n: picked.length })}
                      </button>
                    )}
                    <button
                      onClick={clearAll}
                      disabled={busy}
                      className="inline-flex h-7 items-center gap-1.5 rounded-md px-2 text-xs text-muted transition-colors hover:bg-hover hover:text-fg disabled:opacity-40"
                    >
                      <Trash2 size={13} />
                      {t("Clear List")}
                    </button>
                  </div>
                </div>
                <div className="min-h-0 flex-1 overflow-auto">
                  <table className="w-full text-sm">
                    <thead className="sticky top-0 bg-panel text-[11px] uppercase tracking-wider text-muted">
                      <tr className="border-b border-line">
                        <th className="w-10 px-3 py-2">
                          <input
                            ref={selectAllRef}
                            type="checkbox"
                            checked={allChecked}
                            onChange={toggleAll}
                            aria-label={allChecked ? t("Deselect all") : t("Select all")}
                            className="h-4 w-4 cursor-pointer accent-accent"
                          />
                        </th>
                        <th className="px-4 py-2 text-left font-medium">{t("File Name")}</th>
                        <th className="px-4 py-2 text-left font-medium">Format</th>
                        <th className="px-4 py-2 text-right font-medium">{t("Dimensions")}</th>
                        <th className="px-4 py-2 text-right font-medium">{t("Original Size")}</th>
                        <th className="w-44 px-4 py-2 text-left font-medium">Status</th>
                        <th className="w-28 px-2 py-2" aria-label={t("Actions")} />
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-line">
                      {images.map((img) => (
                        <Row
                          key={img.path}
                          img={img}
                          res={results[img.path]}
                          running={running}
                          onCompare={setCompare}
                          onCopy={copyImage}
                          copied={copiedPath === img.path}
                          onCrop={setCropTarget}
                          cropped={!!crops[img.path]}
                          rotateDeg={rotateDeg}
                          flipH={flipH}
                          flipV={flipV}
                          checked={selected.has(img.path)}
                          onToggle={() => toggleOne(img.path)}
                          onRemove={() => removeImages([img.path])}
                        />
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}

            {dragging && (
              <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-accent-soft">
                <span className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-white shadow-lg">
                  {t("Drop files here")}
                </span>
              </div>
            )}
          </section>
        </main>

        {/* Kanan */}
        <aside className="flex min-h-0 flex-col border-l border-line bg-panel">
          {/* Navigasi tab + penunjuk gelongsor */}
          <div className="shrink-0 px-2 pt-2">
            <div role="tablist" aria-label={t("Settings")} className="relative grid grid-cols-3 rounded-lg bg-hover p-1">
              <span
                aria-hidden
                className="tab-pill pointer-events-none absolute inset-y-1 left-1 w-[calc((100%-0.5rem)/3)] rounded-md bg-panel shadow-sm"
                style={{ transform: `translateX(${TAB_ORDER.indexOf(activeTab) * 100}%)` }}
              />
              {TABS.map((tab) => (
                <button
                  key={tab.key}
                  role="tab"
                  id={`tab-${tab.key}`}
                  aria-selected={activeTab === tab.key}
                  aria-controls={`tabpanel-${tab.key}`}
                  onClick={() => setActiveTab(tab.key)}
                  className={`relative z-10 inline-flex h-8 items-center justify-center gap-1.5 rounded-md text-xs transition-colors ${
                    activeTab === tab.key ? "font-medium text-accent-text" : "text-muted hover:text-fg"
                  }`}
                >
                  {t(tab.label)}
                  {tabDots[tab.key] && (
                    <span aria-hidden title={t("Non-default settings active")} className="h-1.5 w-1.5 rounded-full bg-accent" />
                  )}
                </button>
              ))}
            </div>
          </div>

          <div
            key={activeTab}
            role="tabpanel"
            id={`tabpanel-${activeTab}`}
            aria-labelledby={`tab-${activeTab}`}
            className="tab-fade-in min-h-0 flex-1 space-y-6 overflow-y-auto p-5"
          >
            {activeTab === "format" && (
              <>
            <Group title={t("Quick Presets")}>
              <div className="grid grid-cols-2 gap-1.5">
                {(
                  [
                    ["custom", t("Custom")],
                    ["web", "Web & Blog"],
                    ["ecom", t("E-Commerce")],
                    ["form", t("Forms < 500 KB")],
                  ] as [PresetKey, string][]
                ).map(([key, label]) => (
                  <button
                    key={key}
                    onClick={() => applyPreset(key)}
                    aria-pressed={preset === key}
                    className={`h-8 rounded-md border px-2 text-xs transition-colors ${
                      preset === key
                        ? "border-accent bg-accent-soft font-medium text-accent-text"
                        : "border-line text-muted hover:bg-hover hover:text-fg"
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </Group>

            <Group title={t("Format & Quality")}>
              <select
                aria-label={t("Output Format")}
                value={outputFormat}
                onChange={(e) => setOutputFormat(e.target.value as OutputFormat)}
                className={INPUT}
              >
                {FORMAT_GROUPS.map((g) => (
                  <optgroup key={g.label} label={t(g.label)}>
                    {g.items.map((f) => (
                      <option key={f.value} value={f.value}>
                        {t(f.label)}
                      </option>
                    ))}
                  </optgroup>
                ))}
              </select>
              <p className="text-xs text-muted">
                {pngSelected
                  ? pngLossy
                    ? t("Lossy 256 colors: smaller files, reduced color quality")
                    : t("Lossless: ideal for logos & transparency")
                  : FORMAT_HINTS[outputFormat] && t(FORMAT_HINTS[outputFormat])}
              </p>

              {pngSelected && (
                <div className="flex items-center justify-between">
                  <span id="lossy-label" className="text-xs">
                    {t("High PNG Compression (256 Colors)")}
                  </span>
                  <Switch checked={pngLossy} onChange={() => setPngLossy((v) => !v)} labelId="lossy-label" />
                </div>
              )}

              {!noQuality && (
                <div className="space-y-1">
                  <label htmlFor="target-kb" className="text-xs">
                    {t("Max Target Size (KB)")} <span className="text-muted">{t("— optional")}</span>
                  </label>
                  <input
                    id="target-kb"
                    type="number"
                    min={1}
                    placeholder={t("No target")}
                    value={targetKb}
                    onChange={(e) => setTargetKb(e.target.value)}
                    className={`${INPUT} tabular-nums`}
                  />
                  {targetKb !== "" && (
                    <p className="text-xs text-muted">
                      {t("Quality is tuned automatically (15–95); the quality slider is ignored.")}
                    </p>
                  )}
                </div>
              )}

              <label className={`block space-y-1.5 ${noQuality ? "opacity-40" : ""}`}>
                <span className="flex justify-between text-xs">
                  <span>{t("Quality")}</span>
                  <span className="tabular-nums text-accent-text">{noQuality ? "—" : quality}</span>
                </span>
                <input
                  type="range"
                  min={1}
                  max={100}
                  value={quality}
                  disabled={noQuality}
                  onChange={(e) => setQuality(Number(e.target.value))}
                  className="w-full accent-accent"
                />
              </label>
            </Group>

            <Group title="Output & Metadata">
              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <span id="strip-label" className="text-xs">
                    {t("Strip EXIF/GPS Metadata")}
                  </span>
                  <Switch checked={stripMetadata} onChange={() => setStripMetadata((v) => !v)} labelId="strip-label" />
                </div>
                <p className="text-xs text-muted">
                  {t("EXIF/GPS metadata is stripped automatically when re-encoding")}
                </p>
              </div>

              <fieldset className="space-y-2">
                <legend className="mb-1.5 text-xs">{t("Output Folder")}</legend>
                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="radio"
                    name="outdir"
                    checked={!useCustomDir}
                    onChange={() => setUseCustomDir(false)}
                    className="accent-accent"
                  />
                  {t("/output subfolder next to the source")}
                </label>
                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="radio"
                    name="outdir"
                    checked={useCustomDir}
                    onChange={() => (customOutDir ? setUseCustomDir(true) : void pickOutDir())}
                    className="accent-accent"
                  />
                  {t("Custom folder")}
                </label>
                {useCustomDir && (
                  <button
                    onClick={pickOutDir}
                    title={customOutDir ?? ""}
                    className="h-8 w-full truncate rounded-md border border-line px-2.5 text-left text-xs text-muted transition-colors hover:bg-hover"
                  >
                    {customOutDir ?? t("Select folder…")}
                  </button>
                )}
              </fieldset>

              <div className="space-y-2">
                <label htmlFor="suffix" className="text-xs">
                  {t("File Suffix")}
                </label>
                <input
                  id="suffix"
                  type="text"
                  value={suffix}
                  placeholder="_morph"
                  onChange={(e) => setSuffixInput(e.target.value)}
                  className={INPUT}
                />
                <div className="flex items-center justify-between">
                  <span id="num-label" className="text-xs">
                    {t("Numbering (001_name)")}
                  </span>
                  <Switch checked={useNumbering} onChange={() => setUseNumbering((v) => !v)} labelId="num-label" />
                </div>
              </div>
            </Group>
              </>
            )}

            {activeTab === "size" && (
              <>
            <Group title={t("Max Width")}>
              <div className="space-y-1">
                <label htmlFor="width" className="text-xs">
                  {t("Max Width")}
                </label>
                <select
                  id="width"
                  value={widthChoice}
                  onChange={(e) => setWidthChoice(e.target.value as WidthChoice)}
                  className={INPUT}
                >
                  <option value="orig">{t("Original")}</option>
                  <option value="1920">1920px</option>
                  <option value="1280">1280px</option>
                  <option value="800">800px</option>
                  <option value="custom">{t("Custom…")}</option>
                </select>
                {widthChoice === "custom" && (
                  <div className="flex items-center gap-2">
                    <input
                      type="number"
                      min={16}
                      max={20000}
                      value={customWidth}
                      onChange={(e) => setCustomWidth(Number(e.target.value))}
                      aria-label={t("Custom width in pixels")}
                      className={INPUT + " tabular-nums"}
                    />
                    <span className="text-xs text-muted">px</span>
                  </div>
                )}
                <p className="text-xs text-muted">{t("Images smaller than this width are not enlarged.")}</p>
              </div>
            </Group>

            <Group title={t("Transform Status")}>
              <div className="space-y-2 rounded-lg border border-line bg-elevated p-3">
                {transformActive ? (
                  <ul className="space-y-1 text-xs">
                    <li className="flex justify-between">
                      <span className="text-muted">{t("Rotate")}</span>
                      <span className="tabular-nums text-accent-text">{rotateDeg}°</span>
                    </li>
                    <li className="flex justify-between">
                      <span className="text-muted">{t("Mirror horizontally")}</span>
                      <span className={flipH ? "text-accent-text" : "text-muted"}>{flipH ? t("On") : t("Off")}</span>
                    </li>
                    <li className="flex justify-between">
                      <span className="text-muted">{t("Flip vertically")}</span>
                      <span className={flipV ? "text-accent-text" : "text-muted"}>{flipV ? t("On") : t("Off")}</span>
                    </li>
                    <li className="flex justify-between">
                      <span className="text-muted">Crop</span>
                      <span className={cropCount > 0 ? "text-accent-text" : "text-muted"}>
                        {cropCount > 0 ? cropCount + " fail" : t("None")}
                      </span>
                    </li>
                  </ul>
                ) : (
                  <p className="text-xs text-muted">{t("No transforms — images are processed as-is.")}</p>
                )}
                <button onClick={resetTransforms} disabled={!transformActive} className={BTN + " w-full"}>
                  <RotateCcw size={13} />
                  {t("Reset Transforms")}
                </button>
              </div>
              <p className="text-xs text-muted">{t("Rotate & mirror: toolbar above the file list. Crop: icon on each file.")}</p>
            </Group>
              </>
            )}

            {activeTab === "ai" && (
              <>
            <Group title={t("Remove Background")}>
              <div className="space-y-2 rounded-lg border border-line p-3">
                <div className="flex items-center justify-between">
                  <span id="remove-bg-label" className="text-xs">
                    {t("Remove Background")}
                  </span>
                  <Switch checked={removeBg} onChange={() => setRemoveBg((v) => !v)} labelId="remove-bg-label" />
                </div>

                {removeBg && (
                  <>
                    <div className="flex items-start gap-1.5">
                      <ModelSelect
                        label={t("Background removal model")}
                        value={bgModel}
                        builtin={(Object.keys(BG_MODELS) as BgModelId[]).map((id) => ({
                          id,
                          label: t(BG_MODELS[id].label),
                          sub: `~${BG_MODELS[id].mb} MB`,
                        }))}
                        custom={customBg.map((m) => ({
                          id: m.id,
                          label: m.name,
                          sub: `${m.input_size}×${m.input_size} • ${m.norm_type === "imagenet" ? "ImageNet" : "Standard"} • ${m.size_mb} MB`,
                        }))}
                        onChange={setBgModel}
                        onDelete={(id) => {
                          const m = customBg.find((x) => x.id === id);
                          if (m) void deleteModel(m);
                        }}
                      />
                      <button
                        onClick={() => setBgImportNorm((v) => (v === null ? "standard" : null))}
                        disabled={importing !== null}
                        aria-expanded={bgImportNorm !== null}
                        className={`${BTN} shrink-0`}
                      >
                        {importing === "bg" ? <Loader2 size={13} className="animate-spin" /> : <Plus size={13} />}
                        Import Model
                      </button>
                    </div>
                    {bgImportNorm !== null && (
                      <div className="space-y-1.5 rounded-md border border-line p-2">
                        <p className="text-xs text-muted">{t("Model normalization type:")}</p>
                        <div role="group" aria-label={t("Normalization type")} className="grid grid-cols-2 gap-1.5">
                          {(
                            [
                              ["standard", "Standard / RMBG"],
                              ["imagenet", "ImageNet / BiRefNet"],
                            ] as const
                          ).map(([key, lbl]) => (
                            <button
                              key={key}
                              aria-pressed={bgImportNorm === key}
                              onClick={() => setBgImportNorm(key)}
                              className={bgImportNorm === key ? BTN_ON : BTN}
                            >
                              {lbl}
                            </button>
                          ))}
                        </div>
                        <p className="text-[10px] text-muted">
                          Standard [0..1]: RMBG / U2Net. ImageNet: BiRefNet / BEN2.
                        </p>
                        <button onClick={() => void importModel("bg", bgImportNorm)} className={`${BTN} w-full`}>
                          {t("Choose File & Import")}
                        </button>
                      </div>
                    )}
                    <p className="text-xs text-muted">
                      {pickedCustomBg
                        ? t("Imported model ({size_mb} MB).", { size_mb: pickedCustomBg.size_mb })
                        : t(BG_MODELS[bgModel as BgModelId]?.hint ?? "")}{" "}
                      <button onClick={() => setGuideOpen(true)} className="text-accent-text hover:underline">
                        {t("Find great models →")}
                      </button>
                    </p>
                    {((bgModel === "birefnet-lite" && lowVram) ||
                      (pickedCustomBg && pickedCustomBg.size_mb > HEAVY_MB && lowVram)) && (
                      <p role="status" className="text-xs text-amber-600">
                        {t("VRAM ≤ 4GB detected: runs on CPU automatically to keep the display stable")}
                      </p>
                    )}

                    {bgModelReady === false && (
                      <div className="space-y-1.5">
                        <button
                          onClick={() => void downloadBgModel()}
                          disabled={bgDownloading !== null}
                          className={`${BTN} w-full`}
                        >
                          {t("Download AI Model (~{mb} MB)", { mb: BG_MODELS[bgModel as BgModelId]?.mb ?? "?" })}
                        </button>
                        {bgDownloading === bgModel && bgModelPct !== null && (
                          <div
                            role="progressbar"
                            aria-valuemin={0}
                            aria-valuemax={100}
                            aria-valuenow={Math.round(bgModelPct)}
                            className="h-1.5 w-full overflow-hidden rounded-full bg-line"
                          >
                            <div className="h-full bg-accent transition-[width]" style={{ width: `${bgModelPct}%` }} />
                          </div>
                        )}
                        <p className="text-xs text-muted">{t("Downloaded once; runs 100% offline afterwards.")}</p>
                      </div>
                    )}

                    <div role="group" aria-label={t("Replacement background")} className="grid grid-cols-3 gap-1.5">
                      {(
                        [
                          ["transparent", t("Transparent")],
                          ["white", t("White")],
                          ["custom", t("Color")],
                        ] as const
                      ).map(([key, label]) => (
                        <button
                          key={key}
                          aria-pressed={bgChoice === key}
                          onClick={() => setBgChoice(key)}
                          className={bgChoice === key ? BTN_ON : BTN}
                        >
                          {label}
                        </button>
                      ))}
                    </div>

                    {bgChoice === "custom" && (
                      <label className="flex items-center justify-between gap-2 text-xs">
                        <span>{t("Chosen Color")}</span>
                        <input
                          type="color"
                          aria-label={t("Chosen background color")}
                          value={bgCustomColor}
                          onChange={(e) => setBgCustomColor(e.target.value)}
                          className="h-8 w-14 cursor-pointer rounded border border-line bg-panel"
                        />
                      </label>
                    )}

                    <p className="text-xs text-muted">
                      {bgChoice === "transparent"
                        ? t("Transparent background; JPEG/BMP are switched to PNG automatically.")
                        : t("Subject is composited onto the chosen background color.")}
                    </p>
                  </>
                )}
              </div>
            </Group>

            <Group title={t("Face Restoration")}>
              <div className="space-y-2 rounded-lg border border-line p-3">
                <div className="flex items-center justify-between">
                  <span id="restore-face-label" className="text-xs">
                    {t("Face Restoration")}
                  </span>
                  <Switch checked={restoreFace} onChange={() => setRestoreFace((v) => !v)} labelId="restore-face-label" />
                </div>

                {restoreFace && (
                  <>
                    <div className="flex items-start gap-1.5">
                      {faceModels.length > 0 ? (
                        <ModelSelect
                          label={t("Face restoration model")}
                          value={faceModelId}
                          builtin={[]}
                          custom={faceModels.map((m) => ({
                            id: m.id,
                            label: m.name,
                            sub: `512×512 • ${m.has_fidelity ? "CodeFormer (w)" : "1 input"} • ${m.size_mb} MB`,
                          }))}
                          onChange={setFaceModelId}
                          onDelete={(id) => {
                            const m = faceModels.find((x) => x.id === id);
                            if (m) void deleteFaceModel(m);
                          }}
                        />
                      ) : (
                        <p className="min-w-0 flex-1 self-center text-xs text-muted">{t("No model imported yet.")}</p>
                      )}
                      <button onClick={() => void importFaceModel()} disabled={faceImporting} className={`${BTN} shrink-0`}>
                        {faceImporting ? <Loader2 size={13} className="animate-spin" /> : <Plus size={13} />}
                        Import Model
                      </button>
                    </div>

                    {faceModels.length === 0 ? (
                      <ol className="list-decimal space-y-0.5 rounded-md border border-line p-2 pl-6 text-xs text-muted">
                        <li>
                          {t("Download")} <b className="text-fg">codeformer.onnx</b> {t("or")} <b className="text-fg">GFPGANv1.4.onnx</b>{" "}
                          (input 512×512).
                        </li>
                        <li>{t("Click + Import Model and choose that file (.onnx or .pth).")}</li>
                        <li>
                          <button onClick={() => setGuideOpen(true)} className="text-accent-text hover:underline">
                            {t("Open the AI Guidebook →")}
                          </button>
                        </li>
                      </ol>
                    ) : (
                      <>
                        <label className="block space-y-1.5">
                          <span className="flex justify-between text-xs">
                            <span>{t("Face Balance")}</span>
                            <span className="tabular-nums text-accent-text">{faceFidelity}%</span>
                          </span>
                          <input
                            type="range"
                            min={0}
                            max={100}
                            value={faceFidelity}
                            onChange={(e) => setFaceFidelity(Number(e.target.value))}
                            className="w-full accent-accent"
                          />
                          <span className="flex justify-between text-[10px] text-muted">
                            <span>{t("Original")}</span>
                            <span>{t("Balanced")}</span>
                            <span>{t("Full Restore")}</span>
                          </span>
                        </label>
                        <p className="text-xs text-muted">
                          {t("Best for portraits cropped to the face. Runs before the Upscaler.")}
                        </p>
                        {pickedFace && pickedFace.size_mb > HEAVY_MB && lowVram && (
                          <p role="status" className="text-xs text-amber-600">
                            {t("VRAM ≤ 4GB detected: large models run on CPU automatically to keep the display stable")}
                          </p>
                        )}
                      </>
                    )}
                  </>
                )}
              </div>
            </Group>

            <Group title={t("AI Upscaler")}>
              <div role="group" aria-label={t("Upscale factor")} className="grid grid-cols-3 gap-1.5">
                {([null, 2, 4] as const).map((f) => (
                  <button
                    key={String(f)}
                    aria-pressed={upscaleFactor === f}
                    onClick={() => setUpscaleFactor(f)}
                    className={upscaleFactor === f ? BTN_ON : BTN}
                  >
                    {f === null ? t("Original") : `${f}×`}
                  </button>
                ))}
              </div>

              {upscaleFactor !== null && (
                <div className="space-y-2">
                  <div role="group" aria-label={t("Upscale mode")} className="grid grid-cols-2 gap-1.5">
                    <button
                      aria-pressed={upscaleMode === "fast"}
                      onClick={() => setUpscaleMode("fast")}
                      className={upscaleMode === "fast" ? BTN_ON : BTN}
                    >
                      {t("Fast (CPU)")}
                    </button>
                    <button
                      aria-pressed={upscaleMode === "ai"}
                      onClick={() => setUpscaleMode("ai")}
                      className={upscaleMode === "ai" ? BTN_ON : BTN}
                    >
                      {t("AI Sharp (GPU)")}
                    </button>
                  </div>

                  {upscaleMode === "ai" && (
                    <>
                      <div className="flex items-start gap-1.5">
                        <ModelSelect
                          label={t("AI upscale model")}
                          value={upscaleModel}
                          builtin={(Object.keys(UPSCALE_MODELS) as UpscaleModelId[]).map((id) => ({
                            id,
                            label: t(UPSCALE_MODELS[id].label),
                            sub: `~${UPSCALE_MODELS[id].mb} MB`,
                          }))}
                          custom={customUp.map((m) => ({
                            id: m.id,
                            label: m.name,
                            sub: `${m.scale}× • ${m.size_mb} MB`,
                          }))}
                          onChange={setUpscaleModel}
                          onDelete={(id) => {
                            const m = customUp.find((x) => x.id === id);
                            if (m) void deleteModel(m);
                          }}
                        />
                        <button
                          onClick={() => void importModel("upscale")}
                          disabled={importing !== null}
                          className={`${BTN} shrink-0`}
                        >
                          {importing === "upscale" ? <Loader2 size={13} className="animate-spin" /> : <Plus size={13} />}
                          Import Model
                        </button>
                      </div>
                      <p className="text-xs text-muted">
                        {pickedCustomUp
                          ? t("Imported model • native {scale}× ({size_mb} MB).", { scale: pickedCustomUp.scale ?? "?", size_mb: pickedCustomUp.size_mb })
                          : t(UPSCALE_MODELS[upscaleModel as UpscaleModelId]?.hint ?? "")}{" "}
                        <button onClick={() => setGuideOpen(true)} className="text-accent-text hover:underline">
                          {t("Find great models →")}
                        </button>
                      </p>
                      {pickedCustomUp && pickedCustomUp.size_mb > HEAVY_MB && lowVram && (
                        <p role="status" className="text-xs text-amber-600">
                          {t("VRAM ≤ 4GB detected: large models run on CPU automatically to keep the display stable")}
                        </p>
                      )}

                      <label className="block space-y-1.5">
                        <span className="flex justify-between text-xs">
                          <span>{t("AI Sharpness")}</span>
                          <span className="tabular-nums text-accent-text">{aiStrength}%</span>
                        </span>
                        <input
                          type="range"
                          min={0}
                          max={100}
                          value={aiStrength}
                          onChange={(e) => setAiStrength(Number(e.target.value))}
                          className="w-full accent-accent"
                        />
                        <span className="flex justify-between text-[10px] text-muted">
                          <span>{t("Natural")}</span>
                          <span>{t("Balanced")}</span>
                          <span>{t("Maximum")}</span>
                        </span>
                      </label>

                      {gpu && (
                        <p className="text-xs text-muted">
                          {gpu.available
                            ? t("GPU: {gpu_name} • {backend} Active", { gpu_name: gpu.gpu_name, backend: gpu.backend })
                            : t("No GPU detected • CPU Fallback (slower)")}
                        </p>
                      )}
                      {gpu?.available && gpu.vram_mb < 2048 && (
                        <p role="status" className="text-xs text-amber-600">
                          {t("Only {vram} MB VRAM (< 2048 MB). Large images may be slow.", { vram: gpu.vram_mb })}
                        </p>
                      )}
                      {modelReady === false && (
                        <div className="space-y-1.5">
                          <button onClick={() => void downloadModel()} disabled={modelPct !== null} className={`${BTN} w-full`}>
                            {t("Download AI Model (~{mb} MB)", { mb: UPSCALE_MODELS[upscaleModel as UpscaleModelId]?.mb ?? "?" })}
                          </button>
                          {upscaleDownloading === upscaleModel && modelPct !== null && (
                            <div
                              role="progressbar"
                              aria-valuemin={0}
                              aria-valuemax={100}
                              aria-valuenow={Math.round(modelPct)}
                              className="h-1.5 w-full overflow-hidden rounded-full bg-line"
                            >
                              <div className="h-full bg-accent transition-[width]" style={{ width: `${modelPct}%` }} />
                            </div>
                          )}
                          <p className="text-xs text-muted">{t("Downloaded once; runs 100% offline afterwards.")}</p>
                        </div>
                      )}
                    </>
                  )}
                </div>
              )}
            </Group>
              </>
            )}
          </div>

          {/* Bawah melekat */}
          <div className="shrink-0 space-y-2 border-t border-line bg-panel p-4">
            {batchTotal > 0 && (
              <div className="space-y-1">
                <div className="flex justify-between text-xs text-muted tabular-nums">
                  <span>
                    {batchDone} / {batchTotal} {t("done")}
                  </span>
                  <span>{progressPct}%</span>
                </div>
                <div className="h-1 overflow-hidden rounded-full bg-hover">
                  <div
                    className="h-full bg-gradient-to-r from-accent to-accent-2 transition-all duration-200"
                    style={{ width: `${progressPct}%` }}
                  />
                </div>
              </div>
            )}
            <button
              onClick={processAll}
              disabled={busy || images.length === 0}
              className="flex h-10 w-full items-center justify-center gap-2 rounded-md bg-gradient-to-r from-accent to-accent-2 text-sm font-medium text-white shadow-sm transition-opacity hover:opacity-90 disabled:opacity-40"
            >
              <Zap size={15} />
              {running
                ? t("Processing… ({batchDone}/{batchTotal})", { batchDone, batchTotal })
                : partial
                  ? t("Process Selected ({targets})", { targets: targets.length })
                  : t("Process All")}
            </button>
            <button
              onClick={() => void exportPdf()}
              disabled={busy || exporting || images.length === 0}
              title={partial ? t("Merge {targets} selected images into a PDF", { targets: targets.length }) : t("Merge all images into one PDF")}
              className="flex h-9 w-full items-center justify-center gap-2 rounded-md border border-line text-sm transition-colors hover:bg-hover disabled:opacity-40"
            >
              {exporting ? <Loader2 size={15} className="animate-spin" /> : <FileText size={15} />}
              {exporting ? t("Exporting PDF…") : partial ? t("Merge Selected to PDF ({targets})", { targets: targets.length }) : t("Merge to PDF")}
            </button>
            {pdfInfo && (
              <div role="status" className="space-y-1.5 rounded-md border border-ok/40 bg-ok/10 px-3 py-2 text-xs text-ok">
                <p>
                  {t("PDF saved ·")} <b className="tabular-nums">{formatSize(pdfInfo.size)}</b>
                </p>
                <p className="truncate opacity-70" title={pdfInfo.path}>
                  {pdfInfo.path}
                </p>
                <button
                  onClick={() => void invoke("open_folder", { path: dirname(pdfInfo.path) }).catch((e) => setError(String(e)))}
                  className="flex h-7 w-full items-center justify-center gap-2 rounded border border-ok/40 text-xs transition-colors hover:bg-ok/10"
                >
                  <FolderOpen size={13} />
                  {t("Open Output Folder")}
                </button>
              </div>
            )}
            {finished && (
              <button
                onClick={openOutput}
                className="flex h-9 w-full items-center justify-center gap-2 rounded-md border border-line text-sm transition-colors hover:bg-hover"
              >
                <FolderOpen size={15} />
                {t("Open Output Folder")}
              </button>
            )}
            <p className="pt-1 text-center text-[10px] text-muted/70">{t("Built by Kando")}</p>
          </div>
        </aside>
      </div>

      <footer className="flex h-7 shrink-0 items-center gap-4 border-t border-line bg-panel px-4 text-[11px] text-muted tabular-nums">
        <span className="font-mono text-accent-text">
          {t("Engine: CPU Multi-Core ({a}/{b} Threads)", {
            a: sysStats?.active_threads ?? "—",
            b: sysStats?.total_threads ?? "—",
          })}
        </span>
        <span className="flex items-center gap-2">
          CPU
          <span className="h-1 w-16 overflow-hidden rounded-full bg-hover">
            <span
              className="block h-full rounded-full bg-gradient-to-r from-accent to-accent-2 transition-all duration-300"
              style={{ width: `${Math.min(100, Math.max(0, sysStats?.cpu_usage ?? 0))}%` }}
            />
          </span>
          <span className="text-fg">{Math.round(sysStats?.cpu_usage ?? 0)}%</span>
        </span>
        <span>RAM App: {(sysStats?.app_memory_mb ?? 0).toFixed(1)} MB</span>
        {sysStats && sysStats.gpu_name === "Tiada GPU" ? (
          <span className="opacity-50">GPU: N/A</span>
        ) : (
          <span className="flex items-center gap-2" title={sysStats?.gpu_name}>
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                sysStats?.gpu_active ? "animate-pulse bg-emerald-400" : "bg-muted/40"
              }`}
            />
            {sysStats?.ai_backend && sysStats.ai_backend !== "Idle" ? `AI ${sysStats.ai_backend}:` : "GPU:"}
            <span className="h-1 w-16 overflow-hidden rounded-full bg-hover">
              <span
                className="block h-full rounded-full bg-gradient-to-r from-accent to-accent-2 transition-all duration-300"
                style={{ width: `${Math.min(100, Math.max(0, sysStats?.gpu_usage ?? 0))}%` }}
              />
            </span>
            <span className="text-fg">{Math.round(sysStats?.gpu_usage ?? 0)}%</span>
            <span>VRAM: {sysStats?.gpu_vram_used_mb ?? 0} MB</span>
          </span>
        )}
      </footer>

      {compare && (
        <CompareModal key={compare.output_path} res={compare} onClose={() => setCompare(null)} />
      )}
      {cropTarget && (
        <CropModal
          img={cropTarget}
          initial={crops[cropTarget.path]}
          rotateDeg={rotateDeg}
          flipH={flipH}
          flipV={flipV}
          onOrient={(deg, fh, fv) => {
            setRotateDeg(deg);
            setFlipH(fh);
            setFlipV(fv);
          }}
          onClose={() => setCropTarget(null)}
          onSave={(rect) => {
            setCrops((prev) => ({ ...prev, [cropTarget.path]: rect }));
            setCropTarget(null);
          }}
          onClear={() => {
            setCrops((prev) => {
              const next = { ...prev };
              delete next[cropTarget.path];
              return next;
            });
            setCropTarget(null);
          }}
        />
      )}
    </div>
  );
}

const INPUT =
  "h-8 w-full rounded-md border border-line bg-panel px-2.5 text-sm text-fg focus:border-accent focus:outline-none";
const BTN =
  "inline-flex h-8 items-center justify-center gap-1.5 rounded-md border border-line px-2.5 text-xs font-medium text-fg transition-colors hover:bg-hover disabled:opacity-40";
const BTN_ON =
  "inline-flex h-8 items-center justify-center gap-1.5 rounded-md border border-accent bg-accent-soft px-2.5 text-xs font-medium text-accent-text transition-colors";
const iconBar =
  "inline-flex h-7 w-7 items-center justify-center rounded-md border border-line text-muted transition-colors hover:bg-hover hover:text-fg";
const iconBarOn =
  "inline-flex h-7 w-7 items-center justify-center rounded-md border border-accent bg-accent-soft text-accent-text transition-colors";

function Group({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="space-y-3">
      <h2 className="text-xs font-medium uppercase tracking-wider text-muted">{title}</h2>
      {children}
    </section>
  );
}

function Switch({ checked, onChange, labelId }: { checked: boolean; onChange: () => void; labelId: string }) {
  return (
    <button
      role="switch"
      aria-checked={checked}
      aria-labelledby={labelId}
      onClick={onChange}
      className={`relative h-5 w-9 shrink-0 rounded-full transition-colors ${checked ? "bg-accent" : "bg-line"}`}
    >
      <span
        className={`absolute left-0.5 top-0.5 h-4 w-4 rounded-full bg-white shadow-sm transition-transform ${
          checked ? "translate-x-4" : ""
        }`}
      />
    </button>
  );
}

function StatCard({
  label,
  value,
  icon,
  accent,
}: {
  label: string;
  value: string;
  icon: React.ReactNode;
  accent?: boolean;
}) {
  return (
    <div className="rounded-xl border border-line bg-panel p-4 shadow-sm">
      <div className="flex items-center gap-2 text-xs text-muted">
        <span
          className={`grid h-6 w-6 place-items-center rounded-md ${
            accent ? "bg-ok/10 text-ok" : "bg-accent-soft text-accent-text"
          }`}
        >
          {icon}
        </span>
        {label}
      </div>
      <div className={`mt-2 text-lg font-semibold tabular-nums ${accent ? "text-ok" : "text-fg"}`}>
        {value}
      </div>
    </div>
  );
}

function Row({
  img,
  res,
  running,
  onCompare,
  onCopy,
  copied,
  onCrop,
  cropped,
  rotateDeg,
  flipH,
  flipV,
  checked,
  onToggle,
  onRemove,
}: {
  img: ImageInfo;
  res: ProcessResult | undefined;
  running: boolean;
  onCompare: (r: ProcessResult) => void;
  onCopy: (path: string) => void;
  copied: boolean;
  onCrop: (img: ImageInfo) => void;
  cropped: boolean;
  rotateDeg: number;
  flipH: boolean;
  flipV: boolean;
  checked: boolean;
  onToggle: () => void;
  onRemove: () => void;
}) {
  const iconBtn = "rounded-md p-1.5 transition-colors hover:bg-hover";
  const thumbable = !NO_THUMB.includes(img.format.toLowerCase());
  const transform = previewTransform(rotateDeg, flipH, flipV);
  return (
    <tr className={`group transition-colors hover:bg-hover ${checked ? "bg-accent-soft/50" : ""}`}>
      <td className="w-10 px-3 py-2">
        <input
          type="checkbox"
          checked={checked}
          onChange={onToggle}
          onClick={(e) => e.stopPropagation()}
          aria-label={t("Select {name}", { name: img.name })}
          className="h-4 w-4 cursor-pointer accent-accent"
        />
      </td>
      <td className="max-w-[18rem] px-4 py-2 font-medium">
        <button
          onClick={() => onCrop(img)}
          title={t("Open editing studio: {name}", { name: img.name })}
          aria-label={t("Open editing studio {name}", { name: img.name })}
          className="flex w-full items-center gap-2.5 text-left"
        >
          <span className="grid h-11 w-11 shrink-0 place-items-center overflow-hidden rounded-lg border border-line bg-elevated">
            {thumbable ? (
              <img
                src={convertFileSrc(img.path)}
                alt=""
                loading="lazy"
                draggable={false}
                className="h-full w-full object-cover transition-transform duration-300 ease-out"
                style={{ transform }}
              />
            ) : (
              <ImageIcon size={16} className="text-muted" />
            )}
          </span>
          <span className="min-w-0 flex-1 truncate" title={img.path}>
            {img.name}
          </span>
        </button>
      </td>
      <td className="px-4 py-2">
        <span className="rounded-full bg-accent-soft px-2 py-0.5 font-mono text-[11px] text-accent-text">
          {img.format}
        </span>
      </td>
      <td className="px-4 py-2 text-right tabular-nums text-muted">
        {img.width} × {img.height}
      </td>
      <td className="px-4 py-2 text-right tabular-nums text-muted">{formatSize(img.size_bytes)}</td>
      <td className="px-4 py-2 text-xs">
        {res ? (
          res.success ? (
            <div className="space-y-1">
              <div className="flex items-center gap-1 text-ok">
                <Check size={12} />
                <span className="tabular-nums">
                  {formatSize(res.new_size)} ({res.saved_percent.toFixed(1)}%)
                </span>
              </div>
              <Bar pct={100} color="bg-ok" />
            </div>
          ) : (
            <div className="space-y-1" title={res.error ?? undefined}>
              <div className="flex items-center gap-1 text-bad">
                <AlertTriangle size={12} />
                {t("Error")}
              </div>
              <Bar pct={100} color="bg-bad" />
            </div>
          )
        ) : running ? (
          <div className="space-y-1">
            <div className="flex items-center gap-1 text-muted">
              <Loader2 size={12} className="animate-spin" />
              {t("Processing…")}
            </div>
            <Bar pct={40} color="bg-accent animate-pulse" />
          </div>
        ) : (
          <span className="text-muted">{t("Ready")}</span>
        )}
      </td>
      <td className="px-2 py-2">
        <div className="flex items-center justify-end gap-0.5">
          {res?.success && (
            <>
              <button
                onClick={() => onCopy(res.output_path)}
                title={copied ? t("Copied!") : t("Copy result image to clipboard")}
                aria-label={t("Copy {name}", { name: img.name })}
                className={`${iconBtn} ${copied ? "text-ok" : "text-muted hover:text-fg"}`}
              >
                {copied ? <Check size={15} /> : <Copy size={15} />}
              </button>
              <button
                onClick={() => onCompare(res)}
                title={t("Compare Before vs After")}
                aria-label={t("Compare {name}", { name: img.name })}
                className={`${iconBtn} text-muted hover:text-fg`}
              >
                <Columns2 size={15} />
              </button>
            </>
          )}
          <button
            onClick={() => onCrop(img)}
            title={cropped ? t("Crop set — edit") : t("Crop image")}
            aria-label={`Crop ${img.name}`}
            className={`${iconBtn} ${cropped ? "text-accent-text" : "text-muted hover:text-fg"}`}
          >
            <Crop size={15} />
          </button>
          <button
            onClick={onRemove}
            title={t("Remove from list")}
            aria-label={t("Remove {name} from list", { name: img.name })}
            className={`${iconBtn} text-muted opacity-0 transition-opacity hover:bg-bad/10 hover:text-bad focus-visible:opacity-100 group-hover:opacity-100`}
          >
            <Trash2 size={15} />
          </button>
        </div>
      </td>
    </tr>
  );
}

type CompareMode = "slider" | "side";
type Dim = { w: number; h: number };

const clampZoom = (z: number) => Math.min(8, Math.max(0.05, z));

const ZOOM_STEPS: { label: string; value: number | null }[] = [
  { label: "Fit", value: null },
  { label: "100%", value: 1 },
  { label: "200%", value: 2 },
  { label: "400%", value: 4 },
];

/** Satu panel imej: saiz semula jadi × zum, pan dikongsi antara panel. */
function ComparePane({
  label,
  src,
  size,
  dim,
  onDim,
  zoom,
  pan,
  className,
  onPointerDown,
  onPointerMove,
  onPointerUp,
  badgeRight,
  fixed,
}: {
  label: string;
  src: string;
  size: number;
  dim: Dim | null;
  onDim: (d: Dim) => void;
  zoom: number;
  pan: { x: number; y: number };
  className: string;
  onPointerDown: (e: React.PointerEvent<HTMLDivElement>) => void;
  onPointerMove: (e: React.PointerEvent<HTMLDivElement>) => void;
  onPointerUp: () => void;
  badgeRight?: boolean;
  /** Dimensi paparan tetap (mod Gelangsar): kedua-dua lapisan guna saiz Hasil. */
  fixed?: Dim | null;
}) {
  const shown = fixed ?? dim;
  return (
    <div
      data-pane
      className={`touch-none select-none overflow-hidden bg-panel ${className}`}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
    >
      <div className="absolute inset-0 grid place-items-center">
        <img
          src={src}
          alt={label}
          draggable={false}
          onLoad={(e) => onDim({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
          className="select-none"
          style={
            shown
              ? {
                  width: shown.w * zoom,
                  height: shown.h * zoom,
                  maxWidth: "none",
                  maxHeight: "none",
                  objectFit: fixed ? "fill" : undefined,
                  transform: `translate(${pan.x}px, ${pan.y}px)`,
                }
              : { maxWidth: "100%", maxHeight: "100%" }
          }
        />
      </div>
      <span
        className={`pointer-events-none absolute top-2 z-10 rounded bg-black/70 px-2 py-0.5 text-[11px] tabular-nums text-white ${
          badgeRight ? "right-2" : "left-2"
        }`}
      >
        {label}: {dim ? `${dim.w}×${dim.h}` : "…"} • {formatSize(size)}
      </span>
    </div>
  );
}

function CompareModal({ res, onClose }: { res: ProcessResult; onClose: () => void }) {
  const [mode, setMode] = useState<CompareMode>(() => {
    try {
      return localStorage.getItem("morphimage-compare-mode") === "slider" ? "slider" : "side";
    } catch {
      return "side";
    }
  });
  const [pos, setPos] = useState(50);
  /** null = Fit (muat dalam panel). */
  const [zoom, setZoom] = useState<number | null>(null);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [dims, setDims] = useState<{ before: Dim | null; after: Dim | null }>({ before: null, after: null });
  const [box, setBox] = useState({ w: 0, h: 0 });
  const canvasRef = useRef<HTMLDivElement>(null);
  const drag = useRef<{ x: number; y: number; px: number; py: number } | null>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  useEffect(() => {
    try {
      localStorage.setItem("morphimage-compare-mode", mode);
    } catch {
      /* abaikan */
    }
  }, [mode]);

  // Saiz kanvas untuk kira skala Fit
  useEffect(() => {
    const el = canvasRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setBox({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    setBox({ w: el.clientWidth, h: el.clientHeight });
    return () => ro.disconnect();
  }, []);

  // Lebar satu panel (2 panel + jurang 8px dalam mod bersebelahan)
  const paneW = mode === "side" ? (box.w - 8) / 2 : box.w;
  const fitZoom = useMemo(() => {
    const ds = [dims.before, dims.after].filter(Boolean) as Dim[];
    if (!paneW || !box.h || !ds.length) return 1;
    return Math.min(...ds.map((d) => Math.min(paneW / d.w, box.h / d.h)));
  }, [dims, paneW, box.h]);
  const z = zoom ?? fitZoom;

  // Skrol roda = zum berpusat pada kursor (listener native supaya preventDefault berkesan)
  useEffect(() => {
    const el = canvasRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const cur = zoom ?? fitZoom;
      const next = clampZoom(cur * Math.exp(-e.deltaY * 0.0015));
      const k = next / cur;
      const pane = (e.target as HTMLElement).closest?.("[data-pane]") as HTMLElement | null;
      const r = (pane ?? el).getBoundingClientRect();
      const cx = e.clientX - (r.left + r.width / 2);
      const cy = e.clientY - (r.top + r.height / 2);
      // Pan dikongsi: sauh kursor pada panel yang di-hover, panel lain ikut serentak.
      setPan((p) => ({ x: cx - (cx - p.x) * k, y: cy - (cy - p.y) * k }));
      setZoom(next);
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [zoom, fitZoom]);

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (z <= fitZoom) return; // Fit: tiada ruang untuk digeser
    e.currentTarget.setPointerCapture(e.pointerId);
    drag.current = { x: e.clientX, y: e.clientY, px: pan.x, py: pan.y };
  };
  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const d = drag.current;
    if (!d) return;
    setPan({ x: d.px + e.clientX - d.x, y: d.py + e.clientY - d.y });
  };
  const onPointerUp = () => {
    drag.current = null;
  };

  const before = convertFileSrc(res.source_path);
  const after = convertFileSrc(res.output_path);
  const diff = res.original_size - res.new_size;
  const paneProps = { zoom: z, pan, onPointerDown, onPointerMove, onPointerUp };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={t("Before and After Comparison")}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-6"
      onClick={onClose}
    >
      <div
        className="modal-surface flex h-full max-h-[92vh] w-full max-w-6xl flex-col gap-3 rounded-xl border border-line bg-bg p-4"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
          <div className="flex items-center gap-3">
            <div className="flex rounded-md border border-line p-0.5" role="group" aria-label={t("View mode")}>
              {(
                [
                  { value: "side", label: t("Side by Side") },
                  { value: "slider", label: t("Before/After Slider") },
                ] as const
              ).map((m) => (
                <button
                  key={m.value}
                  onClick={() => {
                    setMode(m.value);
                    setPan({ x: 0, y: 0 });
                  }}
                  aria-pressed={mode === m.value}
                  className={`rounded px-2.5 py-1 text-xs font-medium transition-colors ${
                    mode === m.value ? "bg-accent-soft text-accent-text" : "text-muted hover:bg-hover hover:text-fg"
                  }`}
                >
                  {m.label}
                </button>
              ))}
            </div>
            <span className="text-xs tabular-nums text-muted">
              {diff >= 0 ? t("Saved") : t("Increased")}:{" "}
              <b className={`tabular-nums ${diff >= 0 ? "text-ok" : "text-amber-500"}`}>
                {diff >= 0 ? "" : "+"}
                {formatSize(Math.abs(diff))} ({Math.abs(res.saved_percent).toFixed(1)}%)
              </b>
            </span>
          </div>

          <div className="flex items-center gap-2">
            <div className="flex rounded-md border border-line p-0.5" role="group" aria-label={t("Zoom")}>
              {ZOOM_STEPS.map((s) => (
                <button
                  key={s.label}
                  onClick={() => {
                    setZoom(s.value);
                    setPan({ x: 0, y: 0 });
                  }}
                  aria-pressed={zoom === s.value}
                  className={`rounded px-2 py-1 text-xs font-medium tabular-nums transition-colors ${
                    zoom === s.value ? "bg-accent-soft text-accent-text" : "text-muted hover:bg-hover hover:text-fg"
                  }`}
                >
                  {s.label}
                </button>
              ))}
            </div>
            <span className="w-12 text-right text-xs tabular-nums text-muted">{Math.round(z * 100)}%</span>
            <button
              onClick={onClose}
              aria-label={t("Close")}
              className="rounded p-1.5 text-muted hover:bg-hover hover:text-fg"
            >
              <X size={18} />
            </button>
          </div>
        </div>

        <div
          ref={canvasRef}
          className={`flex min-h-0 flex-1 gap-2 ${z > fitZoom ? "cursor-grab active:cursor-grabbing" : ""}`}
        >
          {mode === "side" ? (
            <>
              <ComparePane
                label={t("Original")}
                src={before}
                size={res.original_size}
                dim={dims.before}
                onDim={(d) => setDims((v) => ({ ...v, before: d }))}
                className="relative flex-1 rounded-lg border border-line"
                {...paneProps}
              />
              <ComparePane
                label={t("Result")}
                src={after}
                size={res.new_size}
                dim={dims.after}
                onDim={(d) => setDims((v) => ({ ...v, after: d }))}
                className="relative flex-1 rounded-lg border border-line"
                badgeRight
                {...paneProps}
              />
            </>
          ) : (
            <div className="relative min-h-0 flex-1 overflow-hidden rounded-lg bg-panel">
              <ComparePane
                label={t("Result")}
                src={after}
                size={res.new_size}
                dim={dims.after}
                onDim={(d) => setDims((v) => ({ ...v, after: d }))}
                fixed={dims.after}
                className="absolute inset-0"
                {...paneProps}
              />
              <div className="absolute inset-0" style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}>
                <ComparePane
                  label={t("Original")}
                  src={before}
                  size={res.original_size}
                  dim={dims.before}
                  onDim={(d) => setDims((v) => ({ ...v, before: d }))}
                  fixed={dims.after}
                  className="absolute inset-0"
                  {...paneProps}
                />
              </div>
              <div
                className="pointer-events-none absolute inset-y-0 w-0.5 bg-white shadow"
                style={{ left: `${pos}%` }}
              />
              {[
                { text: t("Original"), side: "left-2" },
                { text: t("Result"), side: "right-2" },
              ].map((l) => (
                <span
                  key={l.text}
                  className={`pointer-events-none absolute top-2 z-10 rounded bg-black/70 px-2 py-0.5 text-[11px] font-medium text-white ${l.side}`}
                >
                  {l.text}
                </span>
              ))}
            </div>
          )}
        </div>

        {mode === "slider" && (
          <input
            type="range"
            min={0}
            max={100}
            value={pos}
            onChange={(e) => setPos(Number(e.target.value))}
            aria-label={t("Comparison slider")}
            className="w-full accent-accent"
          />
        )}
        <p className="text-center text-[11px] text-muted">
          {t("Scroll to zoom • drag to pan — both panels move together")}
        </p>
      </div>
    </div>
  );
}

const ASPECTS: { key: string; label: string; ratio: number | null }[] = [
  { key: "free", label: "Free", ratio: null },
  { key: "1:1", label: "1:1", ratio: 1 },
  { key: "4:3", label: "4:3", ratio: 4 / 3 },
  { key: "16:9", label: "16:9", ratio: 16 / 9 },
  { key: "9:16", label: "9:16", ratio: 9 / 16 },
];

const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

/** Putar kotak ternormal 90° ikut arah jam. */
function rotRect(r: CropRect, deg: number): CropRect {
  switch (((deg % 360) + 360) % 360) {
    case 90:
      return { x: 1 - r.y - r.height, y: r.x, width: r.height, height: r.width };
    case 180:
      return { x: 1 - r.x - r.width, y: 1 - r.y - r.height, width: r.width, height: r.height };
    case 270:
      return { x: r.y, y: 1 - r.x - r.width, width: r.height, height: r.width };
    default:
      return r;
  }
}

/** Ruang imej asal → ruang paparan. Enjin Rust: crop → putar → cermin → flip. */
function sourceToDisplay(r: CropRect, deg: number, fh: boolean, fv: boolean): CropRect {
  const q = rotRect(r, deg);
  return {
    x: fh ? 1 - q.x - q.width : q.x,
    y: fv ? 1 - q.y - q.height : q.y,
    width: q.width,
    height: q.height,
  };
}

/** Songsang bagi `sourceToDisplay`: ruang paparan → ruang imej asal. */
function displayToSource(r: CropRect, deg: number, fh: boolean, fv: boolean): CropRect {
  return rotRect(
    {
      x: fh ? 1 - r.x - r.width : r.x,
      y: fv ? 1 - r.y - r.height : r.y,
      width: r.width,
      height: r.height,
    },
    360 - (((deg % 360) + 360) % 360),
  );
}

/** Modals pratonton crop: kotak boleh ditarik, koordinat ternormal 0.0–1.0. */
function CropModal({
  img,
  initial,
  rotateDeg,
  flipH,
  flipV,
  onOrient,
  onClose,
  onSave,
  onClear,
}: {
  img: ImageInfo;
  initial: CropRect | undefined;
  rotateDeg: number;
  flipH: boolean;
  flipV: boolean;
  onOrient: (deg: number, fh: boolean, fv: boolean) => void;
  onClose: () => void;
  onSave: (rect: CropRect) => void;
  onClear: () => void;
}) {
  const [rect, setRect] = useState<CropRect>(() =>
    initial
      ? sourceToDisplay(initial, rotateDeg, flipH, flipV)
      : { x: 0, y: 0, width: 1, height: 1 },
  );
  const [aspectKey, setAspectKey] = useState("free");
  const [nat, setNat] = useState({ w: img.width || 1, h: img.height || 1 });
  const [imgOk, setImgOk] = useState(true);
  const boxRef = useRef<HTMLDivElement>(null);
  const ratio = ASPECTS.find((a) => a.key === aspectKey)?.ratio ?? null;

  // Dimensi selepas orientasi: putar 90°/270° menukar lebar & tinggi.
  const swapped = rotateDeg % 180 !== 0;
  const effW = swapped ? nat.h : nat.w;
  const effH = swapped ? nat.w : nat.h;

  /** Tukar orientasi tanpa mengubah kawasan imej yang dipilih. */
  function reorient(deg: number, fh: boolean, fv: boolean) {
    const srcRect = displayToSource(rect, rotateDeg, flipH, flipV);
    setRect(sourceToDisplay(srcRect, deg, fh, fv));
    onOrient(deg, fh, fv);
  }

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  /** Selaraskan tinggi kotak dengan nisbah aspek terpilih (dalam piksel sebenar). */
  function snapToRatio(r: CropRect, ratioNow: number): CropRect {
    const h = (r.width * effW) / (ratioNow * effH);
    if (h <= 1 - r.y) return { ...r, height: Math.max(0.02, h) };
    // Tidak muat secara menegak: kecilkan lebar supaya tinggi muat.
    const w = ((1 - r.y) * ratioNow * effH) / effW;
    return { ...r, width: Math.max(0.02, w), height: 1 - r.y };
  }

  function beginDrag(mode: "move" | "nw" | "ne" | "sw" | "se", e: React.PointerEvent) {
    e.preventDefault();
    e.stopPropagation();
    const start = { ...rect };
    const ratioNow = ratio;
    const origin = { cx: e.clientX, cy: e.clientY };

    const onMove = (ev: PointerEvent) => {
      const b = boxRef.current?.getBoundingClientRect();
      if (!b || b.width === 0 || b.height === 0) return;
      const dx = (ev.clientX - origin.cx) / b.width;
      const dy = (ev.clientY - origin.cy) / b.height;

      if (mode === "move") {
        setRect({
          ...start,
          x: Math.min(Math.max(start.x + dx, 0), 1 - start.width),
          y: Math.min(Math.max(start.y + dy, 0), 1 - start.height),
        });
        return;
      }

      if (ratioNow) {
        // Kunci nisbah: saiz ikut lebar; bucu bertentangan kekal tetap.
        let w = start.width + (mode.includes("w") ? -dx : dx);
        w = Math.max(0.02, w);
        let x = mode.includes("w") ? start.x + start.width - w : start.x;
        let h = (w * effW) / (ratioNow * effH);
        if (h > 1) { h = 1; w = (h * ratioNow * effH) / effW; x = mode.includes("w") ? start.x + start.width - w : start.x; }
        let y = mode.includes("n") ? start.y + start.height - h : start.y;
        x = clamp01(x); y = clamp01(y);
        setRect({ x, y, width: Math.min(w, 1 - x), height: Math.min(h, 1 - y) });
        return;
      }

      let { x, y, width, height } = start;
      if (mode.includes("e")) width = start.width + dx;
      if (mode.includes("s")) height = start.height + dy;
      if (mode.includes("w")) { width = start.width - dx; x = start.x + dx; }
      if (mode.includes("n")) { height = start.height - dy; y = start.y + dy; }
      if (width < 0.02) { width = 0.02; x = start.x + (mode.includes("w") ? start.width - 0.02 : 0); }
      if (height < 0.02) { height = 0.02; y = start.y + (mode.includes("n") ? start.height - 0.02 : 0); }
      setRect({
        x: clamp01(x),
        y: clamp01(y),
        width: Math.min(width, 1 - clamp01(x)),
        height: Math.min(height, 1 - clamp01(y)),
      });
    };
    const onUp = () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  }

  function pickAspect(key: string) {
    setAspectKey(key);
    const r = ASPECTS.find((a) => a.key === key)?.ratio;
    if (r) setRect((prev) => snapToRatio({ ...prev, x: clamp01(prev.x), y: clamp01(prev.y) }, r));
  }

  const src = convertFileSrc(img.path);
  const pxW = Math.round(rect.width * effW);
  const pxH = Math.round(rect.height * effH);

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={t("Editing studio {name}", { name: img.name })}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 p-4 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="modal-surface flex max-h-full w-full max-w-5xl flex-col gap-3 rounded-2xl border border-line bg-bg p-5 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between gap-3">
          <div className="min-w-0">
            <p className="truncate text-sm font-semibold">{t("Editing Studio")}</p>
            <p className="truncate text-xs text-muted" title={img.path}>
              {img.name} · {effW} × {effH} px
            </p>
          </div>
          <button
            onClick={onClose}
            aria-label={t("Close")}
            className="rounded-md p-1.5 text-muted transition-colors hover:bg-hover hover:text-fg"
          >
            <X size={18} />
          </button>
        </div>

        <div className="flex flex-wrap items-center gap-1.5">
          <button
            onClick={() => reorient((rotateDeg + 270) % 360, flipH, flipV)}
            title={t("Rotate 90° left")}
            aria-label={t("Rotate 90 degrees left")}
            className={BTN}
          >
            <RotateCcw size={14} />
            {t("90° Left")}
          </button>
          <button
            onClick={() => reorient((rotateDeg + 90) % 360, flipH, flipV)}
            title={t("Rotate 90° right")}
            aria-label={t("Rotate 90 degrees right")}
            className={BTN}
          >
            <RotateCw size={14} />
            {t("90° Right")}
          </button>
          <button
            onClick={() => reorient(rotateDeg, !flipH, flipV)}
            aria-pressed={flipH}
            title={t("Mirror horizontally")}
            aria-label={t("Mirror horizontally")}
            className={flipH ? BTN_ON : BTN}
          >
            <FlipHorizontal size={14} />
            {t("Mirror")}
          </button>
          <button
            onClick={() => reorient(rotateDeg, flipH, !flipV)}
            aria-pressed={flipV}
            title={t("Flip vertically")}
            aria-label={t("Flip vertically")}
            className={flipV ? BTN_ON : BTN}
          >
            <FlipVertical size={14} />
            Flip
          </button>
          <span className="mx-1 h-5 w-px bg-line" />
          {ASPECTS.map((a) => (
            <button
              key={a.key}
              onClick={() => pickAspect(a.key)}
              aria-pressed={aspectKey === a.key}
              className={`h-8 rounded-md border px-2.5 text-xs transition-colors ${
                aspectKey === a.key
                  ? "border-accent bg-accent-soft text-accent-text"
                  : "border-line text-muted hover:bg-hover"
              }`}
            >
              {t(a.label)}
            </button>
          ))}
          <span className="ml-auto tabular-nums text-xs text-muted">
            {pxW} × {pxH} px
          </span>
        </div>

        <div className="flex min-h-0 flex-1 items-center justify-center overflow-hidden rounded-xl border border-line bg-elevated p-3">
          <div
            ref={boxRef}
            className="relative select-none overflow-hidden rounded-lg bg-black/40"
            style={{
              width: `min(100%, calc(52vh * ${effW / effH}))`,
              aspectRatio: `${effW} / ${effH}`,
            }}
          >
            {imgOk ? (
              <img
                src={src}
                alt={img.name}
                draggable={false}
                onLoad={(e) => {
                  const t = e.currentTarget;
                  if (t.naturalWidth && t.naturalHeight) setNat({ w: t.naturalWidth, h: t.naturalHeight });
                }}
                onError={() => setImgOk(false)}
                className="absolute left-1/2 top-1/2 max-w-none transition-transform duration-300 ease-out"
                style={{
                  transform: `translate(-50%, -50%) ${previewTransform(rotateDeg, flipH, flipV)}`,
                  width: swapped ? `${(effH / effW) * 100}%` : "100%",
                  height: swapped ? `${(effW / effH) * 100}%` : "100%",
                }}
              />
            ) : (
              <div className="flex h-full flex-col items-center justify-center gap-2 text-muted">
                <ImageIcon size={28} />
                <p className="px-4 text-center text-xs">{t("Preview not available for this format")}</p>
              </div>
            )}
            {/* Kotak crop */}
            <div
              onPointerDown={(e) => beginDrag("move", e)}
              className="absolute cursor-move border border-accent shadow-[0_0_0_9999px_rgba(0,0,0,0.55)]"
              style={{
                left: `${rect.x * 100}%`,
                top: `${rect.y * 100}%`,
                width: `${rect.width * 100}%`,
                height: `${rect.height * 100}%`,
              }}
            >
              <span className="pointer-events-none absolute inset-0 grid grid-cols-3 grid-rows-3">
                {Array.from({ length: 9 }, (_, i) => (
                  <span key={i} className="border border-white/15" />
                ))}
              </span>
              {(["nw", "ne", "sw", "se"] as const).map((corner) => (
                <span
                  key={corner}
                  onPointerDown={(e) => beginDrag(corner, e)}
                  className="absolute h-3.5 w-3.5 rounded-full border-2 border-white bg-accent shadow"
                  style={{
                    left: corner.includes("w") ? -7 : undefined,
                    right: corner.includes("e") ? -7 : undefined,
                    top: corner.includes("n") ? -7 : undefined,
                    bottom: corner.includes("s") ? -7 : undefined,
                    cursor: `${corner}-resize`,
                  }}
                />
              ))}
            </div>
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-2">
          <p className="tabular-nums text-xs text-muted">
            x {rect.x.toFixed(2)} · y {rect.y.toFixed(2)} · w {rect.width.toFixed(2)} · h{" "}
            {rect.height.toFixed(2)}
            <span className="ml-2 text-fg">
              {pxW} × {pxH} px
            </span>
          </p>
          <div className="flex gap-2">
            {initial && (
              <button
                onClick={onClear}
                className="rounded-md border border-line px-3 py-1.5 text-sm text-muted transition-colors hover:bg-hover hover:text-fg"
              >
                {t("Remove Crop")}
              </button>
            )}
            <button
              onClick={() => onSave(displayToSource(rect, rotateDeg, flipH, flipV))}
              className="flex items-center gap-2 rounded-md bg-gradient-to-r from-accent to-accent-2 px-3.5 py-1.5 text-sm font-medium text-white shadow-sm transition-opacity hover:opacity-90"
            >
              <Check size={15} />
              {t("Save Changes")}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function Bar({ pct, color }: { pct: number; color: string }) {
  return (
    <div className="h-1 overflow-hidden rounded-full bg-hover">
      <div className={`h-full ${color}`} style={{ width: `${pct}%` }} />
    </div>
  );
}
