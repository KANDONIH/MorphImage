import { useEffect, useRef } from "react";
import { openUrl } from "@tauri-apps/plugin-opener";
import { BookOpen, ExternalLink, X } from "lucide-react";
import { t } from "../i18n";

type Vram = "4GB VRAM OK" | "6GB+ VRAM" | "8GB+ VRAM";
interface GuideModel {
  name: string;
  mb: string;
  use: string;
  vram: Vram;
  /** Pilihan penormalan semasa import (Buang Latar sahaja). */
  norm?: string;
}

const UPSCALERS: GuideModel[] = [
  { name: "realesr-general-x4v3", mb: "~5 MB", use: "Fast & light", vram: "4GB VRAM OK" },
  { name: "RealESRGAN_x4plus_anime_6B", mb: "~18 MB", use: "Anime, logos & illustrations", vram: "4GB VRAM OK" },
  { name: "RealESRGAN_x4plus", mb: "~67 MB", use: "Real photos & portraits", vram: "6GB+ VRAM" },
  { name: "4x-UltraSharp", mb: "~67 MB", use: "Sharpness & blurry text", vram: "6GB+ VRAM" },
  { name: "4x_NMKD-Siax_200k", mb: "~67 MB", use: "Natural skin & fabric texture", vram: "6GB+ VRAM" },
  { name: "HAT-L / SwinIR-L ONNX", mb: "~100–160 MB", use: "Maximum studio-grade quality", vram: "8GB+ VRAM" },
];

const BG_REMOVERS: GuideModel[] = [
  { name: "BRIA RMBG-1.4", mb: "~44 MB", use: "Fast & light for 4GB VRAM", vram: "4GB VRAM OK", norm: "Standard" },
  { name: "BiRefNet_lite (FP16)", mb: "~88 MB", use: "Best fine hair & fur cutouts • MIT License", vram: "6GB+ VRAM", norm: "ImageNet" },
  { name: "BEN2 (ONNX)", mb: "~210 MB", use: "Shadow & complex lighting separation • Apache 2.0", vram: "8GB+ VRAM", norm: "ImageNet" },
  { name: "BRIA RMBG-2.0 (ONNX)", mb: "~220 MB", use: "High accuracy for products & portraits", vram: "8GB+ VRAM", norm: "ImageNet" },
];

const FACE_RESTORERS: GuideModel[] = [
  { name: "CodeFormer (codeformer.onnx)", mb: "~370 MB", use: "Restores extremely blurry faces", vram: "6GB+ VRAM" },
  { name: "GFPGAN v1.4 (GFPGANv1.4.onnx)", mb: "~340 MB", use: "Fast, natural portrait restoration", vram: "6GB+ VRAM" },
];

const VRAM_TONE: Record<Vram, string> = {
  "4GB VRAM OK": "border-ok/40 text-ok",
  "6GB+ VRAM": "border-amber-500/40 text-amber-600",
  "8GB+ VRAM": "border-red-500/40 text-red-500",
};

const HF_ONNX = "https://huggingface.co/models?library=onnx&sort=downloads";
const OPENMODELDB = "https://openmodeldb.info/";
const hfSearch = (name: string) =>
  `https://huggingface.co/models?library=onnx&search=${encodeURIComponent(name.split(" (")[0].split(" / ")[0])}`;

const LINK =
  "inline-flex h-8 items-center gap-1.5 rounded-md border border-line px-2.5 text-xs font-medium text-fg transition-colors hover:bg-hover";

function go(url: string) {
  void openUrl(url).catch(() => {});
}

function ModelTable({ title, items }: { title: string; items: GuideModel[] }) {
  return (
    <section className="space-y-2">
      <h3 className="text-xs font-semibold uppercase tracking-wider text-muted">{title}</h3>
      <ul className="divide-y divide-line rounded-lg border border-line">
        {items.map((m) => (
          <li key={m.name} className="flex flex-wrap items-center gap-x-3 gap-y-1 px-3 py-2.5">
            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-medium">{m.name}</p>
              <p className="text-xs text-muted">
                {m.mb} • {t(m.use)}
                {m.norm && <> {t("• Import as")} <b className="text-fg">{m.norm}</b></>}
              </p>
            </div>
            <span className={`shrink-0 rounded-full border px-2 py-0.5 text-[10px] font-medium ${VRAM_TONE[m.vram]}`}>
              {m.vram}
            </span>
            <button onClick={() => go(hfSearch(m.name))} className="shrink-0 text-xs text-accent-text hover:underline">
              {t("Search →")}
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}

export default function AiGuideModal({ onClose }: { onClose: () => void }) {
  const closeRef = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    closeRef.current?.focus();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="ai-guide-title"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4"
      onClick={onClose}
    >
      <div
        className="modal-surface flex max-h-[90vh] w-full max-w-2xl flex-col rounded-xl border border-line bg-bg shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-2 border-b border-line px-5 py-3">
          <BookOpen size={16} className="text-accent" />
          <h2 id="ai-guide-title" className="text-sm font-semibold">
            {t("AI Guidebook")}
          </h2>
          <button
            ref={closeRef}
            onClick={onClose}
            aria-label={t("Close guide")}
            className="ml-auto inline-flex h-7 w-7 items-center justify-center rounded-md text-muted hover:bg-hover hover:text-fg"
          >
            <X size={15} />
          </button>
        </div>

        <div className="space-y-6 overflow-y-auto px-5 py-4">
          <section className="space-y-2">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-muted">{t("How to Use (3 Easy Steps)")}</h3>
            <ol className="grid gap-2 sm:grid-cols-3">
              {[
                [t("Download an .onnx or .pth file"), t("From Hugging Face or OpenModelDB.")],
                [t("Click + Import Model"), t("In AI Studio, on the Upscaler, Remove Background or Face Restoration card.")],
                [t("Select & Process"), t("Pick the model in the list, then process your images.")],
              ].map(([t, d], i) => (
                <li key={t} className="rounded-lg border border-line p-3">
                  <span className="mb-1 inline-flex h-5 w-5 items-center justify-center rounded-full bg-accent-soft text-[11px] font-semibold text-accent-text">
                    {i + 1}
                  </span>
                  <p className="text-sm font-medium">{t}</p>
                  <p className="text-xs text-muted">{d}</p>
                </li>
              ))}
            </ol>
          </section>

          <section className="space-y-2">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-muted">{t("File Format Notes (.onnx vs .pth)")}</h3>
            <ul className="grid gap-2 sm:grid-cols-2">
              <li className="rounded-lg border border-ok/40 p-3">
                <p className="text-sm font-medium">{t(".onnx (Recommended)")}</p>
                <p className="text-xs text-muted">{t("Works right away with no extra install (0 MB extra storage).")}</p>
              </li>
              <li className="rounded-lg border border-amber-500/40 p-3">
                <p className="text-sm font-medium">.pth (OpenModelDB)</p>
                <p className="text-xs text-muted">
                  {t("Supported automatically if your PC has Python, but installing the")}{" "}
                  <code className="text-fg">torch spandrel onnx</code> {t("libraries takes about ~2.5 GB of disk space. Both .onnx and .pth now work with every AI feature (Upscaler, Remove Background, Face Restoration).")}
                </p>
              </li>
            </ul>
          </section>

          <div className="space-y-4">
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <h3 className="text-sm font-semibold">{t("Best AI Models")}</h3>
              <span className="text-[11px] text-muted">{t("Updated 16/11/2026")}</span>
            </div>
            <ModelTable title="AI Upscaler" items={UPSCALERS} />
            <ModelTable title={t("Remove Background")} items={BG_REMOVERS} />
            <ModelTable title={t("Face Restoration (AI)")} items={FACE_RESTORERS} />
            <p className="text-xs text-muted">
              {t("GPU ≤ 4GB VRAM: models over 60 MB run on CPU automatically so the screen never goes black.")}
            </p>
          </div>
        </div>

        <div className="flex flex-wrap gap-2 border-t border-line px-5 py-3">
          <button onClick={() => go(HF_ONNX)} className={LINK}>
            <ExternalLink size={13} /> Hugging Face (ONNX)
          </button>
          <button onClick={() => go(OPENMODELDB)} className={LINK}>
            <ExternalLink size={13} /> OpenModelDB
          </button>
        </div>
      </div>
    </div>
  );
}
