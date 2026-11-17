import { useEffect, useRef, useState } from "react";
import { Check, ChevronDown, Trash2 } from "lucide-react";
import { t } from "../i18n";

export interface ModelOption {
  id: string;
  label: string;
  /** Teks kecil di bawah label (saiz, skala…). */
  sub?: string;
}

interface Props {
  label: string;
  value: string;
  builtin: ModelOption[];
  custom: ModelOption[];
  onChange: (id: string) => void;
  onDelete: (id: string) => void;
}

export default function ModelSelect({ label, value, builtin, custom, onChange, onDelete }: Props) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => !root.current?.contains(e.target as Node) && setOpen(false);
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const current = [...builtin, ...custom].find((o) => o.id === value);

  const row = (o: ModelOption, deletable: boolean) => (
    <li key={o.id} role="option" aria-selected={o.id === value} className="flex items-center hover:bg-hover">
      <button
        onClick={() => {
          onChange(o.id);
          setOpen(false);
        }}
        className="flex min-w-0 flex-1 items-center gap-2 px-2.5 py-1.5 text-left"
      >
        <Check size={12} className={o.id === value ? "shrink-0 text-accent" : "invisible shrink-0"} />
        <span className="min-w-0">
          <span className="block truncate text-xs">{o.label}</span>
          {o.sub && <span className="block truncate text-[10px] text-muted">{o.sub}</span>}
        </span>
      </button>
      {deletable && (
        <button
          onClick={() => {
            setOpen(false);
            onDelete(o.id);
          }}
          title={t("Delete model")}
          aria-label={t("Delete model {label}", { label: o.label })}
          className="mr-1.5 inline-flex h-6 w-6 shrink-0 items-center justify-center rounded text-muted hover:bg-line hover:text-red-500"
        >
          <Trash2 size={12} />
        </button>
      )}
    </li>
  );

  return (
    <div ref={root} className="relative min-w-0 flex-1">
      <button
        aria-label={label}
        aria-haspopup="listbox"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
        className="inline-flex h-8 w-full items-center justify-between gap-2 rounded-md border border-line bg-panel px-2.5 text-xs font-medium text-fg transition-colors hover:bg-hover"
      >
        <span className="truncate">{current?.label ?? value}</span>
        <ChevronDown size={13} className="shrink-0 text-muted" />
      </button>
      {open && (
        <ul
          role="listbox"
          aria-label={label}
          className="absolute left-0 right-0 z-30 mt-1 max-h-64 overflow-y-auto rounded-md border border-line bg-panel py-1 shadow-lg"
        >
          <li role="presentation" className="px-2.5 pb-0.5 pt-1 text-[10px] font-semibold uppercase tracking-wider text-muted">
            {t("Built-in Models")}
          </li>
          {builtin.map((o) => row(o, false))}
          {custom.length > 0 && (
            <>
              <li role="presentation" className="mt-1 border-t border-line px-2.5 pb-0.5 pt-2 text-[10px] font-semibold uppercase tracking-wider text-muted">
                {t("Imported Models")}
              </li>
              {custom.map((o) => row(o, true))}
            </>
          )}
        </ul>
      )}
    </div>
  );
}
