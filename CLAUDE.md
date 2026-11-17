# Project Rules: Local Desktop Image Utility (Tauri v2 + Rust + React)

## 1. Tech Stack & Strict Versions
- **Desktop Framework:** Tauri v2 ONLY (Do NOT use Tauri v1 APIs like `tauri::api::dialog` or `@tauri-apps/api/tauri`). Use `@tauri-apps/api/core` (`invoke`), `@tauri-apps/plugin-dialog`, and `@tauri-apps/plugin-fs`.
- **Backend:** Rust (Edition 2021+).
- **Frontend:** React 18+, TypeScript, Vite, Tailwind CSS, Lucide Icons.

## 2. Core Architecture Constraints
- **100% Local Processing:** Zero cloud/API calls. All image decoding, resizing, converting, and compressing MUST run in Rust (`src-tauri/src/`).
- **Thin Frontend:** React ONLY handles UI state, native file drop events, settings configuration, and rendering progress updates. Never load raw multi-megabyte image buffers into JS memory for processing.
- **Pure Rust Dependencies Only:** Avoid crates requiring external C/C++ build toolchains (LLVM/CMake/NASM) on Windows unless explicitly requested. Prefer:
  - `image` (decoding, encoding, resizing)
  - `oxipng` (lossless PNG compression)
  - `webp` or `image` webp features (WebP encoding)
  - `rayon` (parallel batch processing)
  - `kamadak-exif` (reading EXIF orientation)
  - `walkdir` (recursive folder scanning)

## 3. Coding & Verification Workflow
- Every time you modify Rust files in `src-tauri/`, ALWAYS run `cargo check` inside `src-tauri/` to verify there are zero compiler or type errors before finishing your turn.
- Respect user CPU limits: Cap Rayon thread pool to `(available_parallelism - 2).max(1)` so the host OS never freezes during heavy batch jobs.
- Always update `src-tauri/capabilities/default.json` whenever adding new Tauri v2 plugins (e.g., `dialog:default`, `fs:default`).