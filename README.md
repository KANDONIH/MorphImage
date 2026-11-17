# MorphImage

A fast, fully offline image workspace for Windows. MorphImage batches and compresses images across many formats, upscales them with AI, restores faces, removes backgrounds, and does it all locally — no cloud, no uploading your photos.

![MorphImage](logo-morphimage.png)

---

## Features

### Batch Image Converter
Convert hundreds of images in one pass. Supported input and output formats include **JPEG, PNG, WebP, AVIF, GIF, BMP, TIFF, ICO, JPEG XL (JXL), SVG, PSD (Photoshop), and RAW**. Files are processed in parallel across all CPU cores, with per-file and overall progress.

### AI Upscaler
Increase resolution using ONNX super-resolution models, including **RealESRGAN x4plus** and **RealESRGAN x4plus Anime 6B**, plus any compatible custom model you import. Models are downloaded on demand and cached locally.

### Face Restoration
Repair blurred or low-quality faces with **CodeFormer** or **GFPGAN** style 512×512 ONNX models. A fidelity slider controls how far the restored face blends toward the original — high fidelity preserves identity, low fidelity cleans up more aggressively.

### Background Removal
Cut subjects out of images with **RMBG-1.4** (default) or **BiRefNet-lite**. Outputs transparent PNGs, and can run in the same job as conversion and upscaling.

### Custom `.onnx` / `.pth` Model Import
Drop your own models into the app. Import `.onnx` models for upscaling, face restoration, or background removal — each model class is tracked separately in its own models folder. PyTorch `.pth` checkpoints can be converted to ONNX from inside the app on import.

### 4GB VRAM Safe Mode & CPU Fallback
Every AI task runs on a **DirectML GPU** when it can, and falls back to **CPU** when it cannot:

| Mode | Behaviour |
| --- | --- |
| **Auto (4GB VRAM Safe Mode)** | Default. Heavy models run on CPU when GPU VRAM is ≤ 4 GB, preventing driver resets on low-VRAM machines. |
| **Force GPU (DirectML)** | Skip Safe Mode and always try the GPU. Faster, but may reset the display driver on low-VRAM GPUs. |
| **Force CPU** | Never touches the GPU. Slowest, but always stable. |

CPU work uses a configurable worker count (default: **total threads − 2**, so Windows stays responsive during large batches).

### Liquid Glass / Sky Frost Themes
A React + Tailwind interface in the Liquid Glass style, with a **Sky Frost** light theme. Interface available in **English and Bahasa Melayu**.

---

## System Requirements

| | Minimum | Recommended |
| --- | --- | --- |
| OS | Windows 10 (64-bit) | Windows 11 (64-bit) |
| RAM | 8 GB | 16 GB |
| GPU | Any (CPU fallback available) | DirectX 12 GPU with 6 GB+ VRAM |
| Disk | ~200 MB installed | + 1–2 GB for AI models |
| WebView2 | Required (pre-installed on Windows 11) | Latest |

AI features run on CPU without a GPU, but are substantially slower. GPU acceleration uses **DirectML** on any DirectX 12-capable graphics card (NVIDIA, AMD, or Intel).

---

## Installation

### Installer (recommended)
1. Download the latest `MorphImage_<version>_x64-setup.exe` from Releases.
2. Run the installer and follow the prompts.
3. Launch **MorphImage** from the Start Menu.

The installer is not code-signed, so Windows SmartScreen may show a warning — choose **More info → Run anyway**.

### AI Models
Models are **not bundled**. On first use of an AI feature, MorphImage downloads the selected model into your model storage folder (see **Settings → AI & Performance → Model Storage**). Custom models can be added with **Import** in the relevant dialog.

---

## Building from Source

**Prerequisites**
- [Node.js](https://nodejs.org/) 18+ and npm
- [Rust](https://rustup.rs/) (stable, MSVC toolchain)
- [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with "Desktop development with C++"
- WebView2 runtime

```bash
# Install frontend dependencies
npm install

# Development (hot reload)
npm run tauri dev

# Production build — installers go to src-tauri/target/release/bundle/
npm run tauri build
```

The Windows installer is produced at:
`src-tauri/target/release/bundle/nsis/MorphImage_<version>_x64-setup.exe`

(`<version>` comes from `package.json` / `tauri.conf.json` — currently `0.1.0`.)

---

## Tech Stack

- **Frontend:** React 19, TypeScript, Vite, Tailwind CSS v4, lucide-react
- **Backend:** Rust, Tauri v2, ONNX Runtime (`ort` with DirectML), `image`, `rayon`, `oxipng`, `resvg`, `psd`, `jxl-oxide`, `rawler`, `lopdf`, `sysinfo`

---

## License

Released under the [MIT License](LICENSE).

Copyright © 2026 Kando / Muhammad Danial Iskandar.
