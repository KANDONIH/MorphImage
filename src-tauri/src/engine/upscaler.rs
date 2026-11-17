use fast_image_resize::images::Image;
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
use half::f16;
use image::{DynamicImage, RgbImage, RgbaImage};
use ort::execution_providers::{DirectMLExecutionProvider, ExecutionProvider};
use ort::session::Session;
use ort::tensor::TensorElementType;
use ort::value::{Tensor, ValueType};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{LazyLock, Mutex, OnceLock, RwLock};
use tauri::{Emitter, Manager};

struct ModelSpec {
    id: &'static str,
    file: &'static str,
    url: &'static str,
    sha256: &'static str,
    bytes: u64,
}

pub const DEFAULT_MODEL: &str = "x4v3";

/// Pinned Real-ESRGAN weights (BSD-3-Clause ONNX exports). All: RGB 0..1 NCHW, dynamic H/W, 4x.
const MODELS: [ModelSpec; 4] = [
    ModelSpec {
        id: "x4v3",
        file: "realesr-general-x4v3.onnx",
        url: "https://huggingface.co/CoderViking/realesr-general-x4v3-onnx/resolve/main/realesr-general-x4v3.onnx",
        sha256: "1940a93ee08283a0a7286183186357b1688fe9fa8ede74604b424586aaddf112",
        bytes: 4_866_417,
    },
    ModelSpec {
        id: "x4plus",
        file: "RealESRGAN_x4plus.onnx",
        url: "https://huggingface.co/anakhiu/realesrgan-onnx/resolve/main/realesrgan_x4plus.onnx",
        sha256: "1c0fc21d1d6d7df7f39c1fa23c73138a87a01ea0b3ac56ebd0ea943e62e12800",
        bytes: 67_051_616,
    },
    ModelSpec {
        id: "ultrasharp",
        file: "4x-UltraSharp.onnx",
        url: "https://huggingface.co/Kim2091/UltraSharp/resolve/main/ONNX/4x-UltraSharp-fp32-opset17.onnx",
        sha256: "c83e0bf4fe1051ec7f2330614aa2edef0ad589d59c718934897a93459b180dd1",
        bytes: 67_002_795,
    },
    ModelSpec {
        id: "anime-6b",
        file: "RealESRGAN_x4plus_anime_6B.onnx",
        url: "https://huggingface.co/RekluzLabs/realesrgan_anime6b.onnx/resolve/main/realesrgan_anime6b.onnx",
        sha256: "45bd54934aeabe8df744c8fdacb9e8846c9b55cb4e60c499db77405d1625a667",
        bytes: 18_352_469,
    },
];

fn spec(id: &str) -> Result<&'static ModelSpec, String> {
    MODELS.iter().find(|m| m.id == id).ok_or_else(|| format!("Model upscale tidak dikenali: {id}"))
}

const TILE: u32 = 256;
const OVERLAP: u32 = 16;
const MODEL_SCALE: u32 = 4;
/// Guards against OOM from huge outputs (width * height of the final buffer).
const MAX_OUT_PIXELS: u64 = 150_000_000;

/// Set once at startup (see lib.rs) so the pipeline can find the model without an AppHandle.
pub static MODEL_DIR: OnceLock<PathBuf> = OnceLock::new();
/// Cached session; the lock also serialises AI runs so rayon workers do not fight over VRAM.
/// Holds (model_id, session); switching model replaces it.
static SESSION: Mutex<Option<(String, Session)>> = Mutex::new(None);
/// true while the ort session is running tiles (read by get_system_stats for the footer).
pub static GPU_AI_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Backend of the most recently built AI session, shown in the footer.
pub static ACTIVE_AI_BACKEND: LazyLock<RwLock<String>> = LazyLock::new(|| RwLock::new(LABEL_IDLE.to_string()));

pub const LABEL_IDLE: &str = "Idle";
pub const LABEL_GPU: &str = "GPU (DirectML)";
pub const LABEL_SAFE: &str = "CPU (4GB VRAM Safe Mode)";
pub const LABEL_OOM: &str = "CPU Fallback (VRAM Full)";
const LABEL_FORCED_CPU: &str = "CPU (Forced)";
const LABEL_NO_DML: &str = "CPU Fallback";

/// User override from Settings → AI & Performance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ExecMode {
    /// DirectML, except heavy models on <= 4 GB VRAM (Safe Mode).
    Auto,
    /// DirectML always (skips Safe Mode); OOM still falls back to CPU.
    Gpu,
    /// CPU session only.
    Cpu,
}

impl ExecMode {
    pub fn parse(s: &str) -> Self {
        match s {
            "gpu" => Self::Gpu,
            "cpu" => Self::Cpu,
            _ => Self::Auto,
        }
    }
    /// Session cache key, so switching mode rebuilds the session.
    pub fn key(self, model: &str) -> String {
        format!("{model}|{self:?}")
    }
}

/// Builds the session for `mode`; shared by upscaler, bg remover and face restorer.
pub fn open_session(path: &Path, heavy: bool, mode: ExecMode) -> Result<Session, String> {
    match mode {
        ExecMode::Cpu => build_cpu_session(path, LABEL_FORCED_CPU),
        ExecMode::Auto if safe_cpu_needed(heavy) => build_cpu_session(path, LABEL_SAFE),
        ExecMode::Auto | ExecMode::Gpu => build_session(path),
    }
}

/// GPUs with this much dedicated VRAM (MB) or less run heavy models on CPU.
pub const SAFE_VRAM_MB: u64 = 4608;
/// Imported models above this size count as heavy.
const HEAVY_MB: f64 = 60.0;

/// Heavy model on <= 4 GB VRAM: never touch DirectML (a 100% VRAM spike resets the WebView2 GPU).
pub fn safe_cpu_needed(heavy: bool) -> bool {
    heavy && {
        let vram = get_gpu_info().vram_mb;
        vram > 0 && vram <= SAFE_VRAM_MB
    }
}

pub fn is_f16(t: &ValueType) -> bool {
    matches!(t, ValueType::Tensor { ty: TensorElementType::Float16, .. })
}

// ---------- User-imported models (models/custom_*/ + models/custom_models.json) ----------

const INDEX_FILE: &str = "custom_models.json";
/// Serialises index writes (import/delete).
static INDEX_LOCK: Mutex<()> = Mutex::new(());

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct CustomModel {
    pub id: String,
    /// "upscale" | "bg"
    pub kind: String,
    pub name: String,
    pub file_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_size: Option<u32>,
    /// "standard" (0..1) | "imagenet"
    pub norm_type: String,
    pub size_mb: f64,
}

impl CustomModel {
    pub fn heavy(&self) -> bool {
        self.size_mb > HEAVY_MB
    }
}

fn sub_dir(kind: &str) -> &'static str {
    match kind {
        "bg" => "custom_bg",
        "face" => "custom_face",
        _ => "custom_upscale",
    }
}

fn models_dir() -> Result<PathBuf, String> {
    MODEL_DIR.get().cloned().ok_or_else(|| "Folder model belum disediakan".to_string())
}

/// Missing file = empty list; a corrupt file is an error so it is never silently overwritten.
fn read_index(dir: &Path) -> Result<Vec<CustomModel>, String> {
    match std::fs::read(dir.join(INDEX_FILE)) {
        Ok(b) => serde_json::from_slice(&b).map_err(|e| format!("{INDEX_FILE} rosak: {e}")),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(e) => Err(e.to_string()),
    }
}

/// Write-then-rename so a crash never leaves a half-written index.
fn write_index(dir: &Path, list: &[CustomModel]) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let tmp = dir.join(format!("{INDEX_FILE}.tmp"));
    std::fs::write(&tmp, serde_json::to_vec_pretty(list).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, dir.join(INDEX_FILE)).map_err(|e| e.to_string())
}

fn custom_file(dir: &Path, m: &CustomModel) -> Result<PathBuf, String> {
    // The index is user-editable: only accept a bare file name.
    if Path::new(&m.file_name).file_name().and_then(|f| f.to_str()) != Some(m.file_name.as_str()) {
        return Err(format!("Nama fail model tidak sah: {}", m.file_name));
    }
    Ok(dir.join(sub_dir(&m.kind)).join(&m.file_name))
}

/// Looks up an imported model of `kind` and returns it with its on-disk path.
pub fn find_custom(id: &str, kind: &str) -> Result<(CustomModel, PathBuf), String> {
    let dir = models_dir()?;
    let m = read_index(&dir)?
        .into_iter()
        .find(|m| m.id == id && m.kind == kind)
        .ok_or_else(|| format!("Model tidak dikenali: {id}"))?;
    let path = custom_file(&dir, &m)?;
    Ok((m, path))
}

/// Copies a user `.onnx` into models/custom_<kind>/ under a fresh id.
/// Returns (id, display name, dest path, size in MB).
pub fn copy_in(source: &str, kind: &str) -> Result<(String, String, PathBuf, f64), String> {
    let src = Path::new(source);
    if !src.extension().is_some_and(|e| e.eq_ignore_ascii_case("onnx")) || !src.is_file() {
        return Err("Sila pilih fail .onnx yang sah".to_string());
    }
    let dir = models_dir()?.join(sub_dir(kind));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let id = format!("custom-{kind}-{nanos:x}");
    let dest = dir.join(format!("{id}.onnx"));
    let bytes = std::fs::copy(src, &dest).map_err(|e| format!("Gagal menyalin model: {e}"))?;
    let name = src.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| id.clone());
    Ok((id, name, dest, (bytes as f64 / 1_048_576.0 * 10.0).round() / 10.0))
}

pub fn add_to_index(m: CustomModel) -> Result<CustomModel, String> {
    let _g = INDEX_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let dir = models_dir()?;
    let mut list = read_index(&dir)?;
    list.push(m.clone());
    write_index(&dir, &list)?;
    Ok(m)
}

/// CPU-only probe session that leaves the footer backend untouched.
pub fn probe_session(path: &Path) -> Result<Session, String> {
    Session::builder().and_then(|b| b.commit_from_file(path)).map_err(|e| e.to_string())
}

/// Accepts a Float32/Float16 NCHW input with 3 (or dynamic) channels; returns its dims.
pub fn probe_input_dims(session: &Session) -> Result<Vec<i64>, String> {
    match session.inputs.first().map(|i| &i.input_type) {
        Some(ValueType::Tensor { ty: TensorElementType::Float32 | TensorElementType::Float16, dimensions, .. })
            if dimensions.len() == 4 && (dimensions[1] == 3 || dimensions[1] == -1) =>
        {
            Ok(dimensions.clone())
        }
        _ => Err("input mesti tensor Float32/Float16 NCHW 3 saluran".to_string()),
    }
}

/// Output length of a 16×16 RGB probe → scale factor (output_width / 16).
fn scale_from_len(len: usize) -> Option<u32> {
    let s = ((len / (3 * 16 * 16)) as f64).sqrt().round() as u32;
    ((2..=8).contains(&s) && (s * s * 3 * 256) as usize == len).then_some(s)
}

/// Copies the model in (`.pth` converted first), then runs a 16×16 test inference on CPU
/// to detect its real scale.
#[tauri::command(async)]
pub fn import_upscale_model(source_path: String) -> Result<CustomModel, String> {
    let (source_path, _tmp) = super::pth_converter::prepare(&source_path, super::pth_converter::PthKind::Upscale)?;
    let (id, name, dest, size_mb) = copy_in(&source_path, "upscale")?;
    let bad = |why: String| format!("Bukan model super-resolution yang sah: {why}");
    let res = probe_session(&dest).map_err(bad).and_then(|session| {
        probe_input_dims(&session).map_err(bad)?;
        let out = run_patch(&session, &[0.5; 3 * 16 * 16], 16, 16).map_err(bad)?;
        let scale = scale_from_len(out.len()).ok_or_else(|| bad("output bukan imej RGB 2×–8× ganda".into()))?;
        add_to_index(CustomModel {
            file_name: format!("{id}.onnx"),
            id,
            kind: "upscale".into(),
            name,
            scale: Some(scale),
            input_size: None,
            norm_type: "standard".into(),
            size_mb,
        })
    });
    if res.is_err() {
        let _ = std::fs::remove_file(&dest);
    }
    res
}

#[tauri::command]
pub fn list_custom_models() -> Result<Vec<CustomModel>, String> {
    read_index(&models_dir()?)
}

#[tauri::command(async)]
pub fn delete_custom_model(id: String) -> Result<(), String> {
    let _g = INDEX_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let dir = models_dir()?;
    let mut list = read_index(&dir)?;
    let pos = list.iter().position(|m| m.id == id).ok_or_else(|| format!("Model tidak dikenali: {id}"))?;
    // A live session keeps the file open (Windows cannot delete it): release both first.
    clear_session();
    super::bg_remover::clear_session();
    super::face_restorer::clear_session();
    match std::fs::remove_file(custom_file(&dir, &list[pos])?) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(format!("Gagal memadam model: {e}")),
        _ => {}
    }
    list.remove(pos);
    write_index(&dir, &list)
}

/// Builds an ort session on DirectML, falling back to CPU. Records the real backend in
/// `ACTIVE_AI_BACKEND` and prints why DirectML failed.
pub fn build_session(path: &std::path::Path) -> Result<Session, String> {
    build_session_on(path, None)
}

/// CPU-only session (DirectML never registered); `label` is shown as the backend.
pub fn build_cpu_session(path: &std::path::Path, label: &str) -> Result<Session, String> {
    build_session_on(path, Some(label))
}

fn build_session_on(path: &std::path::Path, cpu_label: Option<&str>) -> Result<Session, String> {
    let mut builder = Session::builder().map_err(|e| e.to_string())?;
    let backend = if let Some(label) = cpu_label {
        label
    } else {
        match DirectMLExecutionProvider::default().register(&mut builder) {
            Ok(()) => LABEL_GPU,
            Err(e) => {
                eprintln!("DirectML gagal didaftarkan, guna CPU: {e}");
                LABEL_NO_DML
            }
        }
    };
    let session = builder
        .with_memory_pattern(false)
        .and_then(|b| b.with_parallel_execution(false))
        .and_then(|b| b.commit_from_file(path))
        .map_err(|e| e.to_string())?;
    *ACTIVE_AI_BACKEND.write().unwrap_or_else(|p| p.into_inner()) = backend.to_string();
    Ok(session)
}

/// Drops the cached upscaler session so its VRAM is freed immediately.
pub fn clear_session() {
    *SESSION.lock().unwrap_or_else(|p| p.into_inner()) = None;
}

/// True when a `session.run` failure looks like DirectML running out of memory
/// (e.g. 8007000E) while the live backend is the GPU.
pub fn is_gpu_oom(err: &str) -> bool {
    let on_gpu = *ACTIVE_AI_BACKEND.read().unwrap_or_else(|p| p.into_inner()) == LABEL_GPU;
    let e = err.to_lowercase();
    on_gpu && ["8007000e", "memory", "dml", "directml"].iter().any(|k| e.contains(k))
}

/// Resets GPU_AI_ACTIVE on every exit path (error or panic included).
pub struct ActiveGuard;
impl ActiveGuard {
    pub fn new() -> Self {
        GPU_AI_ACTIVE.store(true, Ordering::Relaxed);
        Self
    }
}
impl Drop for ActiveGuard {
    fn drop(&mut self) {
        GPU_AI_ACTIVE.store(false, Ordering::Relaxed);
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UpscaleMode {
    FastLanczos,
    AiRealEsrgan,
}

impl UpscaleMode {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "fast" => Ok(Self::FastLanczos),
            "ai" => Ok(Self::AiRealEsrgan),
            other => Err(format!("Mod upscale tidak dikenali: {other}")),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct GpuInfo {
    pub available: bool,
    pub gpu_name: String,
    pub vram_mb: u64,
    pub backend: String,
}

#[derive(Serialize, Clone)]
struct DownloadProgress {
    model: String,
    downloaded: u64,
    total: u64,
    percent: f32,
}

fn model_path(m: &ModelSpec) -> Result<PathBuf, String> {
    MODEL_DIR
        .get()
        .map(|d| d.join(m.file))
        .ok_or_else(|| "Folder model belum disediakan".to_string())
}

fn cpu_fallback() -> GpuInfo {
    GpuInfo { available: false, gpu_name: "CPU Fallback".into(), vram_mb: 0, backend: "CPU".into() }
}

/// Picks the adapter with the most dedicated VRAM (skips software/Basic Render adapters).
#[cfg(windows)]
fn detect_gpu() -> GpuInfo {
    use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE};
    let mut best: Option<GpuInfo> = None;
    unsafe {
        let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory1>() else { return cpu_fallback() };
        let mut i = 0;
        while let Ok(adapter) = factory.EnumAdapters1(i) {
            i += 1;
            let Ok(d) = adapter.GetDesc1() else { continue };
            let len = d.Description.iter().position(|&c| c == 0).unwrap_or(d.Description.len());
            let name = String::from_utf16_lossy(&d.Description[..len]);
            if d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 || name.contains("Microsoft Basic Render") {
                continue;
            }
            let vram_mb = d.DedicatedVideoMemory as u64 / 1_048_576;
            if best.as_ref().is_none_or(|b| vram_mb > b.vram_mb) {
                best = Some(GpuInfo { available: true, gpu_name: name, vram_mb, backend: "DirectML".into() });
            }
        }
    }
    best.unwrap_or_else(cpu_fallback)
}

#[cfg(not(windows))]
fn detect_gpu() -> GpuInfo {
    cpu_fallback()
}

#[tauri::command]
pub fn get_gpu_info() -> GpuInfo {
    detect_gpu()
}

#[tauri::command]
pub fn check_upscale_model(app: tauri::AppHandle, model: String) -> Result<bool, String> {
    let m = spec(&model)?;
    let dir = app.path().app_data_dir().map_err(|e| e.to_string())?.join("models");
    Ok(std::fs::metadata(dir.join(m.file)).is_ok_and(|f| f.len() == m.bytes))
}

/// One-time download; afterwards the AI mode is fully offline.
#[tauri::command]
pub async fn download_upscale_model(app: tauri::AppHandle, model: String) -> Result<(), String> {
    let m = spec(&model)?;
    let dir = app.path().app_data_dir().map_err(|e| e.to_string())?.join("models");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let final_path = dir.join(m.file);
    let part_path = dir.join(format!("{}.part", m.file));

    let mut resp = reqwest::get(m.url).await.map_err(|e| format!("Muat turun gagal: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("Muat turun gagal: HTTP {}", resp.status()));
    }
    let total = resp.content_length().unwrap_or(m.bytes);
    let mut file = std::fs::File::create(&part_path).map_err(|e| e.to_string())?;
    let mut hasher = Sha256::new();
    let (mut downloaded, mut last_pct) = (0u64, -1i32);

    let result: Result<(), String> = async {
        while let Some(chunk) = resp.chunk().await.map_err(|e| format!("Muat turun gagal: {e}"))? {
            file.write_all(&chunk).map_err(|e| e.to_string())?;
            hasher.update(&chunk);
            downloaded += chunk.len() as u64;
            let percent = (downloaded as f32 / total.max(1) as f32 * 100.0).min(100.0);
            if percent as i32 != last_pct {
                last_pct = percent as i32;
                let _ = app.emit(
                    "model-download-progress",
                    DownloadProgress { model: m.id.to_string(), downloaded, total, percent },
                );
            }
        }
        file.flush().map_err(|e| e.to_string())?;
        if format!("{:x}", hasher.finalize()) != m.sha256 {
            return Err("Fail model rosak (semakan SHA-256 gagal)".to_string());
        }
        Ok(())
    }
    .await;

    drop(file);
    if let Err(e) = result {
        let _ = std::fs::remove_file(&part_path);
        return Err(e);
    }
    std::fs::rename(&part_path, &final_path).map_err(|e| e.to_string())?;
    // New file on disk: drop any stale session for this model.
    let mut guard = SESSION.lock().unwrap_or_else(|p| p.into_inner());
    if guard.as_ref().is_some_and(|(id, _)| id == m.id) {
        *guard = None;
    }
    Ok(())
}

/// Lanczos3 resize on RGBA8 (fast_image_resize premultiplies alpha internally).
pub fn lanczos(rgba: &RgbaImage, w: u32, h: u32) -> Result<RgbaImage, String> {
    let src = Image::from_vec_u8(rgba.width(), rgba.height(), rgba.as_raw().clone(), PixelType::U8x4)
        .map_err(|e| e.to_string())?;
    let mut dst = Image::new(w, h, PixelType::U8x4);
    let opts = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3));
    Resizer::new().resize(&src, &mut dst, &opts).map_err(|e| e.to_string())?;
    RgbaImage::from_raw(w, h, dst.into_vec()).ok_or_else(|| "Saiz penimbal tidak sah".to_string())
}

/// One NCHW patch through the model; Float16 input/output is converted at the boundary.
fn run_patch(session: &Session, patch: &[f32], pw: u32, ph: u32) -> Result<Vec<f32>, String> {
    let shape = [1i64, 3, ph as i64, pw as i64];
    let name = session.inputs[0].name.clone();
    let outputs = if is_f16(&session.inputs[0].input_type) {
        let half: Vec<f16> = patch.iter().map(|&v| f16::from_f32(v)).collect();
        let tensor = Tensor::from_array((shape, half)).map_err(|e| e.to_string())?;
        session.run(ort::inputs![name => tensor].map_err(|e| e.to_string())?)
    } else {
        let tensor = Tensor::from_array((shape, patch.to_vec())).map_err(|e| e.to_string())?;
        session.run(ort::inputs![name => tensor].map_err(|e| e.to_string())?)
    }
    .map_err(|e| e.to_string())?;
    let out = &outputs[0];
    if let Ok((_, d)) = out.try_extract_raw_tensor::<f16>() {
        return Ok(d.iter().map(|v| v.to_f32()).collect());
    }
    let (_, data) = out.try_extract_raw_tensor::<f32>().map_err(|e| e.to_string())?;
    Ok(data.to_vec())
}

/// `s`× super-resolution over 256px tiles. Each tile carries a 16px margin that is discarded
/// after inference, so tile borders never show up as seams in the result.
fn ai_upscale_rgb(rgb: &RgbImage, session: &Session, s: u32) -> Result<RgbImage, String> {
    let (w, h) = rgb.dimensions();
    let mut out = RgbImage::new(w * s, h * s);
    let core = TILE - 2 * OVERLAP;
    let raw = rgb.as_raw();

    for cy in (0..h).step_by(core as usize) {
        for cx in (0..w).step_by(core as usize) {
            let (cw, ch) = (core.min(w - cx), core.min(h - cy));
            let (px0, py0) = (cx.saturating_sub(OVERLAP), cy.saturating_sub(OVERLAP));
            let (px1, py1) = ((cx + cw + OVERLAP).min(w), (cy + ch + OVERLAP).min(h));
            let (pw, ph) = (px1 - px0, py1 - py0);

            // NCHW, RGB, [0,1]
            let plane = (pw * ph) as usize;
            let mut patch = vec![0f32; plane * 3];
            for y in 0..ph {
                for x in 0..pw {
                    let si = (((py0 + y) * w + px0 + x) * 3) as usize;
                    let di = (y * pw + x) as usize;
                    for c in 0..3 {
                        patch[c * plane + di] = raw[si + c] as f32 / 255.0;
                    }
                }
            }

            let res = run_patch(session, &patch, pw, ph)?;
            let ow = pw * s;
            let oplane = (ow * ph * s) as usize;
            if res.len() != oplane * 3 {
                return Err("Output model tidak sepadan dengan saiz jangkaan".to_string());
            }
            // Copy only the core region (skip the margin).
            for y in 0..ch * s {
                for x in 0..cw * s {
                    let si = (((cy - py0) * s + y) * ow + (cx - px0) * s + x) as usize;
                    let px = out.get_pixel_mut(cx * s + x, cy * s + y);
                    for c in 0..3 {
                        px[c] = (res[c * oplane + si].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Built-in or imported upscale model → (path, scale, heavy).
fn resolve_model(model: &str) -> Result<(PathBuf, u32, bool), String> {
    if let Ok(m) = spec(model) {
        return Ok((model_path(m)?, MODEL_SCALE, false));
    }
    let (m, path) = find_custom(model, "upscale")?;
    Ok((path, m.scale.unwrap_or(MODEL_SCALE), m.heavy()))
}

fn ai_upscale(img: &DynamicImage, model: &str, path: &Path, s: u32, heavy: bool, mode: ExecMode) -> Result<RgbaImage, String> {
    if !path.exists() {
        return Err("Model AI belum dimuat turun".to_string());
    }
    let key = mode.key(model);
    let mut guard = SESSION.lock().unwrap_or_else(|p| p.into_inner());
    if guard.as_ref().is_none_or(|(id, _)| *id != key) {
        *guard = None; // free the old session (VRAM) before building the new one
        let session = open_session(path, heavy, mode).map_err(|e| format!("Gagal memuatkan model AI: {e}"))?;
        *guard = Some((key.clone(), session));
    }
    let session = &guard.as_ref().expect("session just set").1;

    let _active = ActiveGuard::new();
    let rgba = img.to_rgba8();
    let rgb = DynamicImage::ImageRgba8(rgba.clone()).to_rgb8();
    let big_rgb = match ai_upscale_rgb(&rgb, session, s) {
        Err(e) if is_gpu_oom(&e) => {
            eprintln!("DirectML kehabisan memori ({e}); beralih ke CPU");
            *guard = None; // release the failed GPU session first
            let cpu = build_cpu_session(path, LABEL_OOM).map_err(|e| format!("Gagal memuatkan model AI (CPU): {e}"))?;
            *guard = Some((key, cpu));
            ai_upscale_rgb(&rgb, &guard.as_ref().expect("session just set").1, s)?
        }
        r => r?,
    };
    let (bw, bh) = big_rgb.dimensions();
    let mut out = DynamicImage::ImageRgb8(big_rgb).to_rgba8();
    // Model has no alpha channel: scale alpha separately with Lanczos3.
    if img.color().has_alpha() {
        let alpha = lanczos(&rgba, bw, bh)?;
        for (o, a) in out.pixels_mut().zip(alpha.pixels()) {
            o[3] = a[3];
        }
    }
    Ok(out)
}

/// Per-pixel RGB blend: alpha 0 = pure Lanczos3, 1 = pure AI. Alpha channel stays Lanczos3.
pub fn blend_rgb(base: &RgbaImage, ai: &RgbaImage, alpha: f32) -> RgbaImage {
    let mut out = base.clone();
    for (o, a) in out.pixels_mut().zip(ai.pixels()) {
        for c in 0..3 {
            o[c] = (o[c] as f32 * (1.0 - alpha) + a[c] as f32 * alpha).round() as u8;
        }
    }
    out
}

pub fn upscale_image(
    img: &DynamicImage,
    factor: u8,
    mode: UpscaleMode,
    ai_strength: u8,
    upscale_model: &str,
    exec: ExecMode,
) -> Result<DynamicImage, String> {
    if factor != 2 && factor != 4 {
        return Err(format!("Faktor upscale tidak sah: {factor} (guna 2 atau 4)"));
    }
    let f = factor as u32;
    let (w, h) = (img.width(), img.height());
    let strength = ai_strength.min(100);
    // 0% skips the ONNX session entirely; everything else runs it.
    let run_ai = mode == UpscaleMode::AiRealEsrgan && strength > 0;
    if run_ai {
        // Only one AI model may hold VRAM at a time (never nested with our own lock).
        // Skipped for CPU Lanczos so a cached bg session survives batch runs.
        super::bg_remover::clear_session();
        super::face_restorer::clear_session();
    }
    let (model_path, model_scale, heavy) = if run_ai { resolve_model(upscale_model)? } else { (PathBuf::new(), f, false) };
    let peak = if run_ai { model_scale.max(f) } else { f } as u64;
    if (w as u64 * peak) * (h as u64 * peak) > MAX_OUT_PIXELS {
        return Err("Hasil upscale terlalu besar; kecilkan imej atau guna faktor 2×".to_string());
    }

    let rgba = if run_ai {
        let big = ai_upscale(img, upscale_model, &model_path, model_scale, heavy, exec)?;
        // Model scale differs from the request (e.g. 2x from a 4x model): Lanczos3 to the target.
        let ai = if f == model_scale { big } else { lanczos(&big, w * f, h * f)? };
        if strength >= 100 {
            ai
        } else {
            let base = lanczos(&img.to_rgba8(), w * f, h * f)?;
            blend_rgb(&base, &ai, strength as f32 / 100.0)
        }
    } else {
        lanczos(&img.to_rgba8(), w * f, h * f)?
    };
    Ok(if img.color().has_alpha() {
        DynamicImage::ImageRgba8(rgba)
    } else {
        DynamicImage::ImageRgb8(DynamicImage::ImageRgba8(rgba).to_rgb8())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_upscale_scales_dims_and_rejects_bad_factor() {
        let img = DynamicImage::ImageRgb8(RgbImage::new(10, 7));
        for f in [2u8, 4] {
            let out = upscale_image(&img, f, UpscaleMode::FastLanczos, 85, DEFAULT_MODEL, ExecMode::Auto).unwrap();
            assert_eq!((out.width(), out.height()), (10 * f as u32, 7 * f as u32));
        }
        assert!(upscale_image(&img, 3, UpscaleMode::FastLanczos, 85, DEFAULT_MODEL, ExecMode::Auto).is_err());
    }

    #[test]
    fn gpu_oom_detection() {
        *ACTIVE_AI_BACKEND.write().unwrap() = LABEL_GPU.into();
        assert!(is_gpu_oom("80070 8007000E Not enough memory resources"));
        assert!(is_gpu_oom("Dml error"));
        assert!(!is_gpu_oom("Output model tidak sepadan"));
        *ACTIVE_AI_BACKEND.write().unwrap() = LABEL_OOM.into();
        assert!(!is_gpu_oom("8007000E out of memory"));
    }

    #[test]
    fn exec_mode_parse_and_key() {
        assert_eq!(ExecMode::parse("gpu"), ExecMode::Gpu);
        assert_eq!(ExecMode::parse("cpu"), ExecMode::Cpu);
        assert_eq!(ExecMode::parse(""), ExecMode::Auto);
        assert_ne!(ExecMode::Gpu.key("x4v3"), ExecMode::Cpu.key("x4v3"));
    }

    #[test]
    fn model_table_is_consistent() {
        assert!(spec("x4plus").is_ok() && spec("ultrasharp").is_ok() && spec("anime-6b").is_ok() && spec("nope").is_err());
        for m in &MODELS {
            assert_eq!(m.sha256.len(), 64);
        }
        let ids: std::collections::HashSet<_> = MODELS.iter().map(|m| m.file).collect();
        assert_eq!(ids.len(), MODELS.len());
    }

    #[test]
    fn probe_scale_from_output_len() {
        assert_eq!(scale_from_len(3 * 32 * 32), Some(2));
        assert_eq!(scale_from_len(3 * 64 * 64), Some(4));
        assert_eq!(scale_from_len(3 * 16 * 16), None); // 1x: not super-resolution
        assert_eq!(scale_from_len(1 * 64 * 64), None); // mask, not RGB
        assert_eq!(scale_from_len(1000), None);
    }

    #[test]
    fn custom_index_roundtrip_and_rejects_traversal() {
        let dir = std::env::temp_dir().join(format!("morph-idx-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        assert!(read_index(&dir).unwrap().is_empty());
        let m = CustomModel {
            id: "custom-bg-1".into(),
            kind: "bg".into(),
            name: "BEN2".into(),
            file_name: "custom-bg-1.onnx".into(),
            scale: None,
            input_size: Some(1024),
            norm_type: "imagenet".into(),
            size_mb: 210.0,
        };
        write_index(&dir, &[m.clone()]).unwrap();
        assert_eq!(read_index(&dir).unwrap(), vec![m.clone()]);
        assert!(m.heavy());
        assert!(custom_file(&dir, &m).unwrap().ends_with("custom_bg/custom-bg-1.onnx"));
        let evil = CustomModel { file_name: "../../evil.onnx".into(), ..m };
        assert!(custom_file(&dir, &evil).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn blend_rgb_keeps_alpha_and_mixes_rgb() {
        let base = RgbaImage::from_pixel(2, 2, image::Rgba([0, 100, 200, 40]));
        let ai = RgbaImage::from_pixel(2, 2, image::Rgba([255, 200, 100, 255]));
        let out = blend_rgb(&base, &ai, 0.5);
        assert_eq!(out.get_pixel(0, 0).0, [128, 150, 150, 40]);
    }
}
