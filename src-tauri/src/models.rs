use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ImageInfo {
    pub path: String,
    pub name: String,
    pub size_bytes: u64,
    pub width: u32,
    pub height: u32,
    pub format: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum OutputFormat {
    Original,
    Jpeg,
    Png,
    Webp,
    Avif,
    Tiff,
    Ico,
    Bmp,
    Gif,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ProcessOptions {
    pub output_format: OutputFormat,
    /// 1–100
    pub quality: u8,
    pub max_width: Option<u32>,
    pub strip_metadata: bool,
    /// None/empty: `<source dir>/output`
    pub output_dir: Option<String>,
    /// PNG output: quantize to 256 colors (lossy) before oxipng.
    #[serde(default)]
    pub png_lossy: bool,
    /// JPEG/WebP: binary-search quality until output <= this many KB.
    #[serde(default)]
    pub target_size_kb: Option<u32>,
    /// Appended to the file stem. Empty = original name.
    #[serde(default)]
    pub custom_suffix: String,
    /// Prefix `001_`, `002_`…
    #[serde(default)]
    pub use_numbering: bool,
    /// 0, 90, 180, 270 (clockwise).
    #[serde(default)]
    pub rotate_deg: u16,
    #[serde(default)]
    pub flip_horizontal: bool,
    #[serde(default)]
    pub flip_vertical: bool,
    #[serde(default)]
    pub crop: Option<CropRect>,
    /// None = off; Some(2) or Some(4). Runs after crop/rotate/flip, before max_width.
    #[serde(default)]
    pub upscale_factor: Option<u8>,
    /// "fast" (Lanczos3 CPU) or "ai" (Real-ESRGAN).
    #[serde(default = "default_upscale_mode")]
    pub upscale_mode: String,
    /// 0–100: blend of Lanczos3 (0) and full Real-ESRGAN (100) in AI mode.
    #[serde(default = "default_ai_strength")]
    pub ai_strength: u8,
    /// AI model: "x4v3" (default), "x4plus" or "anime-6b".
    #[serde(default = "default_upscale_model")]
    pub upscale_model: String,
    /// Cut the subject out with RMBG-1.4. Runs after crop/rotate/flip, before upscale.
    #[serde(default)]
    pub remove_bg: bool,
    /// `#RRGGBB` replacement background; None/empty = transparent cut-out.
    #[serde(default)]
    pub bg_replace_color: Option<String>,
    /// "rmbg-1.4" (default) or "birefnet-lite".
    #[serde(default = "default_bg_model")]
    pub bg_model: String,
    /// Restore faces with an imported 512×512 model. Runs after Remove BG, before upscale.
    #[serde(default)]
    pub restore_face: bool,
    /// Id from face_models.json.
    #[serde(default)]
    pub face_model_id: String,
    /// 0.0–1.0: CodeFormer `w` and blend strength over the original.
    #[serde(default = "default_face_fidelity")]
    pub face_fidelity: f32,
    /// "auto" (4GB VRAM Safe Mode), "gpu" (force DirectML) or "cpu" (force CPU).
    #[serde(default)]
    pub execution_mode: String,
    /// Rayon pool size; None/0 = auto (logical cores - 2).
    #[serde(default)]
    pub worker_threads: Option<usize>,
    /// Write the result over the source file (same stem, ignores output_dir/suffix/numbering).
    #[serde(default)]
    pub overwrite_original: bool,
}

fn default_face_fidelity() -> f32 {
    crate::engine::face_restorer::DEFAULT_FIDELITY
}

fn default_bg_model() -> String {
    crate::engine::bg_remover::DEFAULT_MODEL.to_string()
}

fn default_upscale_mode() -> String {
    "fast".to_string()
}

fn default_upscale_model() -> String {
    crate::engine::upscaler::DEFAULT_MODEL.to_string()
}

fn default_ai_strength() -> u8 {
    85
}

/// Normalized crop region (0.0–1.0 of the oriented source image).
#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub struct CropRect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct BatchProgressEvent {
    pub index: usize,
    pub total: usize,
    pub result: ProcessResult,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SystemStats {
    /// Peratus penggunaan CPU global (0–100).
    pub cpu_usage: f32,
    /// RAM proses aplikasi ini (RSS) dalam MB.
    pub app_memory_mb: f64,
    /// Tugasan Rayon yang sedang berjalan.
    pub active_threads: usize,
    /// Saiz kolam worker Rayon.
    pub total_threads: usize,
    /// Nama GPU utama (DXGI), atau "Tiada GPU".
    pub gpu_name: String,
    /// Peratus penggunaan enjin 3D/Compute GPU utama (0–100).
    pub gpu_usage: f32,
    /// VRAM digunakan (MB) menurut QueryVideoMemoryInfo.
    pub gpu_vram_used_mb: u64,
    /// true semasa sesi AI DirectML sedang memproses tile.
    pub gpu_active: bool,
    /// Last AI session backend: "Idle", "GPU (DirectML)", "CPU Fallback (VRAM Full)", ...
    pub ai_backend: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ProcessResult {
    pub source_path: String,
    pub output_path: String,
    pub original_size: u64,
    pub new_size: u64,
    pub saved_percent: f32,
    pub success: bool,
    pub error: Option<String>,
}
