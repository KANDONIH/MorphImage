use fast_image_resize::images::Image;
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
use half::f16;
use image::{DynamicImage, RgbaImage};
use ort::session::Session;
use ort::value::Tensor;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;
use tauri::{Emitter, Manager};

use super::upscaler::{self, build_cpu_session, copy_in, is_f16, open_session, ActiveGuard, CustomModel, ExecMode, MODEL_DIR};

pub const DEFAULT_MODEL: &str = "rmbg-1.4";

/// Everything that differs between supported segmentation models.
struct ModelSpec {
    id: &'static str,
    file: &'static str,
    url: &'static str,
    /// Pinned SHA-256 of the exact file at `url`.
    sha256: &'static str,
    bytes: u64,
    mean: [f32; 3],
    std: [f32; 3],
    /// Take the last graph output instead of the first.
    last_output: bool,
    /// Output may be logits; squash with sigmoid when it falls outside 0..=1.
    sigmoid_if_needed: bool,
}

const MODELS: [ModelSpec; 2] = [
    ModelSpec {
        id: "rmbg-1.4",
        file: "rmbg-1.4-quantized.onnx",
        url: "https://huggingface.co/briaai/RMBG-1.4/resolve/main/onnx/model_quantized.onnx",
        sha256: "a6648479275dfd0ede0f3a8abc20aa5c437b394681b05e5af6d268250aaf40f3",
        bytes: 44_403_226,
        // (x/255 - 0.5) / 1.0
        mean: [0.5; 3],
        std: [1.0; 3],
        last_output: false,
        sigmoid_if_needed: false,
    },
    ModelSpec {
        id: "birefnet-lite",
        file: "birefnet-lite-fp16.onnx",
        // onnx-community/BiRefNet_lite (MIT). Hash from the HF LFS oid, re-verified on a real download.
        url: "https://huggingface.co/onnx-community/BiRefNet_lite/resolve/main/onnx/model_fp16.onnx",
        sha256: "d39b897ceb16ae654c1731f3dba0cf9b368d9cae74b5a57459b455cc8bfec402",
        bytes: 114_538_221,
        mean: [0.485, 0.456, 0.406],
        std: [0.229, 0.224, 0.225],
        last_output: true,
        sigmoid_if_needed: true,
    },
];

fn spec(id: &str) -> Result<&'static ModelSpec, String> {
    MODELS.iter().find(|m| m.id == id).ok_or_else(|| format!("Model buang latar tidak dikenali: {id}"))
}

/// Built-ins (and imported models with a dynamic size) take 1024×1024 RGB and return a mask at the same size.
const INPUT: u32 = 1024;

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Per-run preprocessing, resolved from a built-in spec or an imported model.
struct Prep {
    size: u32,
    mean: [f32; 3],
    std: [f32; 3],
    last_output: bool,
    sigmoid_if_needed: bool,
}

fn norm(norm_type: &str) -> Result<([f32; 3], [f32; 3]), String> {
    match norm_type {
        "standard" => Ok(([0.0; 3], [1.0; 3])),
        "imagenet" => Ok((IMAGENET_MEAN, IMAGENET_STD)),
        o => Err(format!("Jenis penormalan tidak dikenali: {o}")),
    }
}

/// Built-in or imported bg model → (path, prep, heavy).
fn resolve(model: &str) -> Result<(PathBuf, Prep, bool), String> {
    if let Ok(s) = spec(model) {
        let prep = Prep { size: INPUT, mean: s.mean, std: s.std, last_output: s.last_output, sigmoid_if_needed: s.sigmoid_if_needed };
        return Ok((model_path(s)?, prep, s.id == "birefnet-lite"));
    }
    let (m, path) = upscaler::find_custom(model, "bg")?;
    let (mean, std) = norm(&m.norm_type)?;
    // ImageNet family (BiRefNet/BEN2) puts the final mask last; RMBG/U2Net put it first.
    let prep = Prep {
        size: m.input_size.unwrap_or(INPUT),
        mean,
        std,
        last_output: m.norm_type == "imagenet",
        sigmoid_if_needed: true,
    };
    Ok((path, prep, m.heavy()))
}

/// Fixed square H×W from NCHW dims; dynamic (-1) falls back to 1024.
fn input_size_from(dims: &[i64]) -> Result<u32, String> {
    match (dims[2], dims[3]) {
        (h, w) if h > 0 && h == w && h <= 4096 => Ok(h as u32),
        (h, w) if h <= 0 && w <= 0 => Ok(INPUT),
        (h, w) => Err(format!("saiz input {w}×{h} tidak disokong (perlu segi empat sama)")),
    }
}

/// Copies the model in, reads its input size and dtype on CPU, stores the normalisation preset.
#[tauri::command(async)]
pub fn import_bg_model(source_path: String, norm_preset: String) -> Result<CustomModel, String> {
    norm(&norm_preset)?;
    let (source_path, _tmp) = super::pth_converter::prepare(&source_path, super::pth_converter::PthKind::Background)?;
    let (id, name, dest, size_mb) = copy_in(&source_path, "bg")?;
    let bad = |why: String| format!("Bukan model buang latar yang sah: {why}");
    let res = upscaler::probe_session(&dest).map_err(bad).and_then(|session| {
        let size = input_size_from(&upscaler::probe_input_dims(&session).map_err(bad)?).map_err(bad)?;
        if session.outputs.is_empty() {
            return Err(bad("tiada output".into()));
        }
        upscaler::add_to_index(CustomModel {
            file_name: format!("{id}.onnx"),
            id,
            kind: "bg".into(),
            name,
            scale: None,
            input_size: Some(size),
            norm_type: norm_preset,
            size_mb,
        })
    });
    if res.is_err() {
        let _ = std::fs::remove_file(&dest);
    }
    res
}

/// Cached `(model_id, session)`; the lock also serialises runs so rayon workers do not fight over VRAM.
static SESSION: Mutex<Option<(String, Session)>> = Mutex::new(None);

#[derive(Serialize, Clone)]
struct DownloadProgress {
    model: String,
    downloaded: u64,
    total: u64,
    percent: f32,
}

fn model_path(s: &ModelSpec) -> Result<PathBuf, String> {
    MODEL_DIR
        .get()
        .map(|d| d.join(s.file))
        .ok_or_else(|| "Folder model belum disediakan".to_string())
}

#[tauri::command]
pub fn check_bg_model(app: tauri::AppHandle, model: String) -> Result<bool, String> {
    let s = spec(&model)?;
    let dir = app.path().app_data_dir().map_err(|e| e.to_string())?.join("models");
    Ok(std::fs::metadata(dir.join(s.file)).is_ok_and(|m| m.len() == s.bytes))
}

/// One-time download per model; afterwards background removal is fully offline.
#[tauri::command]
pub async fn download_bg_model(app: tauri::AppHandle, model: String) -> Result<(), String> {
    let s = spec(&model)?;
    let dir = app.path().app_data_dir().map_err(|e| e.to_string())?.join("models");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let final_path = dir.join(s.file);
    let part_path = dir.join(format!("{}.part", s.file));

    let mut resp = reqwest::get(s.url).await.map_err(|e| format!("Muat turun gagal: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("Muat turun gagal: HTTP {}", resp.status()));
    }
    let total = resp.content_length().unwrap_or(s.bytes);
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
                    "bg-model-download-progress",
                    DownloadProgress { model: s.id.to_string(), downloaded, total, percent },
                );
            }
        }
        file.flush().map_err(|e| e.to_string())?;
        if format!("{:x}", hasher.finalize()) != s.sha256 {
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
    // Drop a cached session of this model so the fresh file is reloaded.
    let mut guard = SESSION.lock().unwrap_or_else(|p| p.into_inner());
    if guard.as_ref().is_some_and(|(id, _)| id == s.id) {
        *guard = None;
    }
    Ok(())
}

/// `#RGB`, `#RRGGBB`, or `RRGGBB` → `[r, g, b]`.
pub fn parse_hex_color(s: &str) -> Result<[u8; 3], String> {
    let h = s.trim().trim_start_matches('#');
    let bad = || format!("Kod warna tidak sah: {s}");
    let bytes: Vec<u8> = match h.len() {
        3 => h
            .chars()
            .map(|c| u8::from_str_radix(&format!("{c}{c}"), 16).map_err(|_| bad()))
            .collect::<Result<_, _>>()?,
        6 => (0..3)
            .map(|i| u8::from_str_radix(&h[i * 2..i * 2 + 2], 16).map_err(|_| bad()))
            .collect::<Result<_, _>>()?,
        _ => return Err(bad()),
    };
    Ok([bytes[0], bytes[1], bytes[2]])
}

fn resize_u8(buf: &[u8], w: u32, h: u32, nw: u32, nh: u32, px: PixelType) -> Result<Vec<u8>, String> {
    let src = Image::from_vec_u8(w, h, buf.to_vec(), px).map_err(|e| e.to_string())?;
    let mut dst = Image::new(nw, nh, px);
    let opts = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3));
    Resizer::new().resize(&src, &mut dst, &opts).map_err(|e| e.to_string())?;
    Ok(dst.into_vec())
}

/// Raw model output → 0..=255 mask bytes. Logits (anything outside 0..=1) go through sigmoid
/// when `sigmoid_if_needed`.
fn mask_bytes(data: &[f32], sigmoid_if_needed: bool) -> Vec<u8> {
    let logits = sigmoid_if_needed && data.iter().any(|v| !(0.0..=1.0).contains(v));
    data.iter()
        .map(|&v| {
            let p = if logits { 1.0 / (1.0 + (-v).exp()) } else { v };
            (p.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
        })
        .collect()
}

/// Drops the cached bg-removal session so its VRAM is freed immediately.
pub fn clear_session() {
    *SESSION.lock().unwrap_or_else(|p| p.into_inner()) = None;
}

/// NCHW float input at size², (x/255 - mean) / std per channel.
fn make_patch(rgba: &RgbaImage, s: &Prep) -> Result<Vec<f32>, String> {
    let (w, h) = rgba.dimensions();
    let resized = resize_u8(rgba.as_raw(), w, h, s.size, s.size, PixelType::U8x4)?;
    let plane = (s.size * s.size) as usize;
    let mut patch = vec![0f32; plane * 3];
    for i in 0..plane {
        for c in 0..3 {
            patch[c * plane + i] = (resized[i * 4 + c] as f32 / 255.0 - s.mean[c]) / s.std[c];
        }
    }
    Ok(patch)
}

/// Runs the model (f16 or f32 I/O) and returns the raw mask as f32, size² long.
fn infer(session: &Session, patch: &[f32], s: &Prep) -> Result<Vec<f32>, String> {
    let plane = (s.size * s.size) as usize;
    let shape = [1i64, 3, s.size as i64, s.size as i64];
    let in_name = session.inputs[0].name.clone();
    let outputs = if is_f16(&session.inputs[0].input_type) {
        let half: Vec<f16> = patch.iter().map(|&v| f16::from_f32(v)).collect();
        let tensor = Tensor::from_array((shape, half)).map_err(|e| e.to_string())?;
        session.run(ort::inputs![in_name => tensor].map_err(|e| e.to_string())?)
    } else {
        let tensor = Tensor::from_array((shape, patch.to_vec())).map_err(|e| e.to_string())?;
        session.run(ort::inputs![in_name => tensor].map_err(|e| e.to_string())?)
    }
    .map_err(|e| e.to_string())?;

    let out = if s.last_output { session.outputs.last() } else { session.outputs.first() }
        .ok_or_else(|| "Model tiada output".to_string())?;
    let value = &outputs[out.name.as_str()];
    let data: Vec<f32> = if is_f16(&out.output_type) {
        let (_, d) = value.try_extract_raw_tensor::<f16>().map_err(|e| e.to_string())?;
        d.iter().map(|v| v.to_f32()).collect()
    } else {
        let (_, d) = value.try_extract_raw_tensor::<f32>().map_err(|e| e.to_string())?;
        d.to_vec()
    };
    if data.len() != plane {
        return Err("Output model tidak sepadan dengan saiz jangkaan".to_string());
    }
    Ok(data)
}

/// Cuts the subject out. `bg_color = None` keeps the cut-out transparent; `Some(rgb)`
/// composites the subject onto that colour instead. `model`: a built-in or imported model id.
pub fn remove_background(img: DynamicImage, bg_color: Option<[u8; 3]>, model: &str, mode: ExecMode) -> Result<DynamicImage, String> {
    // First thing, before our own lock (never nested): free the upscaler's VRAM.
    upscaler::clear_session();
    super::face_restorer::clear_session();
    let (path, prep, heavy) = resolve(model)?;
    let s = &prep;
    if !path.exists() {
        return Err("Model buang latar belum dimuat turun".to_string());
    }
    let key = mode.key(model);
    let mut guard = SESSION.lock().unwrap_or_else(|p| p.into_inner());
    if guard.as_ref().is_none_or(|(id, _)| *id != key) {
        // Free the old model first so two sessions never sit in VRAM together.
        *guard = None;
        let session = open_session(&path, heavy, mode).map_err(|e| format!("Gagal memuatkan model buang latar: {e}"))?;
        *guard = Some((key.clone(), session));
    }

    let _active = ActiveGuard::new();
    let mut out = img.into_rgba8();
    let (w, h) = out.dimensions();
    let patch = make_patch(&out, s)?;
    let data = match infer(&guard.as_ref().expect("session just set").1, &patch, s) {
        Err(e) if upscaler::is_gpu_oom(&e) => {
            eprintln!("DirectML kehabisan memori ({e}); beralih ke CPU");
            *guard = None; // release the failed GPU session first
            let cpu = build_cpu_session(&path, upscaler::LABEL_OOM).map_err(|e| format!("Gagal memuatkan model buang latar (CPU): {e}"))?;
            *guard = Some((key, cpu));
            infer(&guard.as_ref().expect("session just set").1, &patch, s)?
        }
        r => r?,
    };
    let mask = resize_u8(&mask_bytes(&data, s.sigmoid_if_needed), s.size, s.size, w, h, PixelType::U8)?;

    match bg_color {
        // Multiply, so a source that already has transparency keeps it.
        None => {
            for (px, &m) in out.pixels_mut().zip(mask.iter()) {
                px[3] = ((px[3] as u16 * m as u16) / 255) as u8;
            }
        }
        Some([br, bg, bb]) => {
            for (px, &m) in out.pixels_mut().zip(mask.iter()) {
                let a = m as f32 / 255.0;
                px[0] = (px[0] as f32 * a + br as f32 * (1.0 - a)).round() as u8;
                px[1] = (px[1] as f32 * a + bg as f32 * (1.0 - a)).round() as u8;
                px[2] = (px[2] as f32 * a + bb as f32 * (1.0 - a)).round() as u8;
                px[3] = 255;
            }
        }
    }
    Ok(DynamicImage::ImageRgba8(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_colors_parse_and_reject() {
        assert_eq!(parse_hex_color("#FFFFFF").unwrap(), [255, 255, 255]);
        assert_eq!(parse_hex_color("00ff80").unwrap(), [0, 255, 128]);
        assert_eq!(parse_hex_color("#fff").unwrap(), [255, 255, 255]);
        assert!(parse_hex_color("#12345").is_err());
        assert!(parse_hex_color("#gggggg").is_err());
    }

    #[test]
    fn model_ids_resolve() {
        assert_eq!(spec(DEFAULT_MODEL).unwrap().id, "rmbg-1.4");
        assert_eq!(spec("birefnet-lite").unwrap().bytes, 114_538_221);
        assert!(spec("nope").is_err());
    }

    #[test]
    fn input_size_probe() {
        assert_eq!(input_size_from(&[1, 3, 320, 320]), Ok(320));
        assert_eq!(input_size_from(&[1, 3, -1, -1]), Ok(1024));
        assert!(input_size_from(&[1, 3, 512, 1024]).is_err());
        assert!(norm("standard").is_ok() && norm("imagenet").is_ok() && norm("x").is_err());
    }

    #[test]
    fn mask_sigmoid_only_for_logits() {
        // Already 0..=1: untouched.
        assert_eq!(mask_bytes(&[0.0, 0.5, 1.0], true), vec![0, 128, 255]);
        // Logits: sigmoid(0) = 0.5, large negative → 0, large positive → 255.
        assert_eq!(mask_bytes(&[-20.0, 0.0, 20.0], true), vec![0, 128, 255]);
        // Flag off: clamp only.
        assert_eq!(mask_bytes(&[-1.0, 2.0], false), vec![0, 255]);
    }

    /// Real-model check. Needs `RMBG_MODEL_DIR` + `RMBG_SRC`; runs each model whose file is present.
    #[test]
    fn real_model_cutout() {
        let (Ok(dir), Ok(src)) = (std::env::var("RMBG_MODEL_DIR"), std::env::var("RMBG_SRC")) else {
            return;
        };
        let _ = MODEL_DIR.set(PathBuf::from(&dir));
        let img = image::open(src).expect("decode");
        let (w, h) = (img.width(), img.height());

        for m in &MODELS {
            if !PathBuf::from(&dir).join(m.file).exists() {
                continue;
            }
            let cut = remove_background(img.clone(), None, m.id, ExecMode::Auto).expect("cutout").to_rgba8();
            assert_eq!((cut.width(), cut.height()), (w, h));
            let corner = cut.get_pixel(0, 0)[3];
            assert!(corner < 40, "{}: corner alpha {corner}", m.id);

            let white = remove_background(img.clone(), Some([255, 255, 255]), m.id, ExecMode::Auto).expect("white").to_rgba8();
            assert_eq!(white.get_pixel(0, 0)[3], 255);
        }
    }
}
