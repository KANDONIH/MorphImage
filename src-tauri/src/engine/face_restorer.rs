//! AI face restoration with user-imported 512×512 ONNX models (CodeFormer, GFPGAN).
//! Models live in `models/custom_face/` with their own `face_models.json` index.

use half::f16;
use image::{DynamicImage, RgbaImage};
use ort::session::Session;
use ort::tensor::TensorElementType;
use ort::value::{DynValue, Tensor, ValueType};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use super::upscaler::{self, build_cpu_session, open_session, ActiveGuard, ExecMode, MODEL_DIR};
use super::bg_remover;

/// Every supported model takes and returns 1×3×512×512 RGB in [-1, 1].
const SIZE: u32 = 512;
const INDEX_FILE: &str = "face_models.json";
pub const DEFAULT_FIDELITY: f32 = 0.7;

static INDEX_LOCK: Mutex<()> = Mutex::new(());
/// Cached `(model_id, session)`; the lock also serialises runs so rayon workers do not fight over VRAM.
static SESSION: Mutex<Option<(String, Session)>> = Mutex::new(None);

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct FaceModel {
    pub id: String,
    pub name: String,
    pub file_name: String,
    pub size_mb: f64,
    /// Image input dtype: "float16" | "float32" | "float64".
    pub dtype: String,
    /// true = CodeFormer-style second input `w` (fidelity weight).
    pub has_fidelity: bool,
}

impl FaceModel {
    fn heavy(&self) -> bool {
        self.size_mb > 60.0
    }
}

fn face_dir() -> Result<PathBuf, String> {
    MODEL_DIR.get().map(|d| d.join("custom_face")).ok_or_else(|| "Folder model belum disediakan".to_string())
}

/// Missing file = empty list; a corrupt file is an error so it is never silently overwritten.
fn read_index(dir: &Path) -> Result<Vec<FaceModel>, String> {
    match std::fs::read(dir.join(INDEX_FILE)) {
        Ok(b) => serde_json::from_slice(&b).map_err(|e| format!("{INDEX_FILE} rosak: {e}")),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(e) => Err(e.to_string()),
    }
}

/// Write-then-rename so a crash never leaves a half-written index.
fn write_index(dir: &Path, list: &[FaceModel]) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let tmp = dir.join(format!("{INDEX_FILE}.tmp"));
    std::fs::write(&tmp, serde_json::to_vec_pretty(list).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, dir.join(INDEX_FILE)).map_err(|e| e.to_string())
}

/// The index is user-editable: only accept a bare file name.
fn model_file(dir: &Path, m: &FaceModel) -> Result<PathBuf, String> {
    if Path::new(&m.file_name).file_name().and_then(|f| f.to_str()) != Some(m.file_name.as_str()) {
        return Err(format!("Nama fail model tidak sah: {}", m.file_name));
    }
    Ok(dir.join(&m.file_name))
}

fn find(id: &str) -> Result<(FaceModel, PathBuf), String> {
    let dir = face_dir()?;
    let m = read_index(&dir)?
        .into_iter()
        .find(|m| m.id == id)
        .ok_or_else(|| "Pilih model Pulih Wajah yang telah diimport".to_string())?;
    let path = model_file(&dir, &m)?;
    Ok((m, path))
}

fn dtype_name(t: &ValueType) -> Option<&'static str> {
    match t {
        ValueType::Tensor { ty: TensorElementType::Float16, .. } => Some("float16"),
        ValueType::Tensor { ty: TensorElementType::Float32, .. } => Some("float32"),
        ValueType::Tensor { ty: TensorElementType::Float64, .. } => Some("float64"),
        _ => None,
    }
}

/// NCHW `[1, 3, 512, 512]`; any dimension may be dynamic (<= 0).
fn accepts_512(dims: &[i64]) -> bool {
    let ok = |d: i64, want: i64| d == want || d <= 0;
    dims.len() == 4 && ok(dims[0], 1) && ok(dims[1], 3) && ok(dims[2], SIZE as i64) && ok(dims[3], SIZE as i64)
}

/// Checks a probe session; returns (image dtype, has fidelity input).
fn validate(session: &Session) -> Result<(&'static str, bool), String> {
    let first = session.inputs.first().ok_or("tiada input")?;
    let dtype = dtype_name(&first.input_type).ok_or("input mesti tensor Float16/Float32/Float64")?;
    match &first.input_type {
        ValueType::Tensor { dimensions, .. } if accepts_512(dimensions) => {}
        _ => return Err("input pertama mesti [1, 3, 512, 512]".into()),
    }
    if session.inputs.len() > 2 {
        return Err(format!("{} input tidak disokong (maksimum 2)", session.inputs.len()));
    }
    if let Some(w) = session.inputs.get(1) {
        dtype_name(&w.input_type).ok_or("input kedua (w) mesti Float16/Float32/Float64")?;
    }
    if session.outputs.is_empty() {
        return Err("tiada output".into());
    }
    Ok((dtype, session.inputs.len() > 1))
}

/// Copies the model in and validates its inputs on a temporary CPU session.
#[tauri::command(async)]
pub fn import_face_model(source_path: String) -> Result<FaceModel, String> {
    let (source_path, _tmp) = super::pth_converter::prepare(&source_path, super::pth_converter::PthKind::Face)?;
    let (id, name, dest, size_mb) = upscaler::copy_in(&source_path, "face")?;
    let res = upscaler::probe_session(&dest)
        .and_then(|session| validate(&session))
        .map_err(|why| format!("Bukan model pulih wajah yang sah: {why}"))
        .and_then(|(dtype, has_fidelity)| {
            let m = FaceModel { file_name: format!("{id}.onnx"), id, name, size_mb, dtype: dtype.into(), has_fidelity };
            let _g = INDEX_LOCK.lock().unwrap_or_else(|p| p.into_inner());
            let dir = face_dir()?;
            let mut list = read_index(&dir)?;
            list.push(m.clone());
            write_index(&dir, &list)?;
            Ok(m)
        });
    if res.is_err() {
        let _ = std::fs::remove_file(&dest);
    }
    res
}

#[tauri::command]
pub fn list_face_models() -> Result<Vec<FaceModel>, String> {
    read_index(&face_dir()?)
}

#[tauri::command(async)]
pub fn delete_face_model(id: String) -> Result<(), String> {
    let _g = INDEX_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let dir = face_dir()?;
    let mut list = read_index(&dir)?;
    let pos = list.iter().position(|m| m.id == id).ok_or_else(|| format!("Model tidak dikenali: {id}"))?;
    // A live session keeps the file open (Windows cannot delete it).
    clear_session();
    match std::fs::remove_file(model_file(&dir, &list[pos])?) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(format!("Gagal memadam model: {e}")),
        _ => {}
    }
    list.remove(pos);
    write_index(&dir, &list)
}

/// Drops the cached face session so its VRAM is freed immediately.
pub fn clear_session() {
    *SESSION.lock().unwrap_or_else(|p| p.into_inner()) = None;
}

/// f32 data → tensor of the dtype the model input declares.
fn tensor_for(t: &ValueType, shape: Vec<i64>, data: &[f32]) -> Result<DynValue, String> {
    match dtype_name(t) {
        Some("float16") => Tensor::from_array((shape, data.iter().map(|&v| f16::from_f32(v)).collect::<Vec<_>>())).map(|t| t.into_dyn()),
        Some("float64") => Tensor::from_array((shape, data.iter().map(|&v| v as f64).collect::<Vec<_>>())).map(|t| t.into_dyn()),
        _ => Tensor::from_array((shape, data.to_vec())).map(|t| t.into_dyn()),
    }
    .map_err(|e| e.to_string())
}

fn infer(session: &Session, patch: &[f32], fidelity: f32) -> Result<Vec<f32>, String> {
    let img_in = &session.inputs[0];
    let mut inputs = vec![(img_in.name.clone(), tensor_for(&img_in.input_type, vec![1, 3, SIZE as i64, SIZE as i64], patch)?)];
    if let Some(w) = session.inputs.get(1) {
        // Scalar exports declare `[]`; the usual CodeFormer export declares `[1]`.
        let scalar = matches!(&w.input_type, ValueType::Tensor { dimensions, .. } if dimensions.is_empty());
        inputs.push((w.name.clone(), tensor_for(&w.input_type, if scalar { vec![] } else { vec![1] }, &[fidelity])?));
    }
    let outputs = session.run(inputs).map_err(|e| e.to_string())?;
    let out = &outputs[0];
    let data: Vec<f32> = if let Ok((_, d)) = out.try_extract_raw_tensor::<f16>() {
        d.iter().map(|v| v.to_f32()).collect()
    } else if let Ok((_, d)) = out.try_extract_raw_tensor::<f64>() {
        d.iter().map(|&v| v as f32).collect()
    } else {
        out.try_extract_raw_tensor::<f32>().map_err(|e| e.to_string())?.1.to_vec()
    };
    if data.len() != (3 * SIZE * SIZE) as usize {
        return Err("Output model tidak sepadan dengan 3×512×512".to_string());
    }
    Ok(data)
}

/// 512×512 RGBA → NCHW RGB in [-1, 1].
fn to_patch(img: &RgbaImage) -> Vec<f32> {
    let plane = (SIZE * SIZE) as usize;
    let mut patch = vec![0f32; plane * 3];
    for (i, px) in img.pixels().enumerate() {
        for c in 0..3 {
            patch[c * plane + i] = (px[c] as f32 / 255.0 - 0.5) / 0.5;
        }
    }
    patch
}

/// NCHW [-1, 1] → opaque 512×512 RGBA.
fn from_output(data: &[f32]) -> RgbaImage {
    let plane = (SIZE * SIZE) as usize;
    RgbaImage::from_fn(SIZE, SIZE, |x, y| {
        let i = (y * SIZE + x) as usize;
        let ch = |c: usize| ((data[c * plane + i] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8;
        image::Rgba([ch(0), ch(1), ch(2), 255])
    })
}

/// Restores the face and blends it over the original by `fidelity` (0 = original, 1 = full AI).
/// CodeFormer-style models also receive `fidelity` as their `w` input.
// ponytail: the whole frame is squashed to 512×512 (no face detection/alignment), so it suits
// cropped portraits; add a face detector + paste-back when group photos must be supported.
pub fn restore_face(img: DynamicImage, model_id: &str, fidelity: f32, mode: ExecMode) -> Result<DynamicImage, String> {
    let fidelity = if fidelity.is_finite() { fidelity.clamp(0.0, 1.0) } else { DEFAULT_FIDELITY };
    if fidelity == 0.0 {
        return Ok(img);
    }
    // First thing, before our own lock (never nested): only one AI model holds VRAM.
    upscaler::clear_session();
    bg_remover::clear_session();
    let (m, path) = find(model_id)?;
    if !path.exists() {
        return Err("Fail model pulih wajah tiada; sila import semula".to_string());
    }
    let key = mode.key(model_id);
    let mut guard = SESSION.lock().unwrap_or_else(|p| p.into_inner());
    if guard.as_ref().is_none_or(|(id, _)| *id != key) {
        *guard = None;
        let session = open_session(&path, m.heavy(), mode).map_err(|e| format!("Gagal memuatkan model pulih wajah: {e}"))?;
        *guard = Some((key.clone(), session));
    }

    let _active = ActiveGuard::new();
    let had_alpha = img.color().has_alpha();
    let orig = img.into_rgba8();
    let (w, h) = orig.dimensions();
    let patch = to_patch(&upscaler::lanczos(&orig, SIZE, SIZE)?);
    let data = match infer(&guard.as_ref().expect("session just set").1, &patch, fidelity) {
        Err(e) if upscaler::is_gpu_oom(&e) => {
            eprintln!("DirectML kehabisan memori ({e}); beralih ke CPU");
            *guard = None; // release the failed GPU session first
            let cpu = build_cpu_session(&path, upscaler::LABEL_OOM)
                .map_err(|e| format!("Gagal memuatkan model pulih wajah (CPU): {e}"))?;
            *guard = Some((key, cpu));
            infer(&guard.as_ref().expect("session just set").1, &patch, fidelity)?
        }
        r => r?,
    };
    drop(guard);

    let restored = upscaler::lanczos(&from_output(&data), w, h)?;
    let out = upscaler::blend_rgb(&orig, &restored, fidelity);
    Ok(if had_alpha {
        DynamicImage::ImageRgba8(out)
    } else {
        DynamicImage::ImageRgb8(DynamicImage::ImageRgba8(out).to_rgb8())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_shape_check() {
        assert!(accepts_512(&[1, 3, 512, 512]));
        assert!(accepts_512(&[-1, 3, -1, -1]));
        assert!(!accepts_512(&[1, 3, 256, 256]));
        assert!(!accepts_512(&[1, 4, 512, 512]));
        assert!(!accepts_512(&[1, 3, 512]));
    }

    #[test]
    fn normalise_round_trip() {
        let img = RgbaImage::from_pixel(SIZE, SIZE, image::Rgba([0, 128, 255, 9]));
        let patch = to_patch(&img);
        assert_eq!(patch[0], -1.0);
        assert_eq!(patch[2 * (SIZE * SIZE) as usize], 1.0);
        assert_eq!(from_output(&patch).get_pixel(7, 7).0, [0, 128, 255, 255]);
        // Out-of-range output is clamped.
        assert_eq!(from_output(&vec![3.0; patch.len()]).get_pixel(0, 0).0, [255, 255, 255, 255]);
    }
}
