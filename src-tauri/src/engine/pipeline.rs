use crate::engine::decode::{decode_any_image, INPUT_ONLY_EXTS};
use crate::engine::{bg_remover, compressor, face_restorer, metadata, upscaler};
use crate::models::{OutputFormat, ProcessOptions, ProcessResult};
use image::imageops::FilterType;
use std::path::Path;

/// Replacement background for Remove BG: None/empty hex = transparent cut-out.
fn bg_color(opts: &ProcessOptions) -> Result<Option<[u8; 3]>, String> {
    match opts.bg_replace_color.as_deref().map(str::trim) {
        None | Some("") => Ok(None),
        Some(hex) => bg_remover::parse_hex_color(hex).map(Some),
    }
}

/// Full pipeline: Decode -> Auto-Orient EXIF -> Resize -> Encode & Compress -> Save.
/// `index`: 1-based position in the batch, used for `use_numbering`.
pub fn process_image(source: &str, opts: &ProcessOptions, index: usize) -> ProcessResult {
    let original_size = match std::fs::metadata(source) {
        Ok(m) => m.len(),
        Err(e) => {
            return error_result(source, 0, e.to_string());
        }
    };

    let img = match prepare_image(source, opts) {
        Ok(i) => i,
        Err(e) => return error_result(source, original_size, e),
    };

    // 4. Determine output extension
    let src_path = Path::new(source);
    let src_ext = src_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("jpg")
        .to_lowercase();

    // Input-only sources (RAW/PSD/SVG/JXL) can't be written back: pick PNG if transparent, else JPEG.
    let input_only = INPUT_ONLY_EXTS.contains(&src_ext.as_str());
    let original_ext = if !input_only {
        src_ext.clone()
    } else if img.to_rgba8().pixels().any(|p| p[3] < 255) {
        "png".to_string()
    } else {
        "jpg".to_string()
    };
    let out_ext = match &opts.output_format {
        OutputFormat::Original => original_ext,
        OutputFormat::Jpeg => "jpg".to_string(),
        OutputFormat::Png => "png".to_string(),
        OutputFormat::Webp => "webp".to_string(),
        OutputFormat::Avif => "avif".to_string(),
        OutputFormat::Tiff => "tiff".to_string(),
        OutputFormat::Ico => "ico".to_string(),
        OutputFormat::Bmp => "bmp".to_string(),
        OutputFormat::Gif => "gif".to_string(),
    };
    // Transparent cut-out into a format without alpha would turn the background black.
    let transparent_cut = match bg_color(opts) {
        Ok(bg) => opts.remove_bg && bg.is_none(),
        Err(e) => return error_result(source, original_size, e),
    };
    let out_ext = if transparent_cut && (out_ext == "jpg" || out_ext == "bmp") {
        "png".to_string()
    } else {
        out_ext
    };
    let bytes = encode_as(&img, &out_ext, opts);

    let bytes = match bytes {
        Ok(b) => b,
        Err(e) => return error_result(source, original_size, e),
    };

    // Smart copy: Original format, no resize, encoded >= original → copy source verbatim
    let stem = src_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("image");
    // ponytail: overwrite keeps the stem; a format change writes `<stem>.<new ext>` beside the source.
    let out_path = if opts.overwrite_original {
        src_path.with_extension(&out_ext).to_string_lossy().into_owned()
    } else {
        let out_dir = match opts.output_dir.as_deref() {
            Some(d) if !d.is_empty() => std::path::PathBuf::from(d),
            _ => src_path.parent().unwrap_or(Path::new(".")).join("output"),
        };
        if let Err(e) = std::fs::create_dir_all(&out_dir) {
            return error_result(source, original_size, e.to_string());
        }
        let num = if opts.use_numbering { format!("{:03}_", index) } else { String::new() };
        let out_stem = format!("{}{}{}", num, stem, opts.custom_suffix);
        unique_path(&out_dir.to_string_lossy(), &out_stem, &out_ext)
    };
    let in_place = opts.overwrite_original && Path::new(&out_path) == src_path;

    let write_result = if matches!(opts.output_format, OutputFormat::Original)
        && !input_only
        && !opts.remove_bg
        && !opts.restore_face
        && opts.max_width.is_none()
        && opts.crop.is_none()
        && opts.upscale_factor.is_none()
        && opts.rotate_deg % 360 == 0
        && !opts.flip_horizontal
        && !opts.flip_vertical
        && opts.target_size_kb.is_none()
        && !opts.png_lossy
        && bytes.len() as u64 >= original_size
    {
        // Encoded is no smaller: in place = leave the source untouched.
        if in_place { Ok(()) } else { std::fs::copy(source, &out_path).map(|_| ()) }
    } else if opts.overwrite_original {
        // Temp + rename so a failed write never leaves a half-written original.
        let tmp = format!("{out_path}.morph-tmp");
        std::fs::write(&tmp, &bytes)
            .and_then(|_| std::fs::rename(&tmp, &out_path))
            .inspect_err(|_| {
                let _ = std::fs::remove_file(&tmp);
            })
    } else {
        std::fs::write(&out_path, &bytes)
    };

    if let Err(e) = write_result {
        return error_result(source, original_size, e.to_string());
    }

    let new_size = std::fs::metadata(&out_path)
        .map(|m| m.len())
        .unwrap_or(bytes.len() as u64);
    let saved_percent = if original_size > 0 {
        ((original_size as f64 - new_size as f64) / original_size as f64 * 100.0) as f32
    } else {
        0.0
    };

    ProcessResult {
        source_path: source.to_string(),
        output_path: out_path,
        original_size,
        new_size,
        saved_percent,
        success: true,
        error: None,
    }
}

/// Decode -> EXIF auto-orient -> Crop -> Rotate -> Flip -> Remove BG -> Restore Face -> Upscale -> Resize.
pub fn prepare_image(source: &str, opts: &ProcessOptions) -> Result<image::DynamicImage, String> {
    let exec = upscaler::ExecMode::parse(&opts.execution_mode);
    let mut img = metadata::auto_orient(decode_any_image(source)?, source);

    if let Some(c) = opts.crop {
        let (w, h) = (img.width() as f32, img.height() as f32);
        let x = (c.x.clamp(0.0, 1.0) * w).round() as u32;
        let y = (c.y.clamp(0.0, 1.0) * h).round() as u32;
        let cw = ((c.width.clamp(0.0, 1.0) * w).round() as u32).clamp(1, img.width().saturating_sub(x).max(1));
        let ch = ((c.height.clamp(0.0, 1.0) * h).round() as u32).clamp(1, img.height().saturating_sub(y).max(1));
        img = img.crop_imm(x.min(img.width() - 1), y.min(img.height() - 1), cw, ch);
    }

    img = match opts.rotate_deg % 360 {
        90 => img.rotate90(),
        180 => img.rotate180(),
        270 => img.rotate270(),
        _ => img,
    };
    if opts.flip_horizontal {
        img = img.fliph();
    }
    if opts.flip_vertical {
        img = img.flipv();
    }

    if opts.remove_bg {
        img = bg_remover::remove_background(img, bg_color(opts)?, &opts.bg_model, exec)?;
    }

    // Before upscale so the face is cleaned up before it is enlarged.
    if opts.restore_face {
        img = face_restorer::restore_face(img, &opts.face_model_id, opts.face_fidelity, exec)?;
    }

    if let Some(f) = opts.upscale_factor {
        let mode = upscaler::UpscaleMode::parse(&opts.upscale_mode)?;
        img = upscaler::upscale_image(&img, f, mode, opts.ai_strength, &opts.upscale_model, exec)?;
    }

    match opts.max_width {
        Some(max_w) if img.width() > max_w => Ok(img.resize(max_w, u32::MAX, FilterType::Lanczos3)),
        _ => Ok(img),
    }
}

/// Encode by extension. JPEG/WebP honour `target_size_kb` via quality search.
pub fn encode_as(img: &image::DynamicImage, ext: &str, opts: &ProcessOptions) -> Result<Vec<u8>, String> {
    let q = opts.quality;
    let target = opts.target_size_kb.map(|kb| kb as usize * 1024);
    let lossy = |enc: &dyn Fn(u8) -> Result<Vec<u8>, String>| match target {
        Some(max) => compressor::fit_quality(max, enc),
        None => enc(q),
    };
    match ext {
        "png" => compressor::encode_png(img, opts.png_lossy),
        "webp" => lossy(&|q| compressor::encode_webp(img, q)),
        "avif" => lossy(&|q| compressor::encode_avif(img, q)),
        "tiff" | "tif" => compressor::encode_tiff(img),
        "ico" => compressor::encode_ico(img),
        "bmp" => compressor::encode_bmp(img),
        "gif" => compressor::encode_gif(img),
        _ => lossy(&|q| compressor::encode_jpeg(img, q)),
    }
}

/// Returns `<dir>/<stem>.<ext>`, adding `_1`, `_2`… on collision.
fn unique_path(dir: &str, stem: &str, ext: &str) -> String {
    let base = format!("{}/{}.{}", dir, stem, ext);
    if !Path::new(&base).exists() {
        return base;
    }
    let mut n = 1u32;
    loop {
        let candidate = format!("{}/{}_{}.{}", dir, stem, n, ext);
        if !Path::new(&candidate).exists() {
            return candidate;
        }
        n += 1;
    }
}

fn error_result(source: &str, original_size: u64, msg: String) -> ProcessResult {
    ProcessResult {
        source_path: source.to_string(),
        output_path: String::new(),
        original_size,
        new_size: 0,
        saved_percent: 0.0,
        success: false,
        error: Some(msg),
    }
}
