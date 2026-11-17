use crate::engine::decode::{ext_of, quick_dimensions, INPUT_ONLY_EXTS};
use crate::models::ImageInfo;
use crate::utils::file_scanner;
use std::path::Path;

#[tauri::command]
pub async fn paste_image_from_clipboard() -> Result<ImageInfo, String> {
    tauri::async_runtime::spawn_blocking(|| {
        let clip = arboard::Clipboard::new()
            .and_then(|mut c| c.get_image())
            .map_err(|e| format!("Tiada imej dalam papan keratan: {e}"))?;
        let buf = image::RgbaImage::from_raw(clip.width as u32, clip.height as u32, clip.bytes.into_owned())
            .ok_or("Data imej papan keratan tidak sah")?;
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_millis());
        let path = std::env::temp_dir().join(format!("morphimage_clip_{stamp}.png"));
        buf.save(&path).map_err(|e| e.to_string())?;
        let size_bytes = std::fs::metadata(&path).map_err(|e| e.to_string())?.len();
        Ok(ImageInfo {
            path: path.to_string_lossy().into_owned(),
            name: path.file_name().unwrap().to_string_lossy().into_owned(),
            size_bytes,
            width: buf.width(),
            height: buf.height(),
            format: "PNG".into(),
        })
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
pub async fn inspect_images(paths: Vec<String>) -> Result<Vec<ImageInfo>, String> {
    let paths = file_scanner::expand_paths(paths);
    let mut results = Vec::with_capacity(paths.len());

    for path_str in paths {
        let path = Path::new(&path_str);

        let size_bytes = std::fs::metadata(&path_str)
            .map_err(|e| format!("{}: {}", path_str, e))?
            .len();

        // SVG/PSD/JXL: header read; RAW: 0x0 (full decode too slow for listing).
        let (width, height) = if INPUT_ONLY_EXTS.contains(&ext_of(&path_str).as_str()) {
            quick_dimensions(&path_str).unwrap_or((0, 0))
        } else {
            image::image_dimensions(&path_str).map_err(|e| format!("{}: {}", path_str, e))?
        };

        let format = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("unknown")
            .to_uppercase();

        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(&path_str)
            .to_string();

        results.push(ImageInfo {
            path: path_str,
            name,
            size_bytes,
            width,
            height,
            format,
        });
    }

    Ok(results)
}
