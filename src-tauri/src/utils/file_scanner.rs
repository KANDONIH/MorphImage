use std::path::Path;
use walkdir::WalkDir;

const IMAGE_EXTS: &[&str] = &[
    "jpg", "jpeg", "png", "webp", "gif", "tiff", "tif", "bmp", "svg", "ico", "jxl", "psd", "cr2",
    "cr3", "nef", "arw", "dng", "raw",
];

fn is_image(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| IMAGE_EXTS.contains(&e.to_lowercase().as_str()))
        .unwrap_or(false)
}

/// Expand a mixed list of file/folder paths into a flat list of image file paths.
pub fn expand_paths(paths: Vec<String>) -> Vec<String> {
    let mut out = Vec::new();
    for p in paths {
        let path = Path::new(&p);
        if path.is_dir() {
            for entry in WalkDir::new(&path)
                .follow_links(true)
                .into_iter()
                .filter_map(|e| e.ok())
            {
                let ep = entry.path();
                if ep.is_file() && is_image(ep) {
                    if let Some(s) = ep.to_str() {
                        out.push(s.to_string());
                    }
                }
            }
        } else if path.is_file() && is_image(path) {
            out.push(p);
        }
    }
    out
}
