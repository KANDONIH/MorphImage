use image::{DynamicImage, RgbaImage};
use std::path::Path;

pub const RAW_EXTS: &[&str] = &["cr2", "cr3", "nef", "arw", "dng", "raw"];
/// Formats we can read but not write.
pub const INPUT_ONLY_EXTS: &[&str] = &["svg", "psd", "jxl", "cr2", "cr3", "nef", "arw", "dng", "raw"];

pub fn ext_of(path: &str) -> String {
    Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase()
}

/// Decode any supported input into a `DynamicImage`.
pub fn decode_any_image(path: &str) -> Result<DynamicImage, String> {
    let ext = ext_of(path);
    match ext.as_str() {
        "svg" => decode_svg(path),
        "psd" => decode_psd(path),
        "jxl" => decode_jxl(path),
        e if RAW_EXTS.contains(&e) => decode_raw(path),
        _ => image::open(path).map_err(|e| e.to_string()),
    }
}

/// Fast dimensions for formats `image` can't read. `None` = unknown (caller uses 0x0).
pub fn quick_dimensions(path: &str) -> Option<(u32, u32)> {
    match ext_of(path).as_str() {
        "svg" => {
            let data = std::fs::read(path).ok()?;
            let tree = resvg::usvg::Tree::from_data(&data, &resvg::usvg::Options::default()).ok()?;
            Some((tree.size().width().ceil() as u32, tree.size().height().ceil() as u32))
        }
        "psd" => {
            // PSD header: 4 sig, 2 ver, 6 reserved, 2 channels, 4 height, 4 width (big-endian).
            use std::io::Read;
            let mut h = [0u8; 22];
            std::fs::File::open(path).ok()?.read_exact(&mut h).ok()?;
            if &h[0..4] != b"8BPS" {
                return None;
            }
            Some((
                u32::from_be_bytes(h[18..22].try_into().ok()?),
                u32::from_be_bytes(h[14..18].try_into().ok()?),
            ))
        }
        "jxl" => {
            let img = jxl_oxide::JxlImage::builder().open(path).ok()?;
            Some((img.width(), img.height()))
        }
        _ => None, // RAW: needs full decode; list stays fast with 0x0.
    }
}

fn decode_svg(path: &str) -> Result<DynamicImage, String> {
    use resvg::{tiny_skia, usvg};
    let data = std::fs::read(path).map_err(|e| e.to_string())?;
    let tree = usvg::Tree::from_data(&data, &usvg::Options::default()).map_err(|e| e.to_string())?;
    let size = tree.size();
    let (w0, h0) = (size.width().max(1.0), size.height().max(1.0));
    // Scale up so the longest side is at least 1024px; cap at 8192px.
    let scale = (1024.0 / w0.max(h0)).max(1.0).min(8192.0 / w0.max(h0));
    let (w, h) = ((w0 * scale).ceil() as u32, (h0 * scale).ceil() as u32);
    let mut pixmap = tiny_skia::Pixmap::new(w, h).ok_or("SVG terlalu besar")?;
    resvg::render(&tree, tiny_skia::Transform::from_scale(scale, scale), &mut pixmap.as_mut());
    // tiny-skia is premultiplied; image expects straight alpha.
    let mut raw = Vec::with_capacity((w * h * 4) as usize);
    for p in pixmap.pixels() {
        let c = p.demultiply();
        raw.extend_from_slice(&[c.red(), c.green(), c.blue(), c.alpha()]);
    }
    RgbaImage::from_raw(w, h, raw)
        .map(DynamicImage::ImageRgba8)
        .ok_or_else(|| "SVG: buffer tidak sah".into())
}

fn decode_psd(path: &str) -> Result<DynamicImage, String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    let psd = psd::Psd::from_bytes(&bytes).map_err(|e| e.to_string())?;
    RgbaImage::from_raw(psd.width(), psd.height(), psd.rgba())
        .map(DynamicImage::ImageRgba8)
        .ok_or_else(|| "PSD: buffer tidak sah".into())
}

fn decode_jxl(path: &str) -> Result<DynamicImage, String> {
    let img = jxl_oxide::JxlImage::builder().open(path).map_err(|e| e.to_string())?;
    let render = img.render_frame(0).map_err(|e| e.to_string())?;
    let (w, h) = (img.width(), img.height());
    let mut stream = render.stream();
    let ch = stream.channels() as usize;
    let mut buf = vec![0u8; w as usize * h as usize * ch];
    stream.write_to_buffer(&mut buf);
    let dynimg = match ch {
        1 => image::GrayImage::from_raw(w, h, buf).map(DynamicImage::ImageLuma8),
        2 => image::GrayAlphaImage::from_raw(w, h, buf).map(DynamicImage::ImageLumaA8),
        3 => image::RgbImage::from_raw(w, h, buf).map(DynamicImage::ImageRgb8),
        4 => RgbaImage::from_raw(w, h, buf).map(DynamicImage::ImageRgba8),
        _ => None,
    };
    dynimg.ok_or_else(|| "JXL: format saluran tidak disokong".into())
}

/// Prefer embedded full-size JPEG; fall back to preview, thumbnail, then sensor develop.
fn decode_raw(path: &str) -> Result<DynamicImage, String> {
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut rawfile = rawler::RawFile::new(path, std::io::BufReader::new(file));
    let embedded = rawler::get_decoder(&mut rawfile).ok().and_then(|d| {
        d.full_image(&mut rawfile)
            .ok()
            .flatten()
            .or_else(|| d.preview_image(&mut rawfile).ok().flatten())
            .or_else(|| d.thumbnail_image(&mut rawfile).ok().flatten())
    });
    let old = match embedded {
        Some(i) => i,
        None => {
            let raw = rawler::decode_file(path).map_err(|e| e.to_string())?;
            rawler::imgop::develop::RawDevelop::default()
                .develop_intermediate(&raw)
                .map_err(|e| e.to_string())?
                .to_dynamic_image()
                .ok_or("RAW: gagal menghasilkan imej")?
        }
    };
    // rawler uses image 0.24: hop through raw RGBA8 bytes into our image 0.25.
    let rgba = old.to_rgba8();
    let (w, h) = (rgba.width(), rgba.height());
    RgbaImage::from_raw(w, h, rgba.into_raw())
        .map(DynamicImage::ImageRgba8)
        .ok_or_else(|| "RAW: buffer tidak sah".into())
}
