use image::{DynamicImage, ImageEncoder, codecs::jpeg::JpegEncoder, codecs::png::PngEncoder};

/// Encode image to JPEG bytes at given quality (1–100).
pub fn encode_jpeg(img: &DynamicImage, quality: u8) -> Result<Vec<u8>, String> {
    let rgb = img.to_rgb8();
    let mut buf = Vec::new();
    let enc = JpegEncoder::new_with_quality(&mut buf, quality);
    enc.write_image(
        rgb.as_raw(),
        rgb.width(),
        rgb.height(),
        image::ExtendedColorType::Rgb8,
    )
    .map_err(|e| e.to_string())?;
    Ok(buf)
}

/// Encode image to PNG bytes, then run oxipng compression (preset 2).
/// `lossy`: quantize to 256 colors (exoquant, Floyd-Steinberg) first.
pub fn encode_png(img: &DynamicImage, lossy: bool) -> Result<Vec<u8>, String> {
    let mut rgba = img.to_rgba8();
    if lossy {
        let w = rgba.width() as usize;
        let px: Vec<exoquant::Color> = rgba
            .pixels()
            .map(|p| exoquant::Color::new(p[0], p[1], p[2], p[3]))
            .collect();
        let (palette, idx) = exoquant::convert_to_indexed(
            &px,
            w,
            256,
            &exoquant::optimizer::KMeans,
            &exoquant::ditherer::FloydSteinberg::new(),
        );
        for (p, &i) in rgba.pixels_mut().zip(idx.iter()) {
            let c = palette[i as usize];
            p.0 = [c.r, c.g, c.b, c.a];
        }
    }
    let mut raw = Vec::new();
    let enc = PngEncoder::new(&mut raw);
    enc.write_image(
        rgba.as_raw(),
        rgba.width(),
        rgba.height(),
        image::ExtendedColorType::Rgba8,
    )
    .map_err(|e| e.to_string())?;

    // Compress with oxipng preset 2 (fast + good ratio)
    let opts = oxipng::Options::from_preset(2);
    oxipng::optimize_from_memory(&raw, &opts).map_err(|e| e.to_string())
}

/// Binary-search quality in 15..=95 (max 6 encodes) for the best result <= `max_bytes`.
/// Falls back to quality 15 if nothing fits.
pub fn fit_quality(
    max_bytes: usize,
    enc: impl Fn(u8) -> Result<Vec<u8>, String>,
) -> Result<Vec<u8>, String> {
    let (mut lo, mut hi) = (15u8, 95u8);
    let mut best = None;
    for _ in 0..6 {
        if lo > hi {
            break;
        }
        let mid = (lo + hi) / 2;
        let out = enc(mid)?;
        if out.len() <= max_bytes {
            best = Some(out);
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    match best {
        Some(b) => Ok(b),
        None => enc(15),
    }
}

fn encode_with(img: &DynamicImage, fmt: image::ImageFormat) -> Result<Vec<u8>, String> {
    let mut buf = std::io::Cursor::new(Vec::new());
    img.write_to(&mut buf, fmt).map_err(|e| e.to_string())?;
    Ok(buf.into_inner())
}

/// AVIF via ravif (pure Rust). Speed 8 keeps batch encodes tolerable.
pub fn encode_avif(img: &DynamicImage, quality: u8) -> Result<Vec<u8>, String> {
    let rgba = img.to_rgba8();
    let mut buf = Vec::new();
    image::codecs::avif::AvifEncoder::new_with_speed_quality(&mut buf, 8, quality.clamp(1, 100))
        .write_image(rgba.as_raw(), rgba.width(), rgba.height(), image::ExtendedColorType::Rgba8)
        .map_err(|e| e.to_string())?;
    Ok(buf)
}

pub fn encode_tiff(img: &DynamicImage) -> Result<Vec<u8>, String> {
    encode_with(&DynamicImage::ImageRgba8(img.to_rgba8()), image::ImageFormat::Tiff)
}

pub fn encode_bmp(img: &DynamicImage) -> Result<Vec<u8>, String> {
    encode_with(&DynamicImage::ImageRgba8(img.to_rgba8()), image::ImageFormat::Bmp)
}

pub fn encode_gif(img: &DynamicImage) -> Result<Vec<u8>, String> {
    encode_with(&DynamicImage::ImageRgba8(img.to_rgba8()), image::ImageFormat::Gif)
}

/// ICO: fit within 256x256 (keep aspect), centre on a transparent square canvas.
pub fn encode_ico(img: &DynamicImage) -> Result<Vec<u8>, String> {
    let fitted = if img.width() > 256 || img.height() > 256 {
        img.resize(256, 256, image::imageops::FilterType::Lanczos3)
    } else {
        img.clone()
    }
    .to_rgba8();
    let side = fitted.width().max(fitted.height());
    let mut canvas = image::RgbaImage::new(side, side);
    image::imageops::overlay(
        &mut canvas,
        &fitted,
        ((side - fitted.width()) / 2) as i64,
        ((side - fitted.height()) / 2) as i64,
    );
    encode_with(&DynamicImage::ImageRgba8(canvas), image::ImageFormat::Ico)
}

/// Encode image to lossy WebP bytes at given quality (1–100) via libwebp.
pub fn encode_webp(img: &DynamicImage, quality: u8) -> Result<Vec<u8>, String> {
    let rgba = img.to_rgba8();
    let enc = webp::Encoder::from_rgba(rgba.as_raw(), rgba.width(), rgba.height());
    Ok(enc.encode(quality.clamp(1, 100) as f32).to_vec())
}
