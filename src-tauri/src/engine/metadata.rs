use image::DynamicImage;
use std::io::BufReader;

/// Read EXIF Orientation tag and apply rotation/flip so the image is correctly oriented.
/// Returns the corrected image. If EXIF absent or unreadable, returns image unchanged.
pub fn auto_orient(img: DynamicImage, path: &str) -> DynamicImage {
    let orientation = read_orientation(path).unwrap_or(1);
    apply_orientation(img, orientation)
}

fn read_orientation(path: &str) -> Option<u32> {
    let file = std::fs::File::open(path).ok()?;
    let mut bufreader = BufReader::new(file);
    let exif_reader = exif::Reader::new();
    let exif = exif_reader.read_from_container(&mut bufreader).ok()?;
    let field = exif.get_field(exif::Tag::Orientation, exif::In::PRIMARY)?;
    match field.value {
        exif::Value::Short(ref v) if !v.is_empty() => Some(v[0] as u32),
        _ => None,
    }
}

fn apply_orientation(img: DynamicImage, orientation: u32) -> DynamicImage {
    match orientation {
        2 => img.fliph(),
        3 => img.rotate180(),
        4 => img.flipv(),
        5 => img.rotate90().fliph(),
        6 => img.rotate90(),
        7 => img.rotate270().fliph(),
        8 => img.rotate270(),
        _ => img, // 1 or unknown: no-op
    }
}
