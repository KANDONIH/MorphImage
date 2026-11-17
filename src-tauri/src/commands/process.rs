use crate::commands::system;
use crate::engine::pipeline;
use crate::models::{BatchProgressEvent, CropRect, ProcessOptions, ProcessResult};
use crate::utils::file_scanner;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tauri::Emitter;

#[tauri::command]
pub async fn process_single_image(
    path: String,
    options: ProcessOptions,
) -> Result<ProcessResult, String> {
    // Run blocking I/O + CPU work off the async executor
    tauri::async_runtime::spawn_blocking(move || pipeline::process_image(&path, &options, 1))
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn copy_image_to_clipboard(path: String) -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(move || {
        let rgba = crate::engine::decode::decode_any_image(&path)?.to_rgba8();
        arboard::Clipboard::new()
            .and_then(|mut c| {
                c.set_image(arboard::ImageData {
                    width: rgba.width() as usize,
                    height: rgba.height() as usize,
                    bytes: std::borrow::Cow::Owned(rgba.into_raw()),
                })
            })
            .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| e.to_string())?
}

/// One full page per image (page size = image size in points), JPEG embedded as-is (DCTDecode).
#[tauri::command]
pub async fn export_images_to_pdf(
    paths: Vec<String>,
    options: ProcessOptions,
    output_pdf_path: String,
    crops: Option<HashMap<String, CropRect>>,
) -> Result<u64, String> {
    tauri::async_runtime::spawn_blocking(move || {
        use lopdf::content::{Content, Operation};
        use lopdf::{dictionary, Document, Object, Stream};

        if paths.is_empty() {
            return Err("Tiada imej untuk dieksport".to_string());
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(system::pool_size(options.worker_threads))
            .build()
            .map_err(|e| e.to_string())?;
        // Order preserved by par_iter().collect().
        let pages: Vec<(u32, u32, Vec<u8>)> = pool.install(|| {
            paths
                .par_iter()
                .map(|p| {
                    let mut opts = options.clone();
                    if let Some(c) = crops.as_ref().and_then(|m| m.get(p)) {
                        opts.crop = Some(*c);
                    }
                    let img = pipeline::prepare_image(p, &opts).map_err(|e| format!("{p}: {e}"))?;
                    let jpg = pipeline::encode_as(&img, "jpg", &opts).map_err(|e| format!("{p}: {e}"))?;
                    Ok((img.width(), img.height(), jpg))
                })
                .collect::<Result<_, String>>()
        })?;

        let mut doc = Document::with_version("1.5");
        let pages_id = doc.new_object_id();
        let mut kids: Vec<Object> = Vec::with_capacity(pages.len());
        for (w, h, jpg) in pages {
            let (wf, hf) = (w as f32, h as f32);
            let img_id = doc.add_object(Stream::new(
                dictionary! {
                    "Type" => "XObject",
                    "Subtype" => "Image",
                    "Width" => w as i64,
                    "Height" => h as i64,
                    "ColorSpace" => "DeviceRGB",
                    "BitsPerComponent" => 8,
                    "Filter" => "DCTDecode",
                },
                jpg,
            ));
            let content = Content {
                operations: vec![
                    Operation::new("q", vec![]),
                    Operation::new("cm", vec![wf.into(), 0.into(), 0.into(), hf.into(), 0.into(), 0.into()]),
                    Operation::new("Do", vec!["Im0".into()]),
                    Operation::new("Q", vec![]),
                ],
            };
            let content_id = doc.add_object(Stream::new(dictionary! {}, content.encode().map_err(|e| e.to_string())?));
            let page_id = doc.add_object(dictionary! {
                "Type" => "Page",
                "Parent" => pages_id,
                "MediaBox" => vec![0.into(), 0.into(), wf.into(), hf.into()],
                "Contents" => content_id,
                "Resources" => dictionary! { "XObject" => dictionary! { "Im0" => img_id } },
            });
            kids.push(page_id.into());
        }
        let count = kids.len() as i64;
        doc.objects.insert(
            pages_id,
            dictionary! { "Type" => "Pages", "Kids" => kids, "Count" => count }.into(),
        );
        let catalog_id = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages_id });
        doc.trailer.set("Root", catalog_id);
        doc.save(&output_pdf_path).map_err(|e| e.to_string())?;
        std::fs::metadata(&output_pdf_path)
            .map(|m| m.len())
            .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
pub async fn process_batch_images(
    app: tauri::AppHandle,
    paths: Vec<String>,
    options: ProcessOptions,
    crops: Option<HashMap<String, CropRect>>,
) -> Result<Vec<ProcessResult>, String> {
    let paths = file_scanner::expand_paths(paths);
    let total = paths.len();

    // Settings choice, else auto cap (leave 2 cores for the OS/UI)
    let threads = system::pool_size(options.worker_threads);
    system::POOL_THREADS.store(threads, Ordering::Relaxed);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|e| e.to_string())?;

    let options = Arc::new(options);
    let crops = Arc::new(crops.unwrap_or_default());
    let counter = Arc::new(AtomicUsize::new(0));

    // Collect results in parallel, emitting progress after each image
    let results: Vec<ProcessResult> = pool.install(|| {
        paths
            .par_iter()
            .enumerate()
            .map(|(i, path)| {
                system::BUSY_THREADS.fetch_add(1, Ordering::Relaxed);
                let mut opts = (*options).clone();
                if let Some(c) = crops.get(path) {
                    opts.crop = Some(*c);
                }
                let result = pipeline::process_image(path, &opts, i + 1);
                system::BUSY_THREADS.fetch_sub(1, Ordering::Relaxed);
                let index = counter.fetch_add(1, Ordering::Relaxed);
                let event = BatchProgressEvent {
                    index,
                    total,
                    result: result.clone(),
                };
                let _ = app.emit("batch-progress", event);
                result
            })
            .collect()
    });

    Ok(results)
}
