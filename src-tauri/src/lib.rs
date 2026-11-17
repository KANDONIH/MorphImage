mod models;
mod commands;
mod engine;
mod utils;

// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

#[tauri::command]
fn open_folder(app: tauri::AppHandle, path: String) -> Result<(), String> {
    use tauri_plugin_opener::OpenerExt;
    app.opener().open_path(path, None::<&str>).map_err(|e| e.to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .manage(commands::system::new_sys_state())
        .setup(|app| {
            use tauri::Manager;
            let dir = app.path().app_data_dir()?.join("models");
            let _ = engine::upscaler::MODEL_DIR.set(dir);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            greet,
            open_folder,
            commands::inspect::inspect_images,
            commands::inspect::paste_image_from_clipboard,
            commands::process::copy_image_to_clipboard,
            commands::process::export_images_to_pdf,
            commands::process::process_single_image,
            commands::process::process_batch_images,
            commands::system::get_system_stats,
            commands::system::get_models_storage_info,
            commands::system::open_models_folder,
            commands::system::get_logical_cores,
            engine::upscaler::get_gpu_info,
            engine::upscaler::check_upscale_model,
            engine::upscaler::download_upscale_model,
            engine::bg_remover::check_bg_model,
            engine::bg_remover::download_bg_model,
            engine::upscaler::list_custom_models,
            engine::upscaler::import_upscale_model,
            engine::upscaler::delete_custom_model,
            engine::bg_remover::import_bg_model,
            engine::face_restorer::import_face_model,
            engine::face_restorer::list_face_models,
            engine::face_restorer::delete_face_model,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
