use crate::engine::upscaler::{ACTIVE_AI_BACKEND, GPU_AI_ACTIVE};
use crate::models::SystemStats;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};

/// Tugasan Rayon yang sedang berjalan (dinaik/turun dalam process_batch_images).
pub static BUSY_THREADS: AtomicUsize = AtomicUsize::new(0);
/// Saiz kolam worker Rayon terakhir yang dibina.
pub static POOL_THREADS: AtomicUsize = AtomicUsize::new(0);

/// Cap thread: tinggalkan 2 teras untuk OS/UI.
pub fn worker_cap() -> usize {
    logical_cores().saturating_sub(2).max(1)
}

pub fn logical_cores() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
}

/// User thread choice from Settings: None/0 = auto cap, else clamped to 1..=logical cores.
pub fn pool_size(requested: Option<usize>) -> usize {
    match requested {
        Some(n) if n > 0 => n.min(logical_cores()),
        _ => worker_cap(),
    }
}

#[derive(serde::Serialize)]
pub struct ModelsStorageInfo {
    pub path: String,
    pub total_bytes: u64,
    pub file_count: usize,
}

fn models_dir(app: &tauri::AppHandle) -> Result<std::path::PathBuf, String> {
    use tauri::Manager;
    app.path().app_data_dir().map(|d| d.join("models")).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn get_models_storage_info(app: tauri::AppHandle) -> Result<ModelsStorageInfo, String> {
    let dir = models_dir(&app)?;
    let (mut total_bytes, mut file_count) = (0u64, 0usize);
    // Missing folder (no model yet) yields 0 bytes.
    for m in walkdir::WalkDir::new(&dir).into_iter().filter_map(Result::ok).filter_map(|e| e.metadata().ok()) {
        if m.is_file() {
            total_bytes += m.len();
            file_count += 1;
        }
    }
    Ok(ModelsStorageInfo { path: dir.to_string_lossy().into_owned(), total_bytes, file_count })
}

#[tauri::command]
pub fn open_models_folder(app: tauri::AppHandle) -> Result<(), String> {
    use tauri_plugin_opener::OpenerExt;
    let dir = models_dir(&app)?;
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    app.opener().open_path(dir.to_string_lossy(), None::<&str>).map_err(|e| e.to_string())
}

/// Logical core count, for the thread slider max.
#[tauri::command]
pub fn get_logical_cores() -> usize {
    logical_cores()
}

#[cfg(test)]
mod pool_tests {
    use super::*;

    #[test]
    fn pool_size_clamps() {
        assert_eq!(pool_size(None), worker_cap());
        assert_eq!(pool_size(Some(0)), worker_cap());
        assert_eq!(pool_size(Some(1)), 1);
        assert_eq!(pool_size(Some(usize::MAX)), logical_cores());
    }
}

/// Telemetri GPU. Pembilang PDH ialah kadar: perlu dua kali kutipan, jadi query disimpan.
#[cfg(windows)]
mod gpu {
    use windows::core::{w, Interface, PCWSTR};
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory1, IDXGIAdapter3, IDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE,
        DXGI_MEMORY_SEGMENT_GROUP_LOCAL, DXGI_QUERY_VIDEO_MEMORY_INFO,
    };
    use windows::Win32::System::Performance::*;

    pub struct Probe {
        /// Nama penyesuai dengan VRAM khusus terbanyak (untuk paparan).
        pub name: String,
        /// Semua penyesuai perkakasan: (pemegang DXGI, "luid_0x..._0x..." untuk padanan PDH).
        adapters: Vec<(IDXGIAdapter3, String)>,
        query: isize,
        counter: isize,
    }
    // ponytail: DXGI ialah free-threaded; akses sentiasa di bawah Mutex.
    unsafe impl Send for Probe {}

    impl Probe {
        /// Semua penyesuai bukan-perisian. Nama paparan = yang VRAM khususnya terbanyak.
        pub fn new() -> Self {
            let mut p = Probe { name: "Tiada GPU".into(), adapters: Vec::new(), query: 0, counter: 0 };
            unsafe {
                let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory1>() else { return p };
                let (mut best, mut i) = (0u64, 0);
                while let Ok(a) = factory.EnumAdapters1(i) {
                    i += 1;
                    let Ok(d) = a.GetDesc1() else { continue };
                    let len = d.Description.iter().position(|&c| c == 0).unwrap_or(d.Description.len());
                    let name = String::from_utf16_lossy(&d.Description[..len]);
                    if d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 || name.contains("Microsoft Basic Render") {
                        continue;
                    }
                    let Ok(a3) = a.cast::<IDXGIAdapter3>() else { continue };
                    let luid = format!("luid_0x{:08x}_0x{:08x}", d.AdapterLuid.HighPart as u32, d.AdapterLuid.LowPart);
                    p.adapters.push((a3, luid));
                    if p.name == "Tiada GPU" || (d.DedicatedVideoMemory as u64) > best {
                        best = d.DedicatedVideoMemory as u64;
                        p.name = name;
                    }
                }
                if p.adapters.is_empty() {
                    return p;
                }
                let (mut q, mut c) = (0isize, 0isize);
                if PdhOpenQueryW(PCWSTR::null(), 0, &mut q) == 0 {
                    if PdhAddEnglishCounterW(q, w!("\\GPU Engine(*)\\Utilization Percentage"), 0, &mut c) == 0 {
                        PdhCollectQueryData(q); // kutipan asas; nilai pertama tiada kadar
                        (p.query, p.counter) = (q, c);
                    } else {
                        PdhCloseQuery(q);
                    }
                }
            }
            p
        }

        /// (peratus enjin 3D+Compute, VRAM digunakan MB): nilai tertinggi merentas semua
        /// penyesuai perkakasan, supaya iGPU/dGPU mana pun yang dipakai DirectML dikesan.
        pub fn sample(&mut self) -> (f32, u64) {
            if self.adapters.is_empty() {
                return (0.0, 0);
            }
            let usage = unsafe { self.usage() }.unwrap_or_default();
            let vram = self
                .adapters
                .iter()
                .map(|(a, _)| unsafe {
                    let mut info: DXGI_QUERY_VIDEO_MEMORY_INFO = std::mem::zeroed();
                    a.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info)
                        .map_or(0, |_| info.CurrentUsage / 1_048_576)
                })
                .max()
                .unwrap_or(0);
            (usage, vram)
        }

        /// Jumlah 3D+Compute setiap penyesuai; pulangkan yang tertinggi.
        unsafe fn usage(&self) -> Option<f32> {
            unsafe {
                if self.query == 0 || PdhCollectQueryData(self.query) != 0 {
                    return None;
                }
                let (mut size, mut count) = (0u32, 0u32);
                if PdhGetFormattedCounterArrayW(self.counter, PDH_FMT_DOUBLE, &mut size, &mut count, None) != PDH_MORE_DATA {
                    return None;
                }
                let mut buf = vec![0u64; (size as usize).div_ceil(8)]; // u64: penjajaran untuk struct item
                let items = buf.as_mut_ptr() as *mut PDH_FMT_COUNTERVALUE_ITEM_W;
                if PdhGetFormattedCounterArrayW(self.counter, PDH_FMT_DOUBLE, &mut size, &mut count, Some(items)) != 0 {
                    return None;
                }
                let mut per_adapter = vec![0f64; self.adapters.len()];
                for it in std::slice::from_raw_parts(items, count as usize) {
                    if it.FmtValue.CStatus > 1 {
                        continue; // bukan VALID_DATA / NEW_DATA
                    }
                    let Some(n) = it.szName.to_string().ok().map(|s| s.to_lowercase()) else { continue };
                    if !(n.contains("engtype_3d") || n.contains("engtype_compute")) {
                        continue;
                    }
                    if let Some(i) = self.adapters.iter().position(|(_, luid)| n.contains(luid.as_str())) {
                        per_adapter[i] += it.FmtValue.Anonymous.doubleValue;
                    }
                }
                let max = per_adapter.into_iter().fold(0.0, f64::max);
                Some(max.clamp(0.0, 100.0) as f32)
            }
        }
    }
}

#[cfg(not(windows))]
mod gpu {
    pub struct Probe {
        pub name: String,
    }
    impl Probe {
        pub fn new() -> Self {
            Probe { name: "Tiada GPU".into() }
        }
        pub fn sample(&mut self) -> (f32, u64) {
            (0.0, 0)
        }
    }
}

pub struct SysState(pub Mutex<System>, Mutex<gpu::Probe>);

pub fn new_sys_state() -> SysState {
    // System disimpan supaya delta CPU dikira antara panggilan, bukan dari sifar.
    SysState(Mutex::new(System::new().expect("sysinfo init")), Mutex::new(gpu::Probe::new()))
}

#[tauri::command]
pub fn get_system_stats(state: tauri::State<'_, SysState>) -> Result<SystemStats, String> {
    let mut sys = state.0.lock().map_err(|e| e.to_string())?;
    let pid = Pid::from_u32(std::process::id());

    sys.refresh_cpu_usage();
    sys.refresh_processes_specifics(
        ProcessesToUpdate::Some(&[pid]),
        false,
        ProcessRefreshKind::nothing().with_memory(),
    );

    let app_memory_mb = sys
        .process(pid)
        .map(|p| p.memory() as f64 / 1_048_576.0)
        .unwrap_or(0.0);

    let pool = POOL_THREADS.load(Ordering::Relaxed);
    let total_threads = if pool == 0 { worker_cap() } else { pool };

    let mut probe = state.1.lock().map_err(|e| e.to_string())?;
    let (gpu_usage, gpu_vram_used_mb) = probe.sample();

    Ok(SystemStats {
        cpu_usage: sys.global_cpu_usage(),
        app_memory_mb,
        active_threads: BUSY_THREADS.load(Ordering::Relaxed).min(total_threads),
        total_threads,
        gpu_name: probe.name.clone(),
        gpu_usage,
        gpu_vram_used_mb,
        gpu_active: GPU_AI_ACTIVE.load(Ordering::Relaxed),
        ai_backend: ACTIVE_AI_BACKEND.read().unwrap_or_else(|p| p.into_inner()).clone(),
    })
}
