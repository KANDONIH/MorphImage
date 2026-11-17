//! Shared `.pth`/`.pt` → `.onnx` conversion via system Python + spandrel.

use std::path::{Path, PathBuf};
use tempfile::TempDir;

pub const PTH_DEPS_MSG: &str = "Penukaran automatik fail .pth memerlukan Python + PyTorch (~2.5 GB ruang storan tambahan). Untuk menjimatkan ruang, muat turun fail .onnx terus dari Hugging Face. Jika anda mahu teruskan penukaran .pth, pasang Python dan jalankan: pip install torch spandrel onnx";

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PthKind {
    Upscale,
    Face,
    Background,
}

impl PthKind {
    fn arg(self) -> &'static str {
        match self {
            PthKind::Upscale => "upscale",
            PthKind::Face => "face",
            PthKind::Background => "bg",
        }
    }
}

/// argv: <input.pth> <output.onnx> <kind>. Exit 3 = missing library.
const PTH_TO_ONNX_PY: &str = r#"import sys, inspect
try:
    import torch, spandrel, onnx
except ImportError:
    sys.exit(3)
src, dst, kind = sys.argv[1:4]
d = spandrel.ModelLoader().load_from_file(src).eval().cpu()
m = d.model
wname = None
if kind == "face":  # CodeFormer: forward(x, w=...) -> second ONNX input "w"
    params = inspect.signature(m.forward).parameters
    wname = next((n for n in ("w", "weight", "fidelity") if n in params), None)

class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, x, *w):
        out = self.m(x, **{wname: w[0].to(x.dtype).squeeze()}) if w else self.m(x)
        return out[0] if isinstance(out, (tuple, list)) else out

if kind == "upscale":
    size, dyn = 64, {"input": {2: "h", 3: "w"}, "output": {2: "h", 3: "w"}}
elif kind == "face":
    size, dyn = 512, None
else:  # ponytail: 1024 unless spandrel demands a larger minimum; no per-arch default table
    size, dyn = max(1024, getattr(d.size_requirements, "minimum", 0) or 0), None
args = (torch.rand(1, 3, size, size),)
names = ["input"]
if wname:
    args += (torch.tensor([0.7], dtype=torch.float64),)
    names.append("w")
kw = dict(opset_version=17, input_names=names, output_names=["output"], dynamic_axes=dyn)
with torch.no_grad():
    try:
        torch.onnx.export(Wrap(m), args, dst, dynamo=False, **kw)
    except TypeError:  # torch < 2.5 has no `dynamo` argument
        torch.onnx.export(Wrap(m), args, dst, **kw)
"#;

/// Converts into a fresh empty temp dir (removed when the `TempDir` drops), so nothing
/// planted beside the script can shadow a Python import. Output keeps the source's stem.
pub fn convert_pth_to_onnx(pth_path: &Path, kind: PthKind) -> Result<(PathBuf, TempDir), String> {
    if !pth_path.is_file() {
        return Err("Sila pilih fail .pth yang sah".to_string());
    }
    let work = tempfile::Builder::new().prefix("morphimage_pth_").tempdir().map_err(|e| e.to_string())?;
    let script = work.path().join("pth_to_onnx.py");
    let stem = pth_path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "converted".into());
    let out = work.path().join(format!("{stem}.onnx"));
    std::fs::write(&script, PTH_TO_ONNX_PY).map_err(|e| e.to_string())?;
    for py in ["python", "py", "python3"] {
        let mut cmd = std::process::Command::new(py);
        cmd.arg(&script).arg(pth_path).arg(&out).arg(kind.arg()).current_dir(work.path());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
        }
        let o = match cmd.output() {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => return Err(format!("Gagal menjalankan Python: {e}")),
            Ok(o) => o,
        };
        return match o.status.code() {
            Some(0) if out.is_file() => Ok((out, work)),
            Some(3) => Err(PTH_DEPS_MSG.to_string()),
            Some(9009) => continue, // Windows Store "python" stub: not really installed
            _ => {
                let err = String::from_utf8_lossy(&o.stderr);
                let mut tail: Vec<&str> = err.lines().rev().take(3).collect();
                tail.reverse();
                Err(format!("Penukaran .pth gagal (seni bina tidak disokong spandrel?): {}", tail.join(" ")))
            }
        };
    }
    Err(PTH_DEPS_MSG.to_string())
}

/// `.pth`/`.pt` are converted (keep the guard alive until copied); anything else passes through.
pub fn prepare(source: &str, kind: PthKind) -> Result<(String, Option<TempDir>), String> {
    let p = Path::new(source);
    if p.extension().is_some_and(|e| e.eq_ignore_ascii_case("pth") || e.eq_ignore_ascii_case("pt")) {
        let (onnx, tmp) = convert_pth_to_onnx(p, kind)?;
        Ok((onnx.to_string_lossy().into_owned(), Some(tmp)))
    } else {
        Ok((source.to_string(), None))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bogus_pth_fails_and_leaves_no_temp_files() {
        let leftovers = || {
            std::fs::read_dir(std::env::temp_dir()).unwrap().flatten()
                .filter(|e| e.file_name().to_string_lossy().starts_with("morphimage_pth_")).count()
        };
        let before = leftovers();
        let fake = std::env::temp_dir().join(format!("morph-fake-{}.pth", std::process::id()));
        std::fs::write(&fake, b"not a model").unwrap();
        for kind in [PthKind::Upscale, PthKind::Face, PthKind::Background] {
            assert!(prepare(&fake.to_string_lossy(), kind).is_err());
        }
        assert_eq!(leftovers(), before);
        let _ = std::fs::remove_file(&fake);
        assert!(prepare("tiada.pt", PthKind::Face).unwrap_err().contains(".pth"));
        assert_eq!(prepare("a.onnx", PthKind::Face).unwrap().0, "a.onnx");
    }
}
