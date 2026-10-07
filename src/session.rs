//! Model registry, model download and ONNX inference.
//!
//! Every local session python-rembg registers is a row in [`MODELS`]: they all
//! normalize the image the same way and take channel 0 of the first output as
//! the mask, so the per-model Python classes collapse into one table.

use std::env;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use image::imageops::FilterType;
use image::{DynamicImage, ImageBuffer, Luma};
use ndarray::Array4;
use ort::value::TensorRef;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub struct ModelSpec {
    pub name: &'static str,
    /// Release asset; the cached file is always `<name>.onnx`.
    pub url: &'static str,
    /// `"<algo>:<hex>"`, skipped when `MODEL_CHECKSUM_DISABLED` is set.
    pub checksum: &'static str,
    pub size: (u32, u32),
    pub mean: [f64; 3],
    pub std: [f64; 3],
    /// Run a sigmoid over the prediction before min/max normalization.
    pub sigmoid: bool,
}

const IMAGENET: ([f64; 3], [f64; 3]) = (
    [0.485, 0.456, 0.406],
    [0.229, 0.224, 0.225],
);
macro_rules! model {
    ($name:literal, $file:literal, $sum:literal, $size:expr, $mean:expr, $std:expr, $sigmoid:expr) => {
        ModelSpec {
            name: $name,
            url: concat!(
                "https://github.com/danielgatis/rembg/releases/download/v0.0.0/",
                $file
            ),
            checksum: $sum,
            size: $size,
            mean: $mean,
            std: $std,
            sigmoid: $sigmoid,
        }
    };
}

pub const MODELS: &[ModelSpec] = &[
    model!(
        "bria-rmbg",
        "bria-rmbg-2.0.onnx",
        "sha256:5b486f08200f513f460da46dd701db5fbb47d79b4be4b708a19444bcd4e79958",
        (1024, 1024),
        IMAGENET.0,
        IMAGENET.1,
        false
    ),
    model!(
        "u2net",
        "u2net.onnx",
        "md5:60024c5c889badc19c04ad937298a77b",
        (320, 320),
        IMAGENET.0,
        IMAGENET.1,
        false
    ),
    model!(
        "u2netp",
        "u2netp.onnx",
        "md5:8e83ca70e441ab06c318d82300c84806",
        (320, 320),
        IMAGENET.0,
        IMAGENET.1,
        false
    ),
    model!(
        "u2net_human_seg",
        "u2net_human_seg.onnx",
        "md5:c09ddc2e0104f800e3e1bb4652583d1f",
        (320, 320),
        IMAGENET.0,
        IMAGENET.1,
        false
    ),
    model!(
        "silueta",
        "silueta.onnx",
        "md5:55e59e0d8062d2f5d013f4725ee84782",
        (320, 320),
        IMAGENET.0,
        IMAGENET.1,
        false
    ),
    model!(
        "isnet-general-use",
        "isnet-general-use.onnx",
        "md5:fc16ebd8b0c10d971d3513d564d01e29",
        (1024, 1024),
        [0.5, 0.5, 0.5],
        [1.0, 1.0, 1.0],
        false
    ),
    model!(
        "isnet-anime",
        "isnet-anime.onnx",
        "md5:6f184e756bb3bd901c8849220a83e38e",
        (1024, 1024),
        [0.485, 0.456, 0.406],
        [1.0, 1.0, 1.0],
        false
    ),
    model!(
        "birefnet-general",
        "BiRefNet-general-epoch_244.onnx",
        "md5:7a35a0141cbbc80de11d9c9a28f52697",
        (1024, 1024),
        IMAGENET.0,
        IMAGENET.1,
        true
    ),
    model!(
        "birefnet-general-lite",
        "BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx",
        "md5:4fab47adc4ff364be1713e97b7e66334",
        (1024, 1024),
        IMAGENET.0,
        IMAGENET.1,
        true
    ),
    model!(
        "birefnet-portrait",
        "BiRefNet-portrait-epoch_150.onnx",
        "md5:c3a64a6abf20250d090cd055f12a3b67",
        (1024, 1024),
        IMAGENET.0,
        IMAGENET.1,
        true
    ),
    model!(
        "birefnet-dis",
        "BiRefNet-DIS-epoch_590.onnx",
        "md5:2d4d44102b446f33a4ebb2e56c051f2b",
        (1024, 1024),
        IMAGENET.0,
        IMAGENET.1,
        true
    ),
    model!(
        "birefnet-hrsod",
        "BiRefNet-HRSOD_DHU-epoch_115.onnx",
        "md5:c017ade5de8a50ff0fd74d790d268dda",
        (1024, 1024),
        IMAGENET.0,
        IMAGENET.1,
        true
    ),
    model!(
        "birefnet-cod",
        "BiRefNet-COD-epoch_125.onnx",
        "md5:f6d0d21ca89d287f17e7afe9f5fd3b45",
        (1024, 1024),
        IMAGENET.0,
        IMAGENET.1,
        true
    ),
    model!(
        "birefnet-massive",
        "BiRefNet-massive-TR_DIS5K_TR_TEs-epoch_420.onnx",
        "md5:33e726a2136a3d59eb0fdf613e31e3e9",
        (1024, 1024),
        IMAGENET.0,
        IMAGENET.1,
        true
    ),
];

pub fn model_names() -> Vec<&'static str> {
    MODELS.iter().map(|m| m.name).collect()
}

pub fn spec(name: &str) -> Result<&'static ModelSpec> {
    MODELS
        .iter()
        .find(|m| m.name == name)
        .ok_or_else(|| format!("No session class found for model '{name}'").into())
}

/// Root directory for rembg data, matching `BaseSession.rembg_home`.
pub fn home() -> PathBuf {
    if let Ok(v) = env::var("U2NET_HOME") {
        return PathBuf::from(v);
    }
    if let Ok(v) = env::var("REMBG_HOME") {
        return PathBuf::from(v);
    }
    match env::var("XDG_DATA_HOME") {
        Ok(xdg) => PathBuf::from(xdg).join("rembg"),
        Err(_) => dirs_home().join(".rembg"),
    }
}

/// The pre-2.0 flat model directory, read-only, kept so old downloads are reused.
fn legacy_home() -> PathBuf {
    if let Ok(v) = env::var("U2NET_HOME") {
        return PathBuf::from(v);
    }
    match env::var("XDG_DATA_HOME") {
        Ok(xdg) => PathBuf::from(xdg).join(".u2net"),
        Err(_) => dirs_home().join(".u2net"),
    }
}

fn dirs_home() -> PathBuf {
    PathBuf::from(env::var("HOME").unwrap_or_else(|_| ".".into()))
}

/// An already-downloaded copy of `name`, in either layout, if any.
fn resolve_existing(name: &str) -> Option<PathBuf> {
    let fname = format!("{name}.onnx");
    [home().join("models").join(name).join(&fname), legacy_home().join(&fname)]
        .into_iter()
        .find(|p| p.exists())
}

/// Path to the model file, downloading it first when it is missing.
pub fn ensure_model(name: &str) -> Result<PathBuf> {
    let spec = spec(name)?;
    if let Some(path) = resolve_existing(name) {
        return Ok(path);
    }

    let dir = home().join("models").join(name);
    fs::create_dir_all(&dir)?;
    let path = dir.join(format!("{name}.onnx"));
    eprintln!("Downloading model: {name}");

    let resp = ureq::get(spec.url)
        .set("User-Agent", "rembg-rs")
        .call()
        .map_err(|e| format!("Error downloading model {name}: {e}"))?;
    let total = resp
        .header("content-length")
        .and_then(|v| v.parse::<u64>().ok());
    let mut reader = resp.into_reader();

    let (algo, want) = spec.checksum.split_once(':').unwrap_or(("", ""));
    let mut hasher: Option<Hasher> = if env::var_os("MODEL_CHECKSUM_DISABLED").is_some() {
        None
    } else {
        Some(Hasher::new(algo)?)
    };

    let part = dir.join(format!("{name}.onnx.part"));
    let mut file = fs::File::create(&part)?;
    let mut buf = vec![0u8; 1 << 16];
    let mut done: u64 = 0;
    let mut last_reported = 0u8;
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        if let Some(h) = hasher.as_mut() {
            h.update(&buf[..n]);
        }
        done += n as u64;
        if let Some(total) = total {
            let pct = (done * 100 / total) as u8;
            if pct >= last_reported + 10 {
                last_reported = pct - pct % 10;
                eprintln!("  {pct}%");
            }
        }
    }
    drop(file);
    drop(reader);

    if let Some(h) = hasher {
        let got = h.hex_digest();
        if !got.eq_ignore_ascii_case(want) {
            let _ = fs::remove_file(&part);
            return Err(format!("Checksum mismatch for {name}: expected {want}, got {got}").into());
        }
    }

    fs::rename(&part, &path)?;
    Ok(path)
}

/// Hashes the whole model file to `hex` for comparison with the table.
enum Hasher {
    Md5(md5::Md5),
    Sha256(sha2::Sha256),
}

use md5::Digest as _;

impl Hasher {
    fn new(algo: &str) -> Result<Hasher> {
        match algo {
            "md5" => Ok(Hasher::Md5(md5::Md5::new())),
            "sha256" => Ok(Hasher::Sha256(sha2::Sha256::new())),
            other => Err(format!("unsupported checksum algorithm '{other}'").into()),
        }
    }

    fn update(&mut self, data: &[u8]) {
        match self {
            Hasher::Md5(h) => h.update(data),
            Hasher::Sha256(h) => h.update(data),
        }
    }

    fn hex_digest(self) -> String {
        match self {
            Hasher::Md5(h) => format!("{:x}", h.finalize()),
            Hasher::Sha256(h) => format!("{:x}", h.finalize()),
        }
    }
}

/// A loaded model, ready to predict masks.
pub struct Session {
    spec: &'static ModelSpec,
    inner: ort::session::Session,
}

impl Session {
    pub fn new(name: &str) -> Result<Session> {
        let spec = spec(name)?;
        let path = ensure_model(name)?;

        let mut builder = ort::session::Session::builder()?;
        if let Ok(threads) = env::var("OMP_NUM_THREADS") {
            if let Ok(n) = threads.parse::<usize>() {
                builder = builder.with_intra_threads(n)?;
            }
        }
        let inner = builder.commit_from_file(path)?;

        Ok(Session { spec, inner })
    }

    /// The model's own name, e.g. `"u2net"`.
    pub fn name(&self) -> &'static str {
        self.spec.name
    }

    /// Runs the model on `img` and returns its mask at the image's size.
    pub fn predict(&mut self, img: &DynamicImage) -> Result<ImageBuffer<Luma<u8>, Vec<u8>>> {
        let (w, h) = self.spec.size;
        let resized = image::imageops::resize(&img.to_rgb8(), w, h, FilterType::Lanczos3);

        // numpy: im_ary / max(im_ary), then per-channel (c - mean) / std.
        let peak = resized
            .pixels()
            .flat_map(|p| p.0)
            .map(|c| c as f64)
            .fold(0.0f64, f64::max)
            .max(1e-6);
        let mut input = Array4::<f32>::zeros((1, 3, h as usize, w as usize));
        for (x, y, p) in resized.enumerate_pixels() {
            for c in 0..3 {
                let v = (p.0[c] as f64 / peak - self.spec.mean[c]) / self.spec.std[c];
                input[[0, c, y as usize, x as usize]] = v as f32;
            }
        }

        let outputs = self
            .inner
            .run(ort::inputs![TensorRef::from_array_view(&input)?])?;
        let out = outputs[0].try_extract_array::<f32>()?;
        let shape = out.shape().to_vec();
        if shape.len() != 4 || shape[1] < 1 {
            return Err(format!("unexpected model output shape {shape:?}").into());
        }
        let (ow, oh) = (shape[3], shape[2]);

        // Channel 0 of the first output, optionally squashed by a sigmoid.
        let mut pred: Vec<f32> = Vec::with_capacity((ow * oh) as usize);
        for y in 0..oh {
            for x in 0..ow {
                let v = out[[0, 0, y, x]];
                pred.push(if self.spec.sigmoid { 1.0 / (1.0 + (-v).exp()) } else { v });
            }
        }

        // Min/max normalization, then *255 truncated to 8 bits as numpy does.
        let mut ma = f32::NEG_INFINITY;
        let mut mi = f32::INFINITY;
        for v in &pred {
            ma = ma.max(*v);
            mi = mi.min(*v);
        }
        let mask = ImageBuffer::from_fn(ow as u32, oh as u32, |x, y| {
            let v = (pred[y as usize * ow + x as usize] - mi) / (ma - mi);
            Luma([(v.clamp(0.0, 1.0) * 255.0) as u8])
        });

        Ok(image::imageops::resize(
            &mask,
            img.width(),
            img.height(),
            FilterType::Lanczos3,
        ))
    }
}

/// Downloads every model (or the named ones), like `rembg-rs d`.
pub fn download_models(names: &[&str]) -> Result<()> {
    let names: Vec<&str> = if names.is_empty() {
        MODELS.iter().map(|m| m.name).collect()
    } else {
        names.to_vec()
    };
    for name in names {
        ensure_model(name)?;
        eprintln!("Downloaded model: {name}");
    }
    Ok(())
}

/// Path to a model file for tests and tooling, without downloading.
#[allow(dead_code)]
pub fn existing_model_path(name: &str) -> Option<PathBuf> {
    resolve_existing(name).filter(|p| Path::new(p).exists())
}
