//! The `remove()` pipeline: decode, fix orientation, predict a mask, then cut
//! the background out with one of the modes python-rembg offers.

use std::io::Cursor;

use image::codecs::png::PngEncoder;
use image::{DynamicImage, GrayImage, Rgba, RgbaImage};

use crate::matting;
use crate::session::{Result, Session};

/// Knobs matching `rembg.remove(...)`'s keyword arguments.
#[derive(Debug, Clone)]
pub struct RemoveOptions {
    pub alpha_matting: bool,
    pub alpha_matting_foreground_threshold: u8,
    pub alpha_matting_background_threshold: u8,
    pub alpha_matting_erode_size: usize,
    pub only_mask: bool,
    pub post_process_mask: bool,
    pub decontaminate: bool,
    pub bgcolor: Option<[u8; 4]>,
}

impl Default for RemoveOptions {
    fn default() -> Self {
        RemoveOptions {
            alpha_matting: false,
            alpha_matting_foreground_threshold: 240,
            alpha_matting_background_threshold: 10,
            alpha_matting_erode_size: 10,
            only_mask: false,
            post_process_mask: false,
            decontaminate: false,
            bgcolor: None,
        }
    }
}

/// Removes the background from encoded image bytes, returning PNG bytes —
/// the `rembg-rs i` code path.
pub fn remove_bytes(data: &[u8], session: &mut Session, opts: &RemoveOptions) -> Result<Vec<u8>> {
    let orientation = orientation(data);
    let img = image::load_from_memory(data)?;
    let img = fix_image_orientation(img, orientation);

    let out = remove(&img, session, opts)?;
    let mut png = Vec::new();
    out.write_with_encoder(PngEncoder::new(&mut png))?;
    Ok(png)
}

/// Removes the background from a decoded image.
pub fn remove(img: &DynamicImage, session: &mut Session, opts: &RemoveOptions) -> Result<DynamicImage> {
    let mut mask = session.predict(img)?;
    if opts.post_process_mask {
        mask = post_process(&mask);
    }

    if opts.only_mask {
        return Ok(DynamicImage::ImageLuma8(mask));
    }

    let cutout = if opts.alpha_matting {
        match matting::alpha_matting_cutout(
            img,
            &mask,
            opts.alpha_matting_foreground_threshold,
            opts.alpha_matting_background_threshold,
            opts.alpha_matting_erode_size,
        ) {
            Ok(cutout) => cutout,
            // Alpha matting already unmixes the foreground color; when it fails
            // to converge fall back to the decontaminated cutout so the edges
            // stay usable, like rembg's `except ValueError` branch.
            Err(_) => matting::decontaminate_cutout(img, &mask),
        }
    } else if opts.decontaminate {
        matting::decontaminate_cutout(img, &mask)
    } else {
        naive_cutout(img, &mask)
    };

    let cutout = match opts.bgcolor {
        Some(color) => apply_background_color(&cutout, color),
        None => cutout,
    };
    Ok(DynamicImage::ImageRgba8(cutout))
}

/// `Image.composite` against a fully transparent image: the mask becomes alpha
/// and the colors pass through untouched.
fn naive_cutout(img: &DynamicImage, mask: &GrayImage) -> RgbaImage {
    let src = img.to_rgba8();
    let (w, h) = (mask.width(), mask.height());
    RgbaImage::from_fn(w, h, |x, y| {
        let p = src.get_pixel(x, y);
        let m = mask.get_pixel(x, y)[0] as u32;
        let scale = |c: u32| ((c * m + 127) / 255) as u8;
        Rgba([scale(p[0] as u32), scale(p[1] as u32), scale(p[2] as u32), scale(p[3] as u32)])
    })
}

/// Standard "over" compositing of the cutout onto a solid color.
fn apply_background_color(cutout: &RgbaImage, color: [u8; 4]) -> RgbaImage {
    let (w, h) = (cutout.width(), cutout.height());
    let bg = Rgba(color);
    RgbaImage::from_fn(w, h, |x, y| {
        let src = *cutout.get_pixel(x, y);
        let (sa, da) = (src[3] as u32, bg[3] as u32);
        let out_a = sa + (da * (255 - sa) + 127) / 255;
        if out_a == 0 {
            return Rgba([0, 0, 0, 0]);
        }
        let mut px = [0u8; 4];
        for c in 0..3 {
            let v = src[c] as u32 * sa + bg[c] as u32 * da * (255 - sa) / 255;
            px[c] = ((v + out_a / 2) / out_a) as u8;
        }
        px[3] = out_a as u8;
        Rgba(px)
    })
}

/// Morphological opening followed by a blur and a threshold, so the mask's
/// boundary comes out smooth instead of stair-stepped.
pub fn post_process(mask: &GrayImage) -> GrayImage {
    let (w, h) = (mask.width() as usize, mask.height() as usize);
    let data = mask.as_raw();

    // skimage's `opening(mask, disk(1))`: grey erosion then grey dilation with
    // a 3x3 cross, reflecting at the borders.
    const CROSS: [(isize, isize); 5] = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)];
    let opened = grey_open(data, w, h, &CROSS);

    // scipy's `gaussian_filter(mask.astype(float64), sigma=2)`.
    let blurred: Vec<f64> = gaussian_filter(&opened.iter().map(|&v| v as f64).collect::<Vec<_>>(), w, h, 2.0, 4.0);

    let out: Vec<u8> = blurred.iter().map(|&v| if v < 127.0 { 0 } else { 255 }).collect();
    GrayImage::from_raw(w as u32, h as u32, out).expect("mask size")
}

fn grey_open(src: &[u8], w: usize, h: usize, offsets: &[(isize, isize)]) -> Vec<u8> {
    let eroded: Vec<u8> = extrema(src, w, h, offsets, true);
    extrema(&eroded, w, h, offsets, false)
}

fn extrema(src: &[u8], w: usize, h: usize, offsets: &[(isize, isize)], min: bool) -> Vec<u8> {
    let reflect = |i: isize, n: usize| -> usize {
        if n == 1 {
            return 0;
        }
        let period = 2 * (n as isize - 1);
        let j = if i < 0 { -1 - i } else { i };
        let mut m = j % period;
        if m >= n as isize {
            m = period - m;
        }
        m as usize
    };

    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut best = if min { u8::MAX } else { u8::MIN };
            for &(dy, dx) in offsets {
                let sy = reflect(y as isize + dy, h);
                let sx = reflect(x as isize + dx, w);
                let v = src[sy * w + sx];
                best = if min { best.min(v) } else { best.max(v) };
            }
            out[y * w + x] = best;
        }
    }
    out
}

/// `scipy.ndimage.gaussian_filter` with its defaults: truncate 4, reflect.
fn gaussian_filter(src: &[f64], w: usize, h: usize, sigma: f64, truncate: f64) -> Vec<f64> {
    let radius = (truncate * sigma + 0.5) as isize;
    let kernel: Vec<f64> = (-radius..=radius)
        .map(|i| (-(i * i) as f64 / (2.0 * sigma * sigma)).exp())
        .collect();
    let sum: f64 = kernel.iter().sum();
    let kernel: Vec<f64> = kernel.iter().map(|v| v / sum).collect();

    let reflect = |i: isize, n: usize| -> usize {
        if n == 1 {
            return 0;
        }
        let period = 2 * (n as isize - 1);
        let j = if i < 0 { -1 - i } else { i };
        let mut m = j % period;
        if m >= n as isize {
            m = period - m;
        }
        m as usize
    };

    // Horizontal pass, then vertical pass.
    let mut tmp = vec![0f64; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (k, &kv) in kernel.iter().enumerate() {
                let sx = reflect(x as isize + k as isize - radius, w);
                acc += kv * src[y * w + sx];
            }
            tmp[y * w + x] = acc;
        }
    }

    let mut out = vec![0f64; w * h];
    for y in 0..h {
        let sy = |i: isize| reflect(i, h);
        for x in 0..w {
            let mut acc = 0.0;
            for (k, &kv) in kernel.iter().enumerate() {
                acc += kv * tmp[sy(y as isize + k as isize - radius) * w + x];
            }
            out[y * w + x] = acc;
        }
    }
    out
}

/// EXIF orientation tag (1..=8), 1 when there is none.
fn orientation(data: &[u8]) -> u8 {
    let reader = exif::Reader::new();
    reader
        .read_from_container(&mut Cursor::new(data))
        .ok()
        .and_then(|exif| {
            exif.get_field(exif::Tag::Orientation, exif::In::PRIMARY)
                .and_then(|f| f.value.get_uint(0))
        })
        .unwrap_or(1) as u8
}

/// `PIL.ImageOps.exif_transpose`.
fn fix_image_orientation(img: DynamicImage, orientation: u8) -> DynamicImage {
    match orientation {
        2 => img.fliph(),
        3 => img.rotate180(),
        4 => img.flipv(),
        5 => transpose(&img, false),
        6 => img.rotate90(),
        7 => transpose(&img, true),
        8 => img.rotate270(),
        _ => img,
    }
}

/// Reflect over the main diagonal (`5`) or the anti-diagonal (`7`).
fn transpose(img: &DynamicImage, anti: bool) -> DynamicImage {
    let (w, h) = (img.width(), img.height());
    let src = img.to_rgba8();
    let out = RgbaImage::from_fn(h, w, |x, y| {
        let (sx, sy) = if anti { (w - 1 - y, h - 1 - x) } else { (y, x) };
        *src.get_pixel(sx, sy)
    });
    DynamicImage::ImageRgba8(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(name: &str) -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/reference")
            .join(name)
    }

    /// `rembg.post_process` = skimage opening + scipy gaussian + threshold.
    #[test]
    fn post_process_matches_python() {
        let input = image::open(reference("pp_input.png"))
            .expect("reference missing; run `python3 tests/gen_reference.py`")
            .to_luma8();
        let expected = image::open(reference("pp_output.png")).unwrap().to_luma8();
        let got = post_process(&input);

        let mut bad = 0usize;
        for (a, b) in got.pixels().zip(expected.pixels()) {
            if a != b {
                bad += 1;
            }
        }
        println!("post_process: {bad} differing pixels of {}", got.pixels().len());
        assert!(bad <= got.pixels().len() / 200, "{bad} pixels differ");
    }
}
