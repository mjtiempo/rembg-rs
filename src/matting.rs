//! Alpha matting: the closed-form solver of Levin et al. (2007) and the
//! multilevel foreground estimator of Germer et al. (2020), both ported from
//! pymatting — the engines behind `rembg -a` and `rembg -dc`.

use image::{DynamicImage, ImageBuffer, Luma, Rgba, RgbaImage};

use crate::session::Result;

// ---------------------------------------------------------------------------
// binary morphology (scipy.ndimage.binary_erosion)
// ---------------------------------------------------------------------------

/// Erode a boolean image. `offsets` are the structure element's coordinates
/// relative to the pixel, `border` the value assumed outside the image.
fn erode_with(src: &[bool], w: usize, h: usize, offsets: &[(isize, isize)], border: bool) -> Vec<bool> {
    let mut out = vec![false; src.len()];
    for y in 0..h {
        for x in 0..w {
            let mut all = true;
            'outer: for &(dy, dx) in offsets {
                let sy = y as isize + dy;
                let sx = x as isize + dx;
                if sy < 0 || sy >= h as isize || sx < 0 || sx >= w as isize {
                    if !border {
                        all = false;
                        break 'outer;
                    }
                } else if !src[sy as usize * w + sx as usize] {
                    all = false;
                    break 'outer;
                }
            }
            out[y * w + x] = all;
        }
    }
    out
}

/// A `k`×`k` square footprint, centered as scipy centers an even-sized one.
fn square_offsets(k: usize) -> Vec<(isize, isize)> {
    let c = k as isize / 2;
    (0..k as isize)
        .flat_map(|dy| (0..k as isize).map(move |dx| (dy - c, dx - c)))
        .collect()
}

/// The 1-connectivity footprint scipy falls back to when `structure=None`.
const CROSS: [(isize, isize); 5] = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)];

/// Builds the three-valued trimap `rembg -a` feeds to the solver: 255 is
/// definite foreground, 0 definite background, 128 the band to be resolved.
pub fn trimap_from_mask(
    mask: &ImageBuffer<Luma<u8>, Vec<u8>>,
    fg_threshold: u8,
    bg_threshold: u8,
    erode_size: usize,
) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    let (w, h) = (mask.width() as usize, mask.height() as usize);
    let is_fg: Vec<bool> = mask.pixels().map(|p| p[0] > fg_threshold).collect();
    let is_bg: Vec<bool> = mask.pixels().map(|p| p[0] < bg_threshold).collect();

    // erode_size == 0 still erodes, with scipy's default cross footprint.
    let (fg_off, bg_off) = if erode_size > 0 {
        (square_offsets(erode_size), square_offsets(erode_size))
    } else {
        (CROSS.to_vec(), CROSS.to_vec())
    };

    let is_fg = erode_with(&is_fg, w, h, &fg_off, false);
    let is_bg = erode_with(&is_bg, w, h, &bg_off, true);

    let mut data = vec![128u8; w * h];
    for i in 0..w * h {
        if is_fg[i] {
            data[i] = 255;
        }
        if is_bg[i] {
            data[i] = 0;
        }
    }
    ImageBuffer::from_raw(w as u32, h as u32, data).expect("mask size")
}

// ---------------------------------------------------------------------------
// closed-form alpha matting
// ---------------------------------------------------------------------------

const EPSILON: f64 = 1e-7;
const WINDOW: usize = 3; // (2r + 1) with r = 1
const WINDOW_AREA: f64 = 9.0;

/// Sparse `L_U` (rows and columns over unknown pixels) plus the right-hand
/// side `-R · m`, built in one pass over every 3×3 window.
///
/// Only rows of unknown pixels are materialized: `L[unknown][:, :]` is all the
/// solver ever reads, which keeps memory proportional to the soft edge instead
/// of to the whole image.
struct UnknownSystem {
    indptr: Vec<usize>,
    indices: Vec<usize>,
    values: Vec<f64>,
    rhs: Vec<f64>,
}

fn build_unknown_system(
    image: &[f64],
    w: usize,
    h: usize,
    is_known: &[bool],
    is_fg: &[bool],
    unknown: &[usize],
    slot: &[usize],
) -> UnknownSystem {
    let m = unknown.len();
    let mut acc = vec![0f64; m * 25];
    let mut rhs = vec![0f64; m];

    // Every window that is not entirely known contributes to its pixels.
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let fully_known = (0..WINDOW as isize).all(|dy| {
                (0..WINDOW as isize).all(|dx| is_known[(y as isize + dy - 1) as usize * w + (x as isize + dx - 1) as usize])
            });
            if fully_known {
                continue;
            }

            // Window colors, centered and normalized per channel.
            let mut c = [[0.0f64; 3]; WINDOW * WINDOW];
            for ch in 0..3 {
                let mut sum = 0.0;
                for dy in 0..WINDOW {
                    for dx in 0..WINDOW {
                        let p = ((y + dy - 1) * w + (x + dx - 1)) * 3 + ch;
                        sum += image[p];
                    }
                }
                let mean = sum / WINDOW_AREA;
                for dy in 0..WINDOW {
                    for dx in 0..WINDOW {
                        let p = ((y + dy - 1) * w + (x + dx - 1)) * 3 + ch;
                        c[dy * WINDOW + dx][ch] = image[p] - mean;
                    }
                }
            }

            // Covariance matrix, its inverse (adjugate / det). pymatting
            // regularizes only the diagonal: `a00 = a11 = a22 = epsilon`.
            let mut a = [[0.0; 3]; 3];
            a[0][0] = EPSILON;
            a[1][1] = EPSILON;
            a[2][2] = EPSILON;
            for v in &c {
                for i in 0..3 {
                    for j in 0..3 {
                        a[i][j] += v[i] * v[j];
                    }
                }
            }
            for row in &mut a {
                for v in row.iter_mut() {
                    *v /= WINDOW_AREA;
                }
            }

            let det = a[0][0] * a[1][2] * a[1][2]
                + a[0][1] * a[0][1] * a[2][2]
                + a[0][2] * a[0][2] * a[1][1]
                - a[0][0] * a[1][1] * a[2][2]
                - 2.0 * a[0][1] * a[0][2] * a[1][2];
            let inv = 1.0 / det;
            let m00 = (a[1][2] * a[1][2] - a[1][1] * a[2][2]) * inv;
            let m01 = (a[0][1] * a[2][2] - a[0][2] * a[1][2]) * inv;
            let m02 = (a[0][2] * a[1][1] - a[0][1] * a[1][2]) * inv;
            let m11 = (a[0][2] * a[0][2] - a[0][0] * a[2][2]) * inv;
            let m12 = (a[0][0] * a[1][2] - a[0][1] * a[0][2]) * inv;
            let m22 = (a[0][1] * a[0][1] - a[0][0] * a[1][1]) * inv;

            for dyi in 0..WINDOW {
                for dxi in 0..WINDOW {
                    let ci = c[dyi * WINDOW + dxi];
                    let proj = [
                        m00 * ci[0] + m01 * ci[1] + m02 * ci[2],
                        m01 * ci[0] + m11 * ci[1] + m12 * ci[2],
                        m02 * ci[0] + m12 * ci[1] + m22 * ci[2],
                    ];

                    let yi = y + dyi - 1;
                    let xi = x + dxi - 1;
                    let i = yi * w + xi;
                    // Only rows the solver reads are accumulated.
                    let Some(row) = slot.get(i).copied().filter(|&s| s != usize::MAX) else {
                        continue;
                    };

                    for dyj in 0..WINDOW {
                        for dxj in 0..WINDOW {
                            let yj = y + dyj - 1;
                            let xj = x + dxj - 1;
                            let j = yj * w + xj;
                            let cj = c[dyj * WINDOW + dxj];
                            let temp = proj[0] * cj[0] + proj[1] * cj[1] + proj[2] * cj[2];
                            let value = if i == j { 1.0 } else { 0.0 } - (1.0 + temp) / WINDOW_AREA;
                            let cell = ((yj as isize - yi as isize + 2) * 5
                                + (xj as isize - xi as isize + 2))
                                as usize;
                            acc[row * 25 + cell] += value;
                        }
                    }
                }
            }
        }
    }

    // Emit CSR rows, splitting unknown columns from the known right-hand side.
    let mut indptr = Vec::with_capacity(m + 1);
    let mut indices = Vec::new();
    let mut values = Vec::new();
    indptr.push(0);
    for (row, &pixel) in unknown.iter().enumerate() {
        let (y, x) = (pixel / w, pixel % w);
        for cell in 0..25 {
            let value = acc[row * 25 + cell];
            if value == 0.0 {
                continue;
            }
            let dy = cell / 5;
            let dx = cell % 5;
            let yj = y as isize + dy as isize - 2;
            let xj = x as isize + dx as isize - 2;
            if yj < 0 || yj >= h as isize || xj < 0 || xj >= w as isize {
                continue;
            }
            let j = yj as usize * w + xj as usize;
            if let Some(&col) = slot.get(j).filter(|&&s| s != usize::MAX) {
                indices.push(col);
                values.push(value);
            } else {
                rhs[row] -= value * if is_fg[j] { 1.0 } else { 0.0 };
            }
        }
        indptr.push(indices.len());
    }

    UnknownSystem { indptr, indices, values, rhs }
}

/// Conjugate gradients, following pymatting's `cg`.
fn cg<F>(mut a: F, b: &[f64], precond: &Cholesky, atol: f64, rtol: f64, maxiter: usize) -> Result<Vec<f64>>
where
    F: FnMut(&[f64]) -> Vec<f64>,
{
    let n = b.len();
    let norm_b = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    let mut x = vec![0.0; n];
    let mut r = b.to_vec();
    let mut norm_r = norm_b;
    if norm_r < atol || norm_r < rtol * norm_b {
        return Ok(x);
    }

    let mut z = precond.apply(&r);
    let mut p = z.clone();
    let mut rz = dot(&r, &z);

    for _ in 0..maxiter {
        let ap = a(&p);
        let p_ap = dot(&p, &ap);
        let alpha = rz / p_ap;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        norm_r = r.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm_r < atol || norm_r < rtol * norm_b {
            return Ok(x);
        }

        z = precond.apply(&r);
        let mut beta = 1.0 / rz;
        rz = dot(&r, &z);
        beta *= rz;
        for i in 0..n {
            p[i] = p[i] * beta + z[i];
        }
    }

    Err(format!("Conjugate gradient descent did not converge within {maxiter} iterations").into())
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Thresholded incomplete Cholesky factor of `L_U`, in CSC layout, used as the
/// CG preconditioner exactly like pymatting's `ichol`.
struct Cholesky {
    values: Vec<f64>,
    rows: Vec<usize>,
    ptr: Vec<usize>,
}

const ICHOL_SHIFTS: [f64; 12] = [0.0, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5];

impl Cholesky {
    /// Forward then backward substitution: `(L L^T) x = b`.
    fn apply(&self, b: &[f64]) -> Vec<f64> {
        let n = self.ptr.len() - 1;
        let mut x = b.to_vec();
        for j in 0..n {
            let k = self.ptr[j];
            let temp = x[j] / self.values[k];
            x[j] = temp;
            for idx in self.ptr[j] + 1..self.ptr[j + 1] {
                x[self.rows[idx]] -= self.values[idx] * temp;
            }
        }
        for i in (0..n).rev() {
            let mut s = x[i];
            for idx in self.ptr[i] + 1..self.ptr[i + 1] {
                s -= self.values[idx] * x[self.rows[idx]];
            }
            x[i] = s / self.values[self.ptr[i]];
        }
        x
    }
}

/// One ichol attempt.
enum IcholAttempt {
    Done(Cholesky),
    /// Matrix not positive definite with this shift: retry with a larger one.
    NotPositiveDefinite,
    /// Fill-in exceeded `max_nnz`.
    TooBig,
}

fn ichol_once(
    av: &[f64],
    ar: &[usize],
    ap: &[usize],
    n: usize,
    shift: f64,
    max_nnz: usize,
) -> IcholAttempt {
    const DISCARD: f64 = 1e-4;
    const REL_DISCARD: f64 = 0.0;
    const KEEP_DISCARDED: bool = true;

    let mut values: Vec<f64> = Vec::new();
    let mut rows: Vec<usize> = Vec::new();
    let mut ptr = vec![0usize; n + 1];

    let mut next = vec![0usize; n]; // s: next stored row index in column k
    let mut first_sub = vec![0usize; n]; // t: first subdiagonal index in column j
    let mut list = vec![usize::MAX; n]; // l: linked list of columns in row k
    let mut a = vec![0f64; n];
    let mut col_abs = vec![0f64; n]; // r[j]: sum(abs(A[j:, j]))
    let mut marked = vec![false; n]; // b[i]
    let mut scratch = vec![0usize; n]; // c
    let mut d = vec![shift; n];

    for j in 0..n {
        for idx in ap[j]..ap[j + 1] {
            let i = ar[idx];
            if i == j {
                d[j] += av[idx];
                first_sub[j] = idx + 1;
            }
            if i >= j {
                col_abs[j] += av[idx].abs();
            }
        }
    }

    let mut scratch_n = 0usize;
    for j in 0..n {
        for idx in first_sub[j]..ap[j + 1] {
            let i = ar[idx];
            let lij = av[idx];
            if lij != 0.0 && i > j {
                a[i] += lij;
                if !marked[i] {
                    marked[i] = true;
                    scratch[scratch_n] = i;
                    scratch_n += 1;
                }
            }
        }

        let mut k = list[j];
        while k != usize::MAX {
            let mut k0 = next[k];
            let k1 = ptr[k + 1];
            let k2 = list[k];
            let l_jk = values[k0];
            k0 += 1;
            if k0 < k1 {
                next[k] = k0;
                let i = rows[k0];
                list[k] = list[i];
                list[i] = k;
                for idx in k0..k1 {
                    let i = rows[idx];
                    a[i] -= values[idx] * l_jk;
                    if !marked[i] {
                        marked[i] = true;
                        scratch[scratch_n] = i;
                        scratch_n += 1;
                    }
                }
            }
            k = k2;
        }

        if d[j] <= 0.0 {
            return IcholAttempt::NotPositiveDefinite;
        }
        if values.len() + 1 + scratch_n > max_nnz {
            return IcholAttempt::TooBig;
        }

        d[j] = d[j].sqrt();
        values.push(d[j]);
        rows.push(j);
        next[j] = values.len();

        let mut sorted = scratch[..scratch_n].to_vec();
        sorted.sort_unstable();
        for &i in &sorted {
            let l_ij = a[i] / d[j];
            if KEEP_DISCARDED {
                d[i] -= l_ij * l_ij;
            }
            let rel = REL_DISCARD * col_abs[j];
            if l_ij.abs() > DISCARD && a[i].abs() > rel {
                if !KEEP_DISCARDED {
                    d[i] -= l_ij * l_ij;
                }
                values.push(l_ij);
                rows.push(i);
            }
            a[i] = 0.0;
            marked[i] = false;
        }
        scratch_n = 0;
        ptr[j + 1] = values.len();

        if ptr[j] + 1 < ptr[j + 1] {
            let i = rows[ptr[j] + 1];
            list[j] = list[i];
            list[i] = j;
        }
    }

    IcholAttempt::Done(Cholesky { values, rows, ptr })
}

fn ichol(av: &[f64], ar: &[usize], ap: &[usize], n: usize) -> Result<Cholesky> {
    let max_nnz = 50_000_000;
    for shift in ICHOL_SHIFTS {
        match ichol_once(av, ar, ap, n, shift, max_nnz) {
            IcholAttempt::Done(chol) => return Ok(chol),
            IcholAttempt::TooBig => {
                return Err(
                    "Thresholded incomplete Cholesky decomposition failed because more than max_nnz non-zero elements were created"
                        .into(),
                )
            }
            IcholAttempt::NotPositiveDefinite => continue,
        }
    }
    Err("Thresholded incomplete Cholesky decomposition failed due to insufficient positive-definiteness of matrix A and diagonal shifts did not help".into())
}

/// Transpose CSR into CSC; `L_U` is symmetric, so the transpose is what ichol
/// consumes as its column-compressed copy of the same matrix.
fn transpose(
    indptr: &[usize],
    indices: &[usize],
    values: &[f64],
    n: usize,
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut counts = vec![0usize; n + 1];
    for &j in indices {
        counts[j + 1] += 1;
    }
    for j in 0..n {
        counts[j + 1] += counts[j];
    }
    let mut fill = counts.clone();
    let mut out_values = vec![0.0; values.len()];
    let mut out_rows = vec![0usize; values.len()];
    for i in 0..n {
        for k in indptr[i]..indptr[i + 1] {
            let j = indices[k];
            out_rows[fill[j]] = i;
            out_values[fill[j]] = values[k];
            fill[j] += 1;
        }
    }
    (counts, out_rows, out_values)
}

/// Solves the closed-form matting problem for the unknown part of `trimap`
/// and returns alpha over the whole image (known pixels keep their trimap
/// value). Mirrors pymatting's `estimate_alpha_cf`.
pub fn estimate_alpha_cf(image: &[f64], w: usize, h: usize, trimap: &[f64]) -> Result<Vec<f64>> {
    let n = w * h;
    let is_fg: Vec<bool> = trimap.iter().map(|&t| t >= 0.9).collect();
    let is_bg: Vec<bool> = trimap.iter().map(|&t| t <= 0.1).collect();
    let is_known: Vec<bool> = is_fg.iter().zip(&is_bg).map(|(f, b)| *f || *b).collect();

    let unknown: Vec<usize> = (0..n).filter(|&i| !is_known[i]).collect();
    if unknown.is_empty() {
        // Nothing to solve; numpy would hand back an all-unknown-free trimap.
        return Ok(trimap.iter().map(|&t| t.clamp(0.0, 1.0)).collect());
    }
    let mut slot = vec![usize::MAX; n];
    for (s, &i) in unknown.iter().enumerate() {
        slot[i] = s;
    }

    let system = build_unknown_system(image, w, h, &is_known, &is_fg, &unknown, &slot);
    let m = unknown.len();
    let (cp, cr, cv) = transpose(&system.indptr, &system.indices, &system.values, m);
    let chol = ichol(&cv, &cr, &cp, m)?;

    let rhs = &system.rhs;
    let x = cg(
        |v| {
            let mut out = vec![0.0; m];
            for row in 0..m {
                let mut acc = 0.0;
                for k in system.indptr[row]..system.indptr[row + 1] {
                    acc += system.values[k] * v[system.indices[k]];
                }
                out[row] = acc;
            }
            out
        },
        rhs,
        &chol,
        0.0,
        1e-7,
        10000,
    )?;

    let mut alpha = vec![0.0f64; n];
    for (s, &i) in unknown.iter().enumerate() {
        alpha[i] = x[s].clamp(0.0, 1.0);
    }
    for i in 0..n {
        if is_fg[i] {
            alpha[i] = 1.0;
        } else if is_bg[i] {
            alpha[i] = 0.0;
        }
    }
    Ok(alpha)
}

// ---------------------------------------------------------------------------
// foreground estimation (pymatting's estimate_foreground_ml)
// ---------------------------------------------------------------------------

/// Recovers the unblended foreground color for an image whose alpha is known.
/// Ported from pymatting's multilevel estimator; f32 throughout, like the
/// numba kernel it comes from.
pub fn estimate_foreground_ml(image: &[f64], alpha: &[f64], w: usize, h: usize) -> Vec<[f32; 3]> {
    const REGULARIZATION: f32 = 1e-5;
    const N_SMALL: i32 = 10;
    const N_BIG: i32 = 2;
    const SMALL_SIZE: usize = 32;
    const GRADIENT_WEIGHT: f32 = 1.0;

    let (w0, h0) = (w, h);
    let image: Vec<[f32; 3]> = image
        .chunks_exact(3)
        .map(|c| [c[0] as f32, c[1] as f32, c[2] as f32])
        .collect();
    let alpha: Vec<f32> = alpha.iter().map(|&a| a as f32).collect();

    let mut f_mean = [0.0f32; 3];
    let mut b_mean = [0.0f32; 3];
    let (mut f_count, mut b_count) = (0u32, 0u32);
    for i in 0..w0 * h0 {
        if alpha[i] > 0.9 {
            for c in 0..3 {
                f_mean[c] += image[i][c];
            }
            f_count += 1;
        }
        if alpha[i] < 0.1 {
            for c in 0..3 {
                b_mean[c] += image[i][c];
            }
            b_count += 1;
        }
    }
    for c in 0..3 {
        f_mean[c] /= f_count as f32 + 1e-5;
        b_mean[c] /= b_count as f32 + 1e-5;
    }

    let mut f_prev = vec![f_mean; 1];
    let mut b_prev = vec![b_mean; 1];
    let (mut pw, mut ph) = (1usize, 1usize);
    let n_levels = (w0.max(h0) as f64).log2().ceil() as i32;

    for i_level in 0..=n_levels {
        let lw = (w0 as f64).powf(i_level as f64 / n_levels as f64).round() as usize;
        let lh = (h0 as f64).powf(i_level as f64 / n_levels as f64).round() as usize;
        let (lw, lh) = (lw.max(1), lh.max(1));

        let level_image = resize_nearest_rgb(&image, w0, h0, lw, lh);
        let level_alpha = resize_nearest_gray(&alpha, w0, h0, lw, lh);
        let mut f = resize_nearest_rgb(&f_prev, pw, ph, lw, lh);
        let mut b = resize_nearest_rgb(&b_prev, pw, ph, lw, lh);

        let n_iter = if lw <= SMALL_SIZE && lh <= SMALL_SIZE {
            N_SMALL
        } else {
            N_BIG
        };
        let dx = [-1isize, 1, 0, 0];
        let dy = [0isize, 0, -1, 1];

        for _ in 0..n_iter {
            for y in 0..lh {
                for x in 0..lw {
                    let i = y * lw + x;
                    let a0 = level_alpha[i];
                    let a1 = 1.0 - a0;
                    let mut a00 = a0 * a0;
                    let a01 = a0 * a1;
                    let mut a11 = a1 * a1;
                    let mut b0 = [0.0f32; 3];
                    let mut b1 = [0.0f32; 3];
                    for c in 0..3 {
                        b0[c] = a0 * level_image[i][c];
                        b1[c] = a1 * level_image[i][c];
                    }

                    for d in 0..4 {
                        let x2 = (x as isize + dx[d]).clamp(0, lw as isize - 1) as usize;
                        let y2 = (y as isize + dy[d]).clamp(0, lh as isize - 1) as usize;
                        let j = y2 * lw + x2;
                        let gradient = (a0 - level_alpha[j]).abs();
                        let da = REGULARIZATION + GRADIENT_WEIGHT * gradient;
                        a00 += da;
                        a11 += da;
                        for c in 0..3 {
                            b0[c] += da * f[j][c];
                            b1[c] += da * b[j][c];
                        }
                    }

                    let det = a00 * a11 - a01 * a01;
                    let inv = 1.0 / det;
                    let (q00, q01, q11) = (inv * a11, inv * -a01, inv * a00);
                    for c in 0..3 {
                        f[i][c] = (q00 * b0[c] + q01 * b1[c]).clamp(0.0, 1.0);
                        b[i][c] = (q01 * b0[c] + q11 * b1[c]).clamp(0.0, 1.0);
                    }
                }
            }
        }

        f_prev = f;
        b_prev = b;
        pw = lw;
        ph = lh;
    }

    f_prev
}

/// Nearest-neighbour resize of an RGB buffer, matching pymatting's kernel.
fn resize_nearest_rgb(src: &[[f32; 3]], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<[f32; 3]> {
    let mut out = vec![[0.0f32; 3]; dw * dh];
    for y in 0..dh {
        let sy = (y * sh / dh).min(sh - 1);
        for x in 0..dw {
            let sx = (x * sw / dw).min(sw - 1);
            out[y * dw + x] = src[sy * sw + sx];
        }
    }
    out
}

/// Nearest-neighbour resize of a single-channel buffer.
fn resize_nearest_gray(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; dw * dh];
    for y in 0..dh {
        let sy = (y * sh / dh).min(sh - 1);
        for x in 0..dw {
            let sx = (x * sw / dw).min(sw - 1);
            out[y * dw + x] = src[sy * sw + sx];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// cutouts
// ---------------------------------------------------------------------------

/// Combines a recovered foreground with alpha into an RGBA image, the way
/// `stack_images` + `np.clip(cutout * 255)` does in Python.
pub fn stack_images(foreground: &[[f32; 3]], alpha: &[f64], w: usize, h: usize) -> RgbaImage {
    ImageBuffer::from_fn(w as u32, h as u32, |x, y| {
        let i = y as usize * w + x as usize;
        let mut px = [0u8; 4];
        for c in 0..3 {
            px[c] = (foreground[i][c] as f64 * 255.0).clamp(0.0, 255.0) as u8;
        }
        px[3] = (alpha[i] * 255.0).clamp(0.0, 255.0) as u8;
        Rgba(px)
    })
}

/// `rembg -a`: solve for alpha, then unmix the foreground color.
pub fn alpha_matting_cutout(
    img: &DynamicImage,
    mask: &ImageBuffer<Luma<u8>, Vec<u8>>,
    fg_threshold: u8,
    bg_threshold: u8,
    erode_size: usize,
) -> Result<RgbaImage> {
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let image: Vec<f64> = rgb
        .pixels()
        .flat_map(|p| [p[0] as f64 / 255.0, p[1] as f64 / 255.0, p[2] as f64 / 255.0])
        .collect();

    let trimap = trimap_from_mask(mask, fg_threshold, bg_threshold, erode_size);
    let trimap: Vec<f64> = trimap.pixels().map(|p| p[0] as f64 / 255.0).collect();

    let alpha = estimate_alpha_cf(&image, w, h, &trimap)?;
    let foreground = estimate_foreground_ml(&image, &alpha, w, h);
    Ok(stack_images(&foreground, &alpha, w, h))
}

/// `rembg -dc`: keep the mask as alpha but recover the true foreground color,
/// so soft edges do not keep the background they were blended with.
pub fn decontaminate_cutout(
    img: &DynamicImage,
    mask: &ImageBuffer<Luma<u8>, Vec<u8>>,
) -> RgbaImage {
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let image: Vec<f64> = rgb
        .pixels()
        .flat_map(|p| [p[0] as f64 / 255.0, p[1] as f64 / 255.0, p[2] as f64 / 255.0])
        .collect();
    let alpha: Vec<f64> = mask.pixels().map(|p| p[0] as f64 / 255.0).collect();

    let foreground = estimate_foreground_ml(&image, &alpha, w, h);
    stack_images(&foreground, &alpha, w, h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// `python3 tests/gen_reference.py` writes these.
    fn reference(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/reference")
            .join(name)
    }

    /// pymatting's `estimate_alpha_cf` on the synthetic scene it was built for.
    #[test]
    fn closed_form_alpha_matches_pymatting() {
        let rgb_img = image::open(reference("cf_image.png"))
            .expect("reference missing; run `python3 tests/gen_reference.py`")
            .to_rgb8();
        let (w, h) = (rgb_img.width() as usize, rgb_img.height() as usize);
        let rgb: Vec<f64> = rgb_img
            .pixels()
            .flat_map(|p| [p[0] as f64 / 255.0, p[1] as f64 / 255.0, p[2] as f64 / 255.0])
            .collect();
        let trimap_img = image::open(reference("cf_trimap.png")).unwrap().to_luma8();
        let trimap: Vec<f64> = trimap_img.pixels().map(|p| p[0] as f64 / 255.0).collect();
        let raw = std::fs::read(reference("cf_alpha.bin")).unwrap();
        let expected: Vec<f64> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
            .collect();
        assert_eq!(expected.len(), w * h);

        let alpha = estimate_alpha_cf(&rgb, w, h, &trimap).unwrap();
        let mut max = 0.0f64;
        let mut sum = 0.0f64;
        for (a, b) in alpha.iter().zip(&expected) {
            let d = (a - b).abs();
            max = max.max(d);
            sum += d;
        }
        let mean = sum / (w * h) as f64;
        println!("alpha diff: mean {mean:.6} max {max:.6}");
        assert!(mean < 1e-3 && max < 5e-2, "mean {mean} max {max}");
    }

    fn i64s(name: &str) -> Vec<i64> {
        std::fs::read(reference(name))
            .unwrap()
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect()
    }

    /// The linear system itself: pymatting's `L_U` in CSC and its `-R·m`.
    #[test]
    fn laplacian_system_matches_pymatting() {
        let rgb_img = image::open(reference("cf_image.png")).unwrap().to_rgb8();
        let (w, h) = (rgb_img.width() as usize, rgb_img.height() as usize);
        let rgb: Vec<f64> = rgb_img
            .pixels()
            .flat_map(|p| [p[0] as f64 / 255.0, p[1] as f64 / 255.0, p[2] as f64 / 255.0])
            .collect();
        let trimap_img = image::open(reference("cf_trimap.png")).unwrap().to_luma8();
        let trimap: Vec<f64> = trimap_img.pixels().map(|p| p[0] as f64 / 255.0).collect();

        let is_fg: Vec<bool> = trimap.iter().map(|&t| t >= 0.9).collect();
        let is_bg: Vec<bool> = trimap.iter().map(|&t| t <= 0.1).collect();
        let is_known: Vec<bool> = is_fg.iter().zip(&is_bg).map(|(f, b)| *f || *b).collect();
        let unknown: Vec<usize> = (0..w * h).filter(|&i| !is_known[i]).collect();
        let mut slot = vec![usize::MAX; w * h];
        for (s, &i) in unknown.iter().enumerate() {
            slot[i] = s;
        }

        let sys = build_unknown_system(&rgb, w, h, &is_known, &is_fg, &unknown, &slot);

        let want_rows: Vec<usize> = i64s("lu_unknown.bin").into_iter().map(|v| v as usize).collect();
        assert_eq!(unknown, want_rows, "unknown pixel order differs");

        // Row by row, both sides sorted by column, so a mismatch in the
        // accumulation shows up before anything else.
        let want_ptr: Vec<usize> = i64s("lucsr_indptr.bin").into_iter().map(|v| v as usize).collect();
        let want_idx: Vec<usize> = i64s("lucsr_indices.bin").into_iter().map(|v| v as usize).collect();
        let want_val = f64s_raw("lucsr_values.bin");
        assert_eq!(sys.indptr, want_ptr, "CSR row offsets differ");

        let mut bad = 0usize;
        let mut shown = 0usize;
        let mut max = 0.0f64;
        for row in 0..unknown.len() {
            let mut ours: Vec<(usize, f64)> = (sys.indptr[row]..sys.indptr[row + 1])
                .map(|k| (sys.indices[k], sys.values[k]))
                .collect();
            ours.sort_by_key(|&(c, _)| c);
            let theirs: Vec<(usize, f64)> = (want_ptr[row]..want_ptr[row + 1])
                .map(|k| (want_idx[k], want_val[k]))
                .collect();
            if ours.iter().map(|&(c, _)| c).ne(theirs.iter().map(|&(c, _)| c)) {
                println!("row {row}: column lists differ ({} vs {})", ours.len(), theirs.len());
                bad += 1;
                continue;
            }
            for ((_, a), (_, b)) in ours.iter().zip(&theirs) {
                let d = (a - b).abs();
                if d > 1e-9 && shown < 10 {
                    println!("  row {row}: ours {a:.6e} theirs {b:.6e} d {d:e}");
                    shown += 1;
                }
                if d > 1e-9 {
                    bad += 1;
                }
                max = max.max(d);
            }
        }
        println!("CSR values: {bad} differing entries, max abs diff {max:e}");
        assert_eq!(bad, 0, "L_U differs from pymatting: {bad} entries");

        // pymatting stores this matrix in CSC, so compare against the
        // transpose of what we built.
        let (cp, cr, cv) = transpose(&sys.indptr, &sys.indices, &sys.values, unknown.len());
        let want_ptr: Vec<usize> = i64s("lu_indptr.bin").into_iter().map(|v| v as usize).collect();
        let want_idx: Vec<usize> = i64s("lu_indices.bin").into_iter().map(|v| v as usize).collect();
        assert_eq!(cp, want_ptr, "CSC offsets differ");
        assert_eq!(cr, want_idx, "CSC row indices differ");

        let want_val = f64s_raw("lu_values.bin");
        assert_eq!(cv.len(), want_val.len(), "CSC nnz differs");
        let mut max = 0.0f64;
        let mut bad = 0usize;
        let mut shown = 0usize;
        for k in 0..cv.len() {
            let d = (cv[k] - want_val[k]).abs();
            if d > 1e-9 && shown < 10 {
                let col = want_ptr.partition_point(|&p| p <= k) - 1;
                println!(
                    "  L[{},{col}] ours {:.6e} theirs {:.6e} d {d:e}",
                    want_idx[k], cv[k], want_val[k]
                );
                shown += 1;
            }
            if d > 1e-9 {
                bad += 1;
            }
            max = max.max(d);
        }
        println!("L_U values: {bad} differing of {}, max abs diff {max:e}", cv.len());
        assert!(max < 1e-9, "max {max:e}");

        let want_rhs = f64s_raw("lu_rhs.bin");
        let (mean, max) = stats(&sys.rhs, &want_rhs);
        println!("rhs: mean {mean:e} max {max:e}");
        assert!(max < 1e-9, "max {max:e}");
    }

    fn f32s(name: &str) -> Vec<f64> {
        std::fs::read(reference(name))
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
            .collect()
    }

    fn f64s_raw(name: &str) -> Vec<f64> {
        std::fs::read(reference(name))
            .unwrap()
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    fn stats(got: &[f64], want: &[f64]) -> (f64, f64) {
        assert_eq!(got.len(), want.len(), "length differs from the reference");
        let mut sum = 0.0f64;
        let mut max = 0.0f64;
        for (a, b) in got.iter().zip(want) {
            let d = (a - b).abs();
            sum += d;
            max = max.max(d);
        }
        (sum / got.len() as f64, max)
    }

    /// Stage-by-stage diff of `rembg -a` on a real photo, so a divergence
    /// names the stage instead of the whole pipeline.
    #[test]
    fn matting_stages_match_python() {
        let img = image::open(reference("am_image.png"))
            .expect("reference missing; run `python3 tests/gen_reference.py`")
            .to_rgb8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        let mask = image::open(reference("am_mask.png")).unwrap().to_luma8();
        let rgb: Vec<f64> = img
            .pixels()
            .flat_map(|p| [p[0] as f64 / 255.0, p[1] as f64 / 255.0, p[2] as f64 / 255.0])
            .collect();

        // 1. trimap: scipy's binary_erosion of the mask.
        let trimap = trimap_from_mask(&mask, 240, 10, 10);
        let expected = image::open(reference("am_trimap.png")).unwrap().to_luma8();
        let bad = trimap
            .pixels()
            .zip(expected.pixels())
            .filter(|(a, b)| a != b)
            .count();
        println!("trimap: {bad} differing pixels");
        assert_eq!(bad, 0, "trimap differs");

        // 2. alpha: pymatting's estimate_alpha_cf.
        let t: Vec<f64> = trimap.pixels().map(|p| p[0] as f64 / 255.0).collect();
        let alpha = estimate_alpha_cf(&rgb, w, h, &t).unwrap();
        let (mean, max) = stats(&alpha, &f32s("am_alpha.bin"));
        println!("alpha: mean {mean:.6} max {max:.6}");
        assert!(mean < 1e-3 && max < 5e-2, "alpha mean {mean} max {max}");

        // 3. foreground: pymatting's estimate_foreground_ml.
        let fg: Vec<f64> = estimate_foreground_ml(&rgb, &alpha, w, h)
            .iter()
            .flat_map(|p| p.iter().map(|v| *v as f64))
            .collect();
        let (mean, max) = stats(&fg, &f32s("am_foreground.bin"));
        println!("foreground: mean {mean:.6} max {max:.6}");
        assert!(mean < 1e-3 && max < 5e-2, "foreground mean {mean} max {max}");

        // 4. cutout: foreground * alpha, quantized like numpy.
        let cutout = alpha_matting_cutout(
            &DynamicImage::ImageRgb8(img),
            &mask,
            240,
            10,
            10,
        )
        .unwrap();
        let expected = image::open(reference("am_cutout.png")).unwrap().to_rgba8();
        let mut sum = 0u64;
        let mut max = 0i32;
        for (a, b) in cutout.pixels().zip(expected.pixels()) {
            for c in 0..4 {
                let d = (a.0[c] as i32 - b.0[c] as i32).abs();
                sum += d as u64;
                max = max.max(d);
            }
        }
        let mean = sum as f64 / (w * h * 4) as f64;
        println!("cutout: mean {mean:.6} max {max}");
        assert!(mean < 1.0 && max <= 8, "cutout mean {mean} max {max}");
    }
}
