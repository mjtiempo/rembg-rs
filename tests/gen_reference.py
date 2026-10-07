"""Generate reference outputs from the installed Python rembg.

Run from the crate root:

    python3 tests/gen_reference.py

Writes tests/reference/<case>.png for every case in tests/cases.json, plus two
intermediate references the Rust tests compare against directly:

  * cf_alpha.bin  -- pymatting's closed-form alpha for a synthetic trimap
  * post_mask.png -- rembg's post_process() of a synthetic mask

Regenerating overwrites the references, so it is how the suite is re-baselined
after an intentional behaviour change.
"""

import json
import pathlib
import struct

import numpy as np
from PIL import Image
from pymatting.alpha.estimate_alpha_cf import estimate_alpha_cf
from rembg import new_session, remove

HERE = pathlib.Path(__file__).parent
ASSETS = HERE / "assets"
REF = HERE / "reference"

CASES = json.loads((HERE / "cases.json").read_text())


def run_cases() -> None:
    REF.mkdir(exist_ok=True)
    for case in CASES:
        opts = {k: v for k, v in case.items() if k not in ("name", "model", "input")}
        opts["bgcolor"] = tuple(opts["bgcolor"]) if "bgcolor" in opts else None
        data = (ASSETS / case["input"]).read_bytes()
        out = remove(data, session=new_session(case["model"]), **opts)
        (REF / f"{case['name']}.png").write_bytes(out)
        print("wrote", case["name"])


def cf_alpha() -> None:
    """pymatting's alpha for a small synthetic scene, as raw f32 LE."""
    rng = np.random.default_rng(7)
    w, h = 64, 48
    # A bright disc on a dark, noisy background: soft edges everywhere.
    yy, xx = np.mgrid[0:h, 0:w]
    disc = ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) < 18.0**2
    image = np.where(disc[..., None], 230.0, 30.0) + rng.uniform(-40, 40, (h, w, 3))
    image = np.clip(image, 0, 255)

    # Quantize to bytes first: the Rust test only sees the PNG.
    image = np.floor(image).astype(np.uint8)
    Image.fromarray(image).save(REF / "cf_image.png")
    image = image.astype(np.float64) / 255.0

    trimap = np.full((h, w), 0.5)
    inner = ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) < 13.0**2
    outer = ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) > 23.0**2
    trimap[inner] = 1.0
    trimap[outer] = 0.0
    Image.fromarray((trimap * 255).astype(np.uint8), mode="L").save(REF / "cf_trimap.png")

    alpha = estimate_alpha_cf(image, trimap)
    (REF / "cf_alpha.bin").write_bytes(
        b"".join(struct.pack("<f", float(v)) for v in alpha.ravel())
    )
    print("wrote cf_alpha")


def matting_stages() -> None:
    """Every intermediate of `-a` on one real photo, so a divergence in the
    Rust port points at a stage instead of at the whole pipeline."""
    from pymatting.foreground.estimate_foreground_ml import estimate_foreground_ml
    from rembg.bg import alpha_matting_cutout
    from scipy.ndimage import binary_erosion

    img = Image.open(ASSETS / "photo.jpg").convert("RGB")
    img.save(REF / "am_image.png")
    mask = new_session("u2net").predict(img)[0]
    mask.save(REF / "am_mask.png")

    fg_t, bg_t, erode = 240, 10, 10
    mask_array = np.asarray(mask)
    structure = np.ones((erode, erode), dtype=np.uint8)
    is_foreground = binary_erosion(mask_array > fg_t, structure=structure)
    is_background = binary_erosion(
        mask_array < bg_t, structure=structure, border_value=1
    )
    trimap = np.full(mask_array.shape, 128, np.uint8)
    trimap[is_foreground] = 255
    trimap[is_background] = 0
    Image.fromarray(trimap, mode="L").save(REF / "am_trimap.png")

    normalized = np.asarray(img, dtype=np.float64) / 255.0
    alpha = estimate_alpha_cf(normalized, trimap / 255.0)
    (REF / "am_alpha.bin").write_bytes(
        b"".join(struct.pack("<f", float(v)) for v in alpha.ravel())
    )

    foreground = estimate_foreground_ml(normalized, alpha)
    (REF / "am_foreground.bin").write_bytes(
        b"".join(struct.pack("<f", float(v)) for v in foreground.ravel())
    )

    alpha_matting_cutout(img, mask, fg_t, bg_t, erode).save(REF / "am_cutout.png")
    print("wrote matting stages")



def post_process_mask() -> None:
    """rembg.post_process() of a synthetic soft mask, as a PNG."""
    from rembg.bg import post_process

    rng = np.random.default_rng(11)
    w, h = 96, 72
    yy, xx = np.mgrid[0:h, 0:w]
    blob = ((xx - w / 2) ** 2 / 1.6 + (yy - h / 2) ** 2) < 22.0**2
    mask = np.where(blob, 230, 20)
    mask = np.clip(mask + rng.integers(-30, 30, mask.shape), 0, 255).astype(np.uint8)
    Image.fromarray(mask, mode="L").save(REF / "pp_input.png")
    Image.fromarray(post_process(mask), mode="L").save(REF / "pp_output.png")
    print("wrote post_process")


def laplacian_system() -> None:
    """pymatting's `L_U` (CSC) and `-R·m` right-hand side for the synthetic
    scene, so the Rust port can diff the linear system itself."""
    from pymatting.laplacian.cf_laplacian import cf_laplacian
    from pymatting.util.util import trimap_split

    image = np.asarray(Image.open(REF / "cf_image.png"), dtype=np.float64) / 255.0
    trimap = np.asarray(Image.open(REF / "cf_trimap.png"), dtype=np.float64) / 255.0
    is_fg, is_bg, is_known, is_unknown = trimap_split(trimap)

    lap = cf_laplacian(image, is_known=is_known)
    rows = np.where(is_unknown)[0]
    lu = lap[rows, :][:, rows].tocsc()
    r = lap[rows, :][:, is_known]
    rhs = -r.dot(is_fg[is_known].astype(np.float64))

    # scipy's CSC: row indices per column.
    (REF / "lu_indptr.bin").write_bytes(lu.indptr.astype("<i8").tobytes())
    (REF / "lu_indices.bin").write_bytes(lu.indices.astype("<i8").tobytes())
    (REF / "lu_values.bin").write_bytes(lu.data.astype("<f8").tobytes())
    (REF / "lu_rhs.bin").write_bytes(rhs.astype("<f8").tobytes())
    (REF / "lu_unknown.bin").write_bytes(rows.astype("<i8").tobytes())

    # Same matrix in CSR, to diff against the port row by row.
    csr = lap[rows, :][:, rows].tocsr()
    csr.sum_duplicates()
    (REF / "lucsr_indptr.bin").write_bytes(csr.indptr.astype("<i8").tobytes())
    (REF / "lucsr_indices.bin").write_bytes(csr.indices.astype("<i8").tobytes())
    (REF / "lucsr_values.bin").write_bytes(csr.data.astype("<f8").tobytes())
    print("wrote laplacian system", lu.shape, lu.nnz)


def resize_ref() -> None:
    """PIL's LANCZOS resize of the photo to u2net's 320x320 input."""
    img = Image.open(REF / "am_image.png").convert("RGB")
    img.resize((320, 320), Image.Resampling.LANCZOS).save(REF / "resize320.png")
    print("wrote resize320")


if __name__ == "__main__":
    run_cases()
    cf_alpha()
    post_process_mask()
    matting_stages()
    laplacian_system()
    resize_ref()
