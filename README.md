# rembg-rs

**v0.0.1** — MIT — © 2026 Mark John Tiempo

A rewrite of [python-rembg](https://github.com/danielgatis/rembg) in Rust:
give it an image, get it back with the background removed. Decode → EXIF
orientation fix → mask prediction → cutout, plus closed-form alpha matting for
clean soft edges, model download with checksum verification, and a CLI that
takes python-rembg's options.

```text
photo.jpg ──▶ decode ──▶ ONNX mask ──▶ cutout ──▶ out.png (transparent)
                              │
                              └──▶ trimap ──▶ closed-form alpha matting  (-a)
```

Everything is verified by pixel-diffing against the installed Python package —
see [Testing](#testing).

## Quickstart

### CLI

```sh
git clone git@github.com:mjtiempo/rembg-rs.git
cd rembg-rs
cargo build --release

./target/release/rembg-rs i photo.jpg out.png       # first run downloads the model
```

or, to get `rembg-rs` on your `PATH`:

```sh
cargo install --path .
rembg-rs i photo.jpg out.png
```

More examples:

```sh
rembg-rs i in.jpg out.png                      # one file
rembg-rs i - < in.jpg > out.png                # stdin → stdout (pipe friendly)
rembg-rs i in.jpg                              # writes in.out.png when stdout is a terminal
rembg-rs i -a --decontaminate in.jpg out.png   # alpha matting + edge decontamination
rembg-rs i --only-mask in.jpg mask.png         # just the greyscale mask
rembg-rs i --post-process-mask in.jpg out.png  # denoise the mask
rembg-rs p ./photos ./cutouts                  # a whole folder, one PNG per input
rembg-rs d                                     # download every model
rembg-rs d u2netp                              # download one model and verify its checksum
```

### Library

```toml
[dependencies]
rembg-rs = "0.0.1"
```

```rust
use rembg_rs::{remove_bytes, RemoveOptions, Session};

let mut session = Session::new("u2net")?;          // downloads on first use
let opts = RemoveOptions {
    alpha_matting: true,
    ..RemoveOptions::default()
};
let png: Vec<u8> = remove_bytes(&input_bytes, &mut session, &opts)?;
```

`remove()` takes an already-decoded `DynamicImage` when you have one.
`RemoveOptions` mirrors `rembg.remove(...)`'s keyword arguments.

## CLI reference

```text
rembg-rs i [OPTIONS] [INPUT] [OUTPUT]     one file (`-` or omitted = stdio)
rembg-rs p [OPTIONS] <INPUT> <OUTPUT>     a folder → one PNG per input, skipping
                                          files already converted
rembg-rs d [MODELS...]                    download and checksum models
```

Options shared by `i` and `p`, matching python-rembg's long names exactly:

| option | default | python-rembg |
|---|---|---|
| `-m, --model <NAME>` | `bria-rmbg` | `-m` |
| `-a, --alpha-matting` | off | `-a` |
| `--alpha-matting-foreground-threshold <0-255>` | `240` | `-af` |
| `--alpha-matting-background-threshold <0-255>` | `10` | `-ab` |
| `--alpha-matting-erode-size <N>` | `10` | `-ae` |
| `--only-mask` | off | `-om` |
| `--post-process-mask` | off | `-ppm` |
| `--decontaminate` | off | `-dc` |
| `--bgcolor <R G B A>` | `0 0 0 0` | `-bgc` |

`p` also takes `-d, --delete-input` (remove each input after conversion).
Files already present in the output folder are skipped, so an interrupted run
resumes where it stopped.

Python's multi-character short flags (`-af`, `-ab`, `-ae`, `-om`, `-ppm`,
`-dc`, `-bgc`) have no equivalent here: clap only accepts single-character
shorts, so those are long-only. `-m`, `-a`, `-d`, `-w` and `-V` keep their
short forms. `-w/--watch`, `--accept-usage-cost` and python-rembg's `-x/--extras`
are accepted for command-line compatibility but ignored (watch mode and extras
are not implemented).

### Behaviour notes

* **stdio** — omitting a path, or passing `-`, selects stdin/stdout. When
  output would go to a terminal it is written to `<input>.out.png` instead of
  dumping PNG bytes at you.
* **Output** is always PNG, RGBA. `--bgcolor` replaces the removed background
  with that colour instead of transparency.
* **Environment** — `U2NET_HOME`/`REMBG_HOME` override the model directory
  (default `~/.local/share/rembg/models/<name>/<name>.onnx`, honouring
  `XDG_DATA_HOME`, with the pre-0.0.0 flat `~/.u2net` layout read as a
  fallback); `MODEL_CHECKSUM_DISABLED=1` skips md5/sha256 verification;
  `OMP_NUM_THREADS` sets intra-op threads for ONNX Runtime.
* **Batch/other inputs** — `p` recursively walks the input folder and skips
  files whose format `image` can't guess.

## Models

Every local, downloadable python-rembg session is a row in `MODELS`
(`src/session.rs`) — one table, one `Session`; the per-model Python classes
collapse into input size, mean/std, a sigmoid flag and a checksum.

| model | input | notes |
|---|---|---|
| `bria-rmbg` | 1024² | **default**; RMBG-2.0, largest and slowest — see [license](#model-licenses) |
| `u2net` | 320² | good general default, small |
| `u2netp` | 320² | u2net's lightweight sibling (~4 MB) |
| `u2net_human_seg` | 320² | people |
| `silueta` | 320² | fine-tuned u2net |
| `isnet-general-use` | 1024² | high-detail edges |
| `isnet-anime` | 1024² | anime/illustration |
| `birefnet-general` | 1024² | BiRefNet, sigmoid output |
| `birefnet-general-lite` | 1024² | smaller BiRefNet backbone |
| `birefnet-portrait` | 1024² | portraits |
| `birefnet-dis` | 1024² | camouflaged objects |
| `birefnet-hrsod` | 1024² | high-resolution salient objects |
| `birefnet-cod` | 1024² | camouflaged object detection |
| `birefnet-massive` | 1024² | trained on DIS5K |

Each download is verified against the md5/sha256 from the Python table.

## Not implemented

Out of the agreed "core + alpha matting" scope:

* `sam` and `vitmatte` refinements (including `-vm`), and their checkpoints
* the `withoutbg` remote session and the `s` service command
* `u2net_cloth_seg` (a session returning three masks), custom `model_path`
  sessions, and `-x/--extras`
* `putalpha`, the `b` batch command and the `m` legacy-directory migration
* watch mode on `p`

## Testing

Two layers, both diffing against the installed Python package:

```sh
python3 tests/make_assets.py   # synthetic test inputs (deterministic, no
python3 tests/gen_reference.py # personal data), then the Python references
cargo test
```

The three inputs in `tests/assets/` are generated by `make_assets.py` — a
soft-edged synthetic "portrait" JPEG, a QR-like pattern and a flat UI
screenshot — so the repository contains no photographs or screenshots of
anyone.

**Stage tests** (`src/matting.rs`, `src/bg.rs`) feed identical pixels to one
stage at a time and require the same answer as Python:

| stage | reference | tolerance |
|---|---|---|
| `post_process` (opening + gaussian + threshold) | `rembg.post_process` | identical pixels |
| trimap (`binary_erosion`) | `rembg.alpha_matting_cutout` | identical pixels |
| cf-Laplacian `L_U` + rhs | `pymatting.cf_laplacian` | identical bits |
| `estimate_alpha_cf` | `pymatting` | identical (f32 dump) |
| `estimate_foreground_ml` | `pymatting` | max 4e-5 |
| cutout quantization | `stack_images` | max 1 |

**End-to-end tests** (`tests/compare.rs`) run all of `tests/cases.json` through
both implementations and compare the PNGs. Output is not expected to be
bit-identical: PIL and `image` resize with different arithmetic (mean 0.17 per
channel on 8-bit input), JPEG decoders round differently, and the two
onnxruntime builds disagree by ~0.2 mean on the predicted mask. Alpha matting
amplifies that, because a one-level mask change moves the trimap boundary.
Hence mean ≤ 2.5/255 and ≤ 1% of channels off by more than 32. The pure floors
are pinned separately: `resize_matches_pil` and `predicted_mask_matches_python`.

The references are only as fresh as the last `gen_reference.py` run; regenerate
them whenever Python's rembg, pymatting or onnxruntime changes.

## License

[MIT](LICENSE) — see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the
required notices covering python-rembg and PyMatting (both MIT, ported here)
and Microsoft ONNX Runtime (MIT, statically linked into the binary), plus the
dependency license summary.

### Model licenses

The weights are downloaded at runtime and are **not** covered by this
repository's MIT license. Check before shipping anything commercially:

| model | license | notes |
|---|---|---|
| `bria-rmbg` | **CC BY-NC 4.0 — non-commercial** | commercial use needs an agreement with BRIA AI. It is the CLI's default model (matching python-rembg); pass `-m u2net` for a permissively licensed default. |
| `u2net`, `u2netp`, `u2net_human_seg`, `silueta` | Apache-2.0 | derived from [U²-Net](https://github.com/xuebinqin/U-2-Net); note that upstream rembg does not document where its `.onnx` files were converted from, so verify provenance for your use case. |
| `birefnet-*` (8 variants) | MIT | [BiRefNet](https://github.com/ZhengPeng7/BiRefNet), © 2024 ZhengPeng |
| `isnet-general-use`, `isnet-anime` | unclear | converted from IS-Net, whose repository carries no license file; distributed via python-rembg's releases. Verify before commercial use. |

## Development notes

* System `cargo`/`rustc` on this machine is broken (Arch rust vs `llvm-libs`
  symbol mismatch); use `~/.cargo/bin/cargo` from rustup.
* The closed-form solver only materialises `L_U` rows for unknown pixels, so
  memory scales with the soft edge instead of the whole image, as pymatting
  keeps the full `n × 25` grid.
* Dependency versions are pinned in `Cargo.lock`; `ort` downloads and links its
  own ONNX Runtime at build time (which is why the binary is ~35 MB).
