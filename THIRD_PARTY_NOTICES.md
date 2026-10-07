# Third-party notices

This product is a from-scratch Rust implementation of two MIT-licensed
projects, and statically links a third. MIT requires that their copyright
notice and permission notice accompany the software, so the full texts are
reproduced below.

## python-rembg (ported)

`src/bg.rs`, `src/session.rs` and `src/main.rs` port the behaviour and
structure of <https://github.com/danielgatis/rembg> v2.0.85 (installed as
`rembg` on this machine and used as the differential test oracle).

```
MIT License

Copyright (c) 2020 Daniel Gatis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## PyMatting (ported)

`src/matting.rs` ports the closed-form matting stack of PyMatting — the
cf-Laplacian, incomplete-Cholesky preconditioner, conjugate gradient and
matting-foreground estimation (`pymatting.laplacian.cf_laplacian`,
`pymatting.preconditioner.ichol`, `pymatting.solver.cg`,
`pymatting.foreground.estimate_foreground_ml`).

```
MIT License

Copyright (c) 2020 PyMatting

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Microsoft ONNX Runtime (statically linked)

`ort` downloads `libonnxruntime.a` at build time and links it into the
binary, so its notice must travel with any executable you distribute.

```
MIT License

Copyright (c) Microsoft Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Rust dependencies

All 250 crates in `Cargo.lock` are permissive; there is no copyleft anywhere
in the tree. Direct dependencies:

| crate | license |
|---|---|
| `ort` / `ort-sys` | MIT OR Apache-2.0 |
| `image` | MIT OR Apache-2.0 |
| `ndarray` | MIT OR Apache-2.0 |
| `clap` | MIT OR Apache-2.0 |
| `ureq` | MIT OR Apache-2.0 |
| `sha2` | MIT OR Apache-2.0 |
| `md-5` | MIT OR Apache-2.0 |
| `kamadak-exif` | BSD-2-Clause |
| `serde`, `serde_json` (dev) | MIT OR Apache-2.0 |

Transitive licenses seen in `Cargo.lock`, all of them permissive: MIT,
Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC, Zlib, 0BSD, Unlicense, CC0-1.0,
NCSA, Unicode-3.0, CDLA-Permissive-2.0. Each crate's exact terms ship in the
crate package (`LICENSE*` at the package root).

## Model weights

The ONNX models are **not** part of this repository — they are downloaded on
first use and are governed by their own licenses, independent of this code's
MIT license. See the table in [README.md](README.md#model-licenses).
