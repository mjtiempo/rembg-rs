"""Deterministic synthetic test images — no personal data, nothing copied
from anyone's photos or screenshots.

Regenerate with:

    python3 tests/make_assets.py

The three inputs exercise different paths:
  photo.jpg  RGB JPEG, soft-edged subject  (JPEG decode + alpha matting)
  qr.png     RGBA with hard edges          (PNG + alpha, thresholded mask)
  screen.png RGBA flat UI shapes           (PNG + alpha, post-processing)
"""

import pathlib

import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).parent
ASSETS = HERE / "assets"


def photo() -> None:
    """A 'portrait': silhouette on a graded, noisy background, JPEG-compressed."""
    w, h = 800, 600
    yy, xx = np.mgrid[0:h, 0:w]
    rng = np.random.default_rng(20261008)

    sky = np.stack(
        [50 + 90 * xx / w, 70 + 70 * yy / h, 130 - 50 * xx / w], axis=-1
    )

    head = ((xx - 400) / 70.0) ** 2 + ((yy - 200) / 85.0) ** 2
    shoulders = ((xx - 400) / 175.0) ** 2 + ((yy - 520) / 220.0) ** 2
    subject = np.minimum(head, shoulders)
    # Soft edge: a ramp a few pixels wide, the case alpha matting exists for.
    edge = 1.0 / (1.0 + np.exp((subject - 1.0) * 12.0))

    tone = np.stack(
        [
            190 + 40 * np.sin(yy / 40.0),
            130 + 40 * np.cos(xx / 55.0),
            110 + 30 * np.sin((xx + yy) / 70.0),
        ],
        axis=-1,
    )
    img = sky * (1.0 - edge[..., None]) + tone * edge[..., None]
    img += rng.normal(0.0, 3.0, img.shape)

    out = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(out).save(ASSETS / "photo.jpg", quality=95)


def qr() -> None:
    """A QR-like pattern: hard black/white modules, fully opaque alpha."""
    n = 21 * 3  # modules, 3px each
    quiet = 4
    scale = 8
    size = (n + 2 * quiet) * scale
    rng = np.random.default_rng(4242)

    img = np.full((size, size, 4), 255, np.uint8)
    modules = rng.integers(0, 2, (n, n)).astype(bool)

    # Finder patterns in three corners, like a real QR code.
    for oy, ox in ((0, 0), (0, n - 7), (n - 7, 0)):
        modules[oy : oy + 7, ox : ox + 7] = False
        modules[oy + 1 : oy + 6, ox + 1 : ox + 6] = True
        modules[oy + 2 : oy + 5, ox + 2 : ox + 5] = False

    for y in range(n):
        for x in range(n):
            if modules[y, x]:
                y0 = (y + quiet) * scale
                x0 = (x + quiet) * scale
                img[y0 : y0 + scale, x0 : x0 + scale] = (0, 0, 0, 255)

    Image.fromarray(img, "RGBA").save(ASSETS / "qr.png")


def screen() -> None:
    """A screenshot-like window: flat UI shapes, translucent selection box."""
    w, h = 640, 480
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:h, 0:w]
    a = np.full((h, w), 0, np.uint8)

    canvas = np.stack([245 - 10 * yy / h, 245 - 10 * yy / h, 248 - 8 * yy / h], -1)
    canvas[0:40, :] = (32, 36, 48)  # title bar
    canvas[40:46, :] = (210, 214, 224)  # toolbar edge

    for i in range(14):  # text-ish lines
        y0 = 70 + i * 26
        length = int(rng.integers(120, 480))
        canvas[y0 : y0 + 8, 60 : 60 + length] = (70, 74, 86)

    canvas[60:160, 440:600] = (250, 170, 60)  # button
    canvas[180:300, 60:400] = (120, 180, 230)  # image pane
    a[190:290, 70:390] = 160  # translucent selection

    out = np.dstack([canvas.astype(np.uint8), a])
    Image.fromarray(out, "RGBA").save(ASSETS / "screen.png")


if __name__ == "__main__":
    ASSETS.mkdir(exist_ok=True)
    photo()
    qr()
    screen()
    print("wrote photo.jpg, qr.png, screen.png")
