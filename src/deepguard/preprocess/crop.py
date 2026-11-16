"""Square face crops and JPEG encoding (D-019.5). One function for training data and the app."""
from __future__ import annotations

import cv2
import numpy as np

CROP_SCALE = 1.3     # crop side = 1.3 x the larger box side
CROP_SIZE = 384      # stored size; 224 is derived by resizing this crop
JPEG_QUALITY = 95


def square_box(box: np.ndarray, scale: float = CROP_SCALE) -> tuple[int, int, int]:
    """Integer square (x0, y0, side) centred on the box; may extend past the frame."""
    cx = (float(box[0]) + float(box[2])) / 2.0
    cy = (float(box[1]) + float(box[3])) / 2.0
    side = max(2, int(round(scale * max(float(box[2]) - float(box[0]),
                                        float(box[3]) - float(box[1])))))
    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    return x0, y0, side


def crop_square(rgb: np.ndarray, box: np.ndarray, scale: float = CROP_SCALE,
                out: int = CROP_SIZE) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Cut the square around a face; parts outside the frame are filled with black.

    Downscaling uses INTER_AREA and upscaling INTER_LINEAR. Returns the crop and
    the (x0, y0, side) square in frame pixels, which is stored in the manifest.
    """
    h, w = rgb.shape[:2]
    x0, y0, side = square_box(box, scale)
    canvas = np.zeros((side, side, 3), np.uint8)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x0 + side), min(h, y0 + side)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - y0: sy1 - y0, sx0 - x0: sx1 - x0] = rgb[sy0:sy1, sx0:sx1]
    interp = cv2.INTER_AREA if side > out else cv2.INTER_LINEAR
    crop = cv2.resize(canvas, (out, out), interpolation=interp)
    return crop, (x0, y0, side)


def encode_jpeg(rgb: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    """RGB -> JPEG bytes with explicit quality and 4:2:0 chroma sampling."""
    params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    if hasattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR"):
        params += [cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420]
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), params)
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def decode_jpeg(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("not a JPEG")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
