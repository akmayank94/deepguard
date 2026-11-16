"""Shared helpers for tests: small synthetic videos written with PyAV."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

MODELS_DIR = Path(os.environ.get("DG_MODELS", Path.home() / "models"))


def write_yuv_video(path: Path, yuv_frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
                    fps: int = 25, colorspace: int | None = None, color_range: int | None = None,
                    pix_fmt: str = "yuv420p") -> Path:
    """Write planar YUV frames losslessly (libx264 qp=0) with optional colour tags."""
    import av

    h, w = yuv_frames[0][0].shape
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=fps)
        s.width, s.height, s.pix_fmt = w, h, pix_fmt
        s.options = {"qp": "0", "preset": "ultrafast"}
        if colorspace is not None:
            s.codec_context.colorspace = colorspace
        if color_range is not None:
            s.codec_context.color_range = color_range
        for y, u, v in yuv_frames:
            if pix_fmt == "yuv444p":
                arr = np.stack([y, u, v])  # PyAV wants (3, h, w) for planar 4:4:4
            else:
                arr = np.concatenate([y, u.reshape(-1, w), v.reshape(-1, w)], axis=0)
            f = av.VideoFrame.from_ndarray(arr, format=pix_fmt)
            if colorspace is not None:
                f.colorspace = colorspace
            if color_range is not None:
                f.color_range = color_range
            for p in s.encode(f):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    return path


def write_rgb_video(path: Path, frames: list[np.ndarray], fps: int = 25, crf: int = 0) -> Path:
    """Write RGB frames as H.264 yuv420p (lossless luma when crf=0)."""
    import av

    h, w = frames[0].shape[:2]
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=fps)
        s.width, s.height, s.pix_fmt = w, h, "yuv420p"
        s.options = {"crf": str(crf), "preset": "ultrafast"}
        for fr in frames:
            for p in s.encode(av.VideoFrame.from_ndarray(fr, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    return path


def frame_index_video(path: Path, n: int, h: int = 64, w: int = 96) -> Path:
    """Video whose frame i has a flat luma value encoding i, to check index order."""
    frames = []
    for i in range(n):
        y = np.full((h, w), 16 + (i * 7) % 200, np.uint8)
        u = np.full((h // 2, w // 2), 128, np.uint8)
        v = np.full((h // 2, w // 2), 128, np.uint8)
        frames.append((y, u, v))
    return write_yuv_video(path, frames)


def model_path(*parts: str) -> Path:
    p = MODELS_DIR.joinpath(*parts)
    if not p.exists():
        pytest.skip(f"model file not available: {p}")
    return p
