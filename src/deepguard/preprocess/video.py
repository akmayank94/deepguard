"""Video decoding with an explicit, library-independent colour conversion (D-018.2, D-019.1).

Why this exists: OpenCV, PyAV and ffmpeg can turn the same YUV video into slightly
different RGB (BT.601 vs BT.709 matrix, limited vs full range, chroma upsampling).
If training crops and app crops are made by different code, the model sees a
distribution shift that no test catches. So every frame that reaches DeepGuard is
converted here, by one function, with the rule written down:

* matrix: the stream's colourspace tag if it has one; otherwise BT.709 for frames
  with height >= 720 and BT.601 below (the convention video players use);
* range: the stream's range tag; otherwise limited ("TV") range, except for the
  yuvj* pixel formats, which are full range by definition;
* chroma: upsampled with bilinear interpolation (float32) to the luma size.

Frame indices are counted in presentation order from 0, the same way the
Phase 1.1 audit counted frames.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from fractions import Fraction
from typing import Iterable, Iterator

import cv2
import numpy as np

# Kr, Kb luma coefficients per matrix name.
MATRICES = {
    "bt601": (0.299, 0.114),
    "bt709": (0.2126, 0.0722),
    "bt2020": (0.2627, 0.0593),
}

# FFmpeg AVColorSpace values (libavutil/pixfmt.h) -> our matrix names.
_AVCOL_SPC = {1: "bt709", 5: "bt601", 6: "bt601", 9: "bt2020", 10: "bt2020"}
# FFmpeg AVColorRange values: 1 = MPEG (limited), 2 = JPEG (full).
_AVCOL_RANGE = {1: "limited", 2: "full"}

_SUBSAMPLED_8BIT = {"yuv420p": (2, 2), "yuvj420p": (2, 2), "yuv422p": (2, 1),
                    "yuvj422p": (2, 1), "yuv444p": (1, 1), "yuvj444p": (1, 1)}


@dataclass(frozen=True)
class VideoInfo:
    """Container-level facts about a video, read without decoding every frame."""
    path: str
    width: int
    height: int
    fps: float
    n_frames_header: int  # 0 when the container does not say
    codec: str
    pix_fmt: str
    colorspace_tag: int
    color_range_tag: int
    duration_s: float

    def as_dict(self) -> dict:
        return asdict(self)


def probe(path: str) -> VideoInfo:
    """Read header metadata of the first video stream."""
    import av

    with av.open(path) as c:
        s = c.streams.video[0]
        cc = s.codec_context
        rate = s.average_rate or s.guessed_rate or Fraction(0)
        dur = float(s.duration * s.time_base) if s.duration else (
            float(c.duration) / 1e6 if c.duration else 0.0)
        return VideoInfo(
            path=str(path), width=cc.width, height=cc.height, fps=float(rate),
            n_frames_header=int(s.frames or 0), codec=cc.name, pix_fmt=str(cc.pix_fmt),
            colorspace_tag=int(getattr(cc, "colorspace", 2) or 2),
            color_range_tag=int(getattr(cc, "color_range", 0) or 0), duration_s=dur,
        )


def choose_conversion(height: int, pix_fmt: str, colorspace_tag: int,
                      color_range_tag: int) -> tuple[str, str]:
    """Return (matrix, range) by the documented rule. Pure function, unit-tested."""
    matrix = _AVCOL_SPC.get(int(colorspace_tag))
    if matrix is None:
        matrix = "bt709" if height >= 720 else "bt601"
    rng = _AVCOL_RANGE.get(int(color_range_tag))
    if rng is None:
        rng = "full" if pix_fmt.startswith("yuvj") else "limited"
    return matrix, rng


def yuv_to_rgb(y: np.ndarray, u: np.ndarray, v: np.ndarray, matrix: str,
               rng: str, bit_depth: int = 8) -> np.ndarray:
    """Convert planar YUV (chroma may be subsampled) to RGB uint8, HxWx3.

    The arithmetic is done in float32 and rounded once at the end, so the result
    equals the analytic conversion to within 0.5 before rounding.
    """
    if matrix not in MATRICES:
        raise ValueError(f"unknown matrix {matrix!r}")
    kr, kb = MATRICES[matrix]
    kg = 1.0 - kr - kb
    h, w = y.shape
    yf = y.astype(np.float32)
    uf = u.astype(np.float32)
    vf = v.astype(np.float32)
    if uf.shape != (h, w):
        uf = cv2.resize(uf, (w, h), interpolation=cv2.INTER_LINEAR)
        vf = cv2.resize(vf, (w, h), interpolation=cv2.INTER_LINEAR)
    scale = float(1 << (bit_depth - 8))
    if rng == "limited":
        yn = (yf - 16.0 * scale) / (219.0 * scale)
        cb = (uf - 128.0 * scale) / (224.0 * scale)
        cr = (vf - 128.0 * scale) / (224.0 * scale)
    elif rng == "full":
        maxv = float((1 << bit_depth) - 1)
        yn = yf / maxv
        cb = (uf - 128.0 * scale) / maxv
        cr = (vf - 128.0 * scale) / maxv
    else:
        raise ValueError(f"unknown range {rng!r}")
    r = yn + 2.0 * (1.0 - kr) * cr
    b = yn + 2.0 * (1.0 - kb) * cb
    g = (yn - kr * r - kb * b) / kg
    rgb = np.stack([r, g, b], axis=-1) * 255.0
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def _frame_planes(frame) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, str]:
    """Return Y, U, V planes as numpy arrays plus bit depth and pixel format name."""
    fmt = frame.format.name
    if fmt not in _SUBSAMPLED_8BIT:
        # 10-bit, NV12 and other layouts: let swscale only *repack* to 8-bit 4:4:4.
        # swscale does not apply a YUV->RGB matrix here, so our rule still decides it.
        frame = frame.reformat(format="yuv444p")
    planes = []
    for p in frame.planes:
        buf = np.frombuffer(p, dtype=np.uint8)
        arr = buf.reshape(p.height, p.line_size)[:, : p.width]
        planes.append(arr)
    # The original format name is returned: it decides the default range (yuvj*).
    return planes[0], planes[1], planes[2], 8, fmt


def frame_to_rgb(frame, override: tuple[str, str] | None = None) -> np.ndarray:
    """Convert one PyAV VideoFrame to RGB uint8 with the explicit rule."""
    y, u, v, depth, fmt = _frame_planes(frame)
    if override is None:
        matrix, rng = choose_conversion(frame.height, fmt, int(frame.colorspace or 2),
                                        int(frame.color_range or 0))
    else:
        matrix, rng = override
    return yuv_to_rgb(y, u, v, matrix, rng, depth)


def frame_to_gray_small(frame, size: tuple[int, int]) -> np.ndarray:
    """Luma plane only, resized with INTER_AREA. Used for fast DFD offset matching."""
    y = _frame_planes(frame)[0]
    return cv2.resize(y, size, interpolation=cv2.INTER_AREA)


def sample_indices(length: int, k: int = 32) -> list[int] | None:
    """k frame indices spread evenly over [0, length), by index, never by time.

    Returns None when length < k: such videos are dropped (D-017.6).
    The i-th index is floor((i + 0.5) * length / k), so the samples sit in the
    middle of k equal segments and never touch the first or last frame.
    """
    if length < k:
        return None
    return [int((i + 0.5) * length / k) for i in range(k)]


def _open_decoder(path: str):
    import av

    c = av.open(path)
    s = c.streams.video[0]
    s.thread_type = "AUTO"
    return c, s


def decode_frames(path: str, indices: Iterable[int],
                  override: tuple[str, str] | None = None) -> dict[int, np.ndarray]:
    """Decode the frames at the given presentation-order indices to RGB.

    Decodes sequentially and stops after the largest requested index. Missing
    indices (video shorter than expected) are simply absent from the result, so
    callers must check the returned keys.
    """
    want = sorted(set(int(i) for i in indices))
    out: dict[int, np.ndarray] = {}
    if not want:
        return out
    wanted = set(want)
    last = want[-1]
    c, s = _open_decoder(path)
    try:
        for idx, frame in enumerate(c.decode(s)):
            if idx in wanted:
                out[idx] = frame_to_rgb(frame, override)
            if idx >= last:
                break
    finally:
        c.close()
    return out


def iter_gray_small(path: str, size: tuple[int, int] = (64, 36)) -> Iterator[np.ndarray]:
    """Yield every frame's downscaled luma plane (for alignment, not for models)."""
    c, s = _open_decoder(path)
    try:
        for frame in c.decode(s):
            yield frame_to_gray_small(frame, size)
    finally:
        c.close()


def count_frames(path: str) -> int:
    """Count decodable frames (presentation order), ignoring the header value."""
    c, s = _open_decoder(path)
    try:
        return sum(1 for _ in c.decode(s))
    finally:
        c.close()


def normalize_width(frame: np.ndarray, mode: str | None, offset: int = 0,
                    multiple: int = 32) -> np.ndarray:
    """Width fix of D-017.3 / D-019.2 for FF++ videos whose width % 32 != 0.

    Measured on the audit metadata: every Face2Face/NeuralTextures fake has width
    floor32(original width) (282/282 targets, c23 and c40); height is unchanged.
    mode None: unchanged. mode "crop": keep columns [offset, offset + W32).
    mode "rescale": resize width to W32 (height unchanged) with INTER_AREA.
    Frames whose width is already a multiple of 32 are returned unchanged.
    """
    h, w = frame.shape[:2]
    target = (w // multiple) * multiple
    if mode is None or target == w:
        return frame
    if mode == "crop":
        if not 0 <= offset <= w - target:
            raise ValueError(f"crop offset {offset} outside 0..{w - target}")
        return np.ascontiguousarray(frame[:, offset: offset + target])
    if mode == "rescale":
        return cv2.resize(frame, (target, h), interpolation=cv2.INTER_AREA)
    raise ValueError(f"unknown width mode {mode!r}")
