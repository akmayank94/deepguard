"""Mock FF++/DFD in the real Kaggle layout, with real faces (skimage astronaut) so the
detectors work. Planted facts the stage-A notebook must recover:
  * FF++ is yuv444p; originals 854 and 640 wide; Face2Face/NeuralTextures fakes of the
    854-wide targets are CENTRE-cropped to 832 (offset 11); other methods keep 854.
  * target 002 has two faces; its fakes change the SMALL face.
  * DFD (yuv420p, 24 fps): fake 01_02__kitchen__ABCD1234 starts at original frame 17;
    fake 01_03__kitchen__EFGH5678 starts at 40; fake 02_01__walk__SHRT0001 has 20 frames.
"""
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage import data

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from conftest import write_rgb_video  # noqa: E402

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "mock_input")
if ROOT.exists():
    shutil.rmtree(ROOT)
FACE = data.astronaut()[30:330, 100:380]          # 300 x 280 head-and-shoulders crop
rng = np.random.default_rng(0)


def bg(h, w, seed):
    r = np.random.default_rng(seed)
    small = r.integers(40, 200, (h // 16 + 2, w // 16 + 2, 3)).astype(np.uint8)
    return cv2.resize(small, (w + 64, h), interpolation=cv2.INTER_CUBIC)


def paste(frame, face, x, y, scale=1.0):
    f = cv2.resize(face, (int(face.shape[1] * scale), int(face.shape[0] * scale)))
    h, w = f.shape[:2]
    frame[y:y + h, x:x + w] = f


def manip(face):
    """Change only the face (rows 40-170, cols 75-180 of the crop): blur + colour shift."""
    out = face.astype(np.int16).copy()
    reg = cv2.GaussianBlur(face[40:170, 75:180], (9, 9), 0).astype(np.int16)
    reg[..., 0] += 35
    out[40:170, 75:180] = reg
    return np.clip(out, 0, 255).astype(np.uint8)


def ffpp_frames(h, w, n, seed, small=False, fake_main=False, fake_small=False):
    B = bg(h, w, seed)
    frames = []
    for i in range(n):
        f = B[:, i % 64: i % 64 + w].copy()
        paste(f, manip(FACE) if fake_main else FACE, 120 + (i % 20), 60, 1.0)
        if small:
            paste(f, manip(FACE[:, ::-1]) if fake_small else FACE[:, ::-1], w - 200, 120, 0.5)
        frames.append(f)
    return frames


def write444(path, frames, fps=25):
    import av
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=fps)
        s.width, s.height, s.pix_fmt = w, h, "yuv444p"
        s.options = {"crf": "18", "preset": "ultrafast"}
        for fr in frames:
            for p in s.encode(av.VideoFrame.from_ndarray(fr, format="rgb24").reformat(format="yuv444p")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)


TARGETS = {"000": (480, 854, False), "001": (480, 640, False), "002": (480, 854, True),
           "003": (360, 640, False)}
PAIRS = {"000": "001", "001": "000", "002": "003", "003": "002"}
N = 80
for comp in ("c23",):
    base = ROOT / "datasets/mynkey/dg-ffpp-c23"
    for t, (h, w, two) in TARGETS.items():
        frames = ffpp_frames(h, w, N, int(t), small=two)
        write444(base / f"original_sequences/youtube/{comp}/videos/{t}.mp4", frames)
        s = PAIRS[t]
        for m in ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"):
            ff = ffpp_frames(h, w, N, int(t), small=two, fake_main=not two, fake_small=two)
            if m in ("Face2Face", "NeuralTextures") and w % 32:
                w32 = w // 32 * 32
                off = (w - w32) // 2
                ff = [np.ascontiguousarray(x[:, off:off + w32]) for x in ff]
            write444(base / f"manipulated_sequences/{m}/{comp}/videos/{t}_{s}.mp4", ff)
    dbase = ROOT / "datasets/mynkey/dg-dfd-c23"
    H, W = 360, 640
    B = bg(H, W, 99)
    orig = []
    for i in range(120):
        f = B[:, i % 64: i % 64 + W].copy()
        paste(f, FACE, 200 + i, 40, 1.0)
        orig.append(f)
    p = dbase / f"original_sequences/actors/{comp}/videos"
    p.mkdir(parents=True, exist_ok=True)
    write_rgb_video(p / "01__kitchen.mp4", orig, fps=24, crf=18)
    write_rgb_video(p / "02__walk.mp4", orig[::-1], fps=24, crf=18)
    q = dbase / f"manipulated_sequences/DeepFakeDetection/{comp}/videos"
    q.mkdir(parents=True, exist_ok=True)

    def dfd_fake(src, start, n):
        out = []
        for i in range(start, start + n):
            f = src[i].copy()
            x = 200 + i if src is orig else 200 + (119 - i)
            paste(f, manip(FACE), x, 40, 1.0)
            out.append(f)
        return out
    write_rgb_video(q / "01_02__kitchen__ABCD1234.mp4", dfd_fake(orig, 17, 90), fps=24, crf=18)
    write_rgb_video(q / "01_03__kitchen__EFGH5678.mp4", dfd_fake(orig, 40, 70), fps=24, crf=18)
    write_rgb_video(q / "02_01__walk__SHRT0001.mp4", dfd_fake(orig[::-1], 5, 20), fps=24, crf=18)
print("mock ready:", sum(1 for _ in ROOT.rglob("*.mp4")), "videos under", ROOT)
