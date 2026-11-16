"""Builds dg_p12a_detector_width_budget.ipynb — Phase 1.2 stage A (D-019 items 1, 2, 4-pilot, 6, 9).

Run on Kaggle with: Accelerator GPU T4 x1, Internet On, Persistence off.
Inputs: mynkey/dg-ffpp-c23 and mynkey/dg-dfd-c23.
"""
import sys

import nbformat as nbf

OUT = sys.argv[1]
COMMIT = sys.argv[2]

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# DeepGuard v2 · Phase 1.2 stage A — detector choice, width test, DFD pilot, budget

This notebook settles the open questions **before** the full preprocessing run (D-019).

| Cell | What it does | D-019 item |
|---|---|---|
| 1 | Settings, installs, GPU check | — |
| 2 | Get the shared `deepguard` package from GitHub at a fixed commit | 1 |
| 3 | Download the face models (SCRFD, ArcFace) and verify their SHA-256 | — |
| 4 | Run the package's unit tests here, with the real models (colour test included) | 1 |
| 5 | Inventory: find every c23 video, read sizes and frame counts | — |
| 6 | **Width test:** were the 282 narrower fakes cropped or rescaled? | 2 |
| 7 | **Detector comparison:** SCRFD vs MediaPipe on 200 videos | 6 |
| 8 | **DFD offset pilot:** can we find where each DFD fake starts in its original? | 4 |
| 9 | **Budget:** time and storage measured on about 1,000 real crops | 9 |
| 10 | Report and zip to download | — |

**Kaggle settings:** Accelerator **GPU T4 ×1**, Internet **On**.
**Inputs:** `mynkey/dg-ffpp-c23` and `mynkey/dg-dfd-c23`.
**Secret (only while the repo is private):** `GITHUB_TOKEN`, a fine-grained GitHub token with read-only *Contents* access to `akmayank94/deepguard`.
Run with **Save Version → Save & Run All**. Expected time: about 45–75 minutes.
""")

# ---------------------------------------------------------------- CELL 1
code(r'''
# ===== CELL 1: Settings, installs, GPU check =====
import os, sys, json, time, random, shutil, subprocess, zipfile, hashlib, platform
from pathlib import Path

T0 = time.time()
def log(msg):
    """Print a message with elapsed time, so progress is visible while a cell runs."""
    m, s = divmod(int(time.time() - T0), 60)
    print(f"[{m:3d}m{s:02d}s] {msg}", flush=True)

# ---------------- settings ----------------
REPO_URL    = "https://github.com/akmayank94/deepguard.git"
REPO_COMMIT = "__COMMIT__"     # the package version this run uses (recorded in every output)
INPUT_ROOT  = Path(os.environ.get("DG_INPUT_ROOT", "/kaggle/input"))
WORK        = Path(os.environ.get("DG_WORK", "/kaggle/working"))
OUT         = WORK / "p12a"                 # every result goes here
MOCK        = os.environ.get("DG_MOCK") == "1"   # local test run on mock data (not on Kaggle)
SEED        = 2026
N_FRAMES    = 32        # frames per video (D-019.5)
WIDTH_SAMPLE  = 60      # width-changed targets checked in cell 6
DET_FFPP, DET_DFD = 100, 100   # videos in the detector comparison (cell 7)
DFD_PILOT   = 50        # DFD fakes in the offset pilot (cell 8)
BUDGET_CROPS = 1000     # crops made in the budget run (cell 9)
DECODE_THREADS = 4
# ------------------------------------------
random.seed(SEED)
OUT.mkdir(parents=True, exist_ok=True); (OUT / "figures").mkdir(exist_ok=True)

def sh(cmd, check=True):
    """Run a shell command, print its last lines, fail loudly if it fails."""
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
    for line in tail: print("   ", line)
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {cmd}")
    return r

if not MOCK:
    log("Installing PyAV, onnxruntime-gpu, pytest ...")
    sh(f"{sys.executable} -m pip install -q av onnxruntime-gpu pytest scikit-image")
import numpy as np, pandas as pd, cv2, av, onnxruntime as ort
try:
    ort.preload_dlls()          # onnxruntime >= 1.21: load CUDA/cuDNN from the pip wheels
except Exception as e:
    print("   preload_dlls not used:", type(e).__name__)
GPU = "CUDAExecutionProvider" in ort.get_available_providers()
log(f"Python {platform.python_version()} | numpy {np.__version__} | OpenCV {cv2.__version__} | "
    f"PyAV {av.__version__} | onnxruntime {ort.__version__} | pandas {pd.__version__}")
log(f"onnxruntime providers: {ort.get_available_providers()}  -> GPU available: {GPU}")
log(f"Input root: {INPUT_ROOT} (exists: {INPUT_ROOT.exists()}) | output: {OUT} | mock: {MOCK}")
assert INPUT_ROOT.exists(), "Attach mynkey/dg-ffpp-c23 and mynkey/dg-dfd-c23 with 'Add Input'."
''')

# ---------------------------------------------------------------- CELL 2
code(r'''
# ===== CELL 2: Get the shared package at the fixed commit (D-018.2, D-018.7) =====
# The same code will later run inside the app, so the notebook never re-implements it.
REPO = WORK / "deepguard"
if MOCK:
    REPO = Path(os.environ["DG_REPO_DIR"])
    log(f"Mock run: using local repo {REPO}")
else:
    token = None
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("GITHUB_TOKEN")
    except Exception:
        log("No GITHUB_TOKEN secret: trying a public clone.")
    url = REPO_URL.replace("https://", f"https://x-access-token:{token}@") if token else REPO_URL
    if REPO.exists(): shutil.rmtree(REPO)
    r = subprocess.run(["git", "clone", "-q", url, str(REPO)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("git clone failed (token missing or wrong?): "
                           + r.stderr.replace(token or "#", "***")[-300:])
    sh(f"git -C {REPO} checkout -q {REPO_COMMIT}")
    sh(f"{sys.executable} -m pip install -q --no-deps -e {REPO}")
HEAD = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                      text=True).stdout.strip()
sys.path.insert(0, str(REPO / "src"))
import deepguard
from deepguard.preprocess import PREPROCESS_VERSION
from deepguard.preprocess import video as V, faces as F, crop as C, align as A, select as S
log(f"deepguard {deepguard.__version__} | preprocess {PREPROCESS_VERSION} | commit {HEAD[:12]}")
if not MOCK:
    assert HEAD.startswith(REPO_COMMIT), f"checked-out commit {HEAD} is not {REPO_COMMIT}"
RUN = {"commit": HEAD, "preprocess_version": PREPROCESS_VERSION, "gpu": GPU,
       "python": platform.python_version(), "onnxruntime": ort.__version__, "pyav": av.__version__,
       "opencv": cv2.__version__, "numpy": np.__version__, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
''')

# ---------------------------------------------------------------- CELL 3
code(r'''
# ===== CELL 3: Face models, verified by SHA-256 =====
# InsightFace model packs from the official GitHub release (non-commercial research licence).
MODELS = Path(os.environ.get("DG_MODELS", WORK / "models"))
PACKS = {"buffalo_l": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
         "buffalo_sc": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip"}
SHA = {"buffalo_l/det_10g.onnx": "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
       "buffalo_l/w600k_r50.onnx": "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
       "buffalo_sc/det_500m.onnx": "5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a"}
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
for pack, url in PACKS.items():
    if not (MODELS / pack).exists():
        log(f"Downloading {pack} ...")
        MODELS.mkdir(parents=True, exist_ok=True)
        z = MODELS / f"{pack}.zip"
        sh(f"curl -sSL -o {z} {url}")
        with zipfile.ZipFile(z) as zz: zz.extractall(MODELS / pack)
        z.unlink()
for rel, want in SHA.items():
    got = sha256(MODELS / rel)
    assert got == want, f"{rel}: SHA-256 {got} != {want}"
    log(f"OK {rel}  sha256 {got[:12]}…")
RUN["models"] = SHA
''')

# ---------------------------------------------------------------- CELL 4
code(r'''
# ===== CELL 4: The package's own tests, here, with the real models (D-019.1) =====
# Includes the colour test: decoded RGB must match the analytic value within ±1.
env = dict(os.environ, DG_MODELS=str(MODELS), PYTHONPATH=str(REPO / "src"))
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:warnings",
                    "-p", "no:cacheprovider", str(REPO / "tests")],
                   capture_output=True, text=True, env=env, cwd=str(REPO))
print("\n".join(r.stdout.strip().splitlines()[-8:]))
assert r.returncode == 0, "Package tests FAILED here: stop and send this output."
RUN["tests"] = r.stdout.strip().splitlines()[-1]
det = F.make_detector("scrfd_10g", str(MODELS))
log(f"SCRFD-10G providers in use: {det.providers}")
RUN["scrfd_providers"] = det.providers
''')

# ---------------------------------------------------------------- CELL 5
code(r'''
# ===== CELL 5: Inventory of c23 videos (header facts only, fast) =====
from concurrent.futures import ThreadPoolExecutor
METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"]
rows = []
for p in sorted(INPUT_ROOT.rglob("*.mp4")):
    s = str(p)
    if "/c23/" not in s: continue
    if "/original_sequences/youtube/" in s:
        rows.append(dict(path=s, dataset="FFPP", method="original", vid=p.stem, target=p.stem, source=""))
    elif "/original_sequences/actors/" in s:
        rows.append(dict(path=s, dataset="DFD", method="original", vid=p.stem,
                         target=p.stem.split("__")[0], source=""))
    elif "/manipulated_sequences/DeepFakeDetection/" in s:
        a, scene, _ = p.stem.split("__")
        t, src = a.split("_")
        rows.append(dict(path=s, dataset="DFD", method="DeepFakeDetection", vid=p.stem, target=t,
                         source=src, original=f"{t}__{scene}"))
    else:
        m = p.parts[-4]
        if m in METHODS:
            t, src = p.stem.split("_")
            rows.append(dict(path=s, dataset="FFPP", method=m, vid=p.stem, target=t, source=src))
inv = pd.DataFrame(rows)
log(f"Found {len(inv)} c23 videos")
def head(path):
    i = V.probe(path)
    return i.width, i.height, i.n_frames_header, i.fps, i.pix_fmt
with ThreadPoolExecutor(8) as ex:
    meta = list(ex.map(head, inv.path))
inv[["width", "height", "n_frames", "fps", "pix_fmt"]] = pd.DataFrame(meta, index=inv.index)
print(inv.groupby(["dataset", "method"]).agg(videos=("vid", "size"), frames_med=("n_frames", "median"),
                                             pix_fmt=("pix_fmt", lambda x: ",".join(sorted(set(x))))))
ff_orig = inv[(inv.dataset == "FFPP") & (inv.method == "original")].set_index("vid")
W_TARGETS = sorted(ff_orig.index[ff_orig.width % 32 != 0])
log(f"FF++ originals whose width is not a multiple of 32: {len(W_TARGETS)} (audit: 282)")
inv.to_csv(OUT / "inventory_c23.csv", index=False)
''')

# ---------------------------------------------------------------- CELL 6
code(r'''
# ===== CELL 6: Width test — cropped or rescaled? (D-019.2) =====
# For width-changed targets: compare each narrow fake frame with (a) every crop offset of the
# original and (b) the rescaled original, on background pixels only (faces are excluded).
def path_of(dataset, method, vid):
    r = inv[(inv.dataset == dataset) & (inv.method == method) & (inv.vid == vid)]
    return None if r.empty else r.iloc[0]
rng = random.Random(SEED)
by_w = {}
for t in W_TARGETS: by_w.setdefault(int(ff_orig.loc[t, "width"]), []).append(t)
sample = []                              # every width value represented, then fill at random
for w_, ts in sorted(by_w.items()): sample.append(rng.choice(ts))
rest = [t for t in W_TARGETS if t not in sample]; rng.shuffle(rest)
sample += rest[: max(0, WIDTH_SAMPLE - len(sample))]
wrows = []
for k, t in enumerate(sample):
    o = ff_orig.loc[t]
    for m in ("Face2Face", "NeuralTextures"):
        fk = inv[(inv.dataset == "FFPP") & (inv.method == m) & (inv.target == t)]
        if fk.empty: continue
        fk = fk.iloc[0]
        idx = V.sample_indices(int(min(o.n_frames, fk.n_frames)), N_FRAMES)
        pick = [idx[4], idx[12], idx[20], idx[28]]
        of, ff = V.decode_frames(o.path, pick), V.decode_frames(fk.path, pick)
        for i in pick:
            boxes = [d.box for d in det.detect(ff[i])]
            r = A.width_candidates(of[i], ff[i], boxes)
            mode, off, mad = r["best"]
            crop_best = min(r["crop"].values())
            wrows.append(dict(target=t, method=m, frame=i, w_orig=int(o.width), w_fake=int(fk.width),
                              mode=mode, offset=off, rule=A.crop_offset_rule(int(o.width), off) if mode == "crop" else "",
                              mad_best=mad, mad_crop=crop_best, mad_rescale=r["rescale"], n_faces=len(boxes)))
    if (k + 1) % 10 == 0: log(f"  width test {k+1}/{len(sample)} targets")
wt = pd.DataFrame(wrows); wt.to_csv(OUT / "width_test.csv", index=False)
print(wt.groupby(["mode", "rule"]).size().rename("frames"))
print(wt.groupby("w_orig").agg(frames=("mode", "size"), crop_share=("mode", lambda x: (x == "crop").mean()),
                               offset_med=("offset", "median"), mad_crop=("mad_crop", "median"),
                               mad_rescale=("mad_rescale", "median")))
top = wt.groupby(["mode", "rule"]).size().sort_values(ascending=False)
(mode, rule), n = top.index[0], int(top.iloc[0])
share = n / len(wt)
WIDTH = {"mode": mode, "offset_rule": rule or None, "share": round(share, 4), "frames": len(wt),
         "targets": int(wt.target.nunique()), "decided": share >= 0.95}
def width_offset(w):
    """Crop offset for an original of width w under the decided rule."""
    d = w - (w // 32) * 32
    return {"left": 0, "right": d, "center": d // 2}.get(WIDTH["offset_rule"], 0)
if WIDTH["decided"] and mode == "crop":
    # check: the centre/left/right rule reproduces the measured offset for every frame
    WIDTH["rule_matches_measured"] = float((wt.offset == wt.w_orig.map(width_offset)).mean())
json.dump(WIDTH, open(OUT / "width_decision.json", "w"), indent=2)
log(f"WIDTH DECISION: {WIDTH}")
if not WIDTH["decided"]:
    print("!! No transform explains >= 95% of frames. Stage B must NOT run; send width_test.csv.")
def fix_width(frames, width):
    """Apply the decided width transform to frames of a video whose width % 32 != 0."""
    if width % 32 == 0 or not WIDTH["decided"]:
        return frames
    return {i: V.normalize_width(f, WIDTH["mode"], width_offset(width)) for i, f in frames.items()}
''')

# ---------------------------------------------------------------- CELL 7
code(r'''
# ===== CELL 7: Detector comparison — SCRFD-10G vs MediaPipe full-range (D-019.6) =====
# Same 200 videos, same 32 frames, same decoder. MediaPipe runs in its own small Python
# environment (subprocess) so its pinned numpy cannot break this notebook.
rng = random.Random(SEED + 1)
ff_all = inv[inv.dataset == "FFPP"]; dfd_all = inv[(inv.dataset == "DFD") & (inv.n_frames >= N_FRAMES)]
pick_ff = ff_all.sample(min(DET_FFPP, len(ff_all)), random_state=SEED)
pick_dfd = dfd_all.sample(min(DET_DFD, len(dfd_all)), random_state=SEED)
jobs = pd.concat([pick_ff, pick_dfd]).reset_index(drop=True)
jobs["indices"] = [V.sample_indices(int(n), N_FRAMES) for n in jobs.n_frames]
jobs = jobs[jobs["indices"].notna()].reset_index(drop=True)
def load(row):
    fr = V.decode_frames(row.path, row.indices)
    return fix_width(fr, int(row.width)) if row.dataset == "FFPP" else fr
def to_list(dets):
    return [[*map(float, d.box), float(d.score)] for d in dets]
def prefetch(rows, fn, depth=2 * DECODE_THREADS):
    """Decode ahead in threads while the GPU detects, keeping at most `depth` videos in memory."""
    rows = list(rows)
    with ThreadPoolExecutor(DECODE_THREADS) as ex:
        timed = lambda r: (time.time(), fn(r), time.time())
        futs = {k: ex.submit(timed, rows[k]) for k in range(min(depth, len(rows)))}
        for k in range(len(rows)):
            a, out, b = futs.pop(k).result()
            if k + depth < len(rows): futs[k + depth] = ex.submit(timed, rows[k + depth])
            yield rows[k], out, b - a
scrfd, t_det, t_dec = {}, [], []
if True:
    for k, (row, frames, dt) in enumerate(prefetch(jobs.itertuples(), load)):
        t_dec.append(dt)
        t = time.time()
        scrfd[row.path] = {str(i): to_list(det.detect(f)) for i, f in frames.items()}
        t_det.append(time.time() - t)
        if (k + 1) % 25 == 0: log(f"  SCRFD {k+1}/{len(jobs)} videos")
json.dump(scrfd, open(OUT / "dets_scrfd.json", "w"))
# ---- MediaPipe in a subprocess ----
MP_DIR = WORK / "mp_env"
mp_script = OUT / "run_mediapipe.py"
mp_script.write_text("""
import json, sys, time
sys.path.insert(0, sys.argv[3])
from deepguard.preprocess import video as V, faces as F
jobs = json.load(open(sys.argv[1])); det = F.MediaPipeDetector(); out = {}; tt = []
for j in jobs:
    fr = V.decode_frames(j['path'], j['indices'])
    if j['fix'] is not None:
        fr = {i: V.normalize_width(f, j['fix'][0], j['fix'][1]) for i, f in fr.items()}
    t = time.time()
    out[j['path']] = {i: [[*map(float, d.box), float(d.score)] for d in det.detect(f)] for i, f in fr.items()}
    tt.append(time.time() - t)
json.dump({'dets': out, 'sec_per_video': tt}, open(sys.argv[2], 'w'))
""")
jl = [{"path": r.path, "indices": r.indices,
       "fix": [WIDTH["mode"], width_offset(int(r.width))] if (r.dataset == "FFPP" and int(r.width) % 32
                                                              and WIDTH["decided"]) else None}
      for r in jobs.itertuples()]
json.dump(jl, open(OUT / "mp_jobs.json", "w"))
mp_ok = True
if MOCK:
    py, env = sys.executable, dict(os.environ)
else:
    log("Installing MediaPipe into its own folder ...")
    r = sh(f"{sys.executable} -m pip install -q --target {MP_DIR} mediapipe==0.10.14 av", check=False)
    mp_ok = r.returncode == 0
    py, env = sys.executable, dict(os.environ, PYTHONPATH=str(MP_DIR))
if mp_ok:
    log("Running MediaPipe on the same frames ...")
    r = subprocess.run([py, str(mp_script), str(OUT / "mp_jobs.json"), str(OUT / "dets_mediapipe.json"),
                        str(REPO / "src")], capture_output=True, text=True, env=env)
    mp_ok = r.returncode == 0
    if not mp_ok: print(r.stderr[-1500:])
mp = json.load(open(OUT / "dets_mediapipe.json")) if mp_ok else None
log(f"MediaPipe available: {mp_ok}")

def tracks_from(d):
    frames = {int(i): [F.Detection(np.array(b[:4], np.float32), b[4]) for b in bs] for i, bs in d.items()}
    tr = F.link_tracks(frames)
    kept = F.filter_tracks(tr, N_FRAMES)
    return (max(kept, key=len) if kept else (max(tr, key=len) if tr else None)), len(tr) - len(kept)
crows = []
for r in jobs.itertuples():
    s_main, s_fp = tracks_from(scrfd[r.path])
    m_main, m_fp = tracks_from(mp["dets"][r.path]) if mp else (None, 0)
    agree = True
    if s_main and m_main:
        shared = set(s_main.dets) & set(m_main.dets)
        agree = bool(shared) and np.mean([F.iou(s_main.dets[i].box, m_main.dets[i].box) for i in shared]) >= 0.3
    union = set(s_main.dets if s_main else {}) | set(m_main.dets if m_main else {})
    crows.append(dict(path=r.path, dataset=r.dataset, method=r.method, agree=agree, union=len(union),
                      scrfd_frames=len(s_main) if s_main else 0, mp_frames=len(m_main) if m_main else 0,
                      scrfd_miss=(len(union - set(s_main.dets)) if s_main else len(union)) / max(1, len(union)) if agree else np.nan,
                      mp_miss=(len(union - set(m_main.dets)) if m_main else len(union)) / max(1, len(union)) if agree else np.nan,
                      scrfd_short_tracks=s_fp, mp_short_tracks=m_fp))
cmp_df = pd.DataFrame(crows); cmp_df.to_csv(OUT / "detector_comparison.csv", index=False)
summ = cmp_df.groupby("dataset").agg(videos=("path", "size"), disagree=("agree", lambda x: int((~x).sum())),
                                     scrfd_miss=("scrfd_miss", "mean"), mp_miss=("mp_miss", "mean"),
                                     no_face_scrfd=("scrfd_frames", lambda x: int((x == 0).sum())),
                                     no_face_mp=("mp_frames", lambda x: int((x == 0).sum())))
print(summ)
DET = {"scrfd_miss": float(cmp_df.scrfd_miss.mean()), "mp_miss": float(cmp_df.mp_miss.mean()) if mp else None,
       "disagree_videos": int((~cmp_df.agree).sum()), "videos": len(cmp_df),
       "scrfd_sec_per_video": float(np.mean(t_det)), "decode_sec_per_video": float(np.mean(t_dec)),
       "mp_sec_per_video": float(np.mean(mp["sec_per_video"])) if mp else None, "mediapipe_available": mp_ok}
if not mp_ok:
    DET["winner"], DET["why"] = "scrfd_10g", "MediaPipe could not run here; comparison incomplete (flag)."
elif abs(DET["scrfd_miss"] - DET["mp_miss"]) > 0.01:
    DET["winner"] = "scrfd_10g" if DET["scrfd_miss"] < DET["mp_miss"] else "mediapipe"
    DET["why"] = "lower miss rate"
else:
    DET["winner"] = "scrfd_10g" if DET["scrfd_sec_per_video"] <= DET["mp_sec_per_video"] else "mediapipe"
    DET["why"] = "miss rates within 1 point; faster"
json.dump(DET, open(OUT / "detector_decision.json", "w"), indent=2)
log(f"DETECTOR DECISION: {DET}")
# contact sheet of disagreements: SCRFD boxes green, MediaPipe boxes red
dis = cmp_df[~cmp_df.agree].head(12)
tiles = []
for r in dis.itertuples():
    j = jobs[jobs.path == r.path].iloc[0]; i = j.indices[16]
    fr = V.decode_frames(r.path, [i])
    if j.dataset == "FFPP": fr = fix_width(fr, int(j.width))
    f = fr[i].copy()
    for b in scrfd[r.path].get(str(i), []): cv2.rectangle(f, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 3)
    for b in (mp["dets"][r.path].get(str(i), []) if mp else []):
        cv2.rectangle(f, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 0), 3)
    tiles.append(cv2.resize(f, (320, int(320 * f.shape[0] / f.shape[1]))))
if tiles:
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    rows_ = [np.hstack(tiles[k:k + 4] + [np.zeros_like(tiles[0])] * (4 - len(tiles[k:k + 4]))) for k in range(0, len(tiles), 4)]
    cv2.imwrite(str(OUT / "figures/detector_disagreements.jpg"), cv2.cvtColor(np.vstack(rows_), cv2.COLOR_RGB2BGR))
    log(f"Saved {len(tiles)} disagreement frames to figures/detector_disagreements.jpg")
''')

# ---------------------------------------------------------------- CELL 8
code(r'''
# ===== CELL 8: DFD offset pilot (D-019.4) =====
# DFD fakes are trimmed sub-clips. Find each fake's start frame in its original from small
# grayscale frames, then measure the full-resolution background difference after alignment.
dfd = inv[inv.dataset == "DFD"]
dfd_orig = dfd[dfd.method == "original"].set_index("vid")
fakes = dfd[(dfd.method == "DeepFakeDetection") & (dfd.n_frames >= N_FRAMES)]
fakes = fakes[fakes.original.isin(dfd_orig.index)].sample(min(DFD_PILOT, len(fakes)), random_state=SEED)
gray_cache = {}
def gray(path):
    if path not in gray_cache: gray_cache[path] = np.stack(list(V.iter_gray_small(path)))
    return gray_cache[path]
prow = []
for k, r in enumerate(fakes.itertuples()):
    o = dfd_orig.loc[r.original]
    res = A.find_offset(gray(r.path), gray(o.path))
    idx = V.sample_indices(int(r.n_frames), N_FRAMES)
    pick = [i for i in (idx[4], idx[16], idx[28]) if i + res.offset < o.n_frames]
    ff = V.decode_frames(r.path, pick); of = V.decode_frames(o.path, [i + res.offset for i in pick])
    of = {i: of[i + res.offset] for i in pick if i + res.offset in of}
    face_tracks = [F.Track(n, {i: d}) for n, (i, d) in
                   enumerate((i, d) for i in pick for d in det.detect(ff[i]))]   # exclude every face
    bg = S.background_diff(ff, of, face_tracks)
    prow.append(dict(fake=r.vid, original=r.original, n_fake=int(r.n_frames), n_orig=int(o.n_frames),
                     offset=res.offset, err=res.err, margin=res.margin, bg_diff=bg))
    if (k + 1) % 10 == 0: log(f"  DFD pilot {k+1}/{len(fakes)}")
    if len(gray_cache) > 40: gray_cache.clear()
pil = pd.DataFrame(prow)
FFPP_BG = 2.0                                          # FF++ median background difference (audit A7)
ALIGN_LIMIT = max(8.0, 2.5 * FFPP_BG)                  # same rule as the audit's "misaligned" flag
pil["aligned"] = pil.bg_diff <= ALIGN_LIMIT
pil.to_csv(OUT / "dfd_offset_pilot.csv", index=False)
print(pil.describe()[["offset", "err", "margin", "bg_diff"]].round(2))
DFDP = {"fakes": len(pil), "aligned_share": float(pil.aligned.mean()), "bg_limit": ALIGN_LIMIT,
        "bg_median_after": float(pil.bg_diff.median()), "offset_zero_share": float((pil.offset == 0).mean()),
        "margin_median": float(pil.margin.median())}
json.dump(DFDP, open(OUT / "dfd_pilot.json", "w"), indent=2)
log(f"DFD PILOT: {DFDP}  (target for stage B: aligned share >= 0.95)")
''')

# ---------------------------------------------------------------- CELL 9
code(r'''
# ===== CELL 9: Budget on ~1,000 real crops (D-019.9) =====
# Runs the real crop path (decode -> detect -> track -> crop -> JPEG) and extrapolates.
DETECTOR = F.make_detector(DET["winner"], str(MODELS)) if DET["winner"] != "scrfd_10g" else det
n_videos = int(np.ceil(BUDGET_CROPS / N_FRAMES))
bud_jobs = pd.concat([ff_all.sample(n_videos // 2, random_state=SEED + 2),
                      dfd_all.sample(min(len(dfd_all), n_videos - n_videos // 2), random_state=SEED + 2)])
brow = []; sizes = []
for r in bud_jobs.itertuples():
    idx = V.sample_indices(int(r.n_frames), N_FRAMES)
    t0 = time.time(); fr = V.decode_frames(r.path, idx)
    if r.dataset == "FFPP": fr = fix_width(fr, int(r.width))
    t1 = time.time(); dets = {i: DETECTOR.detect(f) for i, f in fr.items()}
    t2 = time.time()
    kept = F.filter_tracks(F.link_tracks(dets), N_FRAMES)
    nb = 0; nc = 0
    if kept:
        main = max(kept, key=len)
        for i, d in main.dets.items():
            b = C.encode_jpeg(C.crop_square(fr[i], d.box)[0]); nb += len(b); nc += 1
            if len(sizes) < 16: sizes.append(b)
    t3 = time.time()
    brow.append(dict(dataset=r.dataset, decode_s=t1 - t0, detect_s=t2 - t1, crop_s=t3 - t2, crops=nc, bytes=nb))
bud = pd.DataFrame(brow); bud.to_csv(OUT / "budget_runs.csv", index=False)
per = bud.groupby("dataset").agg(videos=("crops", "size"), crops=("crops", "sum"), bytes=("bytes", "sum"),
                                 decode_s=("decode_s", "mean"), detect_s=("detect_s", "mean"), crop_s=("crop_s", "mean"))
per["kb_per_crop"] = per.bytes / per.crops.clip(lower=1) / 1024
print(per.round(3))
# full run: c23 + c40 for every video; detection only on c23 (c40 reuses the boxes, D-019.5)
counts = inv.groupby("dataset").size()
est = {}
for ds in per.index:
    n = int(counts[ds]); p = per.loc[ds]
    est[ds] = {"videos_per_compression": n, "crops_total": int(n * 2 * N_FRAMES),
               "storage_gb": round(n * 2 * N_FRAMES * p.kb_per_crop / 1024 / 1024, 2),
               "cpu_decode_h_1proc": round(n * 2 * p.decode_s / 3600, 2),
               "gpu_detect_h": round(n * p.detect_s / 3600, 2),
               "crop_jpeg_h_1proc": round(n * 2 * p.crop_s / 3600, 2)}
BUDGET = {"measured_crops": int(bud.crops.sum()), "detector": DET["winner"], "per_dataset": est,
          "storage_gb_total": round(sum(e["storage_gb"] for e in est.values()), 1),
          "note": "decode and crop hours are for one process; Kaggle has 4 CPU cores"}
json.dump(BUDGET, open(OUT / "budget.json", "w"), indent=2)
log(f"BUDGET: {json.dumps(BUDGET, indent=1)}")
for k, b in enumerate(sizes[:8]):
    (OUT / "figures" / f"sample_crop_{k}.jpg").write_bytes(b)
''')

# ---------------------------------------------------------------- CELL 10
code(r'''
# ===== CELL 10: Report + zip to download =====
RUN["finished"] = time.strftime("%Y-%m-%d %H:%M:%S"); RUN["minutes"] = round((time.time() - T0) / 60, 1)
report = {"run": RUN, "width": WIDTH, "detector": DET, "dfd_pilot": DFDP, "budget": BUDGET}
json.dump(report, open(OUT / "stageA_report.json", "w"), indent=2, default=str)
ok = {"tests pass (cell 4)": True,
      "width transform explains >= 95% (cell 6)": WIDTH["decided"],
      "MediaPipe compared (cell 7)": DET["mediapipe_available"],
      "DFD pilot aligned >= 95% (cell 8)": DFDP["aligned_share"] >= 0.95}
print("=" * 60); print("STAGE A SUMMARY — paste this back to Claude"); print("=" * 60)
for k, v in ok.items(): print(f"  [{'PASS' if v else 'CHECK'}] {k}")
print(json.dumps(report, indent=1, default=str))
zp = WORK / "p12a_results.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob("*"):
        if p.is_file() and not p.name.startswith("dets_"):
            z.write(p, p.relative_to(OUT.parent))
log(f"Saved {zp} ({zp.stat().st_size/1e6:.1f} MB). Download it from the Output panel.")
''')

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
for c in nb["cells"]:
    if c.cell_type == "code":
        c.source = c.source.replace("__COMMIT__", COMMIT)
nbf.write(nb, OUT)
print("wrote", OUT, len(cells), "cells")
