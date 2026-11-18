"""Builds dg_p12a2_detectors_tracks_masks.ipynb — Phase 1.2 stage A2 (D-019a, b, c).

Run on Kaggle: GPU T4 x1, Internet On.
Inputs: mynkey/dg-ffpp-c23, mynkey/dg-dfd-c23, mynkey/dg-dfd-masks.
Usage: python tools/build_p12a2_notebook.py OUT.ipynb <package commit>
"""
import sys

import nbformat as nbf

OUT, COMMIT = sys.argv[1], sys.argv[2]
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# DeepGuard v2 · Phase 1.2 stage A2 — three detectors, identity tracks, DFD masks

Re-checks only what stage A left open (D-019a–c). Width (centre crop) and budget are already decided.

| Cell | What it does | D-019 |
|---|---|---|
| 1–4 | Settings, package at a fixed commit, models (SHA-256), package tests | 1 |
| 5 | Inventory: c23 videos and the official DFD masks | — |
| 6 | **Detectors:** SCRFD vs YuNet vs MediaPipe on the same 200 videos as stage A | 6, b |
| 7 | **Tracks:** position-only vs identity-aware linking | c |
| 8 | **DFD masks:** face chosen from the official masks on 300 fakes; agreement with the difference method | 4, a |
| 9 | Report + zip | — |

**Kaggle:** GPU T4 ×1, Internet On. **Inputs:** `dg-ffpp-c23`, `dg-dfd-c23`, `dg-dfd-masks`.
**Save Version → Save & Run All.** Expected time: about 60–80 minutes.
""")

code(r'''
# ===== CELL 1: Settings, installs, GPU check =====
import os, sys, json, time, random, shutil, subprocess, zipfile, hashlib, platform
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
T0 = time.time()
def log(msg):
    """Print a message with elapsed time, so progress is visible while a cell runs."""
    m, s = divmod(int(time.time() - T0), 60)
    print(f"[{m:3d}m{s:02d}s] {msg}", flush=True)

REPO_URL    = "https://github.com/akmayank94/deepguard.git"
REPO_COMMIT = "__COMMIT__"
INPUT_ROOT  = Path(os.environ.get("DG_INPUT_ROOT", "/kaggle/input"))
WORK        = Path(os.environ.get("DG_WORK", "/kaggle/working"))
OUT         = WORK / "p12a2"
MOCK        = os.environ.get("DG_MOCK") == "1"
SEED, N_FRAMES = 2026, 32
DET_FFPP, DET_DFD = 100, 100     # same sample as stage A (same seed)
MASK_SAMPLE, PILOT = 300, 50     # DFD fakes checked with masks; first 50 = the stage A pilot
DECODE_THREADS = 4
MEDIAPIPE_VERSION = "1.2.3"
random.seed(SEED)
OUT.mkdir(parents=True, exist_ok=True); (OUT / "figures").mkdir(exist_ok=True)

def sh(cmd, check=True):
    """Run a shell command, print its last lines, fail loudly if it fails."""
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    for line in (r.stdout + r.stderr).strip().splitlines()[-3:]: print("   ", line)
    if check and r.returncode != 0: raise RuntimeError(f"command failed ({r.returncode}): {cmd}")
    return r

if not MOCK:
    log("Installing PyAV, onnxruntime-gpu, pytest ...")
    sh(f"{sys.executable} -m pip install -q av onnxruntime-gpu pytest scikit-image")
import numpy as np, pandas as pd, cv2, av, onnxruntime as ort
try: ort.preload_dlls()
except Exception as e: print("   preload_dlls not used:", type(e).__name__)
GPU = "CUDAExecutionProvider" in ort.get_available_providers()
log(f"Python {platform.python_version()} | numpy {np.__version__} | OpenCV {cv2.__version__} | PyAV {av.__version__} "
    f"| onnxruntime {ort.__version__} | GPU available: {GPU}")
assert INPUT_ROOT.exists(), "Attach dg-ffpp-c23, dg-dfd-c23 and dg-dfd-masks with 'Add Input'."
''')

code(r'''
# ===== CELL 2: The shared package at the fixed commit =====
REPO = WORK / "deepguard"
if MOCK:
    REPO = Path(os.environ["DG_REPO_DIR"])
else:
    if REPO.exists(): shutil.rmtree(REPO)
    sh(f"git clone -q {REPO_URL} {REPO}")
    sh(f"git -C {REPO} checkout -q {REPO_COMMIT}")
    sh(f"{sys.executable} -m pip install -q --no-deps -e {REPO}")
HEAD = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
sys.path.insert(0, str(REPO / "src"))
from deepguard.preprocess import PREPROCESS_VERSION
from deepguard.preprocess import video as V, faces as F, crop as C, align as A, select as S, identity as I
if not MOCK: assert HEAD.startswith(REPO_COMMIT), f"checked-out {HEAD} is not {REPO_COMMIT}"
log(f"preprocess {PREPROCESS_VERSION} | commit {HEAD[:12]}")
RUN = {"commit": HEAD, "preprocess_version": PREPROCESS_VERSION, "gpu": GPU, "python": platform.python_version(),
       "onnxruntime": ort.__version__, "pyav": av.__version__, "opencv": cv2.__version__, "numpy": np.__version__}
''')

code(r'''
# ===== CELL 3: Models, verified by SHA-256 =====
MODELS = Path(os.environ.get("DG_MODELS", WORK / "models"))
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
def fetch(dest, urls):
    """Download dest from the first URL that works."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for u in urls:
        if sh(f"curl -sSL -f -o {dest} {u}", check=False).returncode == 0: return
packs = {"buffalo_l": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
         "buffalo_sc": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip"}
for pack, url in packs.items():
    if not (MODELS / pack).exists():
        log(f"Downloading {pack} ..."); z = MODELS / f"{pack}.zip"; fetch(z, [url])
        with zipfile.ZipFile(z) as zz: zz.extractall(MODELS / pack)
        z.unlink()
YUNET = MODELS / "yunet/face_detection_yunet_2023mar.onnx"
if not YUNET.exists():
    fetch(YUNET, ["https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
                  "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"])
MPF = MODELS / "mediapipe" / F.MEDIAPIPE_FULL_RANGE
if not MPF.exists() or sha256(MPF) != F.MEDIAPIPE_FULL_RANGE_SHA256:
    fetch(MPF, [f"https://storage.googleapis.com/mediapipe-assets/{F.MEDIAPIPE_FULL_RANGE}"])
    if not MPF.exists() or sha256(MPF) != F.MEDIAPIPE_FULL_RANGE_SHA256:
        log("Asset storage failed; taking the file from the official mediapipe 0.10.14 wheel ...")
        wd = WORK / "mp_wheel"; shutil.rmtree(wd, ignore_errors=True)
        sh(f"{sys.executable} -m pip download -q --no-deps --only-binary=:all: --python-version 3.10 "
           f"--platform manylinux_2_17_x86_64 mediapipe==0.10.14 -d {wd}")
        with zipfile.ZipFile(next(wd.glob("*.whl"))) as zz:
            MPF.write_bytes(zz.read(f"mediapipe/modules/face_detection/{F.MEDIAPIPE_FULL_RANGE}"))
SHA = {"buffalo_l/det_10g.onnx": "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
       "buffalo_l/w600k_r50.onnx": "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
       "buffalo_sc/det_500m.onnx": "5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a",
       "yunet/face_detection_yunet_2023mar.onnx": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
       f"mediapipe/{F.MEDIAPIPE_FULL_RANGE}": F.MEDIAPIPE_FULL_RANGE_SHA256}
for rel, want in SHA.items():
    got = sha256(MODELS / rel); assert got == want, f"{rel}: SHA-256 {got} != {want}"
    log(f"OK {rel}  sha256 {got[:12]}…")
RUN["models"] = SHA
''')

code(r'''
# ===== CELL 4: Package tests here, with the models =====
env = dict(os.environ, DG_MODELS=str(MODELS), PYTHONPATH=str(REPO / "src"))
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:warnings", "-p", "no:cacheprovider",
                    str(REPO / "tests")], capture_output=True, text=True, env=env, cwd=str(REPO))
print("\n".join(r.stdout.strip().splitlines()[-6:]))
assert r.returncode == 0, "Package tests FAILED here: stop and send this output."
RUN["tests"] = r.stdout.strip().splitlines()[-1]
scrfd = F.make_detector("scrfd_10g", str(MODELS))
yunet = F.make_detector("yunet", str(MODELS))
arc = I.ArcFaceEmbedder(str(MODELS / "buffalo_l/w600k_r50.onnx"))
log(f"SCRFD providers: {scrfd.providers}")
''')

code(r'''
# ===== CELL 5: Inventory (c23 videos + official DFD masks) =====
METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"]
rows, masks = [], {}
for p in sorted(INPUT_ROOT.rglob("*.mp4")):
    s = str(p)
    if "/DeepFakeDetection/" in s and "/masks/" in s:
        masks[p.stem] = s; continue
    if "/c23/" not in s: continue
    if "/original_sequences/youtube/" in s:
        rows.append(dict(path=s, dataset="FFPP", method="original", vid=p.stem))
    elif "/original_sequences/actors/" in s:
        rows.append(dict(path=s, dataset="DFD", method="original", vid=p.stem))
    elif "/manipulated_sequences/DeepFakeDetection/" in s:
        a, scene, _ = p.stem.split("__"); t = a.split("_")[0]
        rows.append(dict(path=s, dataset="DFD", method="DeepFakeDetection", vid=p.stem, original=f"{t}__{scene}"))
    elif p.parts[-4] in METHODS:
        rows.append(dict(path=s, dataset="FFPP", method=p.parts[-4], vid=p.stem))
inv = pd.DataFrame(rows)
with ThreadPoolExecutor(8) as ex:
    meta = list(ex.map(lambda q: (lambda i: (i.width, i.height, i.n_frames_header))(V.probe(q)), inv.path))
inv[["width", "height", "n_frames"]] = pd.DataFrame(meta, index=inv.index)
n_fakes = int((inv.method == "DeepFakeDetection").sum())
log(f"c23 videos: {len(inv)} | DFD masks found: {len(masks)} (DFD fakes: {n_fakes})")
HAVE_MASKS = len(masks) > 0
if not HAVE_MASKS: print("!! dg-dfd-masks is not attached: cell 8 will be skipped.")
def fix_width(fr, w):
    """Decided in stage A (D-019.2): centre crop to a multiple of 32."""
    d = w - (w // 32) * 32
    return fr if d == 0 else {i: V.normalize_width(f, "crop", d // 2) for i, f in fr.items()}
''')

code(r'''
# ===== CELL 6: Three detectors on the same 200 videos (D-019.6 / D-019b) =====
ff_all = inv[inv.dataset == "FFPP"]; dfd_all = inv[(inv.dataset == "DFD") & (inv.n_frames >= N_FRAMES)]
jobs = pd.concat([ff_all.sample(min(DET_FFPP, len(ff_all)), random_state=SEED),
                  dfd_all.sample(min(DET_DFD, len(dfd_all)), random_state=SEED)]).reset_index(drop=True)
jobs["indices"] = [V.sample_indices(int(n), N_FRAMES) for n in jobs.n_frames]
def load(r):
    fr = V.decode_frames(r.path, r.indices)
    return fix_width(fr, int(r.width)) if r.dataset == "FFPP" else fr
def prefetch(rows, fn, depth=2 * DECODE_THREADS):
    """Decode ahead in threads, keeping at most `depth` videos in memory."""
    rows = list(rows)
    with ThreadPoolExecutor(DECODE_THREADS) as ex:
        futs = {k: ex.submit(fn, rows[k]) for k in range(min(depth, len(rows)))}
        for k in range(len(rows)):
            out = futs.pop(k).result()
            if k + depth < len(rows): futs[k + depth] = ex.submit(fn, rows[k + depth])
            yield rows[k], out
DETS = {"scrfd": {}, "yunet": {}}; TIME = {"scrfd": [], "yunet": [], "arcface": []}
for k, (r, fr) in enumerate(prefetch(jobs.itertuples(), load)):
    for name, det in (("scrfd", scrfd), ("yunet", yunet)):
        t = time.time(); DETS[name][r.path] = {i: det.detect(f) for i, f in fr.items()}; TIME[name].append(time.time() - t)
    t = time.time(); I.embed_detections(fr, DETS["scrfd"][r.path], arc); TIME["arcface"].append(time.time() - t)
    if (k + 1) % 25 == 0: log(f"  SCRFD + YuNet + ArcFace {k+1}/{len(jobs)} videos")
# MediaPipe (Tasks API) in its own folder, so its dependencies cannot touch this notebook
MP_DIR = WORK / "mp_env"; mp_script = OUT / "run_mediapipe.py"
mp_script.write_text("""
import json, sys, time
sys.path.insert(0, sys.argv[3])
import mediapipe as mp
from deepguard.preprocess import video as V, faces as F
det = F.MediaPipeDetector(model_path=None if hasattr(mp, 'solutions') else sys.argv[4])
out, tt = {}, []
for j in json.load(open(sys.argv[1])):
    fr = V.decode_frames(j['path'], j['indices'])
    if j['crop'] is not None: fr = {i: V.normalize_width(f, 'crop', j['crop']) for i, f in fr.items()}
    t = time.time(); out[j['path']] = {i: [[*map(float, d.box), float(d.score)] for d in det.detect(f)] for i, f in fr.items()}
    tt.append(time.time() - t)
json.dump({'dets': out, 'sec': tt, 'version': mp.__version__, 'api': det.api}, open(sys.argv[2], 'w'))
""")
jl = [{"path": r.path, "indices": r.indices,
       "crop": (int(r.width) - int(r.width) // 32 * 32) // 2 if (r.dataset == "FFPP" and int(r.width) % 32) else None}
      for r in jobs.itertuples()]
json.dump(jl, open(OUT / "mp_jobs.json", "w"))
env = dict(os.environ)
if not MOCK:
    log(f"Installing MediaPipe {MEDIAPIPE_VERSION} into its own folder ...")
    mp_ok = sh(f"{sys.executable} -m pip install -q --target {MP_DIR} mediapipe=={MEDIAPIPE_VERSION} av", check=False).returncode == 0
    env["PYTHONPATH"] = str(MP_DIR)
else:
    mp_ok = True
if mp_ok:
    r = subprocess.run([sys.executable, str(mp_script), str(OUT / "mp_jobs.json"), str(OUT / "dets_mp.json"),
                        str(REPO / "src"), str(MPF)], capture_output=True, text=True, env=env)
    mp_ok = r.returncode == 0
    if not mp_ok: print(r.stderr[-1500:])
if mp_ok:
    mpj = json.load(open(OUT / "dets_mp.json"))
    DETS["mediapipe"] = {p: {int(i): [F.Detection(np.array(b[:4], np.float32), b[4]) for b in bs] for i, bs in d.items()}
                         for p, d in mpj["dets"].items()}
    TIME["mediapipe"] = mpj["sec"]; RUN["mediapipe"] = f"{mpj['version']} ({mpj['api']} API)"
log(f"MediaPipe ran: {mp_ok} {RUN.get('mediapipe', '')}")
NAMES = list(DETS)
# Per frame: group boxes of all detectors (IoU >= 0.3). A face is "present" when >= 2 detectors found it.
recs = []
for r in jobs.itertuples():
    for i in r.indices:
        groups = []                                   # each group: {detector: box}
        for name in NAMES:
            for d in DETS[name][r.path].get(i, []):
                for g in groups:
                    if name not in g and max(F.iou(d.box, b) for b in g.values()) >= 0.3:
                        g[name] = d.box; break
                else:
                    groups.append({name: d.box})
        for g in groups:
            recs.append(dict(path=r.path, dataset=r.dataset, frame=i, n=len(g), **{n: n in g for n in NAMES}))
fr_df = pd.DataFrame(recs); fr_df.to_csv(OUT / "detector_faces.csv", index=False)
need = 2 if len(NAMES) == 3 else 1
present = fr_df[fr_df.n >= need]
rows_ = []
for name in NAMES:
    for ds, g in [("ALL", present)] + list(present.groupby("dataset")):
        rows_.append(dict(detector=name, dataset=ds, faces=len(g), miss_rate=float((~g[name]).mean()),
                          alone=int(((fr_df.n == 1) & fr_df[name] & ((fr_df.dataset == ds) | (ds == "ALL"))).sum()),
                          sec_per_video=float(np.mean(TIME[name]))))
det_tab = pd.DataFrame(rows_); det_tab.to_csv(OUT / "detector_comparison.csv", index=False)
print(det_tab.round(4).to_string(index=False))
allrow = det_tab[det_tab.dataset == "ALL"].set_index("detector")
best = allrow.miss_rate.idxmin()
close = allrow[allrow.miss_rate <= allrow.miss_rate.min() + 0.01]
winner = close.sec_per_video.idxmin() if len(close) > 1 else best
WIN = {"winner": winner, "why": "lowest miss rate" if len(close) == 1 else "miss rates within 1 point; fastest",
       "miss": allrow.miss_rate.round(4).to_dict(), "sec_per_video": allrow.sec_per_video.round(3).to_dict(),
       "alone": allrow.alone.to_dict(), "present_faces": int(len(present)), "detectors": NAMES,
       "complete": len(NAMES) == 3}
json.dump(WIN, open(OUT / "detector_decision.json", "w"), indent=2)
log(f"DETECTOR DECISION: {WIN}")
# contact sheet: frames where detectors disagree (present face missed by someone), boxes coloured per detector
COL = {"scrfd": (0, 255, 0), "yunet": (255, 160, 0), "mediapipe": (255, 0, 0)}
def sheet(tiles, path, cols=4, w=320):
    if not tiles: return
    tiles = [cv2.resize(t, (w, int(w * t.shape[0] / t.shape[1]))) for t in tiles]
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
    img = np.vstack([np.hstack(tiles[k:k + cols]) for k in range(0, len(tiles), cols)])
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
dis = present[~present[NAMES].all(axis=1)].drop_duplicates("path").head(12)
tiles = []
for d in dis.itertuples():
    j = jobs[jobs.path == d.path].iloc[0]
    fr = V.decode_frames(d.path, [d.frame]); fr = fix_width(fr, int(j.width)) if j.dataset == "FFPP" else fr
    f = fr[d.frame].copy()
    for name in NAMES:
        for b in DETS[name][d.path].get(d.frame, []):
            cv2.rectangle(f, (int(b.box[0]), int(b.box[1])), (int(b.box[2]), int(b.box[3])), COL[name], 3)
    tiles.append(f)
sheet(tiles, OUT / "figures/detector_disagreements.jpg")
log(f"Contact sheet: figures/detector_disagreements.jpg ({len(tiles)} frames; green SCRFD, orange YuNet, red MediaPipe)")
''')

code(r'''
# ===== CELL 7: Tracks — position-only vs identity-aware (D-019c), SCRFD detections =====
def main_track(dets, use_emb):
    if not use_emb:
        dets = {i: [F.Detection(d.box, d.score, d.kps) for d in ds] for i, ds in dets.items()}
    tr = F.link_tracks(dets)
    kept = F.filter_tracks(tr, N_FRAMES)
    return tr, kept, (max(kept, key=len) if kept else (max(tr, key=len) if tr else None))
trows, merge_suspects, multi = [], [], []
for r in jobs.itertuples():
    d = DETS["scrfd"][r.path]
    for mode in ("position", "identity"):
        tr, kept, m = main_track(d, mode == "identity")
        trows.append(dict(path=r.path, dataset=r.dataset, mode=mode, tracks=len(tr), kept=len(kept),
                          main=len(m) if m else 0))
        if mode == "identity":
            for t in kept:
                e = [x.emb for x in t.dets.values() if x.emb is not None]
                if len(e) > 1:
                    sims = np.array(e) @ np.array(e).T
                    if sims.min() < 0.2: merge_suspects.append((r.path, t.tid, float(sims.min())))
            if len(kept) >= 2 and r.dataset == "DFD": multi.append((r, kept))
tk = pd.DataFrame(trows); tk.to_csv(OUT / "tracks.csv", index=False)
summ = tk.groupby(["dataset", "mode"]).agg(videos=("path", "size"), main_median=("main", "median"),
                                          main_min=("main", "min"), share_ge30=("main", lambda x: (x >= 30).mean()),
                                          tracks_mean=("tracks", "mean"))
print(summ.round(3))
dfd_id = tk[(tk.dataset == "DFD") & (tk["mode"] == "identity")]
TRK = {"dfd_share_main_ge30": float((dfd_id.main >= 30).mean()), "dfd_main_median": float(dfd_id.main.median()),
       "dfd_share_main_ge30_position": float((tk[(tk.dataset == "DFD") & (tk["mode"] == "position")].main >= 30).mean()),
       "merge_suspects": len(merge_suspects), "multi_person_dfd_videos": len(multi)}
TRK["pass"] = TRK["dfd_share_main_ge30"] >= 0.90 and TRK["merge_suspects"] == 0
json.dump(TRK, open(OUT / "tracks.json", "w"), indent=2)
log(f"TRACKS: {TRK}")
# contact sheet: multi-person DFD videos, one row per track (4 crops spread over the track) -> check by eye
tiles = []
for r, kept in multi[:6]:
    fr = V.decode_frames(r.path, r.indices)
    for t in kept[:3]:
        idx = sorted(t.dets)
        for i in [idx[int(q * (len(idx) - 1))] for q in (0, 0.33, 0.66, 1.0)]:
            tiles.append(C.crop_square(fr[i], t.dets[i].box, out=160)[0])
sheet(tiles, OUT / "figures/identity_tracks_multi_person.jpg", cols=4, w=160)
log("Contact sheet: figures/identity_tracks_multi_person.jpg (each row = one track; a row must show one person)")
''')

code(r'''
# ===== CELL 8: DFD face selection from the official masks (D-019a) =====
MASK = {}
if HAVE_MASKS:
    dfd = inv[inv.dataset == "DFD"]; dfd_orig = dfd[dfd.method == "original"].set_index("vid")
    fakes = dfd[(dfd.method == "DeepFakeDetection") & (dfd.n_frames >= N_FRAMES)]
    pilot = fakes[fakes.original.isin(dfd_orig.index)].sample(min(PILOT, len(fakes)), random_state=SEED)
    rest = fakes.drop(pilot.index).sample(min(MASK_SAMPLE - len(pilot), len(fakes) - len(pilot)), random_state=SEED)
    sample = pd.concat([pilot.assign(pilot=True), rest.assign(pilot=False)])
    mrows, tiles = [], []
    for k, r in enumerate(sample.itertuples()):
        if r.vid not in masks:
            mrows.append(dict(fake=r.vid, pilot=r.pilot, flags="no_mask_file")); continue
        mi = V.probe(masks[r.vid])
        idx = V.sample_indices(int(r.n_frames), N_FRAMES)
        fr = V.decode_frames(r.path, idx)
        mk = {i: S.mask_from_rgb(m) for i, m in V.decode_frames(masks[r.vid], idx).items()}
        same_size = all(mk[i].shape == fr[i].shape[:2] for i in mk)
        if not same_size:   # mask stored at another resolution: scale it to the fake (nearest neighbour)
            mk = {i: cv2.resize(m.astype(np.uint8), fr[i].shape[1::-1], interpolation=cv2.INTER_NEAREST) > 0
                  for i, m in mk.items()}
        dets = {i: scrfd.detect(f) for i, f in fr.items()}
        I.embed_detections(fr, dets, arc)
        kept = F.filter_tracks(F.link_tracks(dets), N_FRAMES)
        sel = S.select_by_mask(mk, kept)
        row = dict(fake=r.vid, pilot=r.pilot, n_fake=int(r.n_frames), n_mask=mi.n_frames_header,
                   mask_wh=f"{mi.width}x{mi.height}", same_size=same_size, tracks=len(kept),
                   mask_frames=sum(int(m.any()) for m in mk.values()), share=sel.best_ratio if sel.tid is not None else 0.0,
                   flags=";".join(sel.flags))
        if r.pilot:   # cross-check with the difference method where the offset is clean
            o = dfd_orig.loc[r.original]
            off = A.find_offset(np.stack(list(V.iter_gray_small(r.path))), np.stack(list(V.iter_gray_small(o.path))))
            row.update(offset=off.offset, off_err=off.err, off_margin=off.margin)
            if off.err < 1 and off.margin >= 1.3:
                of = V.decode_frames(o.path, [i + off.offset for i in idx])
                of = {i: of[i + off.offset] for i in idx if i + off.offset in of}
                dsel = S.select_manipulated({i: fr[i] for i in of}, of, kept)
                row.update(clean=True, agree=(dsel.tid == sel.tid))
            else:
                row.update(clean=False)
        mrows.append(row)
        if len(tiles) < 24 and sel.tid is not None:
            trk = next(t for t in kept if t.tid == sel.tid)
            i = idx[16] if idx[16] in trk.dets else sorted(trk.dets)[0]
            f = fr[i].copy(); f[mk[i]] = (0.5 * f[mk[i]] + [128, 0, 0]).astype(np.uint8)
            b = trk.dets[i].box
            cv2.rectangle(f, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 4)
            tiles.append(f)
        if (k + 1) % 25 == 0: log(f"  masks {k+1}/{len(sample)}")
    mt = pd.DataFrame(mrows); mt.to_csv(OUT / "dfd_masks_check.csv", index=False)
    ok_rows = mt[mt["flags"].fillna("") != "no_mask_file"]
    clean = mt[mt.get("clean", pd.Series(False, index=mt.index)).fillna(False).astype(bool)]
    MASK = {"checked": len(mt), "mask_file_missing": int((mt["flags"] == "no_mask_file").sum()),
            "frame_count_equal": float((ok_rows.n_fake == ok_rows.n_mask).mean()),
            "same_size": float(ok_rows.same_size.mean()),
            "share_ge_0.5": float((ok_rows.share >= 0.5).mean()), "share_median": float(ok_rows.share.median()),
            "flags": ok_rows["flags"].fillna("").str.split(";").explode().replace("", np.nan).dropna().value_counts().to_dict(),
            "pilot_clean": int(len(clean)), "agree_on_clean": float(clean.agree.mean()) if len(clean) else None}
    MASK["pass"] = MASK["share_ge_0.5"] >= 0.95 and (MASK["agree_on_clean"] is None or MASK["agree_on_clean"] >= 0.95)
    json.dump(MASK, open(OUT / "dfd_masks.json", "w"), indent=2, default=str)
    sheet(tiles, OUT / "figures/dfd_mask_selection.jpg")
    log(f"MASKS: {MASK}")
    log("Contact sheet: figures/dfd_mask_selection.jpg (red = mask pixels, green = chosen face)")
''')

code(r'''
# ===== CELL 9: Report + zip =====
RUN["minutes"] = round((time.time() - T0) / 60, 1)
report = {"run": RUN, "detector": WIN, "tracks": TRK, "masks": MASK}
json.dump(report, open(OUT / "stageA2_report.json", "w"), indent=2, default=str)
ok = {"tests pass": True, "all three detectors compared": WIN["complete"],
      "identity tracks: DFD main >= 30/32 in 90%, no merges": TRK["pass"],
      "masks: >= 95% DFD fakes with a face on the mask, >= 95% agreement": bool(MASK.get("pass"))}
print("=" * 60); print("STAGE A2 SUMMARY — paste this back to Claude"); print("=" * 60)
for k, v in ok.items(): print(f"  [{'PASS' if v else 'CHECK'}] {k}")
print(json.dumps(report, indent=1, default=str))
zp = WORK / "p12a2_results.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob("*"):
        if p.is_file() and not p.name.startswith("dets_"): z.write(p, p.relative_to(OUT.parent))
log(f"Saved {zp} ({zp.stat().st_size/1e6:.1f} MB). Download it from the Output panel.")
''')

nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
for c in nb["cells"]:
    if c.cell_type == "code": c.source = c.source.replace("__COMMIT__", COMMIT)
nbf.write(nb, OUT); print("wrote", OUT, len(cells), "cells")
