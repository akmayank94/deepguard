"""Builds dg_p0_dfd_masks.ipynb — Colab: official DFD masks -> Drive (+SHA-256) -> private Kaggle dataset dg-dfd-masks.
D-019a: DFD face selection uses the official masks."""
import sys

import nbformat as nbf

OUT = sys.argv[1]
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# DeepGuard v2 · Official DeepFakeDetection masks → Drive → Kaggle (D-019a)

DFD fakes are trimmed sub-clips, and half of them cannot be matched frame-exactly to their original
(Phase 1.2 stage A). The official **masks** mark the swapped pixels in every fake directly, so they tell us
which face was changed without any time alignment.

| Cell | What it does |
|---|---|
| 1 | Mount Drive, Kaggle login |
| 2 | Get the official download script (secret `FFPP_SCRIPT_URL`), show its options |
| 3 | Download the DFD masks into `DeepGuard/raw/ffpp/` (resumable) |
| 4 | Check: one mask per fake (3,068), same size and frame count as the fake, binary pixels |
| 5 | SHA-256 manifest of the masks |
| 6 | Upload to the private Kaggle dataset `mynkey/dg-dfd-masks` and wait until it is ready |

**Before running:** Colab runtime **CPU**; secrets **`FFPP_SCRIPT_URL`** and **`KAGGLE_API_TOKEN`** with *Notebook access* on
(both already exist from Phase 0). Keep the tab open; if it disconnects, **Run all** again: finished work is skipped.
""")

code(r'''
# ===== CELL 1: Setup — Drive, Kaggle CLI, login check =====
import os, sys, re, json, time, shutil, zipfile, hashlib, random, subprocess, urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

T0 = time.time()
def log(msg):
    """Print with elapsed time so progress is visible."""
    m, s = divmod(int(time.time() - T0), 60)
    print(f"[{m:3d}m{s:02d}s] {msg}", flush=True)

LOCAL_TEST = os.environ.get("DG_LOCAL_TEST") == "1"     # only for offline testing of this notebook
if not LOCAL_TEST:
    from google.colab import drive, userdata
    drive.mount("/content/drive")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle>=1.8.0"], check=True)
    os.environ["KAGGLE_API_TOKEN"] = userdata.get("KAGGLE_API_TOKEN").strip()   # never printed
DRIVE = Path(os.environ.get("DG_DRIVE", "/content/drive/MyDrive"))
RAW = DRIVE / "DeepGuard" / "raw" / "ffpp"               # same folder as the videos (official layout)
STAGE = Path(os.environ.get("DG_STAGE", "/content/stage"))
SCRIPT = Path("/content/ffpp_download.py") if not LOCAL_TEST else Path(os.environ["DG_SCRIPT_PATH"])
STATUS = RAW / "_masks_status.json"
OWNER, SLUG, TITLE = "mynkey", "dg-dfd-masks", "DFD official masks (DeepGuard)"
assert 6 <= len(TITLE) <= 50
EXPECTED = int(os.environ.get("DG_EXPECTED", "3068"))
assert RAW.exists(), f"{RAW} not found: is this the same Google account as Phase 0?"

def kaggle(*args, timeout=None):
    """Run the Kaggle CLI and return (exit code, output text)."""
    r = subprocess.run(["kaggle", *args], capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout + r.stderr).strip()
rc, out = kaggle("datasets", "list", "--mine")
assert rc == 0, f"Kaggle login failed — check the KAGGLE_API_TOKEN secret.\n{out[-500:]}"
log(f"Drive: {RAW} | Kaggle login OK")
''')

code(r'''
# ===== CELL 2: The official download script and its options =====
if not LOCAL_TEST:
    url = userdata.get("FFPP_SCRIPT_URL").strip()
    urllib.request.urlretrieve(url, SCRIPT)
help_txt = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, input="\n").stdout
print(help_txt)
assert "masks" in help_txt, "The script does not list a 'masks' type: send Claude the help text above."
log("Script offers masks.")
''')

code(r'''
# ===== CELL 3: Download the DFD masks (resumable; the script skips files that already exist) =====
POLL = int(os.environ.get("DG_POLL", "30"))
DFD_ROOT = RAW / "manipulated_sequences" / "DeepFakeDetection"
def mask_files():
    """Finished mask videos, wherever the script puts them under .../DeepFakeDetection/*masks*/."""
    out = []
    for f in DFD_ROOT.rglob("*.mp4"):
        if "masks" in f.relative_to(DFD_ROOT).parts[0]:
            try:
                out.append((f, f.stat().st_size))
            except FileNotFoundError:
                pass
    return out
subprocess.run(["pkill", "-f", str(SCRIPT)], capture_output=True); time.sleep(1)
for f in DFD_ROOT.rglob("tmp*"):
    if "masks" in str(f): f.unlink(missing_ok=True)           # half-written files from a crashed run
status = json.load(open(STATUS)) if STATUS.exists() else {}
if not status.get("download"):
    cmd = [sys.executable, str(SCRIPT), str(RAW), "-d", "DeepFakeDetection", "-t", "masks", "-c", "c23",
           "--server", "EU2"]
    logf = open(RAW / "_log_dfd_masks.txt", "a")
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=logf, stderr=subprocess.STDOUT, text=True)
    p.stdin.write("\n\n"); p.stdin.close()                   # accepts the terms prompt
    t0 = time.time()
    while p.poll() is None:
        time.sleep(POLL)
        fs = mask_files(); gb = sum(s for _, s in fs) / 1e9
        log(f"   masks: {len(fs):5d}/{EXPECTED} files, {gb:.2f} GB, {gb*1000/max(time.time()-t0,1):.1f} MB/s")
    logf.close()
    n = len(mask_files())
    status["download"] = p.returncode == 0 and n >= EXPECTED
    json.dump(status, open(STATUS, "w"))
    print(open(RAW / "_log_dfd_masks.txt").read()[-400:])
fs = mask_files()
log(f"Masks on Drive: {len(fs)} files, {sum(s for _, s in fs)/1e9:.2f} GB; folder: "
    f"{sorted({str(f.parent.relative_to(RAW)) for f, _ in fs})}")
assert len(fs) >= EXPECTED, "Fewer masks than expected: re-run this cell (it resumes)."
''')

code(r'''
# ===== CELL 4: Check the masks against the c23 fakes =====
import cv2, numpy as np
fake_dir = DFD_ROOT / "c23" / "videos"
fakes = {f.stem: f for f in fake_dir.glob("*.mp4")}
masks = {f.stem: f for f, _ in mask_files()}
missing, extra = sorted(set(fakes) - set(masks)), sorted(set(masks) - set(fakes))
log(f"c23 fakes: {len(fakes)} | masks: {len(masks)} | fakes without mask: {len(missing)} | masks without fake: {len(extra)}")
if missing: print("   e.g. missing:", missing[:5])
def info(path, n_read=0):
    cap = cv2.VideoCapture(str(path))
    w, h, n = int(cap.get(3)), int(cap.get(4)), int(cap.get(7))
    vals = []
    for _ in range(n_read):
        ok, fr = cap.read()
        if not ok: break
        vals.append(fr[..., 0])
    cap.release()
    return w, h, n, vals
rng = random.Random(2026)
sample = rng.sample(sorted(set(fakes) & set(masks)), min(40, len(set(fakes) & set(masks))))
rows = []
for s in sample:
    fw, fh, fn, _ = info(fakes[s]); mw, mh, mn, vals = info(masks[s], 30)
    v = np.concatenate([x.ravel() for x in vals]) if vals else np.zeros(1)
    rows.append(dict(video=s, fake_wh=f"{fw}x{fh}", mask_wh=f"{mw}x{mh}", fake_frames=fn, mask_frames=mn,
                     same_size=(fw, fh) == (mw, mh), same_frames=fn == mn,
                     binary_share=float(((v < 30) | (v > 225)).mean()), white_share=float((v > 127).mean())))
chk = pd.DataFrame(rows); chk.to_csv(RAW / "_masks_check.csv", index=False)
print(chk.head(10).to_string(index=False))
log(f"Sample of {len(chk)}: same size {chk.same_size.mean():.0%}, same frame count {chk.same_frames.mean():.0%}, "
    f"binary pixels {chk.binary_share.median():.1%} (median), white area {chk.white_share.median():.1%} (median)")
''')

code(r'''
# ===== CELL 5: SHA-256 manifest of the masks =====
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""): h.update(b)
    return h.hexdigest()
man = RAW / "MANIFEST_masks.csv"
files = sorted(f for f, _ in mask_files())
with ThreadPoolExecutor(8) as ex:
    hashes = list(ex.map(sha256, files))
mf = pd.DataFrame({"rel": [str(f.relative_to(RAW)) for f in files], "bytes": [f.stat().st_size for f in files],
                   "sha256": hashes})
mf.to_csv(man, index=False)
log(f"{man.name}: {len(mf)} files, {mf.bytes.sum()/1e9:.2f} GB")
''')

code(r'''
# ===== CELL 6: Upload to the private Kaggle dataset and wait until it is ready =====
ref = f"{OWNER}/{SLUG}"
rc, out = kaggle("datasets", "status", ref)
if rc == 0 and "ready" in out.lower():
    log(f"{SLUG}: already on Kaggle → skipped")
else:
    out_dir = STAGE / SLUG; shutil.rmtree(out_dir, ignore_errors=True); out_dir.mkdir(parents=True)
    zpath = out_dir / "dg_dfd_masks.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for f in files: z.write(f, str(f.relative_to(RAW)))       # keeps the official layout
        z.write(man, man.name)
    meta = {"title": TITLE, "id": ref, "licenses": [{"name": "other"}],
            "subtitle": "Official FaceForensics++ DFD masks. Private; not for redistribution.",
            "description": ("Official DeepFakeDetection masks downloaded with the FaceForensics authors' script "
                            "(access approved 15 Nov 2026), used under the FaceForensics Terms of Use for the "
                            "non-commercial DeepGuard project. PRIVATE — do not share or redistribute.")}
    json.dump(meta, open(out_dir / "dataset-metadata.json", "w"), indent=1)
    log(f"Uploading {zpath.stat().st_size/1e9:.2f} GB …")
    rc, out = kaggle("datasets", "create", "-p", str(out_dir), "-q")
    print("   " + "\n   ".join(out.splitlines()[-4:]))
    assert rc == 0, "Upload failed: send Claude the lines above."
deadline = time.time() + 2 * 3600
while time.time() < deadline:
    rc, out = kaggle("datasets", "status", ref)
    state = out.strip().splitlines()[-1] if out else "?"
    if rc == 0 and "ready" in state.lower():
        break
    print(f"      {SLUG}: {state}", flush=True); time.sleep(int(os.environ.get("DG_POLL", "60")))
log(f"{SLUG}: {state} → https://www.kaggle.com/datasets/{ref}")
''')

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}
nbf.write(nb, OUT)
print("wrote", OUT, len(cells), "cells")
