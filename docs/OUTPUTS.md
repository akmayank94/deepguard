# OUTPUTS.md — DeepGuard v2: measured results, run by run

**Rules:** only numbers produced by a notebook run go here, with the run, the date and where the files are. Interpretation and decisions go in DECISIONS.md.

---

## Phase 0 — Data acquisition

### 0.1 Official FF++ / DFD download (Colab `dg_p0_ffpp_official_fetch.ipynb`, 15 Nov 2026)
- **Run:** 16/16 parts OK, about 2.5 h (EU2 server, 8 parallel parts, 5–11 MB/s).
- **Result:** 18,862 videos, 39.97 GB, at `Drive/DeepGuard/raw/ffpp/`, with `MANIFEST.csv` (SHA-256 per file).

| Set | c23 | c40 |
|---|---|---|
| FF++ originals | 1,000 files, 1.94 GB | 1,000 files, 0.26 GB |
| FF++ fakes (5 methods) | 5,000 files, 9.03 GB | 5,000 files, 1.19 GB |
| DFD originals | 363 files, 2.90 GB | 363 files, 0.41 GB |
| DFD fakes | 3,068 files, 21.27 GB | 3,068 files, 2.98 GB |

### 0.2 Upload to Kaggle (Colab `dg_p0_ffpp_to_kaggle.ipynb`, 15 Nov 2026)
- **Datasets:** four private Kaggle datasets: `mynkey/dg-ffpp-c40`, `dg-ffpp-c23`, `dg-dfd-c40`, `dg-dfd-c23`.
- **Mount path in notebooks:** `/kaggle/input/datasets/mynkey/<dataset>/original_sequences/…`. Kaggle drops the zip-name folder.
- **Integrity:** the audit's SHA-256 manifest matches the Drive manifest **18,862 / 18,862 files** (bytes and hashes), with 0 missing and 0 extra. The Kaggle copy is byte-identical to the official download.

### 0.3 Celeb-DF v2 (Colab `dg_p0_freeze_v1_and_celebdf.ipynb`, Part A, 16 Nov 2026)
- **Source:** the official `Celeb-DF-v2.zip`, 9,952,957,051 bytes, SHA-256 `b22f9d74d4b67bba698677fa2a05f84b26d8250258424a97c940ba94cb126fa2`.
- **Contents:** 6,530 files: Celeb-real 590 (0.95 GB), Celeb-synthesis 5,639 (8.51 GB), YouTube-real 300 (0.70 GB), `List_of_testing_videos.txt`. These match the paper.
- **Stored at:** `Drive/DeepGuard/raw/celebdf/`, with `celebdf_zip_listing.csv` (path, size, CRC32 per file), and the private Kaggle dataset `mynkey/dg-celebdf-v2`.

### 0.4 v1 freeze (D-001; same notebook, Part B, 16 Nov 2026)
- **Kaggle snapshot:** private dataset `mynkey/deepguard-v1-frozen`, 319 files, 1.44 GB:
  - 24 checkpoints;
  - results JSONs, figures and the full code of the original project folder;
  - `MANIFEST.csv` (SHA-256), plus a copy in `Drive/DeepGuard/frozen/v1/`.
- **Live Space** `mynkey/deepguard` at commit `a727641a95911cc596045122ae9a3ce7afa76ac0`:
  - its `model.pt` SHA-256 is `9026cadc…bfc1390`;
  - it is **identical** to `deploy/model.pt` **and to `checkpoints/run_v2_full_b3/checkpoint_best.pt`**.
- **The official v1 model is `run_v2_full_b3`, epoch 52.** Its own results files (`run_v2_full_b3/test_results.json` and `celebdf_official_results_FIXED.json`) give:
  - FF++ validation AUC 0.9158;
  - FF++ test AUC **0.8781**, accuracy 0.823, F1 0.797;
  - per method: Deepfakes 0.913, FaceSwap 0.797, NeuralTextures 0.789, FaceShifter 0.927, DeepFakeDetection 0.966. **Face2Face is missing from the per-method results.**
  - Celeb-DF v2 official test (516 clips: 177 real, 339 fake): clip AUC **0.7238** (the report says 0.7272), frame AUC 0.690.
- **GitHub:** `akmayank94/deepguard`, tag `v1.0` = commit `a1ea01c` (v1 code, the first demo, the Space app).

---

## Phase 1.1 — FF++ / DFD audit (Kaggle `dg_p1_ffpp_audit`, version run 15 Nov 2026)
Files: `ffpp_audit_results.zip`, which contains the report, CSVs and `figures/`.

### A1–A3 Inventory, completeness, integrity
- **Inventory:** 18,862 videos and 0 frame folders. The copy holds **videos only**. 0 unparsed names, 0 unreadable files, 0 header/counted frame-count mismatches.
- **FF++ completeness:** 1,000/1,000 originals and 1,000/1,000 fakes for each of the 5 methods, at both c23 and c40. Names match the official pairs exactly, and c23 and c40 hold the same names.
- **DFD:** 363 originals (28 actors, 16 scenes) and 3,068 fakes; every fake's original is present.
- **Byte-identical duplicates: 14 groups.** All are DFD originals: 7 two-actor scene videos filed under both actors' IDs, at both c23 and c40. Examples: `12__/13__walking_down_indoor_hall_disgust`, `20__/21__walking_and_outside_surprised`.
- **Official-split leakage:** 7 YouTube videos with sequences in different splits: 605/606, 639/640, 755/756, 837/838, 875/876, 954/955, 982/983.

### Metadata (c23; c40 is identical apart from bitrate)

| Set | Videos | Frames (median) | fps | Resolutions | Audio |
|---|---|---|---|---|---|
| FF++ originals | 1,000 | 457.5 (range 287–1,814) | 30 (480), 25 (479), 24 (25), 60 (7), 15 (4), 29 (2), 50 (2), 18 (1) | 29 | none |
| DFD originals | 363 | 850 | 24 only | 1 | none |
| DFD fakes | 3,068 | 724.5 (range 5–1,620) | 24 only | 1 | none |

- **Median bitrate:** c23 ≈ 640 kbps, c40 ≈ 85 kbps (FF++); DFD c23 ≈ 1.5 Mbps, c40 ≈ 0.19 Mbps.
- **No variable-frame-rate files.**

### A4 Frame rate of fakes (H1)
- **Rule:** for Deepfakes, Face2Face, FaceSwap and NeuralTextures, a fake's fps **always equals the source video's fps**.
  - 984/1,000 pairs have target fps = source fps, so the speed changes in only **16 fakes per method** (8 pairs × 2 directions).
  - **FaceShifter instead keeps the target's fps** (16 "= target only", 0 "= source only").
- **Result:** **64 c23 fakes (1.3%) play at a different speed than their real target, from 0.50× to 2.00×.** For example:
- Face2Face 964_174 plays at **2.00×**: a 30 fps target encoded at the source's 60 fps.
- Deepfakes 174_964 plays at **0.50×**: a 60 fps target encoded at 30 fps.

### A5 Fake length (H2), measured against the target and source originals (c23)

| Method | Length of the fake |
|---|---|
| Deepfakes | = target (1,000/1,000) |
| FaceShifter | = target (1,000/1,000) |
| FaceSwap | = min(target, source) (1,000/1,000) |
| NeuralTextures | = min(target, source) (999/1,000; 1 shorter) |
| Face2Face | = **source** (1,000/1,000); **longer than its target in 496 fakes** |

DFD fakes are **trimmed sub-clips** of their originals:
- only 311/3,068 have the original's full length;
- the length ratio has a median of 0.916 and ranges from 0.005 to 1.0;
- **31 fakes have fewer than 50 frames** (5–41 frames), all from the `secret_conversation` scene.

### A6 Metadata-only shortcut AUC (pair-grouped 5-fold CV)

| Fakes vs real | c23 | c40 | Driver |
|---|---|---|---|
| All methods | 0.549 | 0.584 | width |
| Deepfakes | 0.505 | 0.506 | — |
| **Face2Face** | **0.690** | **0.719** | **width** |
| **FaceSwap** | **0.613** | **0.617** | length (n_frames, duration) |
| **NeuralTextures** | **0.749** | **0.766** | **width** |
| FaceShifter | 0.491 | 0.511 | — |

**Cause of the width shortcut:**
- In **282/1,000** Face2Face and NeuralTextures fakes, the width is rounded down to a **multiple of 32** (for example 854 → 832, 656 → 640, 720 → 704). *Corrected 16 Nov 2026 by re-reading `meta.csv`: the fake width equals floor32 of the original in 282/282 cases at both c23 and c40. The first note said "multiple of 16".*
- **Also measured from `meta.csv`:** all FF++ videos are yuv444p and all DFD videos yuv420p; c23 and c40 have identical width, height and frame count for all 9,431 pairs.
- 100% of Face2Face/NeuralTextures fake widths are multiples of 32, against 71.8% of the originals (81.7% are multiples of 16).
- The height is never changed.
- Whether the frames were cropped or rescaled is **not yet known**.

### A7 Visual check (c23; all FF++ fakes except the 282+282 width-changed ones; 600 DFD fakes)

| Method | Checked | Changed pixels (median % of frame) | Background diff (median) | ≥ 2 faces | Manipulated face not largest | "Unchanged" flag | Misaligned |
|---|---|---|---|---|---|---|---|
| Deepfakes | 1,000 | 0.98 | 2.0 | 152 | 3 | 7 | 0 |
| Face2Face | 718 | 0.29 | 2.0 | 116 | 4 | 18 | 0 |
| FaceSwap | 1,000 | 0.75 | 2.0 | 151 | 2 | 4 | 0 |
| NeuralTextures | 718 | 0.20 | 2.0 | 111 | 4 | 29 | 0 |
| FaceShifter | 1,000 | 0.51 | 2.0 | 142 | 8 | 1 | 0 |
| DeepFakeDetection | 600 | 10.82 | 2.0 | 166 | 31 | 103 | 40 |

- **FF++ fakes are frame-aligned with their targets.** Background diff is 2.0 (compression noise only) and 0 are misaligned. **15.1%** of FF++ fakes show 2 or more faces. In **21/4,436** the manipulated face is not the largest.
- **Contact sheets** (`figures/flag_*.jpg`):
  - multi-face cases are real: news insets, two-anchor desks, and DFD meeting scenes with up to 4 people, where only one face is changed;
  - some second "faces" are false detections (logos, toys);
  - the "unchanged" cases are **real but small manipulations** (mouth-only NeuralTextures, small faces), not broken files.
- **DFD:** 60.2% of fakes change more than 5% of the frame at the same frame index. The contact sheets show this is a **time offset** (the person sits elsewhere in the scene), not a large manipulation.

### Phase 1.1 acceptance (D-016)
A1–A7 are all reported. The flags have decisions proposed in D-017, waiting for Mayank's confirmation.
