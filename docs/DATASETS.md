# DATASETS.md — what each DeepGuard v2 dataset really contains

**Purpose:** one reference for how each dataset was made, what is inside it, and the traps that matter for training a detector, so no later step has to guess.
**Rules:** every fact names its source. "Measured" means we computed it ourselves, with the date. Anything not yet verified is marked **to verify** and is checked by the audit notebooks.

---

## 1. FaceForensics++ (FF++) and DeepFakeDetection (DFD)

### 1.1 What it is: videos, not photos
- **Format:** H.264 `.mp4` videos at three quality levels:
  - `raw` (c0): lossless;
  - `c23`: light compression, "HQ";
  - `c40`: heavy compression, "LQ".
- **No images are distributed.** The official README says: "We no longer offer the download of images as you can extract those from the lossless compressed c0 videos." Frames are always extracted by us.
- **Benchmark images are a different thing.** They are 1,000 images with hidden labels for the online leaderboard, and we don't use them.
- **Official folder layout:**

```
original_sequences/youtube/<c23|c40|raw>/videos/000.mp4 … 999.mp4                          (FF++ real)
original_sequences/actors/<c23|c40|raw>/videos/<actor>__<scene>.mp4                         (DFD real)
manipulated_sequences/<Deepfakes|Face2Face|FaceSwap|NeuralTextures|FaceShifter>/<c23|c40|raw>/videos/<target>_<source>.mp4
manipulated_sequences/DeepFakeDetection/<c23|c40|raw>/videos/<target>_<source>__<scene>__<8-char id>.mp4
manipulated_sequences/<method>/masks/videos/…   (binary masks of changed pixels; none for FaceShifter)
```

- **Official sizes:**
  - FF++ (all methods): c23 ≈ 10 GB, c40 ≈ 2 GB, raw ≈ 500 GB.
  - DFD originals: c23 ≈ 3 GB, c40 ≈ 400 MB.
  - DFD fakes: c23 ≈ 22 GB, c40 ≈ 3 GB.

### 1.2 How the 1,000 real videos were made
- **Source:** 1,000 sequences cut from **977 YouTube videos** (paper; dataset README).
  - Downloaded at ≥ 480p, found with tags such as "face", "newscaster" and "interview".
  - The **largest face** was tracked through each clip.
  - The first and last 10 frames were dropped, and every clip kept at least 280 frames.
  - Faces are mostly frontal and unoccluded, and the clips were screened by hand.
  - About 510 frames per video on average (509,914 frames in total).
- **Naming:** files are renamed `000`–`999`. `conversion_dict.json` maps each one to its YouTube ID and clip number.
- **No audio.** The README says: "We only downloaded the source video without audio." The DFD audio is not part of the dataset either.

### 1.3 How the fakes were made
- **Naming:** a fake is called `<target>_<source>`.
  - The **target** video supplies the frames, the background and the length.
  - The **source** video supplies the new face (swap) or the new expressions (reenactment).

| Method | Type | What changes | Whose identity is shown | Masks |
|---|---|---|---|---|
| Deepfakes | learning-based face swap (faceswap GitHub autoencoder, 64 px, Poisson blending in a rectangle) | whole face | **source** | yes (rectangular) |
| FaceSwap | graphics face swap (landmarks, 3D blendshape model, colour correction); **runs only until one of the two videos ends** | whole face | **source** | yes |
| Face2Face | graphics reenactment (76 blendshape coefficients) | expressions | **target** | yes |
| NeuralTextures | learned reenactment (neural texture + rendering network on a 1.7× face box) | mainly the **mouth** | **target** | tracking region only |
| FaceShifter | two-stage GAN face swap, made by its own authors (added Jul 2020) | whole face | **source** | **none** |
| DeepFakeDetection (DFD) | DeepFakes-style swaps on 363 videos of **28 paid, consenting actors in 16 scenes**; **3,068 fakes**. The 8-character ID exists because some actor pairs were recorded several times. | whole face | **source** | yes (taken straight after the swap) |

- **Only one face per fake video is manipulated: the largest tracked face.** Other people in the frame stay real. This confirms what Mayank saw on screen. Measured 15 Nov 2026:
  - 15.1% of FF++ fakes show 2 or more faces (news insets, two-anchor desks).
  - In 21/4,436 FF++ fakes, the manipulated face is **not** the largest. So "largest face" is not a safe rule; select the face by pixel difference.
  - DFD scenes have up to 4 people, with one face changed.
- **Compression:** the official `compress.py` runs `ffmpeg -c:v libx264 -crf 23` (or `-crf 40`) on real and fake videos alike. The paper calls this "quantization parameter", but the script uses CRF.
- **Frame rate (the "weird speed" Mayank noticed):**
  - `compress.py` contains the comment "We take the fps of the source video for manipulated videos".
  - So a fake is made from the **target's frames** but encoded at the **source's fps**, and plays faster or slower than the real target.
  - **Measured 15 Nov 2026 (H1 confirmed):** for Deepfakes, Face2Face, FaceSwap and NeuralTextures, the fake's fps always equals the source's. 984/1,000 pairs have equal fps, so only 16 fakes per method change speed: **64 c23 fakes (1.3%), from 0.50× to 2.00×**. FaceShifter keeps the target's fps.
- **Frame count (measured 15 Nov 2026):**
  - Deepfakes and FaceShifter: equal to the target.
  - FaceSwap and NeuralTextures: equal to `min(target, source)`.
  - **Face2Face: equal to the source**, so in 496 fakes it is longer than the target. What the extra frames contain is still **to verify**.
- **Width (measured; corrected 16 Nov 2026):** every Face2Face and NeuralTextures fake has width = the original's width **rounded down to a multiple of 32**. That changes 282/1,000 targets: exactly the originals whose width is not a multiple of 32 (854 → 832 ×98, 656 → 640 ×66, 600 → 576 ×48, 720 → 704 ×27, 654 → 640 ×20, …). It holds for 282/282 at both c23 and c40. The height is unchanged. **Measured 18 Nov 2026 (Phase 1.2 stage A): they were centre-cropped**, i.e. columns `d//2 … d//2 + floor32(w)` with `d = w − floor32(w)`. This holds in 480/480 test frames from 60 targets; rescaling fits 2–15× worse. The first audit note said "multiple of 16", which is true but incomplete: 720 is already a multiple of 16.
- **Pixel format (measured):** all 6,000 FF++ c23 and c40 videos (originals and fakes) are **yuv444p** (no chroma subsampling); all 3,431 DFD videos are yuv420p.
- **c23 vs c40 (measured):** width, height and frame count are identical for all 9,431 c23/c40 pairs.

### 1.4 Official splits
- **Measured 15 Nov 2026** from `splits/*.json`:
  - 720 / 140 / 140 videos = **360 / 70 / 70 pairs**;
  - both members of a pair are always in the same split;
  - every ID 000–999 appears exactly once.
- Each pair gives two fakes (`a_b` and `b_a`), so each method has **1,000 fakes**.
- **Leakage in the official split (measured 15 Nov 2026** from `conversion_dict.json`):
  - 22 YouTube videos gave more than one sequence.
  - For **7** of them, the sequences sit in **different splits**: the same person, scene and day appear on both sides.

| YouTube ID | Sequences and splits |
|---|---|
| nUNnFudF2pA | 605 train · 606 val |
| pGmDKODT6V4 | 639 train · 640 val |
| tR7z_H1qPMc | 755 **test** · 756 train |
| wERnQNuwQbM | 837 val · 838 train |
| xG9J__APvnY | 875 val · 876 train |
| yvf0HOswcPk | 954 train · 955 **test** |
| zPdvu-SUnbg | 982 val · 983 train |

- **People who recur across different YouTube videos** (for example the same newscaster on different days) cannot be seen in the metadata. Phase 1 finds them by clustering face embeddings.

### 1.5 Traps and the DeepGuard rule for each

| # | Trap | Why it matters | Rule |
|---|---|---|---|
| T1 | Several people in a fake video, only one manipulated | Cropping the wrong face labels a real face "fake" (label noise) | Crop the face whose pixels changed (diff against the original, or the official mask). In real videos any face is fine. At inference, score **every** face and report the most suspicious one. |
| T2 | Fake encoded at the source's fps | Anything measured in seconds or Hz differs between a fake and its real target. Temporal experts (E3 motion, rPPG pulse) could learn "playback speed" instead of "fake". | Sample frames **by index**, never by time. Never use fps or duration as a feature. Add random fps resampling to **both** classes in training (P5). Test every temporal expert with shuffled fps. |
| T3 | FaceSwap fakes can be shorter | Video length becomes a shortcut | Never use length. Pick frame windows inside `min(len_fake, len_original)`. |
| T4 | A fake shares its target's background and camera | The detector cannot use the background, which is good. But a target's real video and its fakes must never be split apart. | Split by pair/target ID (the official splits already do this). |
| T5 | Identity leakage (the 7 cross-split YouTube videos, plus recurring people) | Inflated test scores | Report the official split (for comparison with papers) **and** a clean split with these overlaps removed. Face-embedding clustering in Phase 1. |
| T6 | Reenactment keeps the target's identity; NeuralTextures changes mostly the mouth | Single frames carry weak evidence | Use several frames per face track (E1 temporal pooling) and include mouth-region detail. |
| T7 | Every FF++ video is H.264 at CRF 23/40 | Other sources use other codecs, so compression could become a dataset-level shortcut | Apply the same social-media degradation chain to both classes (D-005.4) and run the metadata shortcut audit. |
| T8 | Third-party copies may be incomplete, re-encoded or renamed | Silent errors | Audit every file (`dg_p1_ffpp_audit.ipynb`) and keep a SHA-256 manifest. |
| T9 | No audio | Nothing to learn from sound here | FF++ is visual-only training data. |
| T10 | Face2Face and NeuralTextures fakes have their width rounded down to a multiple of 32 (282 targets) | Resolution or aspect ratio reveals the fake (metadata AUC 0.69–0.77) | Give every video the same width treatment, real and fake (D-017). |
| T11 | DFD fakes are trimmed sub-clips with an unknown start offset; 31 are under 50 frames. **Measured 18 Nov 2026: only 25/50 sampled fakes match their original frame-exactly at any offset**; 9/50 match no offset (walking, pan and two-actor scenes), so frame differencing cannot select the face reliably for DFD | Comparing frames at the same index fails, so the wrong face could be picked | Find each fake's offset in its original before face selection, or use the official DFD masks. Drop fakes shorter than the frame window. |
| T12 | DFD two-actor scenes: the same original file is filed under both actors (7 videos × 2 compression levels) | Identity leakage and duplicate counting | Deduplicate by SHA-256. A two-actor video belongs to **both** actors in identity splits. |
| T13 | Face2Face fakes follow the source's length (496 are longer than their target) | Frames past the target's end have no matching original | Use only frames inside `min(target, source)` until checked. |

### 1.6 Our copy
- **v1:** used an unofficial FF++ c23 copy downloaded from Hugging Face, not the official script (stated by Mayank, 15 Nov 2026).
- **v2:** official access **approved on 15 Nov 2026** (D-015). The Kaggle copy is byte-identical to the Drive download (18,862/18,862 SHA-256 matches). It is mounted at `/kaggle/input/datasets/mynkey/<dataset>/…`.
  - The official c23 + c40 videos are archived on Drive (`DeepGuard/raw/ffpp/`) with a SHA-256 manifest.
  - The official copy is the **only** FF++ source. It is uploaded to private Kaggle datasets for processing. The third-party Kaggle copy is not used.
- **Audit:** `dg_p1_ffpp_audit.ipynb` checks whichever copy is used. Its numbers go to OUTPUTS.md.

---

## 2. DFDC: Deepfake Detection Challenge (checked 15 Nov 2026 on Mayank's Kaggle account)
- **Status:** a closed Kaggle code competition. Mayank has accepted the rules; the page now shows "Late Submission".
- **Notebook input:** 4.44 GB, 802 files:
  - `train_sample_videos/`: 400 videos + `metadata.json`;
  - `test_videos/`: 400 videos **with no labels** (`sample_submission.csv` holds only a placeholder `label` = 0);
  - `sample_submission.csv`.
- **Full training set:**
  - Not mounted in notebooks. The Data page says it is "not available directly on Kaggle".
  - Download links appear after accepting the rules: `all.zip` (471.84 GB), or **50 chunks `00.zip`–`49.zip`** of 9.03–11.52 GB each.
  - Each chunk holds `.mp4` files plus a `metadata.json` with the columns `filename`, `label` (REAL/FAKE), `original` (the real video a fake was made from) and `split` (always "train").
- **How a chunk downloads (checked 16 Nov 2026):**
  - The link on the Data page (`kaggle.com/c/16880/datadownload/dfdc_train_part_NN.zip`) only works when logged in.
  - It redirects to a **signed Google Cloud Storage URL** (`storage.googleapis.com/kaggle-competitions-detached-data/16880/…?Expires=…&Signature=…`) that is valid for about **39 hours**.
  - A Kaggle notebook can fetch a chunk from such a URL. The URLs are personal credentials while valid: never commit or share them (CI blocks them).
- **A FAKE can be a face swap, a voice swap, or both** (Data page).
  - A voice-only fake looks completely real, so for a visual model that label is wrong.
  - Rule: compare each fake with its `original` frame by frame. If the frames are unchanged, it is a voice-only fake. Exclude it from visual training and keep it for the v2.1 voice expert.
- **Identities:** 486 subjects in the training set (DFDC paper). They must be clustered and split by identity.
- **Labelled test set:** not on Kaggle. Meta's DFDC page offers the full dataset via AWS, a possible later addition. **To verify.**

---

## 3. Sources
- FF++ dataset README, `compress.py`, `splits/*.json`, `conversion_dict.json`: github.com/ondyari/FaceForensics (dataset folder), read 15 Nov 2026.
- Rössler et al., FaceForensics++ (ICCV 2019): arxiv.org/abs/1901.08971.
- Google DFD announcement: ai.googleblog.com/2019/09/contributing-data-to-deepfake-detection.html.
- DFDC Kaggle Data page (Mayank's screenshots, 15 Nov 2026); Dolhansky et al., The DFDC Dataset: arxiv.org/abs/2006.07397.
- Actor-frequency leakage in FF++ splits: arxiv.org/abs/2412.05676.
