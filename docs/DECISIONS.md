# DECISIONS.md — DeepGuard v2: a deepfake and AI-media detector that works on real uploads

**Owner:** Mayank · **Started:** 14 Nov 2026 · **Previous version:** v1 (B.Sc. project, ICET-AICS 2026), live at huggingface.co/spaces/mynkey/deepguard
**Project type:** personal project, no supervisor, no submission deadline (decided 14 Nov 2026). Each phase takes as long as its acceptance test needs, and no longer.
**Purpose of this file:** a factual record of every design decision: what was decided, why, what evidence supports it, and its status. Nothing goes in §0 unless it was measured, run, or checked against a cited source. Same rules as the NIDS project.

Status labels: **DONE** (implemented and verified) · **DECIDED** (agreed, not yet implemented) · **SUPERSEDED** (replaced, with the reason) · **PROPOSED** (direction suggested, not yet agreed) · **PENDING** (waiting on evidence or on access)

---

## 0. Verified facts

### 0.1 What v1 is (from the code and the report)
- Five-branch detector, 22.18 M parameters:
  - A: EfficientNet-B3 (ImageNet) on aligned face crops;
  - B: EfficientNet-B0 (from scratch) on an inter-channel high-frequency "diff image";
  - C: MLP on a 9,369-d "contourlet" vector;
  - D: MLP on 30 PRNU statistics;
  - E: MLP on 32 rPPG statistics.
  - Fusion: softmax attention gate.
- **Data:** FaceForensics++ C23 only, taken from an unofficial copy on Hugging Face, not the official download script (stated by Mayank, 15 Nov 2026). 1,922 clips (977 real / 945 fake), 20 frames each, video-level split 70/15/15 (1,383 / 264 / 275 clips). `preprocess_final.py` covers Deepfakes, Face2Face, FaceShifter, FaceSwap, NeuralTextures; DeepFakeDetection was preprocessed separately (`preprocess_dfd.py`).
- **Training actually used** `lr_backbone 3e-5`, `lr_heads 3e-4` (`main_v2.py`). The report says 5e-5 / 5e-4.
- **Results (report):**
  - FF++ validation AUC 0.9158 (used for checkpoint selection);
  - FF++ test AUC 0.8781;
  - Celeb-DF v2 official test AUC 0.7272, with the F1 threshold tuned to 0.20.

### 0.2 Why v1 under-delivers (measured on 14 Nov 2026)
1. **Branch C never computed a contourlet transform.** `features/extractor.py` receives a `ContourletTransform` and stores it as `self.ct = ct  # kept for API compat`, then never calls it. The features come from `energy.py` / `dccm.py`: one level of separable PKVA low/high filtering into LH, HL and HH bands (horizontal, vertical, diagonal). That is a one-level wavelet-style decomposition, with no Laplacian pyramid and no directional filter bank.
2. **9,261 of the 9,369 features are raw block-energy differences tied to fixed image positions** (8/16/32-pixel blocks on a 224 crop). On a test clip, |value| has p50 = 3, p90 = 28, p99 = 124 and max = 1,207. The other 108 (DCCM) are tiny (|value| ≤ 0.01). The ±500 clip saturates 0% of values, so clipping was not the problem. A position-specific, heavy-tailed 9k-d vector feeding an MLP on 1,383 training clips is a memorisation risk. It matches the gate weight falling from 0.24 to 0.02 and the Spectral+CT-only ablation at AUC ≈ 0.53 (chance).
3. **The branch ablation is not statistically resolved.** With 125 fake and 150 real test clips, AUC 0.878 has standard error ≈ 0.022 (Hanley–McNeil), i.e. a 95% interval of about 0.835–0.921. The full model beats RGB-only by +0.012 test AUC, which is inside that noise.
4. **The gate weight does not show contribution.** Branch B had gate weight 0.50, yet B+C alone scored 0.53 AUC. Branch A (pretrained B3) carries most of the signal (RGB-only test AUC 0.8658).
5. **Memorisation:** training AUC reached 1.000 by epoch 28, against validation 0.916.
6. **PRNU and rPPG are per-clip constants** copied onto all 20 frame rows. rPPG uses 64 frames (≈ 5 s at 12.5 Hz) of C23-compressed video. The report itself says the pulse signal is too weak there (§5.4).
7. **Data age:** every fake comes from 2016–2020 face-swap/reenactment methods. There is no full-frame AI video (Sora/Veo/Kling), no AI images, no lip-sync and no audio. So a modern AI clip falls outside everything v1 has seen, which explains the confident "real" verdicts.
8. **The deployed v1 app did not run the evaluated pipeline.** The old Streamlit app fed zeros to C, D and E, used an unaligned crop and a different diff-image formula. This was fixed in the Hugging Face Space on 14 Nov 2026: the inference pipeline matches the training code with 0.0 difference on every input.
9. **Report-level corrections needed:**
   - the comparison table uses the validation AUC (0.9158) against other methods' test AUCs (use 0.8781);
   - the Celeb-DF threshold was tuned on a subset of the reported test data;
   - Face2Face is printed as "-0.91";
   - §3.9 promises a manipulation-type estimate the model does not produce.

### 0.3 What the field shows now (sources in §5)
- **Lab scores do not survive real uploads.** On Deepfake-Eval-2024 (in-the-wild media from 2024), open-source detectors lose 45–50% AUC. Typical in-the-wild video AUC is ≈ 0.63 (open source) vs ≈ 0.79 (best commercial). A continuously retrained commercial system reached 0.822 video AUC, but only 24% video recall at 1% FPR (BitMind, Jul 2026).
- **Data diversity beats architecture.** Commercial detectors lead mainly through more representative training data, and open-source models close the gap when fine-tuned on in-the-wild data. Community Forensics (CVPR 2025, 2.7 M images from 4,803 generators) shows detection improves as the number of training generators grows.
- **Adapted foundation encoders are the current best base:**
  - Effort (ICML 2025): SVD orthogonal subspace, ≈ 0.19 M trainable parameters, mean AUC 0.917 when trained on FF++ and tested on 7 unseen datasets.
  - GenD (WACV 2026): LayerNorm-tuned CLIP/DINO/Perception Encoder, evaluated across 14 datasets (2019–2025).
  - CLIP + DINOv2 fusion: cross-domain mAUC 92.15, vs 88.86 for CLIP alone and 87.41 for DINOv2 alone (UCF-Net, Sep 2026).
  - DINOv2-L + LoRA beat CLIP-L + LoRA in the NTIRE 2026 robust-deepfake challenge.
  - DF40 (NeurIPS 2024) found CLIP the most generalisable baseline across 40 forgery methods.
- **AI-generated video:**
  - ReStraV (NeurIPS 2025): curvature and step distance of DINOv2 frame-embedding trajectories, with a light classifier. 97.17% accuracy / 98.63% AUROC on VidProM, ≈ 48 ms per video.
  - A 2026 benchmark on Sora-2, Veo-3.1, Kling and Runway found off-the-shelf detectors at ≤ 72% accuracy, and CLIP ViT-L/14 at AUC 0.63.
  - **Datasets carry resolution, fps and bitrate shortcuts** (VidGuard-R1). Benchmarks such as GenVidBench (AAAI 2026, 6 M videos), RobustSora (Sora 2) and CCDF (Veo 3.1 / Sora 2 / Grok) exist.
- **Provenance:**
  - C2PA content credentials can be read and verified in Python (`c2pa-python`).
  - Google's SynthID Detector has been public since 7 Oct 2026 (images, video, audio from Google, OpenAI, NVIDIA and Kakao), but has **no public API**, so a product can only link to it.
- **Audio:** open XLS-R 300M + SLS anti-spoofing models with released checkpoints exist; In-the-Wild audio is the hard benchmark.

### 0.4 Compute and storage facts
- **Kaggle:**
  - ≈ 30 GPU-hours/week (P100 or 2×T4), 12 h per session, idle time counts. v1's training took ≈ 772 min, longer than one session, so checkpoint-resume is mandatory.
  - Notebooks have internet access.
  - `/kaggle/working` is limited to 20 GB (NIDS D-009). A session's temporary disk is larger (≈ 70 GB reported by users), but is wiped when the session ends.
  - **Private-dataset quota: 214.75 GB (= 200 GiB), 28.55 GB already used** (Mayank's account settings, 15 Nov 2026). Private models: a separate 214.75 GB. GPU 30 h/week, TPU 20 h/week.
- **Hugging Face:**
  - Free accounts get **100 GB of private storage**, with best-effort public storage.
  - ZeroGPU is free for 2 Spaces (account older than 30 days), Gradio only. Visitor quota is 2 min/day anonymous and 5 min/day with a free account.
- **Google Drive:** Mayank has 5 TB. Colab mounts Drive natively; Kaggle cannot.
- **Lightning AI:** a one-time credit grant since Sep 2026, so not a recurring option.
- **Mayank's laptop:** 8 GB RAM, RTX 2050 (4 GB VRAM). Used for coding, CPU unit tests and UI work only, never for training or data storage.

### 0.5 Dataset access facts (checked 14 Nov 2026)
| Dataset | Role in v2 | How access works | Size / contents | Terms |
|---|---|---|---|---|
| FaceForensics++ (incl. DeepFakeDetection) | E1 train/val, P1 | Google Form → the authors email a download script (servers EU / EU2 / CA). Videos at raw / c23 / c40. **The form requires a Principal Investigator/Advisor name and email** (checked 15 Nov 2026). **Requested 15 Nov 2026 as an independent project; approved the same day** (script v4; only server EU2 works). | We take c23 + c40 only (raw is very large) | FaceForensics Terms of Use, non-commercial |
| Celeb-DF v2 | P2 test; train part | Google or Tencent form → download link by email. **Form submitted 16 Nov 2026; the Google Drive link was shown straight after submitting.** The official `Celeb-DF-v2.zip` (9,952,957,051 bytes, owner deepfakeforensics@gmail.com) was copied server-side into `Drive/DeepGuard/raw/celebdf/`. | 590 real + 5,639 fake videos (+ 300 YouTube-real) | Celeb-DF Terms to Use |
| DFDC | E1 train subset, P2 | Kaggle competition `deepfake-detection-challenge`, rules accepted 15 Nov 2026. **Confirmed on Mayank's screen:** notebook input = 4.44 GB (400 labelled sample videos + 400 *unlabelled* test videos). Full training set only as download links: `all.zip` 471.84 GB, or 50 chunks `00`–`49.zip` of 9.03–11.52 GB. A FAKE may be a face swap, a voice swap or both (DATASETS.md §2). | 119,154 training clips, 486 subjects | Competition rules |
| DF40 | E1 train, P2/P3 | Preprocessed frames via Google Drive / Baidu links in the GitHub repo (YZY-stack/DF40). Real frames come from FF++ / Celeb-DF. | Train ≈ 50 GB (FF++ domain fakes), test ≈ 93 GB | CC BY-NC 4.0 |
| FakeAVCeleb | E1 lip-sync (Tier B) | Google Form with an academic email and a purpose statement of at least 100 words. **The Drive access expires after 1 week.** No redistribution of the data or anything derived from it. | 500 real + 19,500 fake videos | Non-commercial research |
| Community Forensics | E2 train (images) | HF `OwensLab/CommunityForensics` (streamable). `-Small`: 278 K fakes from 4,803 generators + 278 K redistributable real images. `-Eval`: the comprehensive test set. | Small fits easily; CompEval ≈ 206 GB (51,836 images) | Base cc-by-4.0 (research); Eval cc-by-nc-sa-4.0 |
| GenVideo-100K (DeMamba) | E2/E3 train (video) | Generated videos on ModelScope; real videos from MSR-VTT (in the val zip), Youku, Kinetics-400 | 10,000 samples per category | See repo |
| GenVidBench | E2/E3 train, P3 (Tier B) | genvidbench.github.io (HF copies exist). Pair-1 real videos are given as IDs only, for copyright reasons. | 6.78 M videos, 8 generators | See repo |
| ShareVeo3 | E2/E3 (Veo 3) | HF `WenhaoWang/ShareVeo3` | 6,763 Veo 3 videos from VidProM prompts | See card |
| MSVD clips | Real-video pairing for AI video | Kaggle dataset `sarthakjain004/msvd-clips` (attach directly) | — | See card |
| Deepfake-Eval-2024 | **P4 only (final exam)** | HF gated (`nuriachandra/Deepfake-Eval-2024`): accept the terms, give an institutional email plus a link showing deepfake-detection work. Open to anyone accepting the terms from May 2027. **Requested 16 Nov 2026** (DU email + v1 Space link); approval pending. | ≈ 44–45 h video, 56.5 h audio, 1,975 images | CC-BY-SA-4.0, research |

---

## 1. Decision log

### D-001 — Freeze v1 before any v2 work · DONE 2026-11-16
**Result:**
- **Code:** GitHub tag `v1.0`.
- **Model and results:** Kaggle dataset `deepguard-v1-frozen`, 319 files, with SHA-256 manifest.
- **The v1 baseline for every later comparison is `run_v2_full_b3/checkpoint_best.pt` (epoch 52).** It is byte-identical to the model on the live Space.
- **Report correction to note:** Celeb-DF clip AUC is 0.7238 in the matching-pipeline results file, against 0.7272 in the report.
- Numbers in OUTPUTS.md §0.4.

Original plan:
- GitHub repo `akmayank94/deepguard`, tag `v1.0`, holding the v1 code plus the HF Space code.
- Private Kaggle dataset `deepguard-v1-frozen`: `model.pt`, the v3 checkpoints, results JSONs and `MANIFEST.csv` with SHA-256 checksums.
- The Space `mynkey/deepguard` keeps serving v1 until v2 passes Phase 6. v2 gets its own Space (`mynkey/deepguard-v2`) so the two can be shown side by side.

### D-002 — Product scope · DECIDED 2026-11-14 (visual first)
DeepGuard answers one question for the public: **"Is this real, or made/changed by AI?"** It covers what people actually receive (forwards, reels, screenshots).

| Input | Fake types covered | Version |
|---|---|---|
| Video with faces | face swap, face reenactment, lip-sync | v2.0 |
| Any video | fully AI-generated (text/image-to-video) | v2.0 |
| Image | AI-generated, face-swapped | v2.0 |
| Audio / video soundtrack | cloned or synthetic voice | v2.1 |
| Text | — | out of scope |

### D-003 — Architecture: evidence-based multi-expert system · DECIDED (replaces v1's five-branch net)
```
upload ─► Inspector ─► routes by content ─┬─► E1 Face-manipulation expert   (face tracks)
 (decode, metadata,                       ├─► E2 Generated-media expert     (full frames / image)
  face tracking)                          ├─► E3 Motion-geometry expert     (video only)
                                          ├─► E4 Forensic-signal expert     (Mayank's custom signals, must earn its place)
                                          ├─► E5 Provenance check           (C2PA, metadata, no ML)
                                          └─► E6 Voice expert               (v2.1)
           each expert → calibrated probability + evidence ──► Fusion + uncertainty ──► verdict
                                                                (Likely real / Uncertain / Likely AI or manipulated)
```

**Experts**
- **E1 Face-manipulation expert.** Face crops go into a frozen foundation encoder (DINOv2-L and/or CLIP ViT-L/14) with a small trained adaptation: Effort-style orthogonal SVD residual or LayerNorm tuning (GenD). Frame scores are then pooled by a small temporal attention layer over 16–32 frames of the same face track. It also outputs which frames and regions drove the score.
- **E2 Generated-media expert.** Full frames (video) or the whole image, at native resolution with random crops, never a global resize (resizing erases generator traces). A dual CLIP + DINOv2 embedding feeds a small head.
- **E3 Motion-geometry expert (video).** ReStraV-style descriptors: curvature and step distance of the DINOv2 embedding trajectory over time, computed on the whole frame and also per face track (our extension), with a light classifier. Cheap and complementary to E1/E2.
- **E4 Forensic-signal expert** — Mayank's original idea, rebuilt properly:
  - noise residuals (high-pass / SRM / neighbouring-pixel relationships);
  - a **real** multi-scale, multi-direction filter bank (true contourlet: Laplacian pyramid + directional filter bank, about 3 scales × 8 directions);
  - inter-channel (R/G/B) statistics.
  - Every feature is pooled to **position-invariant** statistics (mean, variance, kurtosis and cross-channel correlation per sub-band), about 200–400 dimensions, not 9,369.
  - **Kept only if it improves P2/P3/P4 in a pre-registered ablation (D-011).** A negative result is reported as a finding.
- **E5 Provenance check (no ML).** C2PA manifest read and validation (signer, tool, AI-generation assertions), container metadata (encoder, creation software), and a link to Google's SynthID Detector. A valid C2PA "created by AI tool" assertion is decisive evidence. Missing provenance is **not** evidence of anything.
- **E6 Voice expert (v2.1).** An open XLS-R 300M + SLS checkpoint, re-evaluated on In-the-Wild, with only the head re-calibrated.

**Fusion and uncertainty**
- A logistic regression (or a small GBM if it wins on validation) on expert scores, fitted on a **calibration split** the experts never trained on, then temperature-scaled.
- Thresholds are fixed on validation for a target false-alarm rate on real media (e.g. 5%).
- **Three-way output:** inputs between the two thresholds are reported as **Uncertain** with the reasons. That is more honest and more useful than a forced 51% "real".

**Why this design:** different fakes leave different evidence (face blending, generator texture, motion physics, sensor noise, provenance). Separate experts can each be trained on the right data, explained in plain language, and upgraded one at a time when a new generator appears, which keeps the system current. It keeps v1's core idea (several forensic signals, fused) but replaces the weak parts with strong pretrained experts and a validated fusion.

### D-004 — Data plan · SUPERSEDED by D-012 / D-013 (same intent, now with access and storage facts)

### D-005 — Leakage and shortcut controls (fixed before any training) · DECIDED
1. **Split units:**
   - faces by source video **and** identity (official splits where they exist);
   - AI media by prompt/source item;
   - every protocol additionally holds out whole **generators** (D-006 P3).
2. **De-duplication across splits:** perceptual hash plus embedding cosine > 0.95 between any two items in different splits → one is removed. Reported as counts.
3. **Shortcut audit (like NIDS A3):** before any model, a classifier trained on **metadata only** (resolution, fps, bitrate, codec, duration, JPEG quality, file size, aspect ratio). If its AUC > 0.60, real and fake are re-encoded to matched distributions until it drops to ≤ 0.60.
4. **Same degradations for both classes:** a random "social-media chain" (resize, H.264 CRF 23–35, JPEG 60–95, crop, screen-record simulation) is applied identically to real and fake in training.
5. **Choice discipline:** thresholds, calibration and model selection use validation only. Each test protocol is run **once per pre-registered phase**.
6. **Time-independent sampling (added 15 Nov 2026, from DATASETS.md T2/T3):** frames are sampled by index, never by seconds. fps, duration and video length are never model inputs. Random fps resampling is applied to both classes. Reason: FF++ fakes are encoded at the source video's fps and FaceSwap fakes can be shorter, so time-based features can separate real from fake without seeing any manipulation.
7. **Right face only:** in fake videos, only the face whose pixels changed (diff against the original, or the official mask) is labelled fake. Other faces in a fake video are excluded, never labelled real or fake by guess. Voice-only DFDC fakes are excluded from visual training.
8. **Honest statistics:** 3 seeds for every headline model, and bootstrap 95% intervals by video/identity (v1's ±0.04 lesson, §0.2.3).

### D-006 — Evaluation protocols and metrics · DECIDED
- **P1 in-domain:** each dataset's own test split.
- **P2 cross-dataset (face):** train on FF++ only → Celeb-DF v2, DFDC, DF40 and others. This is the standard DeepfakeBench protocol, so results compare directly with published methods.
- **P3 cross-generator:** leave one generator family out (AI images/video); also a **time-forward** test (train on generators released before a cut-off date, test on later ones).
- **P4 in-the-wild final exam:** Deepfake-Eval-2024 + DeepGuard-Wild-2026.
- **P5 robustness:** re-compression, resize, crop, screen recording, re-upload chains.
- **Metrics:**
  - AUC and AP for comparison with papers;
  - **for users:** recall at 5% and 1% FPR on real media, the share of "Uncertain" verdicts, accuracy on decided items, and expected calibration error (ECE);
  - **ops:** seconds per 10-s clip, GPU-seconds per request.

### D-007 — Targets (fixed now, judged later; missing one is reported, not hidden) · DECIDED
| Protocol | v1 | Published reference | v2 target |
|---|---|---|---|
| P2 face, mean AUC over unseen datasets | Celeb-DF 0.727 | Effort 0.917 (7 datasets) | ≥ 0.90 |
| P4 Deepfake-Eval-2024 video AUC | not tested | open source ≈ 0.63, commercial ≈ 0.79 | ≥ 0.75 (stretch 0.80) |
| P4 Deepfake-Eval-2024 image AUC | not tested | open source ≈ 0.56, commercial ≈ 0.90 | ≥ 0.80 |
| P3 leave-generator-out video AUC | — | — | ≥ 0.85 |
| Uncertain share on P4 | — | — | ≤ 25%, with ≥ 90% accuracy on decided items |
| Calibration (ECE) | — | — | ≤ 0.05 |
| Latency, 10-s clip on ZeroGPU | ~3–8 s (laptop GPU) | — | ≤ 15 s end-to-end |

### D-008 — Compute and tooling · DECIDED
- **Kaggle:** training, preprocessing and embedding extraction. Every script is resume-safe (checkpoint every ≤ 30 min). Runs use "Save & Run All" in the background. Mixed precision on T4.
- **Colab (free):** fetching form-gated downloads into Drive, pushing archives to Kaggle/HF, and overflow GPU.
- **HF Hub:** model versions (`mynkey/deepguard-e1-face` …) with model cards; private dataset shards (D-013).
- **GitHub:** code with CI (GitHub Actions running unit tests plus the inference-parity test from 14 Nov 2026).
- **Experiment tracking:** Weights & Biases free tier, or a CSV + `OUTPUTS.md` written by the notebooks themselves (NIDS D-024 pattern).
- **Laptop:** editing, CPU tests and UI only.

### D-009 — Product features (what makes people use it and come back) · DECIDED
**v2.0 core**
- Upload a video or image (and paste a link where the platform's terms allow).
- **Three-way verdict** with a calibrated percentage.
- **Evidence cards in plain language,** one per expert, for example:
  - "Face texture inconsistent in 14 of 20 frames";
  - "Motion path curves the way AI video does";
  - "No camera-sensor noise pattern";
  - "Content Credentials: made with <tool>".
  - The text comes from templates filled with measured values. An LLM may rephrase but never decides.
- **Heatmap overlay** on the most suspicious frames, plus the frame timeline (from v1).
- **Shareable report:** a link and a downloadable PDF containing the file's SHA-256, the model version and the date.
- Uploads auto-deleted; nothing stored without opt-in.

**v2.1 reach and engagement**
- **Telegram bot:** forward a video and get the verdict and report link. This is where fakes actually spread in India.
- **"Real or AI?" daily quiz** built from labelled P4-style items. It teaches users, brings them back, and measures human vs model accuracy.
- **Hindi + English** interface.
- **"Was this right?"** button: consented items enter a review queue that feeds the next training round (a data flywheel).
- **Public "Model health" page:** live P4 numbers, version history and known blind spots.

**Developers**
- REST API + Python client.
- **MCP server,** so AI assistants can call DeepGuard as a tool.
- Rate limiting.

**Hosting (free)**
- Inference backend: Gradio API on HF ZeroGPU (`mynkey/deepguard-v2`).
- Product website: Next.js on Vercel Hobby (non-commercial) calling the Space through the official Gradio JS client, the same pattern as flowgraph-nids.vercel.app.

### D-010 — What v2 keeps and drops from v1 · DECIDED
- **Keep:**
  - the multi-signal philosophy;
  - video-level splits and the leakage discipline;
  - MediaPipe face tracking and alignment (re-checked in Phase 1);
  - the inference-parity test;
  - the DeepGuard UI design language.
- **Drop:**
  - the 9,369-d position-specific block-energy vector;
  - EfficientNet-B0 trained from scratch on diff maps;
  - the softmax gate trained on 1.4 k clips as "explanation";
  - rPPG on compressed video (re-tested only inside E4's ablation).

### D-011 — The contourlet question, tested properly · DECIDED
**Hypothesis:** multi-scale, multi-direction sub-band statistics add information that foundation-model embeddings miss, especially after compression.

**Pre-registered test (Phase 3):** E1/E2 embeddings alone vs embeddings + E4 variants:
- (a) a true contourlet (3 scales × 8 directions) with position-invariant statistics;
- (b) a 3-level wavelet with the same statistics;
- (c) noise-residual statistics only;
- (d) all of them.

Same splits and seeds, judged on P2, P3 and P5 (compressed).

**Accepted** if the gain is ≥ +0.02 AUC with a bootstrap interval excluding 0 on at least two protocols. Either outcome goes in the write-up. A clean negative answer is still a contribution, and it finally settles what happened in v1.

### D-012 — Dataset selection · DECIDED 2026-11-14 (facts in §0.5)
**Tier A — requested now, used from Phase 1:**

| Expert | Training / validation | Testing |
|---|---|---|
| E1 faces | FF++ c23 + c40 (all 6 methods incl. DeepFakeDetection), DFDC subset, DF40 train (FF++ domain), Celeb-DF v2 train part | P1 FF++ test; P2 Celeb-DF v2 test, DFDC test subset, DF40 test (unseen methods) |
| E2 images | Community Forensics-Small (fakes from 4,803 generators + its paired reals) | Community Forensics-Eval subset; P3 leave-generator-out |
| E2/E3 video | GenVideo-100K + its real sources (MSR-VTT, Kinetics), ShareVeo3 + MSVD reals | P3 leave-generator-out; time-forward split |
| All | — | **P4: Deepfake-Eval-2024** (never used for training or thresholds) |

**Tier B — only if Phase 3 shows a gap:**
- GenVidBench (cross-source video);
- the full Community Forensics Systematic set;
- RobustSora / CCDF (Sora 2, Veo 3.1, Grok) as extra test sets;
- FakeAVCeleb (lip-sync). Requested only when the Colab fetcher is ready, because the link expires after one week.

**DeepGuard-Wild-2026 (Phase 4):** our own 300–500 labelled 2026 items (C2PA-signed AI outputs, official generator showcase clips, real videos filmed by us with consent). Test only.

**Not used: unofficial re-uploads** of licensed datasets (copies of FF++, Celeb-DF etc. posted by others on HF/Kaggle). *For FF++ this is under review in D-015 (15 Nov 2026).*
- They break the original terms.
- Their integrity cannot be verified.
- Results built on them could not be published honestly.

### D-013 — Storage and data flow: three layers, nothing on the laptop · DECIDED 2026-11-14
**Layer 1 — Raw originals (never modified):**
- **Left where they are** when the source is permanent and online: HF datasets, Kaggle competitions/datasets, ModelScope. Notebooks read them directly; copying would only waste quota.
- **Archived to Google Drive** (`DeepGuard/raw/<dataset>/`) only for form-gated downloads: FF++, Celeb-DF, DF40's Drive links, FakeAVCeleb. A Colab notebook mounts Drive and runs the official script or link straight into it, so the laptop never touches the data. Each dataset gets a `MANIFEST.csv` with SHA-256 checksums.
- For processing, compact Layer-1 archives (FF++ c23/c40, Celeb-DF) are pushed from Colab to **private Kaggle datasets** with the Kaggle API.

**Layer 2 — Processed:**
- Contents: face tracks and aligned crops (JPEG q95; 224 and 384 px), sampled full frames for E2/E3, and a metadata table (resolution, fps, codec, bitrate, duration) for the shortcut audit.
- Stored as tar/parquet shards in a private HF dataset `mynkey/dg-processed`, with a Kaggle copy for fast attaching.
- Budget: ≈ 20 KB per crop, so 1 M crops ≈ 20 GB.

**Layer 3 — Features:**
- Frozen-encoder embeddings in fp16 (1 M × 1,024 × 2 bytes ≈ 2 GB per encoder), plus manifests and frozen split files.
- Stored in a private HF dataset `mynkey/dg-features` plus a Kaggle copy.

**Big sources (DFDC, DF40 test, Community Forensics-Eval) use "stream and reduce":** a Kaggle session pulls one chunk into its temporary disk, writes only Layer 2/3 outputs, deletes the chunk and moves on. Resume-safe by chunk ID.

**Secrets:** the HF write token and Kaggle API key live in Kaggle Secrets / Colab Secrets, never inside a notebook.

**Privacy and licences:**
- Every Layer 1–3 repo that holds licensed data is private.
- Nothing derived from licensed datasets is ever published.
- Things that may be public later: code, model weights (non-commercial licence), our own DeepGuard-Wild items we hold rights to, and aggregate results.

### D-014 — Access requests · DECIDED 2026-11-14
**Rules:**
- Use the Delhi University email for every form.
- Use one consistent project description (below).
- Record the request date and the approval date in §0.5.

**Project description for forms** (about 120 words; FakeAVCeleb needs at least 100):
> I am an M.Sc. Computer Science student at the University of Delhi working on DeepGuard, a non-commercial research project on generalizable deepfake detection. Its first version, a multi-branch detector trained on FaceForensics++ and evaluated on Celeb-DF v2, was presented at ICET-AICS 2026 (Ramanujan College, University of Delhi). Version 2 studies how well detectors built on pretrained vision encoders generalize to unseen manipulation methods and to in-the-wild media, under strict leakage controls (video- and identity-level splits, de-duplication and metadata-shortcut audits). The dataset will be used only to train and evaluate detection models, never to create or improve deepfakes. It will be stored privately, will not be redistributed, and will be cited in any resulting report, publication or demo.

**Advisor fields:** if a form requires a Principal Investigator/Advisor and no faculty member is involved, write "None (independent student project)" and give Mayank's own DU email, rather than naming a teacher. This replaces an earlier draft rule (15 Nov 2026) that named a faculty member as advisor; Mayank chose to apply as an independent project.

**Evidence link for gated access** (Deepfake-Eval-2024): the v1 Space (huggingface.co/spaces/mynkey/deepguard) and, once created, the GitHub repo.

### D-015 — Where our FF++ copy comes from · DECIDED 2026-11-15 (updated: access APPROVED)
**Update, 15 Nov 2026, 12:30:** the FaceForensics authors **approved** the request and sent the download script link (v4 script, server EU2 only for now). FF++ is now used under its official licence.
- **Archive:** the official c23 + c40 videos (FF++ originals, the 5 FF++ methods, DFD originals and DFD fakes) go to `Drive/DeepGuard/raw/ffpp/` via the Colab notebook `dg_p0_ffpp_official_fetch.ipynb`, with a SHA-256 `MANIFEST.csv`. The script link is kept as the Colab secret `FFPP_SCRIPT_URL`, never in code or docs.
- **Single source of truth (Mayank, 15 Nov 2026, 14:43):** the Kaggle copy is **dropped entirely**, with no byte comparison, to avoid any confusion about which copy a result came from.
- The official copy is uploaded from Drive to private Kaggle datasets: `dg-ffpp-c23`, `dg-ffpp-c40`, `dg-dfd-c23`, `dg-dfd-c40`.
- The D-016 audit runs again on those datasets. Its run on the Kaggle copy is discarded and quoted nowhere.
- Download note: the EU2 server gives about 1 MB/s per connection; 4 parallel parts gave about 4 MB/s.
- **Download DONE, 15 Nov 2026** (8 parallel parts, about 2.5 h, 5–11 MB/s). All 16 parts OK with exact official counts (39.97 GB in total):

| Set | c23 | c40 |
|---|---|---|
| FF++ originals | 1,000 files, 1.94 GB | 1,000 files, 0.26 GB |
| FF++ 5 methods | 1,000 each, 9.03 GB | 1,000 each, 1.19 GB |
| DFD originals | 363 files, 2.90 GB | 363 files, 0.41 GB |
| DFD fakes | 3,068 files, 21.27 GB | 3,068 files, 2.98 GB |

- **Upload to Kaggle:** Colab notebook `dg_p0_ffpp_to_kaggle.ipynb`. Uncompressed zips that keep the official layout. `dg-ffpp-c40` goes first as a test. Kaggle login uses the new API token (Colab secret `KAGGLE_API_TOKEN`, Kaggle CLI ≥ 1.8.0), not a legacy `kaggle.json`.

**Earlier decision (Mayank, 15 Nov 2026, before approval):**
- The official form was submitted on 15 Nov 2026 as an independent personal project.
- Work starts now on the Kaggle copy.
- If the request is rejected, the Kaggle copy stays our FF++ source (Option B), disclosed honestly in the write-up.
- If it is approved, the official file lists are used to verify the Kaggle copy.

Options considered:
- **Option A (recommended): apply officially as an independent student**, with no advisor (D-014). If approved, the licence covers the project. The official file lists then let the audit verify any copy, the Kaggle one included.
- **Option B: use the Kaggle copy `stevenadriancorne/faceforensics-c23-c40` without applying.**
  - Fast, and v1 already did the same with a Hugging Face copy.
  - But the uploader had no right to redistribute FF++, so we would hold no licence.
  - Fine for a private personal project. It weakens any paper submission or public weights trained on it, and it has to be stated honestly in the write-up.
- **Either way:** the copy must pass the Phase 1.1 audit (D-016). It stays private, nothing derived from it is redistributed, and it is cited as FF++.
- Supersedes the blanket "not used: unofficial re-uploads" line in D-012 for FF++ only, pending this choice.

### D-016 — Phase 1.1: FF++ audit before any preprocessing · DECIDED 2026-11-15
Notebook `dg_p1_ffpp_audit.ipynb` (Kaggle, CPU, about 1–2 h, resumable). Tested end to end on a synthetic FF++-style set with planted defects; all 7 were caught: unreadable file, unexpected file, fake identical to its original, time-shifted fake, wrong-face manipulation, multi-face video, and fps/length changes.

**Pre-registered checks; every one is reported, and every flag gets a written decision before preprocessing:**
- **A1 Inventory:** every file is parsed (method, compression, IDs). c23 and c40 hold the same names. The output states whether the copy holds videos or frames.
- **A2 Completeness:** 1,000/1,000 originals and 1,000/1,000 fakes per method and compression, with names exactly matching the official pairs. DFD counts are compared with 363 / 3,068.
- **A3 Integrity:** every video decodes; the header frame count is compared with the counted frames; SHA-256 manifest; byte-identical duplicates listed.
- **A4 H1 (fps):** share of fakes whose fps equals the source's rather than the target's, and the playback-speed range.
- **A5 H2 (frames):** fake frames vs the target's (and vs `min(target, source)` for FaceSwap).
- **A6 Shortcut:** metadata-only AUC per compression and method; above 0.60 is flagged.
- **A7 Visual (c23, all FF++ fakes + 600 DFD):** changed-pixel share; background difference (alignment); faces per video; whether the manipulated face is the largest; fakes with almost no change (possibly mislabelled).
- **Official-split leakage:** already measured, 7 YouTube videos across splits (DATASETS.md §1.4).

**Acceptance:** A1–A7 all reported in OUTPUTS.md, with a decision written here for each flag. Preprocessing (crops, face tracks) starts only after that.

**Result (15 Nov 2026):** all A1–A7 reported in OUTPUTS.md (Phase 1.1). The upload is verified byte for byte (18,862/18,862). Status: **DONE**, pending confirmation of the D-017 rules below.

### D-017 — Data rules from the Phase 1.1 audit · DECIDED 2026-11-16 (confirmed by Mayank)
One rule per flag. Numbers are in OUTPUTS.md, Phase 1.1.
1. **Speed (64 fakes, 0.50×–2.00×):** keep these videos. Sampling stays index-based (D-005.6). Mark them `fps_changed` in the master manifest, so results can be checked on them separately.
2. **Fake length** (Face2Face = source length; FaceSwap and NeuralTextures = min; Deepfakes and FaceShifter = target): length is never an input. Frame windows are taken inside `min(len_fake, len_target)`. Phase 1.2 inspects what Face2Face's extra frames contain and records it.
3. **Width shortcut (Face2Face/NeuralTextures, AUC 0.69–0.77):**
   - Phase 1.2 first tests on sample frames whether the 282 fakes were **cropped or rescaled**.
   - Then the **same transform is applied to every video whose width is not a multiple of 16**: all real and fake videos of those 282 targets, across all methods.
   - Acceptance: the metadata-only AUC re-run reaches ≤ 0.60 for every method.
   - FaceSwap's 0.61 comes from length only. Length never reaches the model (rule 2), so no transform is needed.
4. **Right face:** in FF++ fakes, the face is selected by pixel difference against the target original. They are frame-aligned: median background difference 2.0, 0 misaligned. Never use "largest face" (it fails in 21/4,436). Other faces in fake videos are excluded. Detector false positives (logos, toys) are dropped by requiring a face track of at least N frames.
5. **Small manipulations** (the "unchanged" flag, 162 cases; contact sheets show real mouth-only or small-face edits): kept as fakes. Face selection uses the face-level difference ratio, not a whole-frame threshold.
6. **DFD offset:**
   - Before face selection, each DFD fake's start frame in its original is found by matching downscaled frames.
   - If matching fails, the official DFD masks are used (`-t masks`, to be downloaded only if needed).
   - DFD fakes shorter than the frame window (32 frames) are dropped: **19 fakes** per compression level, all `secret_conversation` (measured).
7. **DFD duplicates:** 14 byte-identical groups (7 two-actor scene videos × 2 compression levels). Deduplicate by SHA-256. A two-actor video counts for **both** actors in identity splits.
8. **Official-split leakage (7 YouTube videos):** results are reported on the official split (for comparison with papers) **and** on a clean split that moves the 7 sibling sequences together, plus the face-embedding identity clusters from Phase 1.2.

### D-018 — Base-strength rules: problems to close now, before they become expensive · DECIDED 2026-11-16 (at Mayank's request)
Each rule names the risk, the fix, and the phase where it is checked.
1. **Source-label confounding (the biggest risk).**
   - If reals and fakes come from different sources (for example, old low-resolution MSVD reals against modern 720p Veo fakes), the model learns "which dataset", not "real or fake".
   - Fix:
     - within every training domain, both classes are matched for resolution, fps, codec and bitrate (re-encode both to the same specs);
     - per-dataset label balance is reported;
     - a **dataset-ID probe** (can the embeddings tell which dataset an item came from?) is run alongside the metadata audit.
   - Checked in Phases 1.4–1.5 and 2.
2. **One preprocessing path for training and the app (v1's main lesson).**
   - Use one decoder with an explicit colour conversion (no silent BT.601/BT.709 or full/limited-range differences between OpenCV, PyAV and ffmpeg).
   - Use one face detector and one alignment and crop function with the same margin for every dataset, and one JPEG step.
   - The code is a single package used by the preprocessing notebooks **and** the Space, and the parity test (0.0 difference) runs in CI.
   - Checked from Phase 1.2.
3. **Identity overlap across datasets.**
   - DF40's FF++-domain fakes are built on FF++ videos. Every DF40 item is mapped to its FF++ IDs and must respect the FF++ split; items that cross the split are removed.
   - Face-embedding identity clustering runs **across all datasets** (FF++, DFD, Celeb-DF, DFDC, DF40), not inside each one alone.
   - Checked in Phase 1.5.
4. **A locked test vault.**
   - Split files are frozen with SHA-256 and stored read-only.
   - P2 and P4 test data live in a separate Kaggle dataset that only the evaluation notebook attaches.
   - Every test run is logged in OUTPUTS.md with a run counter, so "run once per phase" (D-005.5) can be shown.
   - Checked from Phase 1.5.
5. **Two wild sets, not one.**
   - DeepGuard-Wild is split into a **dev** part (used from Phase 2 to find failure modes early) and a **locked test** part (used only in Phase 4).
   - Collection starts in Phase 1, not Phase 4.
6. **Real-media false alarms are measured per source.**
   - D-006 metrics gain the false-positive rate per real source (FF++ YouTube, DFD actors, DFDC, Celeb-DF, MSVD/Kinetics, phone/WhatsApp videos).
   - Wrongly calling a real video fake is the most harmful public error.
7. **Reproducibility.**
   - Code lives in GitHub, not only in Kaggle/Colab. Every OUTPUTS entry records the git commit, notebook version, package versions and seed.
   - Kaggle preprocessing and training notebooks use **"Pin to original environment"**.
   - Processed data carries a version ID that points back to the raw SHA-256.
8. **Budgets tracked before they bite.**
   - Storage: Kaggle private datasets ≈ 68.5 / 214.75 GB after FF++; HF private 100 GB.
   - Compute: GPU hours estimated per phase **before** running (for example, the embedding cost per million crops is measured on 1,000 crops first).
   - Both are tracked in OUTPUTS.md.
9. **Licences shape the product.**
   - FF++, Celeb-DF, DF40 and FakeAVCeleb are non-commercial. The public app stays free and non-commercial.
   - The model card states the training-data licences.
   - Feedback items enter training only with explicit consent.
10. **Inputs the public will actually send** (vertical video, screen recordings, no face, many faces, long videos, memes with text, photos of screens) form a routing and robustness checklist in the dev-wild set from Phase 2, not a Phase 6 surprise.

---

## 2. Roadmap — no phase starts before the previous phase's acceptance test passes
No deadline (personal project). Week figures are estimates only; each phase ends when its acceptance test passes. Each phase gets its acceptance test written into this file **before** it runs. Status as of 16 Nov 2026.

| Phase | Status | Est. weeks | Work | Datasets entering | Acceptance test (draft) |
|---|---|---|---|---|---|
| 0 Freeze + setup | **DONE** (16 Nov) | 1 | Accounts and secrets; Drive; FF++ + DFD official (verified); Celeb-DF v2 (verified); DFDC rules + download route; v1 frozen (tag `v1.0` + Kaggle); GitHub repo `akmayank94/deepguard` with CI; OUTPUTS.md, DATASETS.md, TERMS.md. Pending outside the project: Deepfake-Eval-2024 approval (needed only in Phase 4) | FF++ + DFD (done) | v1 checksums verified; v1 tagged; repo + CI green; every Tier-A request sent and logged |
| 1.1 FF++ audit | **DONE** (15 Nov) | — | D-016 | — | All A1–A7 reported; D-017 decided |
| 1.2 FF++/DFD preprocessing | next | 1–2 | Shared preprocessing package (D-018.2); face tracks; right-face selection; width fix; DFD offset alignment; identity clusters; master manifest v1 | — | D-017 rules implemented; metadata-only AUC ≤ 0.60 for every method; parity test passes; random crop sample checked by eye |
| 1.3 Other face datasets | after 1.2 | 1–2 | Audit + preprocess each, using the same checklist as 1.1 | Celeb-DF v2 (when approved), DFDC subset (10–15 chunks, stream and reduce), DF40 | Each dataset audited (A1–A7 equivalent) before its crops are made |
| 1.4 AI-media datasets | after 1.2 (can overlap 1.3) | 1–2 | Audit + matched re-encoding (D-018.1) | Community Forensics-Small (images), GenVideo-100K, ShareVeo3 + MSVD reals | Per-domain metadata AUC ≤ 0.60; dataset-ID probe reported |
| 1.5 Freeze the data | end of Phase 1 | <1 | Cross-dataset de-dup and identity clustering (D-018.3); DF40↔FF++ split check; degradation chain; frozen splits + test vault (D-018.4); dev-wild collection starts | — | 0 cross-split duplicates or identities; splits frozen with SHA-256 |
| 2 Embedding bank + harness | — | 2 | CLIP-L and DINOv2-L embeddings; linear probes; harness for P1–P5; reproduce a published checkpoint | dev-wild (growing) | Harness reproduces published AUC within ±0.02 on ≥ 2 datasets |
| 3 Experts | — | 4–5 | E1 (adapter + temporal pooling), E2, E3, E4 ablation (D-011); 3 seeds each | Phase-1 face + AI-media sets | Each expert beats its linear probe on its protocol (P2 for E1, P3 for E2/E3) |
| 4 Fusion + uncertainty | — | 1–2 | Calibration split, fusion, temperature scaling, three-way thresholds; DeepGuard-Wild locked test finished; **P4 run once** (Deepfake-Eval-2024 + Wild locked test) | Deepfake-Eval-2024, Wild locked test | ECE ≤ 0.05; D-007 targets checked and reported |
| 5 Robustness | — | 1 | P5 degradation curves; failure analysis per generator | — | Curves reported; no expert drops > 0.10 AUC at CRF 30 without being flagged |
| 6 Product v2.0 | — | 3 | Space v2 (API), Vercel site, evidence cards, heatmaps, reports, C2PA, CI + parity tests | — | Parity test PASS; p95 latency ≤ 15 s; site reviewed |
| 7 Reach | — | 2 | Telegram bot, quiz, Hindi UI, feedback queue, MCP server, model-health page | — | Each feature works end-to-end on the public link |
| 8 Voice (v2.1) + release | — | 2 | E6, write-up, model cards, README, v2 freeze | In-the-Wild audio | Release checklist complete |

Phases 6–7 can start in parallel once Phase 4 passes. Waiting for dataset approvals runs in parallel with setup work.

## 3. How v2 will read on a resume (only claims the numbers will support)
"DeepGuard, an open deepfake and AI-media detector for video, images and voice:
- foundation-model experts (DINOv2 / CLIP with parameter-efficient tuning) plus calibrated fusion with an 'uncertain' band;
- leakage- and shortcut-audited training across N datasets and M generators;
- in-the-wild AUC X on Deepfake-Eval-2024 (open-source baseline ≈ 0.63);
- deployed free with a web app, REST API, Telegram bot and MCP server, used by N people."

The numbers are filled in only from OUTPUTS.md.

## 4. Decisions taken 14 Nov 2026
1. Scope: visual-first v2.0 (video + images), voice in v2.1 → D-002.
2. Time: no deadline; take what each phase needs, no more.
3. Personal project, no supervisor. DECISIONS / OUTPUTS / TERMS are still kept, because they are what make the results trustworthy.

## 5. Sources
- Deepfake-Eval-2024 — arxiv.org/abs/2503.02857 · huggingface.co/datasets/nuriachandra/Deepfake-Eval-2024
- BitMind Forensics (continuously evolving detection, Jul 2026) — alphaxiv.org/abs/2607.13234
- Effort, ICML 2025 — arxiv.org/abs/2411.15633 · github.com/YZY-stack/Effort-AIGI-Detection
- GenD, WACV 2026 — arxiv.org/abs/2508.06248
- UCF-Net (CLIP + DINO fusion, Sep 2026) — arxiv.org/abs/2609.07670
- NTIRE 2026 complementary ensembles (DINOv2 + LoRA) — arxiv.org/abs/2604.25889
- DF40, NeurIPS 2024 — arxiv.org/abs/2406.13495 · github.com/YZY-stack/DF40
- Community Forensics, CVPR 2025 — arxiv.org/abs/2411.04125 · huggingface.co/datasets/OwensLab/CommunityForensics
- ReStraV, NeurIPS 2025 — arxiv.org/abs/2507.00583
- TrueVision benchmark (Penn State, Jul 2026) — sites.psu.edu/mcreu/2026/07/22/benchmarking-explainable-deepfake-detection-across-modern-ai-generated-video/
- VidGuard-R1 (dataset shortcuts) — arxiv.org/abs/2510.02282
- GenVidBench — arxiv.org/abs/2501.11340 · genvidbench.github.io
- GenVideo / DeMamba — arxiv.org/abs/2405.19707 · github.com/chenhaoxing/DeMamba
- ShareVeo3 — huggingface.co/datasets/WenhaoWang/ShareVeo3 · ComGenVid — arxiv.org/abs/2603.15026
- RobustSora — arxiv.org/abs/2512.10248 · CCDF — arxiv.org/abs/2610.07939
- FaceForensics++ — github.com/ondyari/FaceForensics
- Celeb-DF — github.com/danmohaha/celeb-deepfakeforensics
- DFDC — kaggle.com/c/deepfake-detection-challenge · arxiv.org/abs/2006.07397
- FakeAVCeleb — github.com/DASH-Lab/FakeAVCeleb
- c2pa-python — github.com/contentauth/c2pa-python
- SynthID Detector public launch (7 Oct 2026) — petapixel.com/2026/10/08/googles-synthid-ai-detector-is-now-freely-available-to-everyone/
- XLS-R + SLS audio checkpoints — huggingface.co/SpeechAntiSpoofingBenchmarks/XLSR-SLS
- HF storage limits — huggingface.co/docs/hub/storage-limits · HF ZeroGPU — huggingface.co/docs/hub/spaces-zerogpu

## 6. Change history
| Date | Change |
|---|---|
| 2026-11-14 | File created. v1 audited from code and report (§0.1–0.2, measured); field review (§0.3); v2 architecture, data, leakage rules, protocols, targets, product and roadmap proposed (D-001–D-011). |
| 2026-11-14 (2) | Mayank's decisions recorded (§4): visual-first, no deadline, personal project. Dataset access facts checked (§0.5); storage facts added (§0.4). D-012 dataset selection, D-013 three-layer storage with no local data, D-014 access-request rules and form text. D-004 superseded. D-001–D-011 moved to DECIDED. |
| 2026-11-15 | Kaggle quota measured (§0.4). FF++ form fields checked: advisor name/email required (§0.5, D-014). DFDC access corrected: full training set comes as chunks, not as notebook input (to confirm). |
| 2026-11-15 (2) | DFDC facts confirmed from Mayank's screen (§0.5). v1 data provenance added (§0.1). D-005 gains time-independent sampling and right-face rules. D-014 advisor rule changed to "independent, no advisor". D-015 FF++ source (PENDING). D-016 Phase 1.1 FF++ audit with pre-registered checks. New companion file DATASETS.md (FF++/DFD/DFDC facts and traps, including 7 official-split leaks measured from the official files). |
| 2026-11-15 (3) | D-015 decided: FF++ form submitted as an independent project; Kaggle copy used now and kept if rejected. |
| 2026-11-15 (4) | FF++ access approved (§0.5). D-015 updated: official copy archived on Drive with a SHA-256 manifest; the Kaggle copy is used for processing only if byte-identical. Colab notebook `dg_p0_ffpp_official_fetch.ipynb` built and tested with a mock download script (resume, count check, manifest; the comparison caught a planted modified file and a planted extra file). |
| 2026-11-15 (5) | Kaggle copy of FF++ dropped (Mayank): official copy only, uploaded to private Kaggle datasets; the D-016 audit is re-run on it. |
| 2026-11-15 (6) | Official download complete: 16/16 parts, exact counts, 39.97 GB (D-015 table). Upload notebook built and tested with a mock Kaggle CLI. |
| 2026-11-15 (7) | All four private Kaggle datasets READY: `mynkey/dg-ffpp-c40`, `dg-ffpp-c23`, `dg-dfd-c40`, `dg-dfd-c23`. Kaggle unpacks the uploaded zips into folders (checked on dg-ffpp-c40). Kaggle titles must be 6–50 characters. Next: run the D-016 audit on these four, and check the upload byte for byte (audit MANIFEST vs Drive MANIFEST). |
| 2026-11-15 (8) | Phase 1.1 audit run on the official copy: upload verified byte for byte (18,862/18,862); results in the new OUTPUTS.md. D-016 DONE. D-017 data rules proposed (speed, length, width shortcut, right face, small edits, DFD offset and duplicates, split leakage). DATASETS.md updated with measured facts and traps T10–T13. |
| 2026-11-16 | D-017 confirmed by Mayank (DECIDED). D-018 base-strength rules added: source-label confounding, one shared preprocessing path, cross-dataset identities (DF40↔FF++), locked test vault, dev/test wild sets, per-source false-positive rate, reproducibility, budgets, licences, public-input checklist. Roadmap split into Phases 1.1–1.5, with status and datasets per phase. |
| 2026-11-16 (2) | Celeb-DF v2 obtained (form → instant Drive link) and copied to Drive. Deepfake-Eval-2024 requested. GitHub repo `akmayank94/deepguard` created (private, empty); pushing waits on linking GitHub to Claude. DFDC chunk links are logged-in Kaggle web links (`kaggle.com/c/16880/datadownload/dfdc_train_part_NN.zip`), so a notebook download method is still to be tested (Phase 1.3). Colab notebook `dg_p0_freeze_v1_and_celebdf.ipynb` built and tested with mocks: Celeb-DF → `dg-celebdf-v2`; v1 freeze → `deepguard-v1-frozen` with MANIFEST and a check of the Space's model against `deploy/model.pt`. D-014: removed a personal name. |
| 2026-11-16 (3) | **Celeb-DF v2 on Kaggle:** `mynkey/dg-celebdf-v2` READY. The zip's SHA-256 is `b22f9d74…cb126fa2` and it holds 6,530 files: 590 Celeb-real, 5,639 Celeb-synthesis, 300 YouTube-real and the test list, exactly as the paper states. **GitHub:** `akmayank94/deepguard` connected. `main` = the v1 snapshot (commit `a1ea01c`); the `v1.0` tag must be created by Mayank in the GitHub UI, because tag pushes are refused from this environment. The v2 structure (`legacy/v1/`, `docs/`, notebooks, `src/deepguard`, CI hygiene checks) is on branch `v2-setup` (commit `47c5dd8`), to be merged after the tag exists. **v1 freeze B1 fixed:** v1's `data/` folder holds dataset *code*, so it is now included; deployed-model provenance was added to `SPACE_CHECK.json`. **DFDC:** a chunk link resolves to a signed `storage.googleapis.com` URL valid about 39 h, so a Kaggle notebook can download chunks from fresh links. TERMS.md created. |
| 2026-11-16 (4) | **v1 freeze DONE (D-001).** `deepguard-v1-frozen` READY (319 files, 1.44 GB). The Space model is identical to `deploy/model.pt` and `run_v2_full_b3/checkpoint_best.pt` (epoch 52, the reported model). Tag `v1.0` created by Mayank on GitHub; `v2-setup` merged into `main`. Phase 0 is complete except the Deepfake-Eval-2024 approval, which only Phase 4 needs. |
