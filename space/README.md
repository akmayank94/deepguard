---
title: DeepGuard Deepfake Video Detector
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.50.0
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
short_description: Five-branch forensic deepfake video detector
tags:
  - deepfake-detection
  - video-forensics
  - computer-vision
  - pytorch
---

# 🛡️ DeepGuard: five-branch forensic deepfake video detector

Upload a short video with a visible face. DeepGuard tracks the face, extracts five independent forensic
signals and returns a real/fake score with a per-frame timeline and per-branch attention weights.

| Branch | Signal | Input | Network |
|---|---|---|---|
| A | Visual appearance | 20 eye-aligned 224×224 face crops | EfficientNet-B3 (ImageNet init) |
| B | Inter-channel high-frequency energy differences | 224×224×3 diff map | EfficientNet-B0 (from scratch) |
| C | Directional filter-bank energy + DCCM | 9,369-d vector | MLP |
| D | Camera-sensor noise (PRNU) statistics | 30-d vector from 64 un-aligned crops | MLP |
| E | Remote photoplethysmography (pulse) | 32-d spectrum features over ~5 s | MLP |

Branch outputs are fused by a learned softmax attention gate. Total: 22.18 M parameters.

## Results (strict video-level split, no source video shared between train/val/test)

| Evaluation | AUC |
|---|---|
| FaceForensics++ C23, validation | 0.9158 |
| FaceForensics++ C23, held-out test | **0.8781** |
| Celeb-DF v2, official test split (cross-dataset, no fine-tuning) | **0.7272** |

## Limitations

- Trained only on FaceForensics++ manipulations (Deepfakes, Face2Face, FaceSwap, FaceShifter,
  NeuralTextures, DeepFakeDetection). Not trained on diffusion-generated video, lip-sync or audio deepfakes.
- One main face per clip; needs at least ~2 s of a clearly visible face. Only the first ~15 s are analysed.
- Attention-gate weights show how the fusion layer weighted each branch. They are not a proof of
  which artifact was found.
- Research prototype: not suitable as sole evidence in legal or official contexts.

## Running locally

```bash
pip install -r requirements.txt gradio==5.50.0
# put the trained checkpoint next to app.py as model.pt
python app.py
```

The inference preprocessing (`deepguard/pipeline.py`) reproduces the training preprocessing
(`preprocess_final.py` + `precompute_features_v2.py`) exactly. On a test clip, every input to the
model matched the original training code with a difference of 0.

## Credits

Mayank and Saloni Nayyar, B.Sc. (H) Computer Science, Keshav Mahavidyalaya, University of Delhi.
Supervisor: Ms. Jyoti Kumari. Presented at ICET-AICS 2026, Ramanujan College, University of Delhi.
