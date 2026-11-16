# DeepGuard v1 — Robust Deepfake Detection (B.Sc. project, frozen)

This is **DeepGuard v1**, frozen as tag `v1.0` on 16 Nov 2026 before work on v2 began (decision D-001).

v1 was presented at **ICET-AICS 2026** (Ramanujan College, University of Delhi). The live demo is the Hugging Face Space [`mynkey/deepguard`](https://huggingface.co/spaces/mynkey/deepguard).

## What v1 is
- **A five-branch video deepfake detector** (22.18 M parameters):
  - A: EfficientNet-B3 on aligned face crops;
  - B: EfficientNet-B0 on an inter-channel high-frequency difference image;
  - C: an MLP on a 9,369-d block-energy + DCCM vector;
  - D: PRNU statistics;
  - E: rPPG statistics;
  - all fused by a softmax attention gate.
- **Training data:** FaceForensics++ C23, video-level split 70/15/15.
- **Reported results:**
  - FF++ validation AUC 0.9158;
  - FF++ test AUC 0.8781;
  - Celeb-DF v2 test AUC 0.7272.

## Layout at this tag
| Path | Contents |
|---|---|
| repo root | Training, feature, evaluation and preprocessing code, as used on the original training machine (`main_v2.py`, `scripts/preprocess_final.py`, `features/`, `contourlet/`, `models/`, `train/`, `eval/`). `app/` is empty in the original project. |
| `deploy/` | The first Streamlit demo (superseded: it fed zeros to branches C–E). The model file is not in git. |
| `space/` | The Gradio app that runs on the Hugging Face Space. Its inference pipeline matches the training code exactly (0.0 difference). |
| `results_figures/` | Figures used in the report. |

## What is not in git
- **Data:** FF++ and Celeb-DF are licensed and never committed.
- **Checkpoints, `model.pt` and results JSONs:** stored in the private Kaggle dataset `mynkey/deepguard-v1-frozen`, with a SHA-256 `MANIFEST.csv`. That dataset is the complete snapshot of the original project folder, including scripts missing from this export (for example `main_v2_aug.py` and `main_v2_spatial.py`).
- **`.env`:** removed. It only held local Windows data paths.

## Known issues (why v2 exists)
These are documented in `docs/DECISIONS.md` on the main branch, §0.2.
- Branch C never called its contourlet transform.
- The 9,369-d vector is position-specific.
- Branch gains sit inside the statistical noise.
- PRNU and rPPG are per-clip constants.
- All training fakes come from 2016–2020 methods.
