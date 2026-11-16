# DeepGuard v2 — deepfake and AI-generated media detector

**Status: Phase 1 (data foundation), in progress.** v1, the B.Sc. project presented at ICET-AICS 2026, is frozen at tag `v1.0` and kept in `legacy/v1/`. Its live demo is [`mynkey/deepguard`](https://huggingface.co/spaces/mynkey/deepguard).

DeepGuard answers one question for the public: **"Is this real, or made or changed by AI?"** It covers face-swapped and reenacted videos, fully AI-generated video, and AI images. Voice comes in v2.1.

## Design in one paragraph
DeepGuard v2 is a multi-expert system:
- face manipulation;
- generated media;
- motion geometry;
- forensic signals;
- provenance (C2PA).

Each expert runs on frozen foundation encoders (CLIP / DINOv2) with light, parameter-efficient adaptation. The experts' scores go through a calibrated fusion that answers **Likely real / Uncertain / Likely AI**, with evidence for each verdict. Every design choice, and the evidence behind it, is recorded in [`docs/DECISIONS.md`](docs/DECISIONS.md).

## How the project is run
- **Decisions before results:** each phase writes its acceptance test into `DECISIONS.md` before it runs, and no phase starts until the previous one passes.
- **Leakage and shortcut controls:**
  - splits by identity, source video and generator;
  - cross-dataset de-duplication;
  - metadata-only shortcut audits;
  - a locked test vault (D-005, D-017, D-018).
- **Only measured numbers are reported**, each with the run that produced it: [`docs/OUTPUTS.md`](docs/OUTPUTS.md).
- **Compute is entirely free cloud:** Kaggle for preprocessing and training, Colab for data transfer, Hugging Face ZeroGPU for the app. Data never lives on a laptop.

## Repository layout
| Path | Contents |
|---|---|
| `docs/` | `DECISIONS.md` (design log), `DATASETS.md` (dataset facts and traps), `OUTPUTS.md` (measured results), `TERMS.md` (glossary) |
| `notebooks/phase0/` | Colab notebooks: official FF++ download → Drive, Drive → private Kaggle datasets, Celeb-DF upload and v1 freeze |
| `notebooks/phase1/` | Kaggle notebooks: the FF++/DFD data audit (Phase 1.1); preprocessing comes next |
| `src/deepguard/` | The shared package used by both the notebooks and the app (D-018.2) |
| `tests/`, `tools/check_repo.py` | CI: tests plus hygiene checks (no data, weights, secrets or notebook outputs in git) |
| `legacy/v1/` | v1 code (the snapshot at tag `v1.0`) |

## Data
Licensed datasets (FaceForensics++, Celeb-DF, DFDC, DF40 …) are obtained from their official sources under their terms. They are stored privately and **never committed or redistributed**. See `docs/DATASETS.md`.
