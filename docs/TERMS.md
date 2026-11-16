# TERMS.md — DeepGuard glossary

Each project term with a plain-language description and where it is used. Kept up to date as terms appear.

| Term | Description | Context in DeepGuard |
|---|---|---|
| Deepfake | Video or image where a real person's face, expression or voice was changed by AI. | Main target of the face expert E1. |
| AI-generated media | An image or video created entirely by a generator (for example Sora, Veo or Stable Diffusion), with no real camera footage. | Target of experts E2 (generated media) and E3 (motion). |
| Face swap | One person's face is replaced with another's (Deepfakes, FaceSwap, FaceShifter, DFD). | The fake shows the **source** person's identity. |
| Face reenactment | The person stays the same but their expressions and mouth are driven by another video (Face2Face, NeuralTextures). | The fake keeps the **target** person's identity. |
| Target / source video | FF++ fake `A_B`: frames and background come from target A; the new face or expressions come from source B. | Used for pairing, splits and face selection (D-017). |
| c0 / c23 / c40 | FF++ H.264 quality levels: lossless, light (CRF 23) and heavy (CRF 40) compression. | We use c23 and c40. |
| CRF | Constant Rate Factor: the H.264 quality setting; higher means more compression. | Degradation chain and robustness tests (P5). |
| fps | Frames per second of a video. | FF++ fakes take the source's fps (A4); never a model input (D-005.6). |
| Frame-aligned | A fake and its original show the same moment at the same frame number. | True for FF++; false for DFD (needs offset alignment, D-017.6). |
| Manifest | A table listing every file with its size and SHA-256 checksum. | Layer-1 integrity (D-013); every Kaggle upload is checked against it. |
| SHA-256 | A fingerprint of a file's bytes; any change gives a different value. | Proves files are complete and unchanged. |
| Layer 1 / 2 / 3 | Raw originals / processed crops and frames / feature embeddings. | Storage plan (D-013). |
| Data leakage | Information from test data reaching training, which inflates scores. | Controlled by D-005, D-017 and D-018. |
| Identity leakage | The same person appears in both training and test data. | 7 official FF++ split leaks; cross-dataset clustering (D-018.3). |
| Shortcut | A property unrelated to manipulation (resolution, length, fps, dataset style) that happens to separate real from fake. | Metadata audit; the Face2Face/NeuralTextures width shortcut (A6). |
| Metadata-only AUC | How well a model using only file properties (no pixels) tells real from fake. Above 0.60 means a shortcut exists. | Phase 1.1 audit (A6). |
| Source-label confounding | Real and fake come from different sources, so the model learns the source instead of the fake. | D-018.1, the biggest risk. |
| AUC | Area under the ROC curve: 0.5 is guessing, 1.0 is perfect separation. | Main paper-comparable metric. |
| FPR / recall at FPR | False-positive rate (real media called fake); recall is the share of fakes caught at a fixed FPR. | User-facing metrics (D-006, D-018.6). |
| ECE | Expected calibration error: how far the stated confidence is from the real accuracy. | Target ≤ 0.05 (D-007). |
| Uncertain band | The range of scores where DeepGuard says "Uncertain" instead of forcing a verdict. | Three-way output (D-003). |
| Calibration split | Data the experts never trained on, used only to fit the fusion and its thresholds. | Phase 4. |
| Expert (E1–E6) | A specialised detector for one kind of evidence (face, generated media, motion, forensic signal, provenance, voice). | v2 architecture (D-003). |
| Foundation encoder | A large pretrained image model (CLIP, DINOv2) used as a frozen feature extractor. | Base of E1, E2 and E3. |
| Effort / LayerNorm tuning | Ways to adapt a frozen encoder by training very few parameters, which generalises better to new fakes. | E1 design (D-003). |
| Embedding | The fixed-length vector an encoder produces for an image or frame. | Layer-3 features. |
| Contourlet transform | A multi-scale, multi-direction image decomposition (Laplacian pyramid + directional filter bank). | v1 never actually ran it; tested properly in D-011. |
| PRNU | Photo-response non-uniformity: a camera sensor's noise fingerprint. | v1 branch D; candidate E4 signal. |
| rPPG | Remote photoplethysmography: a pulse signal read from skin-colour changes in video. | v1 branch E; affected by the fps trap (T2). |
| C2PA / Content Credentials | A signed record inside a file stating which tool created or edited it. | E5 provenance check. |
| SynthID | Google's invisible watermark for AI content; it has a public detector but no API. | Linked from the app (E5). |
| P1–P5 | Evaluation protocols: in-domain, cross-dataset, cross-generator, in-the-wild, robustness. | D-006. |
| Test vault | Test data kept in a separate dataset that only the evaluation notebook can attach, with every run logged. | D-018.4. |
| Dev-wild / locked wild set | Real-world clips for finding failures during development / for the final exam only. | D-018.5. |
| Parity test | Proof that the app and the training pipeline give exactly the same inputs for the same video. | D-018.2; CI. |
| Stream and reduce | Download one chunk, keep only crops and features, delete it, move on. | DFDC and other large sets (D-013). |
| Signed URL | A time-limited download link (it contains `Expires` and `Signature`). | How DFDC chunks are downloaded (Phase 1.3). |
| YUV / colour matrix (BT.601, BT.709) | Videos store brightness (Y) and colour (U, V) separately; turning them into RGB needs a matrix and a value range. Different tools pick different defaults. | DeepGuard converts with one written rule (D-019.1). |
| 4:4:4 / 4:2:0 | How much colour detail is stored: full (4:4:4) or a quarter (4:2:0). | All FF++ videos are 4:4:4, DFD is 4:2:0 (DATASETS.md). |
| SCRFD | A fast, accurate face detector from InsightFace; gives a box and 5 landmarks (eyes, nose, mouth corners). | Candidate detector (D-019.6). Non-commercial licence. |
| MediaPipe face detection | Google's light face detector (full-range model for faces far from the camera). | The other candidate (D-019.6). |
| Face track | The same face followed across the sampled frames of one video. | Short tracks are false detections and are dropped (D-019.3). |
| Background difference | Average pixel change between a fake and its original outside the faces. Near 2 means the frames line up (only compression noise). | Right-face selection and DFD alignment checks. |
| ArcFace embedding | A 512-number fingerprint of a face's identity; two photos of the same person give similar fingerprints. | Identity clusters for leakage-free splits (D-019.8). |
| Contact sheet | A grid of frames saved as one image so a person can check many cases by eye. | Audit flags, detector disagreements, the 200-crop check. |
| Stage A / stage B (Phase 1.2) | Stage A answers the open questions on samples (detector, width, DFD offset, budget); stage B runs the full preprocessing. | Stage B starts only after stage A passes. |
