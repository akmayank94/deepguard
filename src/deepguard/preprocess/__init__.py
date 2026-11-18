"""Shared preprocessing used by every DeepGuard notebook and by the app (D-018.2).

Modules:
    video     decoding with an explicit colour rule, frame sampling, width fix
    faces     face detectors (SCRFD, MediaPipe) and track linking
    crop      square face crops and JPEG encoding
    select    right-face selection by pixel difference against the original
    align     width-transform test and DFD time-offset matching
    identity  ArcFace embeddings and identity clustering
    audit     metadata-only shortcut AUC
    manifest  master-manifest schema and validation
"""
PREPROCESS_VERSION = "1.2.1"
