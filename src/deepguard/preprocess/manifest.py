"""Master manifest v1 schema and validation (D-019.7).

One row per video. A row is either kept (``drop_reason`` empty) or dropped with a
reason; dropped rows stay in the manifest so every count can be traced.
"""
from __future__ import annotations

import pandas as pd

COLUMNS = {
    "key": str,             # unique: dataset/method/compression/vid
    "dataset": str,         # FFPP | DFD
    "method": str,          # original | Deepfakes | ... | DeepFakeDetection
    "compression": str,     # c23 | c40
    "vid": str,             # file stem
    "label": str,           # real | fake
    "target": str,          # FF++ target id / DFD actor of the original
    "source": str,          # FF++ source id / DFD second actor ("" for originals)
    "group": str,           # pair group: FF++ target id, DFD original stem
    "split_official": str,  # train | val | test
    "split_clean": str,     # train | val | test (7 leaks moved + identity clusters)
    "identity_cluster": int,
    "fps_changed": bool,
    "width_fixed": bool,
    "dfd_dup": bool,
    "dfd_offset": int,      # -1 when not DFD or not aligned
    "n_frames": int,
    "n_crops": int,
    "flags": str,           # ';'-separated: low_change, ambiguous, vote_split, ...
    "drop_reason": str,     # "" when kept
    "raw_sha256": str,
    "git_commit": str,
    "preprocess_version": str,
}

SPLITS = {"train", "val", "test"}


def validate(df: pd.DataFrame) -> list[str]:
    """Return a list of problems (empty list = valid)."""
    problems = []
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        return [f"missing columns: {missing}"]
    if df["key"].duplicated().any():
        problems.append(f"duplicate keys: {df.loc[df['key'].duplicated(), 'key'].head(5).tolist()}")
    kept = df[df["drop_reason"].fillna("") == ""]
    for col in ("split_official", "split_clean"):
        bad = kept[~kept[col].isin(SPLITS)]
        if len(bad):
            problems.append(f"{col}: {len(bad)} kept rows with invalid split")
    if not df["label"].isin({"real", "fake"}).all():
        problems.append("label must be real or fake")
    if (kept["n_crops"] <= 0).any():
        problems.append(f"{int((kept['n_crops'] <= 0).sum())} kept rows without crops")
    bad_sha = ~df["raw_sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}")
    if bad_sha.any():
        problems.append(f"{int(bad_sha.sum())} rows with invalid raw_sha256")
    if (df["git_commit"].astype(str).str.len() < 7).any():
        problems.append("git_commit missing")
    for col in ("fps_changed", "width_fixed", "dfd_dup"):
        if df[col].dtype != bool:
            problems.append(f"{col} must be boolean")
    # A pair group never straddles splits (a real video and its fakes stay together).
    for col in ("split_official", "split_clean"):
        n = kept.groupby("group")[col].nunique()
        if (n > 1).any():
            problems.append(f"{col}: {int((n > 1).sum())} groups span several splits")
    return problems
