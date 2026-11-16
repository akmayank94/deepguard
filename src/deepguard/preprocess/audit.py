"""Metadata-only shortcut AUC (D-005.3, D-019.2): can real vs fake be told apart
from properties that are not the manipulation itself? Must stay <= 0.60."""
from __future__ import annotations

import numpy as np
import pandas as pd

SHORTCUT_LIMIT = 0.60


def metadata_auc(df: pd.DataFrame, features: list[str], label: str = "is_fake",
                 group: str = "group", n_splits: int = 5, seed: int = 0) -> dict:
    """Out-of-fold AUC of a gradient-boosted classifier on `features` only.

    Folds are grouped (a real video and its fakes share a group), so a pair is
    never split across train and test. Also reports each feature alone.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    y = df[label].astype(int).to_numpy()
    g = df[group].to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError("need both classes")

    def oof(cols):
        x = df[cols].astype(float).to_numpy()
        p = np.zeros(len(df))
        for tr, te in GroupKFold(n_splits=n_splits).split(x, y, g):
            m = HistGradientBoostingClassifier(max_iter=100, random_state=seed)
            m.fit(x[tr], y[tr])
            p[te] = m.predict_proba(x[te])[:, 1]
        return float(roc_auc_score(y, p))

    auc = oof(features)
    single = {f: oof([f]) for f in features}
    return {"auc": auc, "single": single, "n": int(len(df)),
            "verdict": "ok" if auc <= SHORTCUT_LIMIT else "SHORTCUT"}
