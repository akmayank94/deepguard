"""
scripts/diagnose.py v2
Fixed: PCA was using first 500 samples which were all real.
Now samples randomly from train split.
"""

import sys, os
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(override=True)
sys.path.insert(0, str(Path(__file__).parent.parent))

import h5py
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA

G="\033[92m"; R="\033[91m"; Y="\033[93m"; C="\033[96m"; X="\033[0m"; B="\033[1m"
def _p(m): print(f"  {G}PASS{X}  {m}")
def _f(m): print(f"  {R}FAIL{X}  {m}")
def _i(m): print(f"  {C}INFO{X}  {m}")
def _w(m): print(f"  {Y}WARN{X}  {m}")

def main():
    cache_dir = os.environ.get("CACHE_DIR", str(Path(__file__).parent.parent/"cache"))
    cache_path = Path(cache_dir) / "features_pilot_seed42.h5"
    if not cache_path.exists():
        print(f"Cache not found: {cache_path}"); sys.exit(1)

    print(f"\n{'='*60}")
    print(f"{B}  Diagnostic v2{X}")
    print(f"{'='*60}")

    with h5py.File(cache_path, 'r') as f:

        print(f"\n{B}Check 1: Labels{X}")
        for split in ('train','val','test'):
            labels = f[split]['labels'][:]
            clips  = [c.decode() if isinstance(c,bytes) else c
                      for c in f[split]['clip_names'][:]]
            real_labels = {labels[i] for i,c in enumerate(clips) if 'real' in c.lower()}
            fake_labels = {labels[i] for i,c in enumerate(clips) if 'real' not in c.lower()}
            ok = real_labels=={0} and fake_labels=={1}
            ((_p if ok else _f))(f"{split}: real={real_labels} fake={fake_labels}")

        print(f"\n{B}Check 2: Feature separability (LogReg){X}")
        X_tr = f['train']['handcrafted'][:]
        y_tr = f['train']['labels'][:].astype(int)
        X_va = f['val']['handcrafted'][:]
        y_va = f['val']['labels'][:].astype(int)

        sc = StandardScaler()
        Xs_tr = sc.fit_transform(X_tr)
        Xs_va = sc.transform(X_va)

        lr = LogisticRegression(max_iter=300, C=0.1, random_state=42)
        lr.fit(Xs_tr, y_tr)
        tr_auc = roc_auc_score(y_tr, lr.predict_proba(Xs_tr)[:,1])
        va_auc = roc_auc_score(y_va, lr.predict_proba(Xs_va)[:,1])
        _i(f"LogReg C=0.1 — Train AUC: {tr_auc:.4f}  Val AUC: {va_auc:.4f}")
        if va_auc > 0.60:   _p(f"Features ARE separable (val_auc={va_auc:.4f})")
        elif va_auc > 0.52: _w(f"Weak signal (val_auc={va_auc:.4f})")
        else:               _f(f"NO signal (val_auc={va_auc:.4f})")

        # Also try energy alone and DCCM alone
        X_e = f['train']['energy'][:]
        X_d = f['train']['dccm'][:]
        X_ev = f['val']['energy'][:]
        X_dv = f['val']['dccm'][:]

        sc_e = StandardScaler(); Xe_tr = sc_e.fit_transform(X_e); Xe_va = sc_e.transform(X_ev)
        sc_d = StandardScaler(); Xd_tr = sc_d.fit_transform(X_d); Xd_va = sc_d.transform(X_dv)

        lr_e = LogisticRegression(max_iter=300, C=0.1, random_state=42).fit(Xe_tr, y_tr)
        lr_d = LogisticRegression(max_iter=300, C=0.1, random_state=42).fit(Xd_tr, y_tr)
        e_auc = roc_auc_score(y_va, lr_e.predict_proba(Xe_va)[:,1])
        d_auc = roc_auc_score(y_va, lr_d.predict_proba(Xd_va)[:,1])
        _i(f"Energy alone val AUC: {e_auc:.4f}")
        _i(f"DCCM   alone val AUC: {d_auc:.4f}")

        print(f"\n{B}Check 3: Feature value ranges{X}")
        e_real = X_e[y_tr==0]; e_fake = X_e[y_tr==1]
        d_real = X_d[y_tr==0]; d_fake = X_d[y_tr==1]
        _i(f"Energy real: mean={e_real.mean():.4f} std={e_real.std():.4f} "
           f"min={e_real.min():.4f} max={e_real.max():.4f}")
        _i(f"Energy fake: mean={e_fake.mean():.4f} std={e_fake.std():.4f} "
           f"min={e_fake.min():.4f} max={e_fake.max():.4f}")
        e_diff = abs(e_real.mean()-e_fake.mean())/(abs(e_real.mean())+abs(e_fake.mean())+1e-8)
        _i(f"Energy mean diff ratio: {e_diff*100:.4f}%")
        if e_diff > 0.01: _p(f"Energy features have signal")
        else: _f(f"Energy features still near-identical")

        print(f"\n{B}Check 4: Diff images{X}")
        ri = np.where(y_tr==0)[0]; fi = np.where(y_tr==1)[0]
        np.random.seed(42)
        ri_s = np.random.choice(ri, min(30,len(ri)), replace=False)
        fi_s = np.random.choice(fi, min(30,len(fi)), replace=False)
        r_imgs = f['train']['diff_images'][sorted(ri_s)]
        f_imgs = f['train']['diff_images'][sorted(fi_s)]
        diff_img_gap = abs(r_imgs.mean() - f_imgs.mean())
        _i(f"Real diff_img mean: {r_imgs.mean():.4f}  std: {r_imgs.std():.4f}")
        _i(f"Fake diff_img mean: {f_imgs.mean():.4f}  std: {f_imgs.std():.4f}")
        _i(f"Gap: {diff_img_gap:.4f}")
        if diff_img_gap > 0.01: _p(f"Diff images carry signal ({diff_img_gap:.4f})")
        else: _f(f"Diff images also have no signal")

        print(f"\n{B}Check 5: PCA (random sample, balanced){X}")
        np.random.seed(0)
        ri_pca = np.random.choice(ri, min(200,len(ri)), replace=False)
        fi_pca = np.random.choice(fi, min(200,len(fi)), replace=False)
        idx_pca = np.concatenate([ri_pca, fi_pca])
        X_pca_raw = Xs_tr[idx_pca]
        y_pca = y_tr[idx_pca]
        pca = PCA(n_components=2, random_state=42)
        X2 = pca.fit_transform(X_pca_raw)
        rc = X2[y_pca==0].mean(axis=0)
        fc = X2[y_pca==1].mean(axis=0)
        dist = np.linalg.norm(rc-fc)
        _i(f"Real center: {rc}")
        _i(f"Fake center: {fc}")
        _i(f"PCA cluster distance: {dist:.4f}")
        if dist > 0.5: _p(f"Clear PCA separation")
        elif dist > 0.1: _w(f"Weak PCA separation")
        else: _f(f"No PCA separation")

        print(f"\n{'='*60}")
        print(f"{B}  Summary{X}")
        print(f"{'='*60}")
        print(f"  Handcrafted val AUC : {va_auc:.4f}")
        print(f"  Energy val AUC      : {e_auc:.4f}")
        print(f"  DCCM val AUC        : {d_auc:.4f}")
        print(f"  Diff image gap      : {diff_img_gap:.4f}")
        print(f"  PCA dist            : {dist:.4f}")

        if va_auc > 0.60:
            print(f"\n  {G}Features work. Ready to train.{X}")
        elif e_auc > 0.55 or d_auc > 0.55:
            print(f"\n  {Y}Partial signal. Retrain with current features.{X}")
        else:
            print(f"\n  {R}Features still broken. Deeper fix needed.{X}")

if __name__ == "__main__":
    main()