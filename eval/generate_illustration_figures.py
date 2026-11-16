"""
eval/generate_illustration_figures.py

Generates illustration figures from your ACTUAL preprocessed frames:
    fig3_4_spectral_diff.png  — spectral difference map (real vs fake)
    fig3_5_prnu_extraction.png — PRNU noise residual extraction

Uses actual frames from:
    FF++C23_preprocessed/real/real_000/frames_aligned/
    FF++C23_preprocessed/fake/Deepfakes_000_003/frames_aligned/

Run from project root:
    python eval/generate_illustration_figures.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from scipy.ndimage import uniform_filter, gaussian_filter
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
})

OUT = Path('results_figures')
OUT.mkdir(exist_ok=True)

DATA_ROOT = Path(os.environ.get(
    'DATA_ROOT_V2',
    'D:/DO NOT TOUCH/deepfake_microexpression/data/FF++C23_preprocessed'))

# ── Find actual frames ─────────────────────────────────────────────────────
def find_clip_frame(base_dir, frame_idx=5):
    """Find an actual aligned frame from the dataset."""
    if not base_dir.exists():
        return None
    clips = sorted(base_dir.iterdir())
    for clip in clips:
        aligned = clip / 'frames_aligned'
        if aligned.exists():
            frames = sorted(aligned.glob('*.jpg'))
            if len(frames) > frame_idx:
                return frames[frame_idx]
    return None

real_frame_path = find_clip_frame(DATA_ROOT / 'real', frame_idx=5)
fake_frame_path = find_clip_frame(DATA_ROOT / 'fake', frame_idx=5)

if real_frame_path is None or fake_frame_path is None:
    print("  WARNING: Could not find actual frames — using synthetic illustration")
    USE_REAL_FRAMES = False
else:
    print(f"  Real frame: {real_frame_path}")
    print(f"  Fake frame: {fake_frame_path}")
    USE_REAL_FRAMES = True


def load_frame(path):
    """Load frame as float32 RGB in [0,1]."""
    img = Image.open(path).convert('RGB').resize((224, 224), Image.BILINEAR)
    return np.array(img).astype(np.float32) / 255.0


def make_synthetic_face(fake=False, seed=42):
    """Synthetic face for illustration when real frames unavailable."""
    np.random.seed(seed)
    H, W = 224, 224
    img = np.zeros((H, W, 3), dtype=np.float32)
    img[:,:,0] = np.clip(0.78 + np.random.normal(0, 0.018, (H,W)), 0, 1)
    img[:,:,1] = np.clip(0.62 + np.random.normal(0, 0.018, (H,W)), 0, 1)
    img[:,:,2] = np.clip(0.52 + np.random.normal(0, 0.018, (H,W)), 0, 1)
    img[68:92,  64:92, :] = np.clip(img[68:92,  64:92, :] * 0.35, 0, 1)
    img[68:92, 132:158,:] = np.clip(img[68:92, 132:158,:] * 0.35, 0, 1)
    img[108:130,98:126,:] = np.clip(img[108:130,98:126,:] * 0.65, 0, 1)
    img[143:162,82:142,:] = np.clip(img[143:162,82:142,:] * 0.55, 0, 1)
    if fake:
        np.random.seed(seed + 100)
        img[:,:,0] = np.clip(img[:,:,0]+np.random.normal(0,0.038,(H,W)),0,1)
        img[:,:,1] = np.clip(img[:,:,1]+np.random.normal(0,0.052,(H,W)),0,1)
        img[:,:,2] = np.clip(img[:,:,2]+np.random.normal(0,0.029,(H,W)),0,1)
    return img


def compute_spectral_diff(img_f32):
    """Compute inter-channel high-frequency energy difference map."""
    R, G, B = img_f32[:,:,0], img_f32[:,:,1], img_f32[:,:,2]
    def energy(ch):
        return np.abs(ch - uniform_filter(ch.astype(np.float64), size=3))
    eR, eG, eB = energy(R), energy(G), energy(B)
    diff = np.stack([eR - eB, eR - eG, eG - eB], axis=2)
    mn, mx = diff.min(), diff.max()
    if mx - mn > 1e-8:
        diff = (diff - mn) / (mx - mn)
    return diff.astype(np.float32), eR, eG, eB


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIG 3.4 — SPECTRAL DIFFERENCE MAPS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if USE_REAL_FRAMES:
    real_img = load_frame(real_frame_path)
    fake_img = load_frame(fake_frame_path)
else:
    real_img = make_synthetic_face(fake=False)
    fake_img = make_synthetic_face(fake=True)

real_diff, rR, rG, rB = compute_spectral_diff(real_img)
fake_diff, fR, fG, fB = compute_spectral_diff(fake_img)

fig = plt.figure(figsize=(14, 8))
gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.12)

titles_top = ['Original face\n(224×224 crop)',
              'R-channel\nenergy map',
              'G-channel\nenergy map',
              'Spectral diff map\n[ΔE_R-B, ΔE_R-G, ΔE_G-B]']

# Row 1 — Real
for col, (img, title) in enumerate(zip(
        [real_img, rR, rG, real_diff], titles_top)):
    ax = fig.add_subplot(gs[0, col])
    if col == 0:
        ax.imshow(np.clip(img, 0, 1))
    elif col in [1, 2]:
        vmax = max(rR.max(), rG.max(), rB.max())
        im = ax.imshow(img, cmap='YlOrRd', vmin=0, vmax=vmax)
        if col == 2:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                         label='Energy')
    else:
        ax.imshow(np.clip(img, 0, 1))
    ax.set_title(title, fontsize=10, pad=4)
    ax.axis('off')
    if col == 0:
        ax.text(4, 216, ' REAL ', fontsize=10.5, color='white',
                fontweight='bold',
                bbox=dict(facecolor='#059669', alpha=0.9, pad=3,
                          boxstyle='round'))

# Row 2 — Fake
for col, (img, title) in enumerate(zip(
        [fake_img, fR, fG, fake_diff], titles_top)):
    ax = fig.add_subplot(gs[1, col])
    if col == 0:
        ax.imshow(np.clip(img, 0, 1))
    elif col in [1, 2]:
        vmax = max(fR.max(), fG.max(), fB.max())
        im = ax.imshow(img, cmap='YlOrRd', vmin=0, vmax=vmax)
        if col == 2:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                         label='Energy')
    else:
        ax.imshow(np.clip(img, 0, 1))
    ax.set_title(title if col == 0 else '', fontsize=10, pad=4)
    ax.axis('off')
    if col == 0:
        ax.text(4, 216, ' FAKE (Deepfakes) ', fontsize=10.5,
                color='white', fontweight='bold',
                bbox=dict(facecolor='#DC2626', alpha=0.9, pad=3,
                          boxstyle='round'))

# Row labels
fig.text(0.01, 0.73, 'Real\nFace', va='center', ha='left',
         fontsize=11, fontweight='bold', color='#059669',
         rotation=90)
fig.text(0.01, 0.30, 'Deepfake\nFace', va='center', ha='left',
         fontsize=11, fontweight='bold', color='#DC2626',
         rotation=90)

fig.suptitle(
    'Figure 3.4: Inter-Channel Spectral Difference Map Extraction\n'
    'GAN-generated faces show higher inter-channel energy misalignment '
    'than authentic camera-captured faces',
    fontsize=12, fontweight='bold', y=1.01)

plt.savefig(OUT / 'fig3_4_spectral_diff.png')
plt.close()
print("✓ Fig 3.4 — Spectral diff maps")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIG 3.5 — PRNU NOISE RESIDUAL EXTRACTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Use real frame for PRNU illustration
img_f = real_img.copy()

# Denoising (Gaussian filter — same as feature extractor)
denoised = np.stack(
    [gaussian_filter(img_f[:,:,c].astype(np.float64), sigma=1.2)
     for c in range(3)], axis=2).astype(np.float32)

noise = img_f - denoised

# Visualise noise (scale to [0,1] for display)
noise_vis = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)

# Per-channel noise stats
R_n = noise[:,:,0].flatten()
G_n = noise[:,:,1].flatten()
B_n = noise[:,:,2].flatten()
corr_rg = float(np.corrcoef(R_n, G_n)[0,1])
corr_rb = float(np.corrcoef(R_n, B_n)[0,1])
corr_gb = float(np.corrcoef(G_n, B_n)[0,1])

fig, axes = plt.subplots(1, 4, figsize=(15, 4.5))
fig.suptitle(
    'Figure 3.5: PRNU Noise Residual Extraction Pipeline\n'
    'Camera sensor noise is physically absent in GAN-generated faces',
    fontsize=12, fontweight='bold', y=1.02)

# Panel 1: original
axes[0].imshow(np.clip(img_f, 0, 1))
axes[0].set_title('1. Input face crop\n(aligned, 224×224)', fontsize=10.5)
axes[0].axis('off')
axes[0].text(4, 216, ' REAL ', fontsize=9, color='white',
             fontweight='bold',
             bbox=dict(facecolor='#059669', alpha=0.88, pad=2,
                       boxstyle='round'))

# Panel 2: denoised
axes[1].imshow(np.clip(denoised, 0, 1))
axes[1].set_title('2. Denoised image\n(Gaussian σ=1.2)', fontsize=10.5)
axes[1].axis('off')

# Panel 3: noise residual
im_n = axes[2].imshow(noise_vis, cmap='RdBu_r', vmin=0, vmax=1)
axes[2].set_title('3. Noise residual\n(input − denoised)', fontsize=10.5)
axes[2].axis('off')
plt.colorbar(im_n, ax=axes[2], fraction=0.046, pad=0.04,
             label='Normalised residual')

# Panel 4: inter-channel correlation scatter
np.random.seed(0)
sample = np.random.choice(len(R_n), min(2000, len(R_n)), replace=False)
axes[3].scatter(R_n[sample], G_n[sample],
                alpha=0.25, s=4, color='#1D4ED8', label='R vs G')
axes[3].scatter(R_n[sample], B_n[sample],
                alpha=0.20, s=4, color='#DC2626', label='R vs B')
axes[3].set_xlabel('R-channel noise', fontsize=10)
axes[3].set_ylabel('G / B channel noise', fontsize=10)
axes[3].set_title(
    f'4. Inter-channel correlation\nR↔G: {corr_rg:.3f}  '
    f'R↔B: {corr_rb:.3f}', fontsize=10.5)
axes[3].legend(fontsize=9, framealpha=0.9)
axes[3].grid(True, alpha=0.25)
axes[3].spines['top'].set_visible(False)
axes[3].spines['right'].set_visible(False)

# Arrow between panels
for i in range(3):
    fig.text(0.24 + i*0.185, 0.50, '→', fontsize=18,
             va='center', ha='center', color='#6B7280')

plt.tight_layout()
plt.savefig(OUT / 'fig3_5_prnu_extraction.png')
plt.close()
print("✓ Fig 3.5 — PRNU extraction")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIG — REAL vs FAKE SIDE BY SIDE (Fig 1.1)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if USE_REAL_FRAMES:
    fig, axes = plt.subplots(1, 2, figsize=(9, 5))
    fig.suptitle(
        'Figure 1.1: Authentic vs Deepfake — Visual Comparison\n'
        'Differences are subtle and often imperceptible to the naked eye',
        fontsize=12, fontweight='bold')

    axes[0].imshow(np.clip(real_img, 0, 1))
    axes[0].set_title('Authentic video frame', fontsize=12, pad=8)
    axes[0].axis('off')
    axes[0].text(4, 216, ' ORIGINAL ', fontsize=11, color='white',
                 fontweight='bold',
                 bbox=dict(facecolor='#059669', alpha=0.9, pad=4,
                           boxstyle='round'))

    axes[1].imshow(np.clip(fake_img, 0, 1))
    axes[1].set_title('Deepfake-manipulated frame', fontsize=12, pad=8)
    axes[1].axis('off')
    axes[1].text(4, 216, ' DEEPFAKE ', fontsize=11, color='white',
                 fontweight='bold',
                 bbox=dict(facecolor='#DC2626', alpha=0.9, pad=4,
                           boxstyle='round'))

    plt.tight_layout()
    plt.savefig(OUT / 'fig1_1_real_vs_fake.png')
    plt.close()
    print("✓ Fig 1.1 — Real vs fake comparison")