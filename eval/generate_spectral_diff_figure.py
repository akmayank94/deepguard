"""
eval/generate_spectral_diff_figure.py

Generates Figure 3.4: Spectral difference map extraction
Shows 4 images in a row for BOTH a real and a fake clip:
    (1) Original face crop
    (2) E_R energy map
    (3) E_G energy map
    (4) Final stacked diff map [E_R-E_B, E_R-E_G, E_G-E_B]

Uses actual frames from FF++C23_preprocessed.

Run from project root:
    python eval/generate_spectral_diff_figure.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from PIL import Image
from scipy.ndimage import uniform_filter
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


def find_best_frame(base_dir, frame_idx=8):
    """Find a good aligned frame — picks middle frame for stable face."""
    clips = sorted(base_dir.iterdir()) if base_dir.exists() else []
    for clip in clips:
        aligned = clip / 'frames_aligned'
        if aligned.exists():
            frames = sorted(aligned.glob('*.jpg'))
            if len(frames) > frame_idx:
                return frames[frame_idx], clip.name
    return None, None


def load_frame(path):
    img = Image.open(path).convert('RGB').resize((224, 224), Image.BILINEAR)
    return np.array(img).astype(np.float32) / 255.0


def compute_energy(channel_f32):
    """High-frequency energy via high-pass filter."""
    smoothed = uniform_filter(channel_f32.astype(np.float64), size=3)
    return np.abs(channel_f32.astype(np.float64) - smoothed).astype(np.float32)


def compute_diff_map(img_f32):
    R, G, B = img_f32[:,:,0], img_f32[:,:,1], img_f32[:,:,2]
    eR, eG, eB = compute_energy(R), compute_energy(G), compute_energy(B)
    diff = np.stack([eR - eB, eR - eG, eG - eB], axis=2)
    mn, mx = diff.min(), diff.max()
    if mx - mn > 1e-8:
        diff = (diff - mn) / (mx - mn)
    return diff.astype(np.float32), eR, eG, eB


# ── Load frames ───────────────────────────────────────────────────────────
real_path, real_clip = find_best_frame(DATA_ROOT / 'real', frame_idx=8)
fake_path, fake_clip = find_best_frame(DATA_ROOT / 'fake', frame_idx=8)

if real_path is None or fake_path is None:
    print("ERROR: Could not find frames. Check DATA_ROOT_V2 in .env")
    exit(1)

print(f"Real clip : {real_clip}")
print(f"Fake clip : {fake_clip}")

real_img = load_frame(real_path)
fake_img = load_frame(fake_path)

real_diff, rER, rEG, rEB = compute_diff_map(real_img)
fake_diff, fER, fEG, fEB = compute_diff_map(fake_img)

# ── Plot ───────────────────────────────────────────────────────────────────
col_titles = [
    'Original face crop\n(224 × 224 aligned)',
    'E_R — Red channel\nenergy map',
    'E_G — Green channel\nenergy map',
    'Spectral diff map\n[E_R−E_B, E_R−E_G, E_G−E_B]',
]

fig = plt.figure(figsize=(15, 8.5)) # Slightly taller figure to accommodate the title
# Reduced top margin to 0.76 to make room for the 3-line suptitle
gs  = gridspec.GridSpec(2, 4, figure=fig,
                         hspace=0.30, wspace=0.10,
                         left=0.08, right=0.96,
                         top=0.76, bottom=0.08)

# Consistent vmax for energy maps across both rows
e_vmax = max(rER.max(), rEG.max(), fER.max(), fEG.max()) * 0.85

for row, (img, diff, eR, eG, label, color, clip_name) in enumerate([
    (real_img, real_diff, rER, rEG, 'REAL',  '#059669', real_clip),
    (fake_img, fake_diff, fER, fEG, 'FAKE',  '#DC2626', fake_clip),
]):
    for col, data in enumerate([img, eR, eG, diff]):
        ax = fig.add_subplot(gs[row, col])

        if col == 0:
            ax.imshow(np.clip(data, 0, 1))
            # Fixed Label badge - controlled padding using boxstyle
            ax.text(10, 208, f' {label} ', fontsize=10,
                    color='white', fontweight='bold',
                    bbox=dict(facecolor=color, alpha=0.92,
                              edgecolor='none', boxstyle='round,pad=0.4'))
            
            # Clip name small - shifted up slightly so it doesn't fall off the image
            short = clip_name[:28] + ('…' if len(clip_name) > 28 else '')
            ax.text(112, 218, short, ha='center', fontsize=6.5,
                    color='#d1d5db', clip_on=True)

        elif col in [1, 2]:
            im = ax.imshow(data, cmap='YlOrRd', vmin=0, vmax=e_vmax)
            if col == 2 and row == 0:
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Energy magnitude', fontsize=9)
                cbar.ax.tick_params(labelsize=8)

        else:   # col == 3  — diff map
            ax.imshow(np.clip(data, 0, 1))
            # Fixed Intensity annotation - controlled padding using boxstyle
            mean_intensity = data.mean()
            intensity_label = 'Low misalignment' if mean_intensity < 0.38 \
                              else 'Higher misalignment'
            ax.text(10, 208, f' {intensity_label} ', fontsize=8.5,
                    color='white', fontweight='bold',
                    bbox=dict(facecolor=color, alpha=0.80,
                              edgecolor='none', boxstyle='round,pad=0.4'))

        ax.axis('off')

        # Column titles only on first row
        if row == 0:
            ax.set_title(col_titles[col], fontsize=11,
                         pad=12, fontweight='bold')

# Row labels on left - shifted slightly to avoid overlapping axes
fig.text(0.02, 0.65, 'Real face', va='center', ha='left',
         fontsize=12, fontweight='bold', color='#059669', rotation=90)
fig.text(0.02, 0.25, 'Deepfake face', va='center', ha='left',
         fontsize=12, fontweight='bold', color='#DC2626', rotation=90)

fig.suptitle(
    'Figure 3.4: Inter-Channel Spectral Difference Map Extraction\n'
    'High-frequency energy differences between colour channels reveal '
    'subtle misalignments introduced by GAN synthesis\n'
    'not present in authentic camera-captured faces.',
    fontsize=13, fontweight='bold', y=0.96)

out_path = OUT / 'fig3_4_spectral_diff.png'
plt.savefig(out_path)
plt.close()
print(f"\n✓ Saved: {out_path.absolute()}")