"""
eval/plot_results.py — generates all 7 paper figures.
Run AFTER evaluate_final.py and celebdf_eval.py.

fig1 — FF++ ROC curves per manipulation type
fig2 — Training curve with Celeb-DF reference line
fig3 — Per-type AUC bar chart
fig4 — Confusion matrix
fig5 — FF++ results table
fig6 — Cross-dataset ROC + all metrics comparison
fig7 — Comprehensive metrics table (both datasets)
"""

import sys, os, json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(override=True)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

model_dir    = os.environ.get("MODEL_DIR", "checkpoints")
run_dir      = Path(model_dir) / "run_full_hybrid"
out_dir      = run_dir / "figures"
out_dir.mkdir(exist_ok=True)

def _load(p):
    p = Path(p)
    return json.load(open(p)) if p.exists() else {}

results  = _load(run_dir / "final_results.json")
celebdf  = _load(run_dir / "celebdf_results.json")
history  = _load(run_dir / "history.json") if (run_dir/"history.json").exists() else []

if not results:
    print("ERROR: Run eval/evaluate_final.py first."); sys.exit(1)

COLORS = {
    'overall':        '#1f4e79',
    'Deepfakes':      '#2e75b6',
    'Face2Face':      '#ed7d31',
    'FaceShifter':    '#a9d18e',
    'FaceSwap':       '#ffc000',
    'NeuralTextures': '#7030a0',
}
MT_ORDER = ['Deepfakes','Face2Face','FaceShifter','FaceSwap','NeuralTextures']

plt.rcParams.update({
    'font.family':'DejaVu Sans','font.size':11,
    'axes.spines.top':False,'axes.spines.right':False,
    'axes.grid':True,'grid.alpha':0.3,'grid.linestyle':'--',
})

def save(fig, name):
    for ext in ('pdf','png'):
        fig.savefig(out_dir/f'{name}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {name}")

# ── helpers ──────────────────────────────────────────────────────────────────
def style_table(table, rows, highlight_last=True):
    table.auto_set_font_size(False); table.set_fontsize(10.5)
    for (r,c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor('#1f4e79')
            cell.set_text_props(color='white', fontweight='bold')
        elif highlight_last and r == len(rows)-1:
            cell.set_facecolor('#fff2cc')
            cell.set_text_props(fontweight='bold')
        else:
            cell.set_facecolor('#f8f9fa' if r%2==0 else 'white')
        cell.set_edgecolor('#dee2e6')

# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 — FF++ ROC curves
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 1: FF++ ROC curves...")
roc = results.get('roc_data', {})
fig, ax = plt.subplots(figsize=(7,6))
for name in ['overall']+MT_ORDER:
    if name not in roc: continue
    d = roc[name]
    lw = 2.5 if name=='overall' else 1.5
    ls = '-'  if name=='overall' else '--'
    ax.plot(d['fpr'], d['tpr'],
            color=COLORS.get(name,'gray'), lw=lw, ls=ls,
            label=f"{'Overall' if name=='overall' else name} (AUC={d['auc']:.3f})")
ax.plot([0,1],[0,1],'k:',lw=1,alpha=0.5,label='Random (AUC=0.500)')
ax.set_xlabel('False Positive Rate',fontsize=12)
ax.set_ylabel('True Positive Rate',fontsize=12)
ax.set_title('ROC Curves — FF++ Test Set (TTA + Clip Vote)',fontsize=13,fontweight='bold')
ax.legend(loc='lower right',fontsize=9,framealpha=0.9)
ax.set_xlim([-0.01,1.01]); ax.set_ylim([-0.01,1.01])
fig.tight_layout(); save(fig,'fig1_roc_curves')

# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 — Learning curve
# ─────────────────────────────────────────────────────────────────────────────
if history:
    print("Fig 2: Learning curve...")
    eps  = [h['ep'] for h in history]
    tr   = [h['tr'] for h in history]
    val  = [h['va'] for h in history]
    best_ep = eps[int(np.argmax(val))]; best_va = max(val)
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(eps, tr,  color='#2e75b6', lw=2, label='Train AUC')
    ax.plot(eps, val, color='#ed7d31', lw=2, label='Val AUC (FF++)')
    ax.axvline(x=5,  color='gray',   ls=':',  lw=1.5, alpha=0.7, label='Backbone unfrozen (ep.5)')
    ax.axvline(x=best_ep, color='green', ls='--', lw=1.5, alpha=0.7,
               label=f'Best val AUC={best_va:.3f} (ep.{best_ep})')
    ax.scatter([best_ep],[best_va],color='green',s=80,zorder=5)
    if celebdf and 'Test' in celebdf:
        cel_auc = celebdf['Test']['clip_level_tuned']['auc']
        ax.axhline(y=cel_auc, color='#7030a0', ls='-.', lw=1.5, alpha=0.8,
                   label=f'Celeb-DF v2 zero-shot AUC={cel_auc:.3f}')
    ax.set_xlabel('Epoch',fontsize=12); ax.set_ylabel('AUC',fontsize=12)
    ax.set_title('Training Curve with Cross-Dataset Reference',fontsize=13,fontweight='bold')
    ax.legend(fontsize=10,framealpha=0.9); ax.set_ylim([0.4,1.02])
    fig.tight_layout(); save(fig,'fig2_learning_curve')

# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 — Per-type AUC bar chart
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 3: Per-type AUC bar chart...")
clip_per = results.get('clip_level',{}).get('test',{}).get('per_type',{})
std_per  = results.get('standard',{}).get('test',{}).get('per_type',{})
types    = [mt for mt in MT_ORDER if mt in clip_per]
x = np.arange(len(types)); w = 0.35
fig, ax = plt.subplots(figsize=(9,5))
b1 = ax.bar(x-w/2, [std_per.get(t,{}).get('auc',0) for t in types],
            w, label='Standard (thr=0.5)', color='#2e75b6', alpha=0.85)
b2 = ax.bar(x+w/2, [clip_per.get(t,{}).get('auc',0) for t in types],
            w, label='TTA + Clip Vote',    color='#ed7d31', alpha=0.85)
for bar in list(b1)+list(b2):
    h=bar.get_height()
    ax.text(bar.get_x()+bar.get_width()/2, h+0.005, f'{h:.3f}',
            ha='center',va='bottom',fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels(types,fontsize=10)
ax.set_ylabel('AUC',fontsize=12); ax.set_ylim([0.6,1.02])
ax.set_title('Per-Manipulation AUC — FF++ Test Set',fontsize=13,fontweight='bold')
ax.legend(fontsize=10,framealpha=0.9)
ax.axhline(y=0.8,color='gray',ls=':',lw=1,alpha=0.5)
fig.tight_layout(); save(fig,'fig3_per_type_auc')

# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 — Confusion matrix
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 4: Confusion matrix...")
cm = np.array(results.get('confusion_matrix',[[0,0],[0,0]]))
if cm.size == 4:
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Real','Fake'],fontsize=12)
    ax.set_yticklabels(['Real','Fake'],fontsize=12)
    ax.set_xlabel('Predicted',fontsize=12); ax.set_ylabel('Actual',fontsize=12)
    ax.set_title('Confusion Matrix — FF++ Test Set (TTA)',fontsize=13,fontweight='bold')
    thresh = cm.max()/2.0
    for i in range(2):
        for j in range(2):
            ax.text(j,i,f'{cm[i,j]:,}',ha='center',va='center',fontsize=14,
                    fontweight='bold',color='white' if cm[i,j]>thresh else 'black')
    fig.tight_layout(); save(fig,'fig4_confusion_matrix')

# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 — FF++ results summary table
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 5: Results summary table...")
std   = results.get('standard',{}).get('test',{}).get('overall',{})
tuned = results.get('threshold_tuned',{}).get('test',{}).get('overall',{})
tta   = results.get('tta',{}).get('test',{}).get('overall',{})
clip  = results.get('clip_level',{}).get('test',{}).get('overall',{})

rows5 = [['Method','AUC','F1','Accuracy']]
if std:   rows5.append(['Standard (thr=0.50)',
    f"{std.get('auc',0):.4f}",f"{std.get('f1',0):.4f}",f"{std.get('acc',0):.4f}"])
if tuned: rows5.append([f"Threshold tuning (thr={tuned.get('threshold',0.5):.2f})",
    f"{tuned.get('auc',0):.4f}",f"{tuned.get('f1',0):.4f}",f"{tuned.get('acc',0):.4f}"])
if tta:   rows5.append(['+ Test-Time Augmentation',
    f"{tta.get('auc',0):.4f}",f"{tta.get('f1',0):.4f}",f"{tta.get('acc',0):.4f}"])
if clip:  rows5.append(['+ Clip-Level Vote (Final)',
    f"{clip.get('auc',0):.4f}",f"{clip.get('f1',0):.4f}",f"{clip.get('acc',0):.4f}"])

fig, ax = plt.subplots(figsize=(10,3.5)); ax.axis('off')
t5 = ax.table(cellText=rows5[1:],colLabels=rows5[0],
              cellLoc='center',loc='center',bbox=[0,0,1,1])
style_table(t5, rows5)
ax.set_title('FF++ Test Set Results — Hybrid Deepfake Detector',
             fontsize=13,fontweight='bold',pad=20)
fig.tight_layout(); save(fig,'fig5_results_table')

# ─────────────────────────────────────────────────────────────────────────────
# Fig 6 — Cross-dataset comparison (ROC + bar chart)
# ─────────────────────────────────────────────────────────────────────────────
if celebdf and 'Test' in celebdf:
    print("Fig 6: Cross-dataset comparison...")
    cel      = celebdf['Test']
    cel_clip = cel.get('clip_level_tuned',{})
    cel_fr   = cel.get('frame_level',{})

    fig, axes = plt.subplots(1,2,figsize=(13,5))

    # Left — ROC comparison
    ax = axes[0]
    if 'overall' in roc:
        ax.plot(roc['overall']['fpr'], roc['overall']['tpr'],
                color='#2e75b6', lw=2.5,
                label=f"FF++ Test (AUC={roc['overall']['auc']:.3f})")
    if 'roc' in cel:
        ax.plot(cel['roc']['fpr'], cel['roc']['tpr'],
                color='#ed7d31', lw=2.5, ls='--',
                label=f"Celeb-DF v2 zero-shot (AUC={cel_clip.get('auc',0):.3f})")
    ax.plot([0,1],[0,1],'k:',lw=1,alpha=0.5,label='Random (AUC=0.500)')
    ax.set_xlabel('False Positive Rate',fontsize=12)
    ax.set_ylabel('True Positive Rate',fontsize=12)
    ax.set_title('ROC: FF++ vs Celeb-DF v2 (Zero-Shot)',fontsize=12,fontweight='bold')
    ax.legend(loc='lower right',fontsize=10,framealpha=0.9)
    ax.set_xlim([-0.01,1.01]); ax.set_ylim([-0.01,1.01])

    # Right — All metrics bar chart
    ax2 = axes[1]
    metrics = ['AUC','F1','Accuracy']
    ff_vals = [clip.get('auc',0), clip.get('f1',0), clip.get('acc',0)]
    cd_vals = [cel_clip.get('auc',0), cel_clip.get('f1',0), cel_clip.get('acc',0)]
    x2=np.arange(len(metrics)); w2=0.35
    b1=ax2.bar(x2-w2/2, ff_vals, w2, label='FF++ Test (clip-level)',
               color='#2e75b6', alpha=0.85)
    b2=ax2.bar(x2+w2/2, cd_vals, w2, label='Celeb-DF v2 (zero-shot)',
               color='#ed7d31', alpha=0.85)
    for bar in list(b1)+list(b2):
        h=bar.get_height()
        ax2.text(bar.get_x()+bar.get_width()/2,h+0.01,f'{h:.3f}',
                 ha='center',va='bottom',fontsize=10,fontweight='bold')
    ax2.set_xticks(x2); ax2.set_xticklabels(metrics,fontsize=12)
    ax2.set_ylabel('Score',fontsize=12); ax2.set_ylim([0.0,1.15])
    ax2.set_title('All Metrics: FF++ vs Celeb-DF v2',fontsize=12,fontweight='bold')
    ax2.legend(fontsize=10,framealpha=0.9)
    ax2.axhline(y=0.7,color='gray',ls=':',lw=1,alpha=0.5)

    fig.suptitle('Cross-Dataset Generalization',fontsize=14,fontweight='bold',y=1.02)
    fig.tight_layout(); save(fig,'fig6_cross_dataset')

# ─────────────────────────────────────────────────────────────────────────────
# Fig 7 — Comprehensive all-metrics table (both datasets)
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 7: Comprehensive metrics table...")
cel_clip = celebdf.get('Test',{}).get('clip_level_tuned',{}) if celebdf else {}
cel_fr   = celebdf.get('Test',{}).get('frame_level',{}) if celebdf else {}

rows7 = [['Dataset / Evaluation','AUC','F1','Accuracy','Level']]
if std:
    rows7.append(['FF++ — Standard (thr=0.50)',
        f"{std.get('auc',0):.4f}",f"{std.get('f1',0):.4f}",
        f"{std.get('acc',0):.4f}",'Frame'])
if clip:
    rows7.append(['FF++ — TTA + Clip Vote (best)',
        f"{clip.get('auc',0):.4f}",f"{clip.get('f1',0):.4f}",
        f"{clip.get('acc',0):.4f}",'Clip'])
if cel_fr:
    rows7.append(['Celeb-DF v2 — Frame-level (zero-shot)',
        f"{cel_fr.get('auc',0):.4f}",f"{cel_fr.get('f1',0):.4f}",
        f"{cel_fr.get('acc',0):.4f}",'Frame'])
if cel_clip:
    rows7.append(['Celeb-DF v2 — Clip-level (zero-shot)',
        f"{cel_clip.get('auc',0):.4f}",f"{cel_clip.get('f1',0):.4f}",
        f"{cel_clip.get('acc',0):.4f}",'Clip'])

row_bg = ['#f8f9fa','#dce9f5','#fff3cd','#ffd700']
fig, ax = plt.subplots(figsize=(12,4)); ax.axis('off')
t7 = ax.table(cellText=rows7[1:],colLabels=rows7[0],
              cellLoc='center',loc='center',bbox=[0,0,1,1])
t7.auto_set_font_size(False); t7.set_fontsize(10.5)
for (r,c), cell in t7.get_celld().items():
    if r == 0:
        cell.set_facecolor('#1f4e79')
        cell.set_text_props(color='white',fontweight='bold')
    elif 1 <= r <= len(row_bg):
        cell.set_facecolor(row_bg[r-1])
        if r in (2,4): cell.set_text_props(fontweight='bold')
    cell.set_edgecolor('#dee2e6')
ax.set_title('Complete Evaluation Results — FF++ and Celeb-DF v2',
             fontsize=13,fontweight='bold',pad=20)
fig.tight_layout(); save(fig,'fig7_all_metrics_table')

print(f"\n  All 7 figures saved to: {out_dir}")
print(f"  Formats: .pdf (journal) + .png (preview)")