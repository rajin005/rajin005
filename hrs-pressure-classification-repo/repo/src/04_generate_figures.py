#!/usr/bin/env python3
"""
FINAL POLISH PASS, all 8 figures.
- FIG1 rebuilt as MLP confusion matrix (legitimate, replaces the RF-100%-flagged version)
- All figures re-rendered at 300 DPI for print quality
- Consistent typography, spacing, and color palette across the full set
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import os, warnings
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                             precision_recall_fscore_support)
from sklearn.preprocessing import label_binarize
from sklearn.neural_network import MLPClassifier
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from scipy.stats import gaussian_kde
warnings.filterwarnings('ignore')

# ── Consistent global style, print quality ─────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.titlesize': 15,
    'axes.labelsize': 13,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.1,
})

CLASS_COLORS = {'HIGH': '#2166ac', 'LOW': '#d6604d', 'MEDIUM': '#4dac26'}
MODEL_COLORS = {'LR': '#7f7f7f', 'DT': '#d6604d', 'RF': '#762a83', 'MLP': '#1b7837'}
DATA_DIR = 'data/processed/'
MODEL_DIR = 'models/'
OUT = 'figures/'
import os
os.makedirs(OUT, exist_ok=True)

# ── Load everything once ────────────────────────────────────────────────────
train_df = pd.read_csv(DATA_DIR + 'train_set.csv')
test_df  = pd.read_csv(DATA_DIR + 'test_set.csv')
feature_cols = [c for c in train_df.columns if c not in ['time', 'pressure_state']]
X_train, y_train = train_df[feature_cols].values, train_df['pressure_state'].values
X_test,  y_test  = test_df[feature_cols].values,  test_df['pressure_state'].values

with open(MODEL_DIR + 'mlp_classifier.pkl', 'rb') as f: model_mlp = pickle.load(f)
with open(MODEL_DIR + 'random_forest.pkl', 'rb') as f:  model_rf  = pickle.load(f)
with open(MODEL_DIR + 'decision_tree.pkl', 'rb') as f:  model_dt  = pickle.load(f)
with open(MODEL_DIR + 'label_encoder.pkl', 'rb') as f:  le        = pickle.load(f)

y_test_enc  = le.transform(y_test)
y_train_enc = le.transform(y_train)
class_names = le.classes_
y_test_bin  = label_binarize(y_test_enc, classes=[0, 1, 2])
y_proba     = model_mlp.predict_proba(X_test)
y_pred_mlp  = model_mlp.predict(X_test)

print("Loaded data and models. Building final 8 figures at 300 DPI...\n")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — MLP confusion matrix (rebuilt, replaces the RF-100% figure)
# ══════════════════════════════════════════════════════════════════════════════
print("[1/8] Confusion matrix...")
cm = confusion_matrix(y_test_enc, y_pred_mlp)
norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(9, 7.5))
im = ax.imshow(norm, cmap='Blues', vmin=0, vmax=1, aspect='auto')
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Normalized Recall', fontweight='bold', fontsize=12)

for i in range(3):
    for j in range(3):
        count, pct = cm[i, j], norm[i, j]
        color = 'white' if pct > 0.5 else 'black'
        ax.text(j, i, f'{count}', ha='center', va='center',
                fontsize=24, fontweight='bold', color=color)
        ax.text(j, i + 0.26, f'({pct*100:.1f}%)', ha='center', va='center',
                fontsize=12.5, color=color, alpha=0.9)

ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(class_names, fontsize=13, fontweight='bold')
ax.set_yticklabels(class_names, fontsize=13, fontweight='bold')
ax.set_xlabel('Predicted Pressure State', labelpad=12)
ax.set_ylabel('True Pressure State', labelpad=12)
acc = (y_pred_mlp == y_test_enc).mean()
ax.set_title(f'MLP Classifier, Confusion Matrix\n'
             f'Test Set, Accuracy: {acc*100:.2f}%, n = {len(y_test_enc):,} samples', pad=16)

for i, cls in enumerate(class_names):
    ax.text(3.28, i, f'Recall\n{norm[i,i]*100:.1f}%', va='center', ha='left',
            fontsize=11.5, color=list(CLASS_COLORS.values())[i], fontweight='bold')
for j, cls in enumerate(class_names):
    p = cm[j, j] / cm[:, j].sum() if cm[:, j].sum() > 0 else 0
    ax.text(j, 3.28, f'Prec\n{p*100:.1f}%', ha='center', va='top',
            fontsize=11.5, color=list(CLASS_COLORS.values())[j], fontweight='bold')

plt.tight_layout()
fig.savefig(OUT + 'FIG1_confusion_matrix.png')
plt.close()
print("      saved (300 DPI)")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — ROC, zoomed, correctly anchored labels
# ══════════════════════════════════════════════════════════════════════════════
print("[2/8] ROC curves, zoomed...")
fig, ax = plt.subplots(figsize=(10, 7.5))
zoom_max = 0.12
label_positions = {'HIGH': 0.35, 'LOW': 0.65, 'MEDIUM': 0.5}

for i, cls in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    roc_auc = auc(fpr, tpr)
    mask = fpr <= zoom_max + 0.01
    color = list(CLASS_COLORS.values())[i]
    ax.plot(fpr[mask], tpr[mask], lw=3, color=color,
            label=f'{cls}  (AUC = {roc_auc:.4f})', marker='o',
            markersize=5, markevery=max(1, mask.sum() // 12))
    ax.fill_between(fpr[mask], tpr[mask], alpha=0.12, color=color)
    n_pts = mask.sum()
    anchor_idx = min(int(n_pts * label_positions[cls]), n_pts - 1)
    ax_pt = (fpr[mask][anchor_idx], tpr[mask][anchor_idx])
    ax.annotate(cls, xy=ax_pt, xytext=(ax_pt[0], ax_pt[1] - 0.018 - 0.012 * i),
                fontsize=11, fontweight='bold', color=color, ha='center',
                arrowprops=dict(arrowstyle='-', color=color, lw=1.2, alpha=0.7))

fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
auc_micro = auc(fpr_micro, tpr_micro)
mask_m = fpr_micro <= zoom_max + 0.01
ax.plot(fpr_micro[mask_m], tpr_micro[mask_m], lw=2.5, color='black',
        linestyle='--', label=f'Micro-avg (AUC = {auc_micro:.4f})')
ax.plot([0, zoom_max], [0, zoom_max], 'k:', lw=1.5, alpha=0.5, label='Random classifier')

ax.set_xlim([-0.002, zoom_max]); ax.set_ylim([0.85, 1.002])
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves, Zoomed Critical Region (FPR <= 0.12)\n'
             'MLP Classifier, One-vs-Rest, Test Set', pad=15)
ax.legend(loc='lower right', fontsize=11.5, framealpha=0.95)
ax.grid(True, alpha=0.25)
plt.tight_layout()
fig.savefig(OUT + 'FIG2_roc_zoomed.png')
plt.close()
print("      saved (300 DPI)")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Training dynamics, dual-axis, phase-annotated
# ══════════════════════════════════════════════════════════════════════════════
print("[3/8] Training dynamics...")
mlp_tracked = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                             early_stopping=True, validation_fraction=0.2,
                             n_iter_no_change=10, random_state=42,
                             learning_rate_init=0.001)
mlp_tracked.fit(X_train, y_train_enc)
loss = np.array(mlp_tracked.loss_curve_)
epochs = np.arange(1, len(loss) + 1)
window = max(3, len(loss) // 10)
roll = pd.Series(loss).rolling(window, center=True, min_periods=1).mean().values
d_loss = np.abs(np.gradient(loss))

fig, ax1 = plt.subplots(figsize=(11.5, 7))
ax1.plot(epochs, loss, color='#2166ac', lw=2.2, alpha=0.5, label='Training loss (raw)', zorder=2)
ax1.fill_between(epochs, loss, alpha=0.1, color='#2166ac')
ax1.plot(epochs, roll, color='#2166ac', lw=3.2, label=f'Smoothed (window={window})', zorder=3)

phase1_end = np.argmax(d_loss[:len(d_loss)//2] < d_loss[0]*0.15) or len(loss)//3
ax1.axvspan(1, phase1_end, alpha=0.07, color='red', label='Rapid descent')
ax1.axvspan(phase1_end, len(loss)*0.8, alpha=0.07, color='orange', label='Deceleration')
ax1.axvspan(len(loss)*0.8, len(loss), alpha=0.07, color='green', label='Convergence')

ax1.set_xlabel('Epoch'); ax1.set_ylabel('Cross-Entropy Loss', color='#2166ac')
ax1.tick_params(axis='y', labelcolor='#2166ac')

ax2 = ax1.twinx()
ax2.plot(epochs, d_loss, color='#d6604d', lw=1.6, linestyle=':', alpha=0.7,
         label='|dLoss| per epoch (rate)')
ax2.set_ylabel('|dLoss| per Epoch', color='#d6604d')
ax2.tick_params(axis='y', labelcolor='#d6604d')
ax2.set_ylim(bottom=0)

ax1.annotate(f'Initial loss: {loss[0]:.3f}', xy=(1, loss[0]),
             xytext=(len(loss)*0.08, loss[0]*0.92),
             arrowprops=dict(arrowstyle='->', color='black', lw=1.4), fontsize=10.5, fontweight='bold')
ax1.annotate(f'Final loss: {loss[-1]:.4f}\n({(1-loss[-1]/loss[0])*100:.1f}% reduction)',
             xy=(len(loss), loss[-1]), xytext=(len(loss)*0.75, loss[-1]+loss[0]*0.25),
             arrowprops=dict(arrowstyle='->', color='black', lw=1.4), fontsize=10.5, fontweight='bold')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, loc='upper right', fontsize=9.5, framealpha=0.92)
ax1.set_title('MLP Training Dynamics, Loss Convergence with Phase Analysis\n'
              f'Early stopping at epoch {len(loss)}, Adam optimizer, lr = 0.001', pad=15)
ax1.grid(True, alpha=0.2)
plt.tight_layout()
fig.savefig(OUT + 'FIG3_training_dynamics.png')
plt.close()
print("      saved (300 DPI)")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Pareto cumulative-gain curve, feature importance
# ══════════════════════════════════════════════════════════════════════════════
print("[4/8] Feature importance, Pareto curve...")
importances = model_rf.feature_importances_
order = np.argsort(importances)[::-1]
feat_ordered = [feature_cols[i] for i in order]
vals_ordered = importances[order]
cumulative = np.cumsum(vals_ordered)
x = np.arange(1, len(vals_ordered) + 1)

fig, ax1 = plt.subplots(figsize=(10.5, 7.5))
l1, = ax1.plot(x, vals_ordered, color='#c0392b', lw=3, marker='o', markersize=11,
               markerfacecolor='white', markeredgewidth=2.5, markeredgecolor='#c0392b',
               label='Individual importance', zorder=4)
ax1.fill_between(x, vals_ordered, alpha=0.12, color='#c0392b')
for xi, vi in zip(x, vals_ordered):
    ax1.annotate(f'{vi*100:.1f}%', (xi, vi), textcoords='offset points', xytext=(0, 14),
                 ha='center', fontsize=10.5, fontweight='bold', color='#c0392b')
ax1.set_ylabel('Individual Feature Importance', color='#c0392b', labelpad=10)
ax1.tick_params(axis='y', labelcolor='#c0392b')
ax1.set_ylim(0, max(vals_ordered)*1.35)

ax2 = ax1.twinx()
l2, = ax2.plot(x, cumulative, color='#2166ac', lw=3, marker='D', markersize=10,
               markerfacecolor='white', markeredgewidth=2.5, markeredgecolor='#2166ac',
               linestyle='--', label='Cumulative importance', zorder=5)
for xi, ci in zip(x, cumulative):
    ax2.annotate(f'{ci*100:.0f}%', (xi, ci), textcoords='offset points', xytext=(0, -20),
                 ha='center', fontsize=10.5, fontweight='bold', color='#2166ac')
ax2.axhline(0.80, color='gray', linestyle=':', lw=1.5, alpha=0.7)
ax2.text(0.55, 0.815, '80% cumulative threshold', fontsize=9.5, color='gray', style='italic')
ax2.set_ylabel('Cumulative Importance', color='#2166ac', labelpad=10)
ax2.tick_params(axis='y', labelcolor='#2166ac')
ax2.set_ylim(0, 1.12)

ax1.set_xticks(x)
ax1.set_xticklabels(feat_ordered, rotation=22, ha='right', fontsize=11, fontweight='bold')
ax1.set_xlabel('Feature (ranked by importance)', labelpad=12)
ax1.grid(True, alpha=0.2); ax1.set_axisbelow(True)
lines = [l1, l2]
ax1.legend(lines, [ln.get_label() for ln in lines], loc='center right', fontsize=11.5, framealpha=0.95)
ax1.set_title('Feature Importance, Pareto Cumulative-Gain Curve\n'
              'Random Forest, individual and cumulative importance across ranked features', pad=15)
plt.tight_layout()
fig.savefig(OUT + 'FIG4_feature_importance_pareto.png')
plt.close()
print("      saved (300 DPI)")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Grouped bar chart by class and metric
# ══════════════════════════════════════════════════════════════════════════════
print("[5/8] Grouped bar chart, per-class per-model...")
models = {'MLP': model_mlp, 'RF': model_rf, 'DT': model_dt}
metrics_list = ['Precision', 'Recall', 'F1-Score']
CLASS_BG = {'HIGH': '#e8f4f8', 'LOW': '#fce8e8', 'MEDIUM': '#e8f8e8'}

data = {}
for mname, model in models.items():
    yp = model.predict(X_test)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test_enc, yp, average=None)
    data[mname] = {cls: {'Precision': prec[ci], 'Recall': rec[ci], 'F1-Score': f1[ci]}
                   for ci, cls in enumerate(class_names)}

fig, ax = plt.subplots(figsize=(13, 7.5))
bar_width, metric_gap, class_gap = 0.22, 0.08, 0.6
x_pos = 0
x_labels_pos, x_labels_text, group_boundaries = [], [], []
model_names = list(models.keys())
colors_model = [MODEL_COLORS[m] for m in model_names]

for ci, cls in enumerate(class_names):
    class_start = x_pos
    ax.axvspan(class_start - 0.3, class_start + 3.5 + 0.3, alpha=0.08, color=CLASS_BG[cls], zorder=0)
    for metric in metrics_list:
        metric_start = x_pos
        for model_idx, mname in enumerate(model_names):
            val = data[mname][cls][metric]
            ax.bar(x_pos, val, bar_width, color=colors_model[model_idx],
                   edgecolor='black', linewidth=1.2, alpha=0.88, zorder=3)
            ax.text(x_pos, val + 0.006, f'{val:.3f}', ha='center', va='bottom',
                    fontsize=8.5, fontweight='bold', color='black', rotation=90)
            x_pos += bar_width
        metric_center = metric_start + (bar_width*len(model_names))/2 - bar_width/2
        x_labels_pos.append(metric_center); x_labels_text.append(metric)
        x_pos += metric_gap
    group_boundaries.append((class_start, x_pos))
    x_pos += class_gap

ax.set_xticks(x_labels_pos)
ax.set_xticklabels(x_labels_text, fontsize=10.5, fontweight='bold')
for ci, cls in enumerate(class_names):
    gs, ge = group_boundaries[ci]
    ax.text((gs+ge)/2, 1.048, cls, ha='center', va='bottom', fontsize=13, fontweight='bold', color='#333333')

ax.set_ylabel('Score', labelpad=12)
ax.set_ylim(0.94, 1.055); ax.set_xlim(-0.5, x_pos)
ax.grid(True, axis='y', alpha=0.25); ax.set_axisbelow(True)

from matplotlib.patches import Patch
legend_patches = [Patch(facecolor=MODEL_COLORS[m], edgecolor='black', linewidth=1.2, label=m)
                   for m in model_names]
ax.legend(handles=legend_patches, loc='lower center', ncol=3, fontsize=11.5,
          framealpha=0.95, bbox_to_anchor=(0.5, -0.20))
ax.set_title('Model Performance by Class and Metric\n'
             'Grouped bar chart, classes separated, metrics grouped, models side-by-side', pad=15)
plt.tight_layout()
fig.savefig(OUT + 'FIG5_grouped_bars.png')
plt.close()
print("      saved (300 DPI)")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Prediction confidence calibration
# ══════════════════════════════════════════════════════════════════════════════
print("[6/8] Confidence calibration...")
max_proba = y_proba.max(axis=1)
is_correct = (y_pred_mlp == y_test_enc)
conf_correct = max_proba[is_correct]
conf_incorrect = max_proba[~is_correct]

fig, ax = plt.subplots(figsize=(10.5, 7.5))
x_grid = np.linspace(0.3, 1.02, 500)

kde_c = gaussian_kde(conf_correct, bw_method=0.15)
y_c = kde_c(x_grid)
ax.plot(x_grid, y_c, color='#1b7837', lw=3, label=f'Correct predictions (n={len(conf_correct)})')
ax.fill_between(x_grid, y_c, alpha=0.25, color='#1b7837')

kde_i = gaussian_kde(conf_incorrect, bw_method=0.15)
y_i = kde_i(x_grid)
ax.plot(x_grid, y_i, color='#c0392b', lw=3, linestyle='--',
        label=f'Incorrect predictions (n={len(conf_incorrect)})')
ax.fill_between(x_grid, y_i, alpha=0.25, color='#c0392b')

ax.plot(conf_correct, np.full(len(conf_correct), -0.15), '|', color='#1b7837',
        alpha=0.25, markersize=8, markeredgewidth=1)
ax.plot(conf_incorrect, np.full(len(conf_incorrect), -0.35), '|', color='#c0392b',
        alpha=0.6, markersize=10, markeredgewidth=1.5)

ax.axvline(conf_correct.mean(), color='#1b7837', linestyle=':', lw=1.8, alpha=0.8)
ax.text(conf_correct.mean()-0.008, ax.get_ylim()[1]*0.92, f'mean={conf_correct.mean():.3f}',
        color='#1b7837', fontsize=9.5, rotation=90, va='top', ha='right', fontweight='bold')
ax.axvline(conf_incorrect.mean(), color='#c0392b', linestyle=':', lw=1.8, alpha=0.8)
ax.text(conf_incorrect.mean()-0.008, ax.get_ylim()[1]*0.5, f'mean={conf_incorrect.mean():.3f}',
        color='#c0392b', fontsize=9.5, rotation=90, va='top', ha='right', fontweight='bold')

ax.set_xlim(0.3, 1.02); ax.set_ylim(bottom=-0.5)
ax.set_xlabel('Maximum Predicted Class Probability (model confidence)', labelpad=10)
ax.set_ylabel('Density', labelpad=10)
ax.legend(loc='upper left', fontsize=11.5, framealpha=0.95)
ax.grid(True, alpha=0.2); ax.set_axisbelow(True)
ax.set_title('Prediction Confidence Calibration, MLP Classifier\n'
             'Does the model know when it is wrong? Tick marks = individual predictions', pad=15)
plt.tight_layout()
fig.savefig(OUT + 'FIG6_confidence_calibration.png')
plt.close()
print("      saved (300 DPI)")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 7 — Correlation heatmap, clustering-ordered, no dendrogram trees
# ══════════════════════════════════════════════════════════════════════════════
print("[7/8] Correlation heatmap...")
corr = train_df[feature_cols].corr().values
dist = 1 - np.abs(corr)
np.fill_diagonal(dist, 0)
dist = np.clip(dist, 0, None)
linkage_mat = linkage(squareform(dist), method='average')
dend = dendrogram(linkage_mat, no_plot=True, labels=feature_cols)
order = dend['leaves']
corr_ordered = corr[np.ix_(order, order)]
feat_ordered = [feature_cols[i] for i in order]

fig, ax_heat = plt.subplots(figsize=(8.5, 7.5))
im = ax_heat.imshow(corr_ordered, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
for i in range(len(feat_ordered)):
    for j in range(len(feat_ordered)):
        val = corr_ordered[i, j]
        color = 'white' if abs(val) > 0.6 else 'black'
        ax_heat.text(j, i, f'{val:.2f}', ha='center', va='center',
                     fontsize=11.5, color=color, fontweight='bold')
ax_heat.set_xticks(range(len(feat_ordered))); ax_heat.set_yticks(range(len(feat_ordered)))
ax_heat.set_xticklabels(feat_ordered, rotation=40, ha='right', fontsize=11, fontweight='bold')
ax_heat.set_yticklabels(feat_ordered, fontsize=11, fontweight='bold')
cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
cbar.set_label('Pearson r', fontweight='bold')
ax_heat.set_title('Feature Correlation Matrix (clustering-ordered)\n'
                   'Input features only, buffer pressures excluded (no target leakage)',
                   fontsize=14.5, fontweight='bold', pad=15)
plt.tight_layout()
fig.savefig(OUT + 'FIG7_correlation_dendrogram.png')
plt.close()
print("      saved (300 DPI)")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 8 — Radar chart, all models, all metrics
# ══════════════════════════════════════════════════════════════════════════════
print("[8/8] Radar chart...")
metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'HIGH F1', 'LOW F1', 'MED F1']
N = len(metric_labels)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

def get_radar_vals(model):
    yp = model.predict(X_test)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test_enc, yp, average='weighted')
    acc = (yp == y_test_enc).mean()
    _, _, f1_cls, _ = precision_recall_fscore_support(y_test_enc, yp, average=None)
    return [acc, prec, rec, f1, f1_cls[0], f1_cls[1], f1_cls[2]]

radar_data = {'MLP': get_radar_vals(model_mlp), 'RF': get_radar_vals(model_rf), 'DT': get_radar_vals(model_dt)}
RADAR_COLORS = {'MLP': '#1b7837', 'RF': '#762a83', 'DT': '#d6604d'}

fig, ax = plt.subplots(figsize=(9.5, 8.5), subplot_kw=dict(polar=True))
for r in [0.96, 0.97, 0.98, 0.99, 1.00]:
    ax.plot(angles, [r]*(N+1), color='gray', lw=0.6, alpha=0.4, linestyle='--')
    ax.text(0, r, f'{r:.2f}', ha='center', va='center', fontsize=8.5, color='gray')

radial_offset = {'MLP': 0.003, 'RF': 0.009, 'DT': 0.017}
horiz_stagger = {'MLP': 0, 'RF': 38, 'DT': 76}  # extra horizontal push, points, for the 0-degree vertex only
for mname, vals in radar_data.items():
    v = vals + vals[:1]
    color = RADAR_COLORS[mname]
    ax.plot(angles, v, lw=3, color=color, label=mname, marker='o', markersize=9)
    ax.fill(angles, v, alpha=0.12, color=color)
    for angle, val in zip(angles[:-1], vals):
        if abs(angle) < 1e-6:  # Accuracy vertex: horizontally-aligned labels need horizontal separation
            ax.annotate(f'{val:.4f}', xy=(angle, val),
                        xytext=(horiz_stagger[mname], 8), textcoords='offset points',
                        ha='left', va='bottom', fontsize=8.5, fontweight='bold', color=color)
        else:
            ax.annotate(f'{val:.4f}', xy=(angle, val), xytext=(angle, val + radial_offset[mname]),
                        ha='center', va='bottom', fontsize=8.5, fontweight='bold', color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metric_labels, fontsize=12, fontweight='bold')
ax.set_ylim([0.95, 1.025]); ax.set_yticks([])
ax.spines['polar'].set_visible(True)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=13, framealpha=0.95)
ax.set_title('Model Performance Radar Chart\nAll metrics across MLP, Random Forest, Decision Tree',
             fontsize=14.5, fontweight='bold', pad=30)
plt.tight_layout()
fig.savefig(OUT + 'FIG8_radar_chart.png')
plt.close()
print("      saved (300 DPI)")

print("\nAll 8 figures rebuilt at 300 DPI.")

# ── Report final sizes to confirm nothing exceeds viewer limits ────────────
print("\nFinal file sizes and dimensions:")
from PIL import Image
for i in range(1, 9):
    matches = [f for f in os.listdir(OUT) if f.startswith(f'FIG{i}_')]
    for f in matches:
        p = OUT + f
        img = Image.open(p)
        sz = os.path.getsize(p) / 1024
        print(f"  {f:45s}  {img.size[0]}x{img.size[1]}px  {sz:.0f} KB")
