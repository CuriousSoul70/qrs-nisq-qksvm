"""
plot_results.py
────────────────────────────────────────────────
Generates all figures for the paper from results.pkl.

Version 2.0 — Produces 8 figures:

  Original (updated for 5 datasets):
    fig1_accuracy_vs_noise.png      — accuracy curves per dataset/noise type
    fig2_qrs_comparison.png         — QRS bar chart per dataset
    fig3_noiseless_accuracy.png     — noiseless accuracy with error bars
    fig4_qrs_heatmap.png            — QRS heatmap across all 5 datasets

  New figures:
    fig5_depth_scaling.png          — QRS vs circuit depth (Exp B)
    fig6_classical_comparison.png   — QRS vs CRS side by side (Exp C)
    fig7_frobenius_vs_qrs.png       — kernel stability correlation (Exp D)
    fig8_metric_validation.png      — QRS vs WCD vs AUC-noise (Exp E)

Run this AFTER experiment.py has finished.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ── Load saved results ────────────────────────────────────────
with open('results.pkl', 'rb') as f:
    data = pickle.load(f)

summary_A   = data['summary_A']
summary_B   = data['summary_B']
summary_C   = data['summary_C']
frob_summary = data['frob_summary']

DATASETS              = data['DATASETS']
NOISE_TYPES           = data['NOISE_TYPES']
NOISE_LEVELS          = data['NOISE_LEVELS']
CLASSICAL_NOISE_TYPES = data['CLASSICAL_NOISE_TYPES']
CLASSICAL_NOISE_LEVELS = data['CLASSICAL_NOISE_LEVELS']
CONFIGS               = data['CONFIGS']
DEPTH_SCALING_DEPTHS  = data['DEPTH_SCALING_DEPTHS']
DS_LABELS             = data['DS_LABELS']
NT_LABELS             = data['NT_LABELS']
CFG_NAMES             = list(CONFIGS.keys())

# ── Visual style ──────────────────────────────────────────────
COLORS = {
    'Baseline (depth=2)':    '#2563EB',
    'Shallow (depth=1)':     '#16A34A',
    'Noise-Aware (depth=2)': '#DC2626',
}
MARKERS = {
    'Baseline (depth=2)':    'o',
    'Shallow (depth=1)':     's',
    'Noise-Aware (depth=2)': '^',
}
DEPTH_COLORS = {1: '#16A34A', 2: '#2563EB', 3: '#F59E0B', 4: '#DC2626'}
CLASSICAL_COLORS = {
    'RBF SVM':             '#7C3AED',
    'Logistic Regression': '#0891B2',
}


# ─────────────────────────────────────────────────────────────
# FIG 1 — Accuracy vs Noise (now 5 datasets × 3 noise types)
# Layout: one figure per dataset (5 figures saved as fig1a–fig1e)
# to keep each subplot readable. Each is a 1×3 grid.
# ─────────────────────────────────────────────────────────────

for ds_idx, ds in enumerate(DATASETS):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(
        f'Accuracy vs Noise Level — {DS_LABELS[ds]}',
        fontsize=13, fontweight='bold'
    )

    for c, nt in enumerate(NOISE_TYPES):
        ax = axes[c]
        for cfg in CFG_NAMES:
            means = [summary_A[ds][cfg]['acc_matrix'][nt][nl][0]
                     for nl in NOISE_LEVELS]
            stds  = [summary_A[ds][cfg]['acc_matrix'][nt][nl][1]
                     for nl in NOISE_LEVELS]
            ax.errorbar(
                NOISE_LEVELS, means, yerr=stds,
                label=cfg, color=COLORS[cfg], marker=MARKERS[cfg],
                linewidth=1.8, markersize=6, capsize=4, elinewidth=1.2
            )
            nl_mean = summary_A[ds][cfg]['noiseless_mean']
            ax.axhline(nl_mean, color=COLORS[cfg], linestyle='--',
                       linewidth=0.7, alpha=0.4)

        ax.set_title(NT_LABELS[nt], fontsize=10, fontweight='bold')
        ax.set_xlabel('Noise Level (p)', fontsize=9)
        ax.set_ylabel('Test Accuracy', fontsize=9)
        ax.set_ylim(0.0, 1.12)
        ax.set_xticks(NOISE_LEVELS)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
        if c == 2:
            ax.legend(fontsize=7.5, loc='lower left',
                      framealpha=0.9, edgecolor='gray')

    plt.tight_layout()
    suffix = ['a', 'b', 'c', 'd', 'e'][ds_idx]
    fname  = f'fig1{suffix}_accuracy_vs_noise_{ds}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")


# ─────────────────────────────────────────────────────────────
# FIG 2 — QRS Comparison Bar Chart (all 5 datasets)
# Layout: 5 subplots in a 2×3 grid (last cell empty)
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle(
    'Quantum Robustness Score (QRS) — All Datasets',
    fontsize=13, fontweight='bold'
)
axes_flat = axes.flatten()

x     = np.arange(len(NOISE_TYPES))
width = 0.25

for ci, ds in enumerate(DATASETS):
    ax = axes_flat[ci]
    for bi, cfg in enumerate(CFG_NAMES):
        qrs_vals = [summary_A[ds][cfg]['qrs_per_noise'][nt] for nt in NOISE_TYPES]
        bars = ax.bar(
            x + bi * width, qrs_vals, width,
            label=cfg, color=COLORS[cfg], alpha=0.85,
            edgecolor='black', linewidth=0.5
        )
        for bar, val in zip(bars, qrs_vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f'{val:.2f}', ha='center', va='bottom',
                    fontsize=7, fontweight='bold')

    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8,
               alpha=0.6, label='Perfect QRS')
    ax.set_title(DS_LABELS[ds], fontsize=11, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([NT_LABELS[nt] for nt in NOISE_TYPES], fontsize=8)
    ax.set_ylabel('QRS', fontsize=9)
    ax.set_ylim(0.0, 1.22)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.legend(fontsize=7)

# Hide the unused 6th subplot
axes_flat[5].set_visible(False)

plt.tight_layout()
plt.savefig('fig2_qrs_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig2_qrs_comparison.png")


# ─────────────────────────────────────────────────────────────
# FIG 3 — Noiseless Accuracy with Error Bars (all 5 datasets)
# ─────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 5))
x     = np.arange(len(DATASETS))
width = 0.25

for bi, cfg in enumerate(CFG_NAMES):
    means = [summary_A[ds][cfg]['noiseless_mean'] for ds in DATASETS]
    stds  = [summary_A[ds][cfg]['noiseless_std']  for ds in DATASETS]
    bars  = ax.bar(
        x + bi * width, means, width, yerr=stds,
        label=cfg, color=COLORS[cfg], alpha=0.85,
        capsize=5, edgecolor='black', linewidth=0.5,
        error_kw={'elinewidth': 1.2}
    )
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + s + 0.012,
                f'{m:.3f}', ha='center', va='bottom',
                fontsize=7.5, fontweight='bold')

ax.set_title('Noiseless Classification Accuracy (Mean ± Std, 5 Seeds)',
             fontsize=12, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels([DS_LABELS[d] for d in DATASETS], fontsize=10)
ax.set_ylabel('Test Accuracy', fontsize=10)
ax.set_ylim(0.0, 1.18)
ax.legend(fontsize=9)
ax.grid(True, axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('fig3_noiseless_accuracy.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig3_noiseless_accuracy.png")


# ─────────────────────────────────────────────────────────────
# FIG 4 — Average QRS Heatmap (all 5 datasets)
# ─────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))
qrs_matrix = np.array([
    [summary_A[ds][cfg]['avg_qrs'] for cfg in CFG_NAMES]
    for ds in DATASETS
])
im = ax.imshow(qrs_matrix, cmap='RdYlGn', vmin=0.6, vmax=1.0, aspect='auto')
plt.colorbar(im, ax=ax, label='Average QRS')

ax.set_xticks(range(len(CFG_NAMES)))
ax.set_xticklabels(CFG_NAMES, fontsize=9, rotation=15, ha='right')
ax.set_yticks(range(len(DATASETS)))
ax.set_yticklabels([DS_LABELS[d] for d in DATASETS], fontsize=10)

for i, ds in enumerate(DATASETS):
    best_j = int(np.argmax(qrs_matrix[i]))
    for j in range(len(CFG_NAMES)):
        label = f"{qrs_matrix[i, j]:.3f}"
        if j == best_j:
            label += "  ★"
        ax.text(j, i, label, ha='center', va='center',
                fontsize=10, fontweight='bold')

ax.set_title('Average QRS Heatmap  (★ = best config per dataset)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('fig4_qrs_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig4_qrs_heatmap.png")


# ─────────────────────────────────────────────────────────────
# FIG 5 — Depth Scaling Study
#
# One subplot per dataset.
# X-axis: circuit depth (1–4)
# Y-axis: Avg QRS and noiseless accuracy (dual lines)
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle(
    'QRS and Noiseless Accuracy vs Circuit Depth (Depth Scaling Study)',
    fontsize=13, fontweight='bold'
)
axes_flat = axes.flatten()

for ci, ds in enumerate(DATASETS):
    ax   = axes_flat[ci]
    ax2  = ax.twinx()   # second y-axis for noiseless accuracy

    avg_qrs = [summary_B[ds][d]['avg_qrs']        for d in DEPTH_SCALING_DEPTHS]
    nl_acc  = [summary_B[ds][d]['noiseless_mean']  for d in DEPTH_SCALING_DEPTHS]
    nl_std  = [summary_B[ds][d]['noiseless_std']   for d in DEPTH_SCALING_DEPTHS]

    line1, = ax.plot(DEPTH_SCALING_DEPTHS, avg_qrs,
                     color='#DC2626', marker='o', linewidth=2,
                     markersize=7, label='Avg QRS')
    line2  = ax2.errorbar(DEPTH_SCALING_DEPTHS, nl_acc, yerr=nl_std,
                          color='#2563EB', marker='s', linewidth=2,
                          markersize=7, capsize=4, label='Noiseless Acc')

    ax.set_title(DS_LABELS[ds], fontsize=11, fontweight='bold')
    ax.set_xlabel('Circuit Depth (reps)', fontsize=9)
    ax.set_ylabel('Avg QRS', fontsize=9, color='#DC2626')
    ax2.set_ylabel('Noiseless Accuracy', fontsize=9, color='#2563EB')
    ax.set_xticks(DEPTH_SCALING_DEPTHS)
    ax.set_ylim(0.5, 1.05)
    ax2.set_ylim(0.5, 1.05)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3, linestyle='--')

    lines  = [line1, line2]
    labels = ['Avg QRS', 'Noiseless Acc']
    ax.legend(lines, labels, fontsize=8, loc='lower left')

axes_flat[5].set_visible(False)
plt.tight_layout()
plt.savefig('fig5_depth_scaling.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig5_depth_scaling.png")


# ─────────────────────────────────────────────────────────────
# FIG 6 — Classical Baseline Comparison
#
# Side-by-side bar chart comparing Avg QRS (quantum) vs Avg CRS
# (classical) across all datasets.
# Shows whether QRS captures uniquely quantum behavior.
# ─────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(13, 6))

classical_model_names = list(summary_C[DATASETS[0]].keys())
n_groups = len(DATASETS)
n_bars   = len(CFG_NAMES) + len(classical_model_names)
width    = 0.12
x        = np.arange(n_groups)

all_labels  = CFG_NAMES + classical_model_names
all_colors  = ([COLORS[c] for c in CFG_NAMES] +
               [CLASSICAL_COLORS[m] for m in classical_model_names])
all_hatches = ['', '', '', '///', '///']

for bi, (label, color, hatch) in enumerate(zip(all_labels, all_colors, all_hatches)):
    if label in CFG_NAMES:
        vals = [summary_A[ds][label]['avg_qrs'] for ds in DATASETS]
    else:
        vals = [summary_C[ds][label]['avg_crs'] for ds in DATASETS]

    bars = ax.bar(
        x + bi * width, vals, width,
        label=label, color=color, alpha=0.85,
        edgecolor='black', linewidth=0.5, hatch=hatch
    )

ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_title(
    'Quantum QRS vs Classical CRS — Average Robustness Comparison',
    fontsize=12, fontweight='bold'
)
ax.set_xticks(x + (n_bars - 1) * width / 2)
ax.set_xticklabels([DS_LABELS[d] for d in DATASETS], fontsize=10)
ax.set_ylabel('Average Robustness Score', fontsize=10)
ax.set_ylim(0.0, 1.22)
ax.grid(True, axis='y', alpha=0.3, linestyle='--')
ax.legend(fontsize=8, ncol=2, loc='lower right')

# Add divider between quantum and classical groups
mid_x = (len(CFG_NAMES) - 0.5) * width
for xi in x:
    ax.axvline(xi + mid_x, color='black', linewidth=0.6,
               linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig('fig6_classical_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig6_classical_comparison.png")


# ─────────────────────────────────────────────────────────────
# FIG 7 — Kernel Stability: Frobenius Norm vs QRS
#
# Scatter plot of Frobenius norm (x) vs QRS (y) for all
# dataset × config × noise_type × noise_level combinations.
# Each config gets a different color/marker.
# A negative correlation validates that kernel degradation
# (high Frobenius norm) predicts lower QRS.
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle(
    'Kernel Stability: Frobenius Norm vs QRS  (Exp D validation)',
    fontsize=13, fontweight='bold'
)

for ci, nt in enumerate(NOISE_TYPES):
    ax = axes[ci]
    for cfg in CFG_NAMES:
        frob_vals = []
        qrs_vals  = []
        for ds in DATASETS:
            for nl in NOISE_LEVELS:
                frob_mean = frob_summary[ds][cfg][nt][nl]['mean']
                qrs_val   = summary_A[ds][cfg]['qrs_per_noise'][nt]
                frob_vals.append(frob_mean)
                qrs_vals.append(qrs_val)

        ax.scatter(frob_vals, qrs_vals,
                   color=COLORS[cfg], marker=MARKERS[cfg],
                   label=cfg, alpha=0.75, s=55, edgecolors='black',
                   linewidths=0.4)

    # Fit and plot a trend line across all points
    all_frob = []
    all_qrs  = []
    for cfg in CFG_NAMES:
        for ds in DATASETS:
            for nl in NOISE_LEVELS:
                all_frob.append(frob_summary[ds][cfg][nt][nl]['mean'])
                all_qrs.append(summary_A[ds][cfg]['qrs_per_noise'][nt])
    z   = np.polyfit(all_frob, all_qrs, 1)
    p   = np.poly1d(z)
    xf  = np.linspace(min(all_frob), max(all_frob), 100)
    ax.plot(xf, p(xf), 'k--', linewidth=1.2, alpha=0.5, label='Trend')

    # Pearson r annotation
    r = np.corrcoef(all_frob, all_qrs)[0, 1]
    ax.text(0.05, 0.08, f'r = {r:.3f}', transform=ax.transAxes,
            fontsize=9, color='black',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    ax.set_title(NT_LABELS[nt], fontsize=10, fontweight='bold')
    ax.set_xlabel('Frobenius Norm (normalized)', fontsize=9)
    ax.set_ylabel('QRS', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3, linestyle='--')
    if ci == 0:
        ax.legend(fontsize=7.5)

plt.tight_layout()
plt.savefig('fig7_frobenius_vs_qrs.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig7_frobenius_vs_qrs.png")


# ─────────────────────────────────────────────────────────────
# FIG 8 — Metric Validation: QRS vs Worst-Case Drop vs AUC-Noise
#
# For each dataset, three grouped bars per configuration:
#   QRS (blue), 1-WCD (orange, inverted so higher=better),
#   AUC-noise normalized (green).
# Shows that QRS is consistent with or provides better
# discrimination than the alternative metrics.
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle(
    'Metric Validation: QRS vs Worst-Case Drop vs AUC-Noise  (Exp E)',
    fontsize=13, fontweight='bold'
)
axes_flat = axes.flatten()

x_cfg = np.arange(len(CFG_NAMES))
width = 0.25
metric_colors = ['#2563EB', '#F59E0B', '#16A34A']
metric_labels = ['Avg QRS', '1 − Avg WCD  (higher=better)', 'Avg AUC-Noise']

for ci, ds in enumerate(DATASETS):
    ax = axes_flat[ci]

    qrs_vals = [summary_A[ds][cfg]['avg_qrs'] for cfg in CFG_NAMES]
    wcd_vals = [1.0 - summary_A[ds][cfg]['avg_wcd'] for cfg in CFG_NAMES]  # invert
    auc_vals = [summary_A[ds][cfg]['avg_auc'] for cfg in CFG_NAMES]

    for mi, (vals, color, label) in enumerate(
            zip([qrs_vals, wcd_vals, auc_vals], metric_colors, metric_labels)):
        bars = ax.bar(
            x_cfg + mi * width, vals, width,
            label=label, color=color, alpha=0.85,
            edgecolor='black', linewidth=0.5
        )
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f'{v:.2f}', ha='center', va='bottom',
                    fontsize=7, fontweight='bold')

    ax.set_title(DS_LABELS[ds], fontsize=11, fontweight='bold')
    ax.set_xticks(x_cfg + width)
    # Shorten config labels for readability
    short_labels = ['Baseline\n(d=2)', 'Shallow\n(d=1)', 'Noise-\nAware']
    ax.set_xticklabels(short_labels, fontsize=8)
    ax.set_ylabel('Metric Value', fontsize=9)
    ax.set_ylim(0.0, 1.25)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    if ci == 0:
        ax.legend(fontsize=7.5, loc='lower right')

axes_flat[5].set_visible(False)
plt.tight_layout()
plt.savefig('fig8_metric_validation.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig8_metric_validation.png")

print("\nAll figures saved. Ready for paper.")
print("Figures produced:")
print("  fig1a–e : Accuracy vs noise (one per dataset)")
print("  fig2    : QRS comparison bar chart")
print("  fig3    : Noiseless accuracy")
print("  fig4    : QRS heatmap")
print("  fig5    : Depth scaling study")
print("  fig6    : Classical baseline comparison")
print("  fig7    : Frobenius norm vs QRS scatter")
print("  fig8    : Metric validation (QRS vs WCD vs AUC)")
