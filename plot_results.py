"""
plot_results.py
────────────────────────────────────────────────
Generates all figures for the paper from results.pkl.

Run this AFTER experiment.py has finished.

Produces 4 figures:
  fig1_accuracy_vs_noise.png  — 3×3 accuracy curves with error bars
  fig2_qrs_comparison.png     — QRS bar chart per dataset
  fig3_noiseless_accuracy.png — Noiseless accuracy with error bars
  fig4_qrs_heatmap.png        — QRS heatmap (best config at a glance)
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Load saved results ────────────────────────────────────────
with open('results.pkl', 'rb') as f:
    data = pickle.load(f)

all_results  = data['all_results']
summary      = data['summary']
DATASETS     = data['DATASETS']
NOISE_TYPES  = data['NOISE_TYPES']
NOISE_LEVELS = data['NOISE_LEVELS']
CONFIGS      = data['CONFIGS']
CFG_NAMES    = list(CONFIGS.keys())

# ── Display labels ────────────────────────────────────────────
DS_LABELS = {
    'iris':          'Iris',
    'breast_cancer': 'Breast Cancer',
    'wine':          'Wine'
}
NT_LABELS = {
    'depolarizing':  'Depolarizing',
    'bit_flip':      'Bit-Flip',
    'phase_damping': 'Phase Damping'
}

# ── Consistent style for each configuration ───────────────────
COLORS  = {
    'Baseline (depth=2)':    '#2563EB',   # blue
    'Shallow (depth=1)':     '#16A34A',   # green
    'Noise-Aware (depth=2)': '#DC2626',   # red
}
MARKERS = {
    'Baseline (depth=2)':    'o',
    'Shallow (depth=1)':     's',
    'Noise-Aware (depth=2)': '^',
}


# ─────────────────────────────────────────────────────────────
# FIGURE 1 — Accuracy vs Noise Level (3 × 3 subplot grid)
#
# Rows = datasets (Iris, Breast Cancer, Wine)
# Cols = noise models (Depolarizing, Bit-Flip, Phase Damping)
# Lines = circuit configurations (3 lines per subplot)
# Error bars = ± std across 5 seeds
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(3, 3, figsize=(14, 11))
fig.suptitle(
    'Test Accuracy vs Noise Level — All Datasets and Noise Models',
    fontsize=14, fontweight='bold', y=1.01
)

for r, ds in enumerate(DATASETS):
    for c, nt in enumerate(NOISE_TYPES):
        ax = axes[r][c]

        for cfg in CFG_NAMES:
            means = [summary[ds][cfg]['acc_matrix'][nt][nl][0]
                     for nl in NOISE_LEVELS]
            stds  = [summary[ds][cfg]['acc_matrix'][nt][nl][1]
                     for nl in NOISE_LEVELS]

            ax.errorbar(
                NOISE_LEVELS, means, yerr=stds,
                label=cfg,
                color=COLORS[cfg],
                marker=MARKERS[cfg],
                linewidth=1.8,
                markersize=6,
                capsize=4,
                elinewidth=1.2
            )

        # Also plot the noiseless baseline as a horizontal dashed line
        for cfg in CFG_NAMES:
            nl_mean = summary[ds][cfg]['noiseless_mean']
            ax.axhline(
                nl_mean,
                color=COLORS[cfg],
                linestyle='--',
                linewidth=0.7,
                alpha=0.4
            )

        ax.set_title(
            f'{DS_LABELS[ds]} — {NT_LABELS[nt]}',
            fontsize=9.5, fontweight='bold'
        )
        ax.set_xlabel('Noise Level (p)', fontsize=8.5)
        ax.set_ylabel('Test Accuracy', fontsize=8.5)
        ax.set_ylim(0.0, 1.12)
        ax.set_xticks(NOISE_LEVELS)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Legend only on top-right subplot
        if r == 0 and c == 2:
            ax.legend(fontsize=7.5, loc='lower left',
                      framealpha=0.9, edgecolor='gray')

plt.tight_layout()
plt.savefig('fig1_accuracy_vs_noise.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig1_accuracy_vs_noise.png")


# ─────────────────────────────────────────────────────────────
# FIGURE 2 — QRS Comparison Bar Chart
#
# One subplot per dataset.
# Grouped bars: one group per noise model.
# Three bars per group: one per configuration.
# Annotated with QRS values on top of each bar.
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle(
    'Quantum Robustness Score (QRS) by Configuration and Noise Model',
    fontsize=13, fontweight='bold'
)

x     = np.arange(len(NOISE_TYPES))
width = 0.25

for ci, ds in enumerate(DATASETS):
    ax = axes[ci]

    for bi, cfg in enumerate(CFG_NAMES):
        qrs_vals = [summary[ds][cfg]['qrs_per_noise'][nt]
                    for nt in NOISE_TYPES]
        bars = ax.bar(
            x + bi * width,
            qrs_vals,
            width,
            label=cfg,
            color=COLORS[cfg],
            alpha=0.85,
            edgecolor='black',
            linewidth=0.5
        )
        for bar, val in zip(bars, qrs_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f'{val:.2f}',
                ha='center', va='bottom',
                fontsize=7.5, fontweight='bold'
            )

    # Reference line at QRS = 1.0 (perfect resilience)
    ax.axhline(1.0, color='gray', linestyle='--',
               linewidth=0.8, alpha=0.6, label='Perfect QRS')

    ax.set_title(DS_LABELS[ds], fontsize=11, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(
        [NT_LABELS[nt] for nt in NOISE_TYPES],
        fontsize=8.5
    )
    ax.set_ylabel('QRS  (higher = more robust)', fontsize=9)
    ax.set_ylim(0.0, 1.18)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.legend(fontsize=7.5)

plt.tight_layout()
plt.savefig('fig2_qrs_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig2_qrs_comparison.png")


# ─────────────────────────────────────────────────────────────
# FIGURE 3 — Noiseless Accuracy with Error Bars
#
# Grouped bar chart.
# One group per dataset, three bars per group.
# Error bars = std across 5 seeds.
# ─────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 5))

x     = np.arange(len(DATASETS))
width = 0.25

for bi, cfg in enumerate(CFG_NAMES):
    means = [summary[ds][cfg]['noiseless_mean'] for ds in DATASETS]
    stds  = [summary[ds][cfg]['noiseless_std']  for ds in DATASETS]

    bars = ax.bar(
        x + bi * width,
        means,
        width,
        yerr=stds,
        label=cfg,
        color=COLORS[cfg],
        alpha=0.85,
        capsize=5,
        edgecolor='black',
        linewidth=0.5,
        error_kw={'elinewidth': 1.2}
    )
    for bar, m, s in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + s + 0.012,
            f'{m:.3f}',
            ha='center', va='bottom',
            fontsize=8.5, fontweight='bold'
        )

ax.set_title(
    'Noiseless Classification Accuracy (Mean ± Std across 5 Seeds)',
    fontsize=12, fontweight='bold'
)
ax.set_xticks(x + width)
ax.set_xticklabels(
    [DS_LABELS[d] for d in DATASETS],
    fontsize=11
)
ax.set_ylabel('Test Accuracy', fontsize=10)
ax.set_ylim(0.0, 1.15)
ax.legend(fontsize=9)
ax.grid(True, axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('fig3_noiseless_accuracy.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig3_noiseless_accuracy.png")


# ─────────────────────────────────────────────────────────────
# FIGURE 4 — Average QRS Heatmap
#
# Rows = datasets, Columns = configurations.
# Cell color encodes Avg QRS (green = robust, red = fragile).
# Best cell per row is marked with ★.
# ─────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 4))

qrs_matrix = np.array([
    [summary[ds][cfg]['avg_qrs'] for cfg in CFG_NAMES]
    for ds in DATASETS
])

im = ax.imshow(
    qrs_matrix,
    cmap='RdYlGn',
    vmin=0.6, vmax=1.0,
    aspect='auto'
)
plt.colorbar(im, ax=ax, label='Average QRS')

ax.set_xticks(range(len(CFG_NAMES)))
ax.set_xticklabels(CFG_NAMES, fontsize=9, rotation=15, ha='right')
ax.set_yticks(range(len(DATASETS)))
ax.set_yticklabels(
    [DS_LABELS[d] for d in DATASETS],
    fontsize=10
)

for i, ds in enumerate(DATASETS):
    best_j = int(np.argmax(qrs_matrix[i]))
    for j in range(len(CFG_NAMES)):
        label = f"{qrs_matrix[i, j]:.3f}"
        if j == best_j:
            label += "  ★"
        ax.text(
            j, i, label,
            ha='center', va='center',
            fontsize=11, fontweight='bold'
        )

ax.set_title(
    'Average QRS Heatmap  (★ = best config per dataset)',
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
plt.savefig('fig4_qrs_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig4_qrs_heatmap.png")

print("\nAll figures saved. Ready for paper.")
