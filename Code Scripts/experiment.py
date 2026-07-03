"""
experiment.py
────────────────────────────────────────────────
Main experiment runner for:
  "Noise-Resilient Quantum Machine Learning Models
   for Small-Scale Classification Tasks"

Version 2.0 — Updated experiments:
  A. Core QKSVM experiments (original 3 datasets + 2 new ones)
  B. Depth scaling study: depths 1–4
  C. Classical baseline comparison (RBF SVM, Logistic Regression)
  D. Kernel stability analysis (Frobenius norm)
  E. QRS validation study (QRS vs worst-case drop vs AUC-noise)

Run this first, then run plot_results.py.
Compatible with Qiskit 2.x
"""

import pickle
import warnings
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore', category=DeprecationWarning, module='qiskit')
warnings.filterwarnings('ignore', category=FutureWarning)

from qiskit.circuit.library import ZZFeatureMap

from utils import (
    load_dataset,
    split_dataset,
    compute_kernel,
    apply_noise_to_kernel,
    apply_noise_to_features,
    compute_qrs,
    compute_worst_case_drop,
    compute_auc_noise,
    compute_frobenius_norm,
    get_classical_models,
    GATE_COUNT,
)


# ─────────────────────────────────────────────────────────────
# EXPERIMENT CONFIGURATION
# ─────────────────────────────────────────────────────────────

# All five datasets
DATASETS     = ['iris', 'breast_cancer', 'wine', 'digits', 'adult']

NOISE_TYPES  = ['depolarizing', 'bit_flip', 'phase_damping']
NOISE_LEVELS = [0.01, 0.05, 0.10]
SEEDS        = [0, 1, 2, 3, 4]

N_QUBITS     = 4      # all datasets reduced to 4 features via PCA
TEST_SIZE    = 0.25
SVM_C        = 1.0
NOISE_AWARE_TRAIN_LEVEL = 0.03

# ── Experiment A: Core QKSVM configurations (original 3) ─────
CONFIGS = {
    'Baseline (depth=2)':    {'reps': 2, 'noise_aware': False},
    'Shallow (depth=1)':     {'reps': 1, 'noise_aware': False},
    'Noise-Aware (depth=2)': {'reps': 2, 'noise_aware': True},
}

# ── Experiment B: Depth scaling (depths 1–4) ─────────────────
DEPTH_SCALING_DEPTHS = [1, 2, 3, 4]

# ── Experiment C: Classical noise types (feature-level) ──────
CLASSICAL_NOISE_TYPES  = ['gaussian', 'bit_flip']
CLASSICAL_NOISE_LEVELS = [0.01, 0.05, 0.10]


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _make_feature_map(reps: int) -> ZZFeatureMap:
    return ZZFeatureMap(feature_dimension=N_QUBITS, reps=reps)


def _print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────
# EXPERIMENT A — Core QKSVM experiments
#
# Exactly mirrors the original Stage 3 design but now runs
# on all 5 datasets (Iris, Breast Cancer, Wine, Digits, Adult).
#
# For each dataset × config × noise_type × noise_level × seed:
#   1. Compute ideal quantum kernel
#   2. Train SVM on noiseless (or noise-aware) kernel
#   3. Test on both noiseless and noisy kernels
#   4. Record accuracy
#   5. Record Frobenius norm of kernel perturbation (Exp D)
# ─────────────────────────────────────────────────────────────

def run_experiment_A():
    _print_header("EXPERIMENT A — Core QKSVM (all 5 datasets)")

    # Storage: results[ds][cfg][noise_type][noise_level] → list of accs
    results = {}
    frobenius = {}   # frobenius[ds][cfg][noise_type][noise_level] → list

    for ds in DATASETS:
        results[ds]   = {}
        frobenius[ds] = {}
        for cfg in CONFIGS:
            results[ds][cfg] = {
                'noiseless': [],
                **{nt: {nl: [] for nl in NOISE_LEVELS} for nt in NOISE_TYPES}
            }
            frobenius[ds][cfg] = {
                nt: {nl: [] for nl in NOISE_LEVELS} for nt in NOISE_TYPES
            }

    for ds_name in DATASETS:
        print(f"\n  Dataset : {ds_name.upper()}")

        for cfg_name, cfg in CONFIGS.items():
            reps        = cfg['reps']
            noise_aware = cfg['noise_aware']
            feature_map = _make_feature_map(reps)
            print(f"    Config : {cfg_name}")

            for seed in SEEDS:
                X, y = load_dataset(ds_name, n_features=N_QUBITS, seed=seed)
                X_tr, X_te, y_tr, y_te = split_dataset(X, y, TEST_SIZE, seed)

                K_tr_ideal = compute_kernel(X_tr, X_tr, feature_map)
                K_te_ideal = compute_kernel(X_te, X_tr, feature_map)

                if noise_aware:
                    K_tr_train = apply_noise_to_kernel(
                        K_tr_ideal, 'depolarizing',
                        NOISE_AWARE_TRAIN_LEVEL, reps, seed, N_QUBITS
                    )
                    K_tr_train = (K_tr_train + K_tr_train.T) / 2.0
                else:
                    K_tr_train = K_tr_ideal

                svm = SVC(kernel='precomputed', C=SVM_C)
                svm.fit(K_tr_train, y_tr)

                # Noiseless accuracy
                acc_nl = accuracy_score(y_te, svm.predict(K_te_ideal))
                results[ds_name][cfg_name]['noiseless'].append(acc_nl)

                # Noisy accuracy + Frobenius norm
                for nt in NOISE_TYPES:
                    for nl in NOISE_LEVELS:
                        K_te_noisy = apply_noise_to_kernel(
                            K_te_ideal, nt, nl, reps, seed, N_QUBITS
                        )
                        acc_n = accuracy_score(y_te, svm.predict(K_te_noisy))
                        results[ds_name][cfg_name][nt][nl].append(acc_n)

                        frob = compute_frobenius_norm(K_te_ideal, K_te_noisy)
                        frobenius[ds_name][cfg_name][nt][nl].append(frob)

                print(f"      seed {seed}  noiseless={acc_nl:.3f}")

    return results, frobenius


# ─────────────────────────────────────────────────────────────
# EXPERIMENT B — Depth Scaling Study
#
# Extends the circuit depth from {1,2} to {1,2,3,4}.
# For each depth, a non-noise-aware QKSVM is trained and
# evaluated under all noise types and levels.
# Dataset: all 5, but primarily reported for Breast Cancer and
# Wine where QRS differences were largest.
#
# This formalizes the "shallow = more robust" observation into
# a quantitative QRS-vs-depth scaling relationship.
# ─────────────────────────────────────────────────────────────

def run_experiment_B():
    _print_header("EXPERIMENT B — Depth Scaling Study (depths 1–4)")

    # depth_results[ds][depth]['noiseless'] or [nt][nl] → list
    depth_results = {}
    for ds in DATASETS:
        depth_results[ds] = {}
        for d in DEPTH_SCALING_DEPTHS:
            depth_results[ds][d] = {
                'noiseless': [],
                **{nt: {nl: [] for nl in NOISE_LEVELS} for nt in NOISE_TYPES}
            }

    for ds_name in DATASETS:
        print(f"\n  Dataset : {ds_name.upper()}")
        for depth in DEPTH_SCALING_DEPTHS:
            feature_map = _make_feature_map(depth)
            print(f"    Depth={depth}")
            for seed in SEEDS:
                X, y = load_dataset(ds_name, n_features=N_QUBITS, seed=seed)
                X_tr, X_te, y_tr, y_te = split_dataset(X, y, TEST_SIZE, seed)

                K_tr = compute_kernel(X_tr, X_tr, feature_map)
                K_te = compute_kernel(X_te, X_tr, feature_map)

                svm = SVC(kernel='precomputed', C=SVM_C)
                svm.fit(K_tr, y_tr)

                acc_nl = accuracy_score(y_te, svm.predict(K_te))
                depth_results[ds_name][depth]['noiseless'].append(acc_nl)

                for nt in NOISE_TYPES:
                    for nl in NOISE_LEVELS:
                        K_te_noisy = apply_noise_to_kernel(
                            K_te, nt, nl, depth, seed, N_QUBITS
                        )
                        acc_n = accuracy_score(y_te, svm.predict(K_te_noisy))
                        depth_results[ds_name][depth][nt][nl].append(acc_n)

            print(f"      noiseless mean = "
                  f"{np.mean(depth_results[ds_name][depth]['noiseless']):.3f}")

    return depth_results


# ─────────────────────────────────────────────────────────────
# EXPERIMENT C — Classical Baseline Comparison
#
# Trains RBF SVM and Logistic Regression on clean features.
# At test time, injects feature-level noise (Gaussian and
# bit-flip) and measures accuracy degradation.
# Computes CRS (Classical Robustness Score) using the same
# QRS formula applied to classical accuracy retention.
#
# This establishes whether QRS captures uniquely quantum
# behavior or reflects general model fragility patterns.
# ─────────────────────────────────────────────────────────────

def run_experiment_C():
    _print_header("EXPERIMENT C — Classical Baseline Comparison")

    classical_models = get_classical_models()

    # classical_results[ds][model_name][noise_type][noise_level] → list
    classical_results = {}
    for ds in DATASETS:
        classical_results[ds] = {}
        for model_name in classical_models:
            classical_results[ds][model_name] = {
                'noiseless': [],
                **{nt: {nl: [] for nl in CLASSICAL_NOISE_LEVELS}
                   for nt in CLASSICAL_NOISE_TYPES}
            }

    for ds_name in DATASETS:
        print(f"\n  Dataset : {ds_name.upper()}")
        for model_name, model_template in classical_models.items():
            print(f"    Model : {model_name}")
            for seed in SEEDS:
                X, y = load_dataset(ds_name, n_features=N_QUBITS, seed=seed)
                X_tr, X_te, y_tr, y_te = split_dataset(X, y, TEST_SIZE, seed)

                # Clone the model for this seed (reset any fitted state)
                import copy
                model = copy.deepcopy(model_template)
                model.fit(X_tr, y_tr)

                # Noiseless accuracy
                acc_nl = accuracy_score(y_te, model.predict(X_te))
                classical_results[ds_name][model_name]['noiseless'].append(acc_nl)

                # Noisy accuracy
                for nt in CLASSICAL_NOISE_TYPES:
                    for nl in CLASSICAL_NOISE_LEVELS:
                        X_te_noisy = apply_noise_to_features(X_te, nt, nl, seed)
                        acc_n = accuracy_score(y_te, model.predict(X_te_noisy))
                        classical_results[ds_name][model_name][nt][nl].append(acc_n)

                print(f"      seed {seed}  noiseless={acc_nl:.3f}")

    return classical_results


# ─────────────────────────────────────────────────────────────
# SUMMARISE FUNCTIONS
# ─────────────────────────────────────────────────────────────

def summarise_A(results):
    """
    Aggregate Experiment A raw results into means, stds, QRS, and
    all three robustness metrics (QRS, worst-case drop, AUC-noise).
    """
    summary = {}
    for ds in DATASETS:
        summary[ds] = {}
        for cfg in CONFIGS:
            r = results[ds][cfg]
            nl_mean = float(np.mean(r['noiseless']))
            nl_std  = float(np.std(r['noiseless']))

            acc_matrix    = {}
            qrs_per_noise = {}
            wcd_per_noise = {}
            auc_per_noise = {}

            for nt in NOISE_TYPES:
                acc_matrix[nt] = {}
                noisy_means    = []
                for nl in NOISE_LEVELS:
                    m = float(np.mean(r[nt][nl]))
                    s = float(np.std(r[nt][nl]))
                    acc_matrix[nt][nl] = (m, s)
                    noisy_means.append(m)

                qrs_per_noise[nt] = compute_qrs(nl_mean, noisy_means)
                wcd_per_noise[nt] = compute_worst_case_drop(nl_mean, noisy_means)
                auc_per_noise[nt] = compute_auc_noise(nl_mean, noisy_means, NOISE_LEVELS)

            avg_qrs = float(np.mean(list(qrs_per_noise.values())))
            avg_wcd = float(np.mean(list(wcd_per_noise.values())))
            avg_auc = float(np.mean(list(auc_per_noise.values())))

            summary[ds][cfg] = {
                'noiseless_mean': nl_mean,
                'noiseless_std':  nl_std,
                'qrs_per_noise':  qrs_per_noise,
                'wcd_per_noise':  wcd_per_noise,
                'auc_per_noise':  auc_per_noise,
                'avg_qrs':        avg_qrs,
                'avg_wcd':        avg_wcd,
                'avg_auc':        avg_auc,
                'acc_matrix':     acc_matrix,
            }
    return summary


def summarise_B(depth_results):
    """Aggregate Experiment B depth scaling results."""
    summary = {}
    for ds in DATASETS:
        summary[ds] = {}
        for depth in DEPTH_SCALING_DEPTHS:
            r       = depth_results[ds][depth]
            nl_mean = float(np.mean(r['noiseless']))
            nl_std  = float(np.std(r['noiseless']))

            qrs_per_noise = {}
            for nt in NOISE_TYPES:
                noisy_means = [float(np.mean(r[nt][nl])) for nl in NOISE_LEVELS]
                qrs_per_noise[nt] = compute_qrs(nl_mean, noisy_means)

            avg_qrs = float(np.mean(list(qrs_per_noise.values())))

            summary[ds][depth] = {
                'noiseless_mean': nl_mean,
                'noiseless_std':  nl_std,
                'qrs_per_noise':  qrs_per_noise,
                'avg_qrs':        avg_qrs,
            }
    return summary


def summarise_C(classical_results):
    """Aggregate Experiment C classical baseline results + CRS."""
    summary = {}
    classical_models = get_classical_models()

    for ds in DATASETS:
        summary[ds] = {}
        for model_name in classical_models:
            r       = classical_results[ds][model_name]
            nl_mean = float(np.mean(r['noiseless']))
            nl_std  = float(np.std(r['noiseless']))

            crs_per_noise = {}
            acc_matrix    = {}
            for nt in CLASSICAL_NOISE_TYPES:
                acc_matrix[nt] = {}
                noisy_means    = []
                for nl in CLASSICAL_NOISE_LEVELS:
                    m = float(np.mean(r[nt][nl]))
                    s = float(np.std(r[nt][nl]))
                    acc_matrix[nt][nl] = (m, s)
                    noisy_means.append(m)
                # CRS uses the same formula as QRS
                crs_per_noise[nt] = compute_qrs(nl_mean, noisy_means)

            avg_crs = float(np.mean(list(crs_per_noise.values())))

            summary[ds][model_name] = {
                'noiseless_mean': nl_mean,
                'noiseless_std':  nl_std,
                'crs_per_noise':  crs_per_noise,
                'avg_crs':        avg_crs,
                'acc_matrix':     acc_matrix,
            }
    return summary


def summarise_frobenius(frobenius):
    """Aggregate Frobenius norm results from Experiment A."""
    frob_summary = {}
    for ds in DATASETS:
        frob_summary[ds] = {}
        for cfg in CONFIGS:
            frob_summary[ds][cfg] = {}
            for nt in NOISE_TYPES:
                frob_summary[ds][cfg][nt] = {}
                for nl in NOISE_LEVELS:
                    vals = frobenius[ds][cfg][nt][nl]
                    frob_summary[ds][cfg][nt][nl] = {
                        'mean': float(np.mean(vals)),
                        'std':  float(np.std(vals)),
                    }
    return frob_summary


# ─────────────────────────────────────────────────────────────
# PRINT RESULTS
# ─────────────────────────────────────────────────────────────

DS_LABELS = {
    'iris':          'Iris',
    'breast_cancer': 'Breast Cancer',
    'wine':          'Wine',
    'digits':        'Digits (0 vs 1)',
    'adult':         'Adult (Income)',
}
NT_LABELS = {
    'depolarizing':  'Depolarizing',
    'bit_flip':      'Bit-Flip',
    'phase_damping': 'Phase Damping',
}


def print_summary_A(summary_A):
    print("\n\n" + "="*70)
    print("  TABLE 1: Noiseless Accuracy — All Datasets (Mean ± Std, 5 seeds)")
    print("="*70)
    header = f"{'Config':<28}"
    for ds in DATASETS:
        header += f"  {DS_LABELS[ds]:>18}"
    print(header)
    print("-"*70)
    for cfg in CONFIGS:
        row = f"{cfg:<28}"
        for ds in DATASETS:
            m = summary_A[ds][cfg]['noiseless_mean']
            s = summary_A[ds][cfg]['noiseless_std']
            row += f"  {m:.3f}±{s:.3f}".rjust(18)
        print(row)

    print("\n\n" + "="*70)
    print("  TABLE 2: QRS per Noise Type (all datasets)")
    print("="*70)
    for ds in DATASETS:
        print(f"\n  Dataset: {DS_LABELS[ds]}")
        print(f"  {'Config':<28} {'Depol':>8} {'Bit-Flip':>10} "
              f"{'Phase':>8} {'Avg QRS':>10}")
        print("  " + "-"*65)
        for cfg in CONFIGS:
            q   = summary_A[ds][cfg]['qrs_per_noise']
            avg = summary_A[ds][cfg]['avg_qrs']
            print(f"  {cfg:<28} "
                  f"{q['depolarizing']:>8.4f} "
                  f"{q['bit_flip']:>10.4f} "
                  f"{q['phase_damping']:>8.4f} "
                  f"{avg:>10.4f}")

    print("\n\n" + "="*70)
    print("  TABLE E: QRS vs Worst-Case Drop vs AUC-Noise (Metric Validation)")
    print("="*70)
    for ds in DATASETS:
        print(f"\n  Dataset: {DS_LABELS[ds]}")
        print(f"  {'Config':<28} {'Avg QRS':>10} {'Avg WCD':>10} {'Avg AUC':>10}")
        print("  " + "-"*58)
        for cfg in CONFIGS:
            avg_qrs = summary_A[ds][cfg]['avg_qrs']
            avg_wcd = summary_A[ds][cfg]['avg_wcd']
            avg_auc = summary_A[ds][cfg]['avg_auc']
            print(f"  {cfg:<28} {avg_qrs:>10.4f} {avg_wcd:>10.4f} {avg_auc:>10.4f}")


def print_summary_B(summary_B):
    print("\n\n" + "="*70)
    print("  TABLE B: Depth Scaling — Avg QRS by Depth (all datasets)")
    print("="*70)
    header = f"{'Depth':<10}"
    for ds in DATASETS:
        header += f"  {DS_LABELS[ds]:>18}"
    print(header)
    print("-"*70)
    for depth in DEPTH_SCALING_DEPTHS:
        row = f"depth={depth:<4}"
        for ds in DATASETS:
            avg = summary_B[ds][depth]['avg_qrs']
            nl  = summary_B[ds][depth]['noiseless_mean']
            row += f"  QRS={avg:.3f}/acc={nl:.3f}".rjust(18)
        print(row)


def print_summary_C(summary_C):
    print("\n\n" + "="*70)
    print("  TABLE C: Classical Baseline CRS vs Quantum QRS")
    print("="*70)
    for ds in DATASETS:
        print(f"\n  Dataset: {DS_LABELS[ds]}")
        print(f"  {'Model':<28} {'Gaussian CRS':>14} {'Bit-Flip CRS':>14} {'Avg CRS':>10}")
        print("  " + "-"*68)
        for model_name in summary_C[ds]:
            c   = summary_C[ds][model_name]['crs_per_noise']
            avg = summary_C[ds][model_name]['avg_crs']
            print(f"  {model_name:<28} "
                  f"{c['gaussian']:>14.4f} "
                  f"{c['bit_flip']:>14.4f} "
                  f"{avg:>10.4f}")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':

    total_A = (len(DATASETS) * len(CONFIGS) *
               len(NOISE_TYPES) * len(NOISE_LEVELS) * len(SEEDS))
    total_B = (len(DATASETS) * len(DEPTH_SCALING_DEPTHS) *
               len(NOISE_TYPES) * len(NOISE_LEVELS) * len(SEEDS))
    total_C = (len(DATASETS) * 2 *   # 2 classical models
               len(CLASSICAL_NOISE_TYPES) * len(CLASSICAL_NOISE_LEVELS) * len(SEEDS))

    print("=" * 60)
    print("  QML NOISE RESILIENCE EXPERIMENTS v2.0")
    print("=" * 60)
    print(f"  Datasets       : {DATASETS}")
    print(f"  Seeds          : {SEEDS}")
    print(f"  Exp A runs     : {total_A}  (+ noiseless baselines)")
    print(f"  Exp B runs     : {total_B}  (depth scaling)")
    print(f"  Exp C runs     : {total_C}  (classical baselines)")
    print(f"  Estimated time : 5–15 minutes on a standard laptop")

    # ── Run all experiments ───────────────────────────────────
    print("\n[1/3] Running Experiment A (core QKSVM)...")
    results_A, frobenius_A = run_experiment_A()

    print("\n[2/3] Running Experiment B (depth scaling)...")
    results_B = run_experiment_B()

    print("\n[3/3] Running Experiment C (classical baselines)...")
    results_C = run_experiment_C()

    # ── Summarise ─────────────────────────────────────────────
    print("\nSummarising results...")
    summary_A   = summarise_A(results_A)
    summary_B   = summarise_B(results_B)
    summary_C   = summarise_C(results_C)
    frob_summary = summarise_frobenius(frobenius_A)

    # ── Print tables ──────────────────────────────────────────
    print_summary_A(summary_A)
    print_summary_B(summary_B)
    print_summary_C(summary_C)

    # ── Save everything ───────────────────────────────────────
    save_data = {
        # Raw results
        'results_A':        results_A,
        'results_B':        results_B,
        'results_C':        results_C,
        'frobenius_A':      frobenius_A,
        # Summaries
        'summary_A':        summary_A,
        'summary_B':        summary_B,
        'summary_C':        summary_C,
        'frob_summary':     frob_summary,
        # Config constants (needed by plot_results.py)
        'DATASETS':         DATASETS,
        'NOISE_TYPES':      NOISE_TYPES,
        'NOISE_LEVELS':     NOISE_LEVELS,
        'CLASSICAL_NOISE_TYPES':  CLASSICAL_NOISE_TYPES,
        'CLASSICAL_NOISE_LEVELS': CLASSICAL_NOISE_LEVELS,
        'SEEDS':            SEEDS,
        'CONFIGS':          CONFIGS,
        'DEPTH_SCALING_DEPTHS': DEPTH_SCALING_DEPTHS,
        'DS_LABELS':        DS_LABELS,
        'NT_LABELS':        NT_LABELS,
    }

    with open('results.pkl', 'wb') as f:
        pickle.dump(save_data, f)

    print("\n\nAll results saved to results.pkl")
    print("Now run:  python plot_results.py")
