"""
experiment.py
────────────────────────────────────────────────
Main experiment runner for:
  "Noise-Resilient Quantum Machine Learning Models
   for Small-Scale Classification Tasks"

What this file does:
  1. Loads 3 benchmark datasets (Iris, Breast Cancer, Wine)
  2. Builds a Quantum Kernel SVM for 3 circuit configurations
  3. Evaluates performance under 3 noise models × 3 noise levels
  4. Repeats everything across 5 random seeds
  5. Computes QRS (Quantum Robustness Score) for each setting
  6. Saves all results to results.pkl for plotting

Run this first, then run plot_results.py.

Compatible with Qiskit 2.x
"""

import pickle
import warnings
import numpy as np
from sklearn.svm import SVC

# Suppress Qiskit 2.1 deprecation warning for ZZFeatureMap
# (it still works fine; replacement API arrives in Qiskit 3.0)
warnings.filterwarnings('ignore', category=DeprecationWarning, module='qiskit')
from sklearn.metrics import accuracy_score
from qiskit.circuit.library import ZZFeatureMap

from utils import (
    load_dataset,
    split_dataset,
    compute_kernel,
    apply_noise_to_kernel,
    compute_qrs,
)

# ─────────────────────────────────────────────────────────────
# EXPERIMENT CONFIGURATION
# Change these if you want to run a smaller/larger experiment.
# ─────────────────────────────────────────────────────────────

DATASETS      = ['iris', 'breast_cancer', 'wine']
NOISE_TYPES   = ['depolarizing', 'bit_flip', 'phase_damping']
NOISE_LEVELS  = [0.01, 0.05, 0.10]
SEEDS         = [0, 1, 2, 3, 4]       # 5 runs per experiment
N_QUBITS      = 4                     # = number of features after PCA
TEST_SIZE     = 0.25
SVM_C         = 1.0                   # SVM regularization constant

# Three circuit configurations (your independent variable):
#   depth=2 → 24 gates (more expressive, more noise-prone)
#   depth=1 → 12 gates (shallower, more noise-resilient)
#   noise-aware → trains on noisy kernel (novel training strategy)
CONFIGS = {
    'Baseline (depth=2)':    {'reps': 2, 'noise_aware': False},
    'Shallow (depth=1)':     {'reps': 1, 'noise_aware': False},
    'Noise-Aware (depth=2)': {'reps': 2, 'noise_aware': True},
}

# Noise level used during noise-aware training
NOISE_AWARE_TRAIN_LEVEL = 0.03


# ─────────────────────────────────────────────────────────────
# RESULT STORAGE STRUCTURE
#
# all_results[dataset][config][noise_type][noise_level]
#   -> list of accuracy values (one per seed)
#
# all_results[dataset][config]['noiseless']
#   -> list of noiseless accuracy values (one per seed)
# ─────────────────────────────────────────────────────────────

def initialise_results():
    results = {}
    for ds in DATASETS:
        results[ds] = {}
        for cfg in CONFIGS:
            results[ds][cfg] = {
                'noiseless': [],
                **{nt: {nl: [] for nl in NOISE_LEVELS}
                   for nt in NOISE_TYPES}
            }
    return results


# ─────────────────────────────────────────────────────────────
# CORE EXPERIMENT LOOP
# ─────────────────────────────────────────────────────────────

def run_experiments():
    all_results = initialise_results()

    for ds_name in DATASETS:
        print(f"\n{'='*55}")
        print(f"  DATASET : {ds_name.upper()}")
        print(f"{'='*55}")

        for cfg_name, cfg in CONFIGS.items():
            reps        = cfg['reps']
            noise_aware = cfg['noise_aware']

            print(f"\n  Config  : {cfg_name}")

            feature_map = ZZFeatureMap(
                feature_dimension=N_QUBITS,
                reps=reps
            )

            for seed in SEEDS:

                # ── Load and split data ──────────────────────
                X, y = load_dataset(ds_name, n_features=N_QUBITS, seed=seed)
                X_tr, X_te, y_tr, y_te = split_dataset(
                    X, y, test_size=TEST_SIZE, seed=seed
                )

                # ── Compute ideal (noiseless) kernel matrices ─
                K_tr_ideal = compute_kernel(X_tr, X_tr, feature_map)
                K_te_ideal = compute_kernel(X_te, X_tr, feature_map)

                # ── Choose training kernel ────────────────────
                if noise_aware:
                    # Noise-aware strategy: train on a moderately
                    # noisy kernel so the model learns to handle noise
                    K_tr_train = apply_noise_to_kernel(
                        K_tr_ideal,
                        noise_type='depolarizing',
                        noise_level=NOISE_AWARE_TRAIN_LEVEL,
                        circuit_depth=reps,
                        seed=seed,
                        n_qubits=N_QUBITS
                    )
                    # Kernel matrix must be symmetric — enforce it
                    K_tr_train = (K_tr_train + K_tr_train.T) / 2.0
                else:
                    K_tr_train = K_tr_ideal

                # ── Train SVM on chosen training kernel ──────
                svm = SVC(kernel='precomputed', C=SVM_C)
                svm.fit(K_tr_train, y_tr)

                # ── Evaluate: noiseless ───────────────────────
                acc_nl = accuracy_score(y_te, svm.predict(K_te_ideal))
                all_results[ds_name][cfg_name]['noiseless'].append(acc_nl)

                # ── Evaluate: noisy (all combinations) ───────
                for nt in NOISE_TYPES:
                    for nl in NOISE_LEVELS:
                        K_te_noisy = apply_noise_to_kernel(
                            K_te_ideal,
                            noise_type=nt,
                            noise_level=nl,
                            circuit_depth=reps,
                            seed=seed,
                            n_qubits=N_QUBITS
                        )
                        acc_n = accuracy_score(y_te, svm.predict(K_te_noisy))
                        all_results[ds_name][cfg_name][nt][nl].append(acc_n)

                print(f"    Seed {seed}  |  noiseless acc = {acc_nl:.3f}")

    return all_results


# ─────────────────────────────────────────────────────────────
# SUMMARISE: compute means, stds, and QRS
# ─────────────────────────────────────────────────────────────

def summarise(all_results):
    """
    Aggregate raw seed-level results into summary statistics.

    Returns
    -------
    summary[dataset][config] = {
        'noiseless_mean' : float,
        'noiseless_std'  : float,
        'qrs_per_noise'  : {noise_type: qrs_value},
        'avg_qrs'        : float,
        'acc_matrix'     : {noise_type: {noise_level: (mean, std)}}
    }
    """
    summary = {}

    for ds in DATASETS:
        summary[ds] = {}
        for cfg in CONFIGS:
            r = all_results[ds][cfg]

            noiseless_mean = float(np.mean(r['noiseless']))
            noiseless_std  = float(np.std(r['noiseless']))

            acc_matrix    = {}
            qrs_per_noise = {}

            for nt in NOISE_TYPES:
                acc_matrix[nt] = {}
                noisy_means    = []

                for nl in NOISE_LEVELS:
                    vals = r[nt][nl]
                    m    = float(np.mean(vals))
                    s    = float(np.std(vals))
                    acc_matrix[nt][nl] = (m, s)
                    noisy_means.append(m)

                qrs_per_noise[nt] = compute_qrs(noiseless_mean, noisy_means)

            avg_qrs = float(np.mean(list(qrs_per_noise.values())))

            summary[ds][cfg] = {
                'noiseless_mean': noiseless_mean,
                'noiseless_std':  noiseless_std,
                'qrs_per_noise':  qrs_per_noise,
                'avg_qrs':        avg_qrs,
                'acc_matrix':     acc_matrix,
            }

    return summary


# ─────────────────────────────────────────────────────────────
# PRINT RESULTS TABLE TO CONSOLE
# ─────────────────────────────────────────────────────────────

def print_results(summary):
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

    print("\n\n" + "="*65)
    print("  TABLE 1: Noiseless Accuracy (Mean ± Std, 5 seeds)")
    print("="*65)
    header = f"{'Config':<28} {'Iris':>12} {'Breast Cancer':>16} {'Wine':>10}"
    print(header)
    print("-"*65)
    for cfg in CONFIGS:
        row = f"{cfg:<28}"
        for ds in DATASETS:
            m = summary[ds][cfg]['noiseless_mean']
            s = summary[ds][cfg]['noiseless_std']
            row += f"  {m:.3f}±{s:.3f}".rjust(13)
        print(row)

    print("\n\n" + "="*65)
    print("  TABLE 2: QRS per Noise Type")
    print("="*65)
    for ds in DATASETS:
        print(f"\n  Dataset: {DS_LABELS[ds]}")
        print(f"  {'Config':<28} {'Depol':>8} {'Bit-Flip':>10} "
              f"{'Phase':>8} {'Avg QRS':>10}")
        print("  " + "-"*65)
        for cfg in CONFIGS:
            q   = summary[ds][cfg]['qrs_per_noise']
            avg = summary[ds][cfg]['avg_qrs']
            print(f"  {cfg:<28} "
                  f"{q['depolarizing']:>8.4f} "
                  f"{q['bit_flip']:>10.4f} "
                  f"{q['phase_damping']:>8.4f} "
                  f"{avg:>10.4f}")

    print("\n\n" + "="*65)
    print("  TABLE 3: Best Config per Noise Type per Dataset")
    print("="*65)
    for ds in DATASETS:
        print(f"\n  {DS_LABELS[ds]}:")
        for nt in NOISE_TYPES:
            best_cfg = max(
                CONFIGS.keys(),
                key=lambda c: summary[ds][c]['qrs_per_noise'][nt]
            )
            best_val = summary[ds][best_cfg]['qrs_per_noise'][nt]
            print(f"    {NT_LABELS[nt]:<22}: {best_cfg}  (QRS = {best_val:.4f})")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':

    print("Starting experiments...")
    print(f"  Datasets      : {DATASETS}")
    print(f"  Configs       : {list(CONFIGS.keys())}")
    print(f"  Noise types   : {NOISE_TYPES}")
    print(f"  Noise levels  : {NOISE_LEVELS}")
    print(f"  Seeds         : {SEEDS}")
    print(f"  Total runs    : "
          f"{len(DATASETS)*len(CONFIGS)*len(NOISE_TYPES)*len(NOISE_LEVELS)*len(SEEDS)}"
          f" (+ {len(DATASETS)*len(CONFIGS)*len(SEEDS)} noiseless)")

    # Run all experiments
    all_results = run_experiments()

    # Compute summary statistics
    summary = summarise(all_results)

    # Print tables to console
    print_results(summary)

    # Save everything to disk for plot_results.py
    save_data = {
        'all_results': all_results,
        'summary':     summary,
        'DATASETS':    DATASETS,
        'NOISE_TYPES': NOISE_TYPES,
        'NOISE_LEVELS':NOISE_LEVELS,
        'SEEDS':       SEEDS,
        'CONFIGS':     CONFIGS,
    }
    with open('results.pkl', 'wb') as f:
        pickle.dump(save_data, f)

    print("\n\nResults saved to results.pkl")
    print("Now run:  python plot_results.py")
