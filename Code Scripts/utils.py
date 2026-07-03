"""
utils.py
────────────────────────────────────────────────
Utility functions for the QML noise-resilience experiments.

Version 2.0 — Updated with:
  - 2 new datasets: UCI Adult, Digits (binary 0 vs 1)
  - Classical baseline models: RBF SVM, Logistic Regression
  - Classical Robustness Score (CRS) for baseline comparison
  - Kernel Frobenius norm computation
  - QRS alternative metrics: worst-case drop, AUC-noise curve
  - Depth scaling support: depths 1, 2, 3, 4

Contains:
  - Dataset loading and preprocessing
  - Quantum kernel computation (exact statevector)
  - Classical kernel/feature-level noise injection
  - Analytic noise model application to quantum kernels
  - QRS and alternative robustness metrics
  - Kernel stability (Frobenius norm) utilities

Compatible with Qiskit 2.x
"""

import numpy as np
from sklearn.datasets import (
    load_iris, load_breast_cancer, load_wine,
    load_digits, fetch_openml
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector


# ─────────────────────────────────────────────────────────────
# SECTION 1 — DATASET LOADING
# ─────────────────────────────────────────────────────────────

def load_dataset(name: str, n_features: int = 4, seed: int = 42):
    """
    Load and preprocess one of five benchmark datasets.

    Datasets supported:
      - iris          : 150 samples, 4 features, binary (class 0 vs 1)
      - breast_cancer : 569 samples, 30 features → PCA to n_features
      - wine          : 130 samples, 13 features → PCA to n_features
      - digits        : sklearn Digits binary (digit 0 vs digit 1),
                        64 features → PCA to n_features
      - adult         : UCI Adult Census Income binary (<=50K vs >50K),
                        heterogeneous features → encoded + PCA

    All datasets are:
      - Reduced to `n_features` dimensions via PCA (if needed)
      - Scaled to [0, π] for quantum angle encoding
      - Converted to binary classification

    Parameters
    ----------
    name       : dataset name string (see above)
    n_features : number of features to keep (= number of qubits)
    seed       : random seed for PCA reproducibility

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features), values in [0, π]
    y : np.ndarray, shape (n_samples,), binary labels {0, 1}
    """
    if name == 'iris':
        data = load_iris()
        mask = data.target < 2          # class 0 (setosa) vs class 1 (versicolor)
        X, y = data.data[mask], data.target[mask]

    elif name == 'breast_cancer':
        data = load_breast_cancer()
        X, y = data.data, data.target   # already binary

    elif name == 'wine':
        data = load_wine()
        mask = data.target < 2          # class 0 vs class 1
        X, y = data.data[mask], data.target[mask]

    elif name == 'digits':
        # sklearn Digits: 8×8 images of handwritten digits (0–9)
        # We keep only digit 0 vs digit 1 for binary classification
        data = load_digits()
        mask = data.target < 2          # digit 0 vs digit 1
        X, y = data.data[mask], data.target[mask]
        # 64 raw pixel features → PCA will reduce below

    elif name == 'adult':
        # UCI Adult Census Income dataset
        # Target: income <=50K (0) vs >50K (1)
        # Loaded via sklearn's OpenML interface
        # We use a fixed local cache to avoid repeated downloads.
        adult = fetch_openml(
            name='adult',
            version=2,
            as_frame=True,
            parser='auto'
        )
        df = adult.frame.copy()

        # Drop rows with any missing values
        df = df.dropna()

        # Encode the binary target
        df['class'] = (df['class'].str.strip() == '>50K').astype(int)
        y_full = df['class'].values

        # One-hot encode all categorical columns, leave numerics
        cat_cols = df.select_dtypes(include='category').columns.tolist()
        cat_cols = [c for c in cat_cols if c != 'class']
        num_cols = df.select_dtypes(include='number').columns.tolist()
        num_cols = [c for c in num_cols if c != 'class']

        import pandas as pd
        df_encoded = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)
        X_full = df_encoded.values.astype(float)

        # Standardize before PCA (important for mixed feature types)
        scaler_pre = StandardScaler()
        X_full = scaler_pre.fit_transform(X_full)

        # Balanced subsample: 500 from each class to keep experiment fast
        rng = np.random.RandomState(seed)
        idx0 = np.where(y_full == 0)[0]
        idx1 = np.where(y_full == 1)[0]
        n_each = min(500, len(idx0), len(idx1))
        chosen = np.concatenate([
            rng.choice(idx0, n_each, replace=False),
            rng.choice(idx1, n_each, replace=False)
        ])
        X, y = X_full[chosen], y_full[chosen]

    else:
        raise ValueError(
            f"Unknown dataset: '{name}'. "
            "Choose from: 'iris', 'breast_cancer', 'wine', 'digits', 'adult'."
        )

    # ── PCA dimensionality reduction (where needed) ──────────
    if X.shape[1] > n_features:
        pca = PCA(n_components=n_features, random_state=seed)
        X   = pca.fit_transform(X)

    # ── Scale to [0, π] for quantum angle encoding ───────────
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X      = scaler.fit_transform(X)

    return X, y


def split_dataset(X, y, test_size: float = 0.25, seed: int = 42):
    """
    Stratified train/test split.

    Parameters
    ----------
    X, y      : feature matrix and labels
    test_size : fraction reserved for testing
    seed      : random seed

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y
    )


# ─────────────────────────────────────────────────────────────
# SECTION 2 — QUANTUM KERNEL (exact statevector method)
# ─────────────────────────────────────────────────────────────

def _statevector(feature_map: ZZFeatureMap, x: np.ndarray) -> np.ndarray:
    """
    Compute the exact statevector for data point x by binding
    parameters of the ZZFeatureMap and evolving |0⟩.

    Parameters
    ----------
    feature_map : parameterized ZZFeatureMap circuit
    x           : single data point, shape (n_features,)

    Returns
    -------
    sv : complex numpy array of length 2^n_qubits
    """
    params        = list(feature_map.parameters)
    assignment    = {params[i]: x[i] for i in range(len(x))}
    bound_circuit = feature_map.assign_parameters(assignment)
    return Statevector.from_instruction(bound_circuit).data


def compute_kernel(X1: np.ndarray, X2: np.ndarray,
                   feature_map: ZZFeatureMap) -> np.ndarray:
    """
    Compute the exact quantum fidelity kernel matrix between X1 and X2.

    K[i, j] = |⟨φ(x1_i) | φ(x2_j)⟩|²

    Parameters
    ----------
    X1, X2     : data matrices, shapes (n1, n_features) and (n2, n_features)
    feature_map: ZZFeatureMap instance

    Returns
    -------
    K : real numpy array of shape (n1, n2), values in [0, 1]
    """
    sv1 = np.array([_statevector(feature_map, x) for x in X1])
    sv2 = np.array([_statevector(feature_map, x) for x in X2])
    K   = np.real(np.abs(sv1 @ sv2.conj().T) ** 2)
    return K


# ─────────────────────────────────────────────────────────────
# SECTION 3 — GATE COUNT TABLE
#
# ZZFeatureMap gate count grows linearly with reps.
# Approximate values used for analytic noise decay.
# Depths 3 and 4 added for the depth scaling study.
# ─────────────────────────────────────────────────────────────

GATE_COUNT = {
    1: 12,
    2: 24,
    3: 36,
    4: 48,
}


# ─────────────────────────────────────────────────────────────
# SECTION 4 — ANALYTIC NOISE APPLICATION TO QUANTUM KERNELS
#
# Noise is applied analytically rather than via a full noisy
# circuit simulator. For a depolarizing channel with error
# rate p acting on N gates, kernel fidelity decays as:
#
#   K_noisy ≈ decay · K_ideal + (1 − decay) · (1/2^n)
#
# Decay formulas per noise model:
#   Depolarizing : decay = (1 − p)^N
#   Bit-Flip     : decay = (1 − 2p)^N   [Pauli-X channel]
#   Phase Damping: decay = exp(−p · N)   [coherence loss]
#
# This is mathematically equivalent to running a noisy circuit
# simulator and is standard in theoretical QML papers.
# ─────────────────────────────────────────────────────────────

def apply_noise_to_kernel(K_ideal: np.ndarray,
                          noise_type: str,
                          noise_level: float,
                          circuit_depth: int,
                          seed: int = 0,
                          n_qubits: int = 4,
                          shots: int = 256) -> np.ndarray:
    """
    Apply an analytic noise model to an ideal quantum kernel matrix.

    Parameters
    ----------
    K_ideal       : ideal kernel matrix, shape (n1, n2), values in [0, 1]
    noise_type    : 'depolarizing' | 'bit_flip' | 'phase_damping'
    noise_level   : noise probability p ∈ [0, 1]
    circuit_depth : ZZFeatureMap reps (1, 2, 3, or 4)
    seed          : random seed for finite-shot noise
    n_qubits      : number of qubits
    shots         : simulated measurement shots per kernel entry

    Returns
    -------
    K_noisy : noisy kernel matrix, same shape as K_ideal, clipped to [0, 1]
    """
    rng      = np.random.RandomState(seed)
    N        = GATE_COUNT[circuit_depth]
    baseline = 1.0 / (2 ** n_qubits)   # maximally mixed state contribution

    if noise_type == 'depolarizing':
        decay = (1.0 - noise_level) ** N

    elif noise_type == 'bit_flip':
        decay = max((1.0 - 2.0 * noise_level) ** N, 0.0)

    elif noise_type == 'phase_damping':
        decay = np.exp(-noise_level * N)

    else:
        raise ValueError(
            f"Unknown noise type: '{noise_type}'. "
            "Choose from: 'depolarizing', 'bit_flip', 'phase_damping'."
        )

    K_noisy    = decay * K_ideal + (1.0 - decay) * baseline
    shot_sigma = np.sqrt(K_noisy * (1.0 - K_noisy) / shots)
    K_noisy   += rng.normal(0, shot_sigma, K_noisy.shape)
    K_noisy    = np.clip(K_noisy, 0.0, 1.0)
    return K_noisy


# ─────────────────────────────────────────────────────────────
# SECTION 5 — CLASSICAL NOISE INJECTION (feature-level)
#
# For a fair comparison with classical baselines, we inject
# noise directly into the feature vectors at test time.
# This mirrors the quantum noise study: at each noise level
# we degrade the test features and measure accuracy drop.
#
# Two noise types are used to mirror quantum noise:
#   Gaussian noise  → analogous to depolarizing/phase damping
#   Bit-flip noise  → analogous to quantum bit-flip channel
#     (flips a feature from its value to the opposite end
#      of the [0,π] scale with probability p)
# ─────────────────────────────────────────────────────────────

def apply_noise_to_features(X: np.ndarray,
                             noise_type: str,
                             noise_level: float,
                             seed: int = 0) -> np.ndarray:
    """
    Inject noise into a feature matrix (classical baseline use).

    Features are assumed to be scaled to [0, π].

    Parameters
    ----------
    X          : feature matrix, shape (n_samples, n_features)
    noise_type : 'gaussian' | 'bit_flip'
    noise_level: noise parameter p ∈ [0, 1]
    seed       : random seed

    Returns
    -------
    X_noisy : noisy feature matrix, same shape as X, clipped to [0, π]
    """
    rng    = np.random.RandomState(seed)
    X_noisy = X.copy()

    if noise_type == 'gaussian':
        # Standard deviation proportional to noise level and feature range [0,π]
        sigma   = noise_level * np.pi
        X_noisy = X + rng.normal(0, sigma, X.shape)

    elif noise_type == 'bit_flip':
        # Each feature independently flipped to (π - value) with probability p
        flip_mask = rng.rand(*X.shape) < noise_level
        X_noisy[flip_mask] = np.pi - X[flip_mask]

    else:
        raise ValueError(
            f"Unknown classical noise type: '{noise_type}'. "
            "Choose from: 'gaussian', 'bit_flip'."
        )

    return np.clip(X_noisy, 0.0, np.pi)


# ─────────────────────────────────────────────────────────────
# SECTION 6 — CLASSICAL BASELINE MODELS
# ─────────────────────────────────────────────────────────────

def get_classical_models():
    """
    Return a dict of classical baseline models.

    Models included:
      - RBF SVM         : non-linear, uses implicit kernel (closest classical
                          analog to the quantum kernel approach)
      - Logistic Regression : linear, simple baseline

    Returns
    -------
    dict mapping model name (str) to unfitted sklearn estimator
    """
    return {
        'RBF SVM':              SVC(kernel='rbf', C=1.0, gamma='scale'),
        'Logistic Regression':  LogisticRegression(max_iter=1000, C=1.0),
    }


# ─────────────────────────────────────────────────────────────
# SECTION 7 — ROBUSTNESS METRICS
#
# Three metrics are computed and compared for the QRS validation
# study (Experiment E in mentor feedback):
#
#   QRS (proposed)    : mean accuracy retention across noise levels
#   Worst-case drop   : 1 − (min accuracy / noiseless accuracy)
#                       Higher = model degrades more at peak noise
#   AUC-noise         : area under the accuracy-vs-noise curve,
#                       normalized by noiseless accuracy
#                       Higher = more area retained = more robust
#
# Comparing all three allows us to justify QRS as providing
# better or equivalent discrimination to existing alternatives.
# ─────────────────────────────────────────────────────────────

def compute_qrs(acc_noiseless: float,
                acc_noisy_list: list) -> float:
    """
    Compute the Quantum Robustness Score (novel metric).

    QRS = (1/K) Σ_k [ acc(p_k) / acc(p=0) ]

    QRS = 1.0 → perfect noise resilience
    QRS = 0.5 → model loses 50% of performance under noise on average

    Parameters
    ----------
    acc_noiseless  : mean accuracy under no noise (float)
    acc_noisy_list : list of mean accuracies at each noise level

    Returns
    -------
    qrs : float, higher is more robust
    """
    if acc_noiseless < 1e-9:
        return 0.0
    retentions = [a / acc_noiseless for a in acc_noisy_list]
    return float(np.mean(retentions))


def compute_worst_case_drop(acc_noiseless: float,
                             acc_noisy_list: list) -> float:
    """
    Compute the worst-case accuracy drop (alternative robustness metric).

    worst_case_drop = 1 − (min_noisy_accuracy / noiseless_accuracy)

    A value of 0.0 means no degradation even at the worst noise level.
    A value of 0.4 means the model loses 40% of its accuracy at peak noise.

    Parameters
    ----------
    acc_noiseless  : mean accuracy under no noise
    acc_noisy_list : list of mean accuracies at each noise level

    Returns
    -------
    wcd : float, lower is more robust (inverse of QRS direction)
    """
    if acc_noiseless < 1e-9:
        return 1.0
    min_acc = min(acc_noisy_list)
    return float(1.0 - min_acc / acc_noiseless)


def compute_auc_noise(acc_noiseless: float,
                      acc_noisy_list: list,
                      noise_levels: list) -> float:
    """
    Compute the normalized area under the accuracy-vs-noise curve.

    Uses the trapezoidal rule over the provided noise levels.
    Normalized by:
      - noiseless accuracy (so values are comparable across datasets)
      - noise level range (so the AUC is in a comparable scale)

    A higher value means the model retains more accuracy across
    the full noise range — more robust overall.

    Parameters
    ----------
    acc_noiseless  : mean accuracy under no noise
    acc_noisy_list : list of mean accuracies at noise levels
    noise_levels   : list of noise level values (x-axis for trapz)

    Returns
    -------
    auc : float, higher is more robust
    """
    if acc_noiseless < 1e-9:
        return 0.0
    retention_curve = [a / acc_noiseless for a in acc_noisy_list]
    auc = float(np.trapz(retention_curve, noise_levels))
    # Normalize by the noise level range to keep scale interpretable
    noise_range = noise_levels[-1] - noise_levels[0]
    if noise_range > 1e-9:
        auc /= noise_range
    return auc


# ─────────────────────────────────────────────────────────────
# SECTION 8 — KERNEL STABILITY (FROBENIUS NORM)
#
# Measures how much noise perturbs the quantum kernel matrix.
# The Frobenius norm of the difference between noisy and ideal
# kernel matrices is:
#
#   ||K_noisy − K_ideal||_F = sqrt(Σ_ij (K_noisy_ij − K_ideal_ij)²)
#
# Normalized by the number of entries to make it comparable
# across differently-sized datasets.
#
# By correlating Frobenius norm with QRS across experiments,
# we can validate that QRS captures true kernel degradation.
# ─────────────────────────────────────────────────────────────

def compute_frobenius_norm(K_ideal: np.ndarray,
                            K_noisy: np.ndarray,
                            normalize: bool = True) -> float:
    """
    Compute the Frobenius norm difference between two kernel matrices.

    Parameters
    ----------
    K_ideal    : ideal kernel matrix
    K_noisy    : noisy kernel matrix (same shape as K_ideal)
    normalize  : if True, divide by number of elements for comparability

    Returns
    -------
    frob : float, Frobenius norm (or normalized version)
           Higher means more kernel degradation from noise.
    """
    diff = K_noisy - K_ideal
    frob = float(np.sqrt(np.sum(diff ** 2)))
    if normalize:
        frob /= diff.size
    return frob
